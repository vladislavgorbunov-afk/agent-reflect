from __future__ import annotations

from pathlib import Path

import pytest

from agent_reflect.cursor_index import regenerate

CURSOR_HEADER = (
    "---\n"
    "description: Cross-session memory carried over from Claude Code — facts and preferences\n"
    "alwaysApply: true\n"
    "---\n"
    "\n"
    "# Carried-over memory\n"
    "\n"
    "These are facts established in earlier sessions. Read the individual file under "
    "`.cursor/memory/` before relying on an entry.\n"
    "\n"
)

STALE_MDC = (
    CURSOR_HEADER
    + "- [user_language.md](.cursor/memory/user_language.md) — предпочтительный язык общения: русский\n"
    + "- [stale.md](.cursor/memory/stale.md) — больше нет в MEMORY.md\n"
)


def _project(tmp_path: Path) -> Path:
    project = tmp_path / "project"
    rules = project / ".cursor" / "rules"
    rules.mkdir(parents=True)
    (rules / "claude-memory.mdc").write_text(STALE_MDC, encoding="utf-8")
    return project


def _memory(tmp_path: Path, index: bytes | None) -> Path:
    memory = tmp_path / "memory"
    memory.mkdir()
    if index is not None:
        (memory / "MEMORY.md").write_bytes(index)
    return memory


def _mdc(project: Path) -> Path:
    return project / ".cursor" / "rules" / "claude-memory.mdc"


def test_regenerate_rewrites_links_keeps_header(tmp_path):
    project = _project(tmp_path)
    memory = _memory(
        tmp_path,
        "# Memory Index\n\n- [a.md](a.md) — x\n- [user_language.md](user_language.md) — предпочтительный язык общения: русский\n".encode(),
    )

    assert regenerate(project, memory) is True

    text = _mdc(project).read_text(encoding="utf-8")
    assert text == (
        CURSOR_HEADER
        + "- [a.md](.cursor/memory/a.md) — x\n"
        + "- [user_language.md](.cursor/memory/user_language.md) — предпочтительный язык общения: русский\n"
    )
    assert "stale.md" not in text


def test_missing_mdc_is_noop(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    memory = _memory(tmp_path, "- [a.md](a.md) — x\n".encode())

    assert regenerate(project, memory) is False
    assert not (project / ".cursor").exists()


def test_index_heading_not_copied(tmp_path):
    project = _project(tmp_path)
    memory = _memory(tmp_path, "# Memory Index\n\n- [a.md](a.md) — x\n".encode())

    assert regenerate(project, memory) is True

    text = _mdc(project).read_text(encoding="utf-8")
    assert "# Memory Index" not in text
    assert text == CURSOR_HEADER + "- [a.md](.cursor/memory/a.md) — x\n"


@pytest.mark.parametrize(
    "index",
    [None, "- [a.md](a.md) — x\n".encode("utf-8") + b"\xff\xfe\n"],
    ids=["missing-memory-md", "non-utf8-memory-md"],
)
def test_unreadable_index_leaves_mdc_untouched(tmp_path, index):
    project = _project(tmp_path)
    before = _mdc(project).read_bytes()
    memory = _memory(tmp_path, index)

    assert regenerate(project, memory) is False
    assert _mdc(project).read_bytes() == before
    assert (memory / "MEMORY.md").exists() == (index is not None)


def test_sync_leaves_the_mdc_alone_while_cursor_sync_is_off(tmp_path):
    import dataclasses

    from agent_reflect.config import Config
    from agent_reflect.cursor_index import sync

    project = _project(tmp_path)
    memory = _memory(tmp_path, "# Memory Index\n\n- [a.md](a.md) — x\n".encode())
    before = _mdc(project).read_bytes()
    cfg = dataclasses.replace(Config.under(tmp_path), cursor_sync=False)

    assert sync(cfg, project, memory) is False
    assert _mdc(project).read_bytes() == before


def test_sync_regenerates_while_cursor_sync_is_on(tmp_path):
    from agent_reflect.config import Config
    from agent_reflect.cursor_index import sync

    project = _project(tmp_path)
    memory = _memory(tmp_path, "# Memory Index\n\n- [a.md](a.md) — x\n".encode())

    assert sync(Config.under(tmp_path), project, memory) is True
    assert _mdc(project).read_text(encoding="utf-8") == CURSOR_HEADER + "- [a.md](.cursor/memory/a.md) — x\n"
