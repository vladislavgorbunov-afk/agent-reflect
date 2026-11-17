from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

from agent_reflect.config import Config
from agent_reflect.skills import (
    EditError,
    Target,
    apply_edit_text,
    apply_personal,
    backup,
    list_backups,
    registry,
    rollback,
)

SKILL = "---\nname: {n}\ndescription: Does {n} things\n---\n\n# {n}\n\nUse the old rule here.\n"


def _skill(root: Path, name: str, text: str | None = None) -> Path:
    path = root / name / "SKILL.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(SKILL.format(n=name) if text is None else text, encoding="utf-8")
    return path


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-C", str(repo), "-c", "user.email=t@t", "-c", "user.name=t", *args],
        check=True,
        capture_output=True,
    )


def _repo(path: Path, tracked: list[str]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    _git(path, "init", "-q")
    for rel in tracked:
        f = path / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(SKILL.format(n=f.parent.name), encoding="utf-8")
    if tracked:
        _git(path, "add", "-A")
        _git(path, "commit", "-q", "-m", "init")


@pytest.fixture
def cfg(tmp_path: Path) -> Config:
    return Config.under(tmp_path)


def _personal(cfg: Config, name: str = "alpha") -> Target:
    return next(t for t in registry(cfg) if t.key == f"personal:{name}")


def test_registry_personal_excludes_synced_and_symlinks(cfg: Config, tmp_path: Path) -> None:
    _skill(cfg.personal_skills_dir, "alpha")
    _skill(cfg.personal_skills_dir, "synced")
    real = _skill(tmp_path / "elsewhere", "linked")
    (cfg.personal_skills_dir / "linked").symlink_to(real.parent)
    (cfg.personal_skills_dir / "stray-file.md").write_text("x", encoding="utf-8")
    cfg.repos.clear()

    targets = registry(cfg)

    assert [t.key for t in targets] == ["personal:alpha"]
    t = targets[0]
    assert t.kind == "personal"
    assert t.path == cfg.personal_skills_dir / "alpha" / "SKILL.md"
    assert t.repo is None and t.rel_path is None
    assert t.description == "Does alpha things"


def test_registry_repo_uses_tracked_files_only(cfg: Config) -> None:
    webapp = cfg.repos["webapp"]
    _repo(webapp, [".claude/skills/a/SKILL.md", "CLAUDE.md"])
    untracked = webapp / ".claude/skills/b/SKILL.md"
    untracked.parent.mkdir(parents=True)
    untracked.write_text(SKILL.format(n="b"), encoding="utf-8")
    qa = cfg.repos["toolbox"]
    _repo(qa, ["plugins/qa/skills/qa-suite/SKILL.md", "README.md"])

    by_key = {t.key: t for t in registry(cfg)}

    assert set(by_key) == {
        "repo:webapp:a",
        "repo:webapp:CLAUDE.md",
        "repo:toolbox:qa-suite",
    }
    a = by_key["repo:webapp:a"]
    assert (a.kind, a.repo, a.rel_path) == ("repo", "webapp", ".claude/skills/a/SKILL.md")
    assert a.path == webapp / ".claude/skills/a/SKILL.md"
    assert a.description == "Does a things"
    rules = by_key["repo:webapp:CLAUDE.md"]
    assert rules.description == "project rules"
    assert rules.rel_path == "CLAUDE.md"
    assert by_key["repo:toolbox:qa-suite"].rel_path == "plugins/qa/skills/qa-suite/SKILL.md"


def test_registry_skips_tracked_symlinked_skill(cfg: Config, tmp_path: Path) -> None:
    webapp = cfg.repos["webapp"]
    webapp.mkdir(parents=True)
    _git(webapp, "init", "-q")
    real = _skill(tmp_path / "elsewhere", "linked")
    (webapp / ".claude/skills").mkdir(parents=True)
    (webapp / ".claude/skills/linked").symlink_to(real.parent)
    _git(webapp, "add", "-A")
    cfg.repos.pop("toolbox")

    assert registry(cfg) == []


def test_registry_never_raises_on_missing_repo_or_git_failure(cfg: Config) -> None:
    _skill(cfg.personal_skills_dir, "alpha")
    # webapp dir does not exist; toolbox exists but is not a git repo
    cfg.repos["toolbox"].mkdir(parents=True)

    def boom(*a, **k):
        raise OSError("no git")

    assert [t.key for t in registry(cfg)] == ["personal:alpha"]
    assert [t.key for t in registry(cfg, run=boom)] == ["personal:alpha"]


def test_cursor_copy_resolution(cfg: Config, tmp_path: Path) -> None:
    _skill(cfg.personal_skills_dir, "alpha")
    _skill(cfg.cursor_skills_dir, "alpha")
    _skill(cfg.personal_skills_dir, "beta")
    real = _skill(tmp_path / "elsewhere", "beta")
    (cfg.cursor_skills_dir / "beta").symlink_to(real.parent)
    _skill(cfg.personal_skills_dir, "gamma")  # no cursor copy at all
    webapp = cfg.repos["webapp"]
    _repo(webapp, [".claude/skills/a/SKILL.md", ".claude/skills/q/SKILL.md"])
    _skill(webapp / ".cursor/skills", "a")
    (webapp / ".cursor/skills/q").symlink_to(real.parent)
    cfg.repos.pop("toolbox")

    by_key = {t.key: t for t in registry(cfg)}

    assert by_key["personal:alpha"].cursor_copy == cfg.cursor_skills_dir / "alpha" / "SKILL.md"
    assert by_key["personal:beta"].cursor_copy is None
    assert by_key["personal:gamma"].cursor_copy is None
    assert by_key["repo:webapp:a"].cursor_copy == webapp / ".cursor/skills/a/SKILL.md"
    assert by_key["repo:webapp:q"].cursor_copy is None


def test_registry_cursor_copy_only_for_webapp(cfg: Config) -> None:
    qa = cfg.repos["toolbox"]
    _repo(qa, ["plugins/qa/skills/qa-suite/SKILL.md"])
    _skill(qa / ".cursor/skills", "qa-suite")
    cfg.repos.pop("webapp")

    (t,) = registry(cfg)

    assert t.cursor_copy is None


def test_description_is_lenient(cfg: Config) -> None:
    _skill(cfg.personal_skills_dir, "none", "# no frontmatter\n")
    _skill(cfg.personal_skills_dir, "quoted", '---\nname: quoted\ndescription: "Quoted one"\n---\nbody\n')
    _skill(cfg.personal_skills_dir, "crlf", "---\r\nname: crlf\r\ndescription: Win\r\n---\r\nbody\r\n")
    _skill(cfg.personal_skills_dir, "folded", "---\nname: folded\ndescription: >\n  Folded text\n---\nbody\n")
    cfg.repos.clear()

    by_key = {t.key: t.description for t in registry(cfg)}

    assert by_key["personal:none"] == ""
    assert by_key["personal:quoted"] == "Quoted one"
    assert by_key["personal:crlf"] == "Win"
    assert isinstance(by_key["personal:folded"], str)


def test_apply_edit_text_exactly_once() -> None:
    text = SKILL.format(n="x")
    out = apply_edit_text(text, "old rule", "new rule", skill=True)
    assert out == text.replace("old rule", "new rule")
    with pytest.raises(EditError):
        apply_edit_text(text, "missing", "z", skill=True)
    with pytest.raises(EditError):
        apply_edit_text(text + "old rule again\n", "old rule", "z", skill=True)
    with pytest.raises(EditError):
        apply_edit_text(text, "", "z", skill=True)


def test_apply_edit_rejects_broken_frontmatter() -> None:
    text = SKILL.format(n="x")
    with pytest.raises(EditError):
        apply_edit_text(text, "description: Does x things\n", "", skill=True)
    with pytest.raises(EditError):
        apply_edit_text(text, "name: x\n", "", skill=True)
    with pytest.raises(EditError):
        apply_edit_text(text, "---\nname", "name", skill=True)
    # a plain CLAUDE.md has no frontmatter and that is fine
    assert apply_edit_text("rules\nold\n", "old", "new", skill=False) == "rules\nnew\n"


def test_apply_edit_text_preserves_crlf() -> None:
    text = SKILL.format(n="x").replace("\n", "\r\n")
    out = apply_edit_text(text, "old rule", "new rule", skill=True)
    assert out == text.replace("old rule", "new rule")
    assert out.count("\r\n") == text.count("\r\n") and "\n" not in out.replace("\r\n", "")


def test_apply_personal_writes_both_copies_then_rollback_restores(cfg: Config) -> None:
    main = _skill(cfg.personal_skills_dir, "alpha")
    cursor = _skill(cfg.cursor_skills_dir, "alpha")
    cfg.repos.clear()
    original = main.read_bytes()
    target = _personal(cfg)

    notes = apply_personal(cfg, "run1", target, "old rule", "new rule")

    assert notes == []
    assert "new rule" in main.read_text() and "old rule" not in main.read_text()
    assert "new rule" in cursor.read_text()
    assert list_backups(cfg) == ["run1"]

    restored = rollback(cfg, "run1")

    assert sorted(restored) == sorted([main, cursor])
    assert main.read_bytes() == original
    assert cursor.read_bytes() == original


def test_apply_personal_cursor_diverged_note(cfg: Config) -> None:
    main = _skill(cfg.personal_skills_dir, "alpha")
    cursor = _skill(cfg.cursor_skills_dir, "alpha", SKILL.format(n="alpha").replace("old rule", "other"))
    cfg.repos.clear()
    cursor_before = cursor.read_bytes()

    notes = apply_personal(cfg, "run1", _personal(cfg), "old rule", "new rule")

    assert notes == [f"cursor copy diverged: {cursor}"]
    assert "new rule" in main.read_text()
    assert cursor.read_bytes() == cursor_before


def test_apply_personal_cursor_write_failure_is_a_note(cfg: Config, monkeypatch) -> None:
    from agent_reflect import skills

    main = _skill(cfg.personal_skills_dir, "alpha")
    cursor = _skill(cfg.cursor_skills_dir, "alpha")
    cfg.repos.clear()
    real = skills.write_atomic

    def flaky(path: Path, text: str) -> None:
        if path == cursor:
            raise OSError("read-only file system")
        real(path, text)

    monkeypatch.setattr(skills, "write_atomic", flaky)
    notes = apply_personal(cfg, "run1", _personal(cfg), "old rule", "new rule")

    assert "new rule" in main.read_text() and "old rule" in cursor.read_text()
    assert len(notes) == 1 and notes[0].startswith(f"cursor copy not written: {cursor}")


def test_apply_personal_failure_leaves_files_untouched(cfg: Config) -> None:
    main = _skill(cfg.personal_skills_dir, "alpha")
    cfg.repos.clear()
    before = main.read_bytes()

    with pytest.raises(EditError):
        apply_personal(cfg, "run1", _personal(cfg), "no such text", "x")

    assert main.read_bytes() == before


def test_apply_personal_refuses_repo_target(cfg: Config) -> None:
    _repo(cfg.repos["webapp"], [".claude/skills/a/SKILL.md"])
    cfg.repos.pop("toolbox")
    (t,) = registry(cfg)
    before = t.path.read_bytes()

    with pytest.raises(EditError):
        apply_personal(cfg, "run1", t, "old rule", "new rule")

    assert t.path.read_bytes() == before


def test_apply_personal_keeps_mode_and_crlf(cfg: Config) -> None:
    text = SKILL.format(n="alpha").replace("\n", "\r\n")
    main = _skill(cfg.personal_skills_dir, "alpha")
    main.write_bytes(text.encode("utf-8"))
    main.chmod(0o750)
    cfg.repos.clear()

    apply_personal(cfg, "run1", _personal(cfg), "old rule", "new rule")

    assert main.read_bytes() == text.replace("old rule", "new rule").encode("utf-8")
    assert os.stat(main).st_mode & 0o777 == 0o750
    assert [p.name for p in main.parent.iterdir()] == ["SKILL.md"]  # no temp leftovers


def test_backup_layout_and_manifest(cfg: Config) -> None:
    main = _skill(cfg.personal_skills_dir, "alpha")
    cursor = _skill(cfg.cursor_skills_dir, "alpha")
    cfg.repos.clear()

    backup(cfg, "run1", _personal(cfg))

    folder = cfg.backups_dir / "run1" / "personal__alpha"
    manifest = json.loads((folder / "manifest.json").read_text())
    assert [m["src"] for m in manifest] == [str(main), str(cursor)]
    for m in manifest:
        assert Path(m["saved"]).read_bytes() == Path(m["src"]).read_bytes()
        assert folder in Path(m["saved"]).parents


def test_second_backup_in_same_run_keeps_the_original(cfg: Config) -> None:
    main = _skill(cfg.personal_skills_dir, "alpha")
    cfg.repos.clear()
    original = main.read_bytes()
    target = _personal(cfg)

    apply_personal(cfg, "run1", target, "old rule", "mid rule")
    apply_personal(cfg, "run1", target, "mid rule", "last rule")
    rollback(cfg, "run1")

    assert main.read_bytes() == original


def test_rollback_unknown_run_and_list_backups(cfg: Config) -> None:
    assert list_backups(cfg) == []
    assert rollback(cfg, "nope") == []
    _skill(cfg.personal_skills_dir, "alpha")
    cfg.repos.clear()
    t = _personal(cfg)
    backup(cfg, "2026-10-20", t)
    backup(cfg, "2026-10-13", t)
    assert list_backups(cfg) == ["2026-10-13", "2026-10-20"]


def test_registry_git_calls_time_out_and_a_timeout_is_a_git_failure(cfg: Config) -> None:
    import subprocess

    _skill(cfg.personal_skills_dir, "alpha")
    seen = []

    def hung(argv, **kw):
        seen.append(kw.get("timeout"))
        raise subprocess.TimeoutExpired(argv, kw.get("timeout"))

    assert [t.key for t in registry(cfg, run=hung)] == ["personal:alpha"]
    assert seen == [30, 30]


def test_no_cursor_copies_while_cursor_sync_is_off(cfg: Config) -> None:
    import dataclasses

    off = dataclasses.replace(cfg, cursor_sync=False)
    _skill(off.personal_skills_dir, "alpha")
    _skill(off.cursor_skills_dir, "alpha")
    webapp = off.repos["webapp"]
    _repo(webapp, [".claude/skills/a/SKILL.md"])
    _skill(webapp / ".cursor/skills", "a")
    off.repos.pop("toolbox")

    targets = registry(off)

    assert {t.key for t in targets} >= {"personal:alpha", "repo:webapp:a"}
    assert all(t.cursor_copy is None for t in targets)
