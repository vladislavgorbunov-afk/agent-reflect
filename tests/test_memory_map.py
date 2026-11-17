from __future__ import annotations

from datetime import date, timedelta, timezone
from pathlib import Path

import pytest

from agent_reflect import chart, i18n, memory_map
from agent_reflect.config import Config
from agent_reflect.ledger import Lesson
from agent_reflect.projects import sanitize

FIRST, LAST = date(2026, 11, 10), date(2026, 11, 16)
IN_WEEK, BEFORE = "2026-11-12T10:00:00Z", "2026-11-01T10:00:00Z"


@pytest.fixture
def cfg(tmp_path):
    return Config.under(tmp_path)


@pytest.fixture
def folder(cfg) -> Path:
    path = cfg.projects_dir / sanitize(str(cfg.repos["webapp"])) / "memory"
    path.mkdir(parents=True)
    return path


def lesson(cfg, folder, lesson_id: str, status: str, **kw) -> Lesson:
    base = dict(id=lesson_id, project_root=str(cfg.repos["webapp"]), memory_dir=str(folder),
                file=f"auto_{lesson_id}.md", title="t", hook="h", cue="c", lesson="l", why="w", confidence="high",
                repeats=1, evidence=[], status=status, sha256="", created_at=BEFORE, updated_at=BEFORE)
    base.update(kw)
    return Lesson(**base)


def write(folder: Path, index: str, *files: str) -> None:
    (folder / "MEMORY.md").write_text(index, encoding="utf-8")
    for name in files:
        (folder / name).write_text("x", encoding="utf-8")


INDEX = """# Memory Index

## Mine
- [Deploy](deploy.md) — how we deploy
- [Old](./gone.md) — its file is gone
- [Notes](../notes/x.md) — a note outside the folder
- [auto_eeee5555.md](auto_eeee5555.md) — a lesson the user edited

## Lessons
- [auto_aaaa1111.md](auto_aaaa1111.md) — live, with a hook
- [auto_bbbb2222.md](auto_bbbb2222.md) — live
"""


@pytest.fixture
def lessons(cfg, folder) -> dict[str, Lesson]:
    write(folder, INDEX, "deploy.md", "orphan.md", "auto_aaaa1111.md", "auto_bbbb2222.md", "auto_cccc3333.md",
          "auto_dddd4444.md", "auto_eeee5555.md")
    found = [lesson(cfg, folder, "aaaa1111", "live", trigger="git push", live_since=IN_WEEK),
             lesson(cfg, folder, "bbbb2222", "live", live_since=BEFORE),
             lesson(cfg, folder, "cccc3333", "staged", created_at=IN_WEEK),
             lesson(cfg, folder, "dddd4444", "retired"),
             lesson(cfg, folder, "eeee5555", "user_owned"),
             lesson(cfg, folder, "ffff6666", "promoted", updated_at=IN_WEEK),
             lesson(cfg, folder, "gggg7777", "promoted", updated_at=BEFORE),
             lesson(cfg, folder, "hhhh8888", "removed", updated_at=IN_WEEK),
             lesson(cfg, folder, "iiii9999", "rejected")]
    return {item.id: item for item in found}


def test_another_folders_lessons_are_not_counted(cfg, folder, lessons):
    elsewhere = {**lessons, "jjjj0000": lesson(cfg, Path("/elsewhere/memory"), "jjjj0000", "live", trigger="x")}
    assert picture(cfg, folder, elsewhere) == picture(cfg, folder, lessons)


def picture(cfg, folder, lessons, reads=None) -> memory_map.Picture:
    return memory_map.picture(cfg, lessons, str(folder), "webapp", reads, FIRST, LAST, timezone.utc)


def test_index_lines_are_told_apart_by_the_file_they_link_to(cfg, folder, lessons):
    p = picture(cfg, folder, lessons)
    # deploy, gone, the note outside and the edited lesson are the user's; headings and prose are no lines
    assert (p.user_lines, p.lesson_lines) == (4, 2)


def test_files_of_the_user_and_the_links_that_miss(cfg, folder, lessons):
    p = picture(cfg, folder, lessons)
    # deploy.md, orphan.md and the edited lesson's file; the waiting and retired lessons' files are agent-reflect's
    assert p.user_files == 3
    assert p.orphans == 1  # orphan.md: no line links to it
    assert p.dead == 1  # gone.md; the note outside the folder is no file of it


def test_lessons_counted_by_status_and_hook(cfg, folder, lessons):
    p = picture(cfg, folder, lessons)
    assert (p.live, p.hooked, p.waiting, p.retired) == (2, 1, 1, 1)
    assert (p.promoted, p.removed, p.rejected, p.owned) == (2, 1, 1, 1)


def test_a_path_pattern_counts_as_a_hook_too(cfg, folder, lessons):
    lessons["bbbb2222"] = lesson(cfg, folder, "bbbb2222", "live", path_trigger=r"\.env$")
    assert picture(cfg, folder, lessons).hooked == 2


def test_this_week_counts_only_the_steps_taken_on_the_digest_days(cfg, folder, lessons):
    p = picture(cfg, folder, lessons)
    assert (p.new_this_week, p.live_this_week, p.promoted_this_week) == (1, 1, 1)


def test_the_week_ends_with_its_last_local_day(cfg, folder, lessons):
    lessons["cccc3333"] = lesson(cfg, folder, "cccc3333", "staged", created_at="2026-11-16T23:30:00Z")
    assert picture(cfg, folder, lessons).new_this_week == 1
    plus_one = timezone(timedelta(hours=1))  # 00:30 on the 17th there: past the digest's days
    assert memory_map.picture(cfg, lessons, str(folder), "webapp", None, FIRST, LAST, plus_one).new_this_week == 0


def test_unread_files_come_from_the_scan_of_this_folder(cfg, folder, lessons):
    reads = {"days": 30, "locations": [
        {"name": "notes", "files": [{"name": "deploy.md", "reads": 0}, {"name": "x.md", "reads": 0}]},
        {"name": "webapp", "files": [{"name": "deploy.md", "reads": 2}, {"name": "orphan.md", "reads": 0}]},
        {"name": "toolbox", "files": [{"name": "a.md", "reads": 0}, {"name": "b.md", "reads": 0}]}]}
    p = picture(cfg, folder, lessons, reads)
    assert (p.unread, p.days) == (1, 30)


def test_without_the_scan_unread_is_unknown(cfg, folder, lessons):
    assert picture(cfg, folder, lessons).unread is None


def test_block_is_empty_with_no_live_or_waiting_lesson(cfg, folder, lessons):
    gone = {i: l for i, l in lessons.items() if l.status not in ("live", "staged")}
    assert memory_map.block(cfg, gone, None, FIRST, LAST, timezone.utc) == []


def test_block_draws_the_map_the_path_and_the_gauges(cfg, folder, lessons):
    reads = {"days": 30, "locations": [{"name": "webapp", "files": [{"name": "deploy.md", "reads": 0}]}]}
    block = memory_map.block(cfg, lessons, reads, FIRST, LAST, timezone.utc)
    assert block[:3] == ["## 🧠 Память", "", "```text"] and block[-1] == "```"
    assert block[3:37] == [
        "🧠 webapp",
        "┌─ MEMORY.md · каждая сессия загружает его целиком ───────┐",
        "│ 4 твоих строки                    2 строки уроков из 40 │",
        "└───┬─────────────────────────────────┬───────────────────┘",
        "    │                                 │",
        "    ▼                                 ▼",
        "  3 твоих файла                     2 auto_*.md в индексе",
        "  1 не открывали за 30 дней         1 с шаблоном для хука",
        "  1 без строки: агенту не видны     1 без шаблона",
        "  1 ссылка на файл, которого нет",
        "                                    + 1 auto_*.md без строки: ждёт повтора",
        "                                    + 1 auto_*.md без строки: снят",
        "",
        "путь урока                                          сейчас  за неделю",
        "  ночной разбор · повторяющиеся поправки недели",
        "          │ новый урок",
        "          ▼",
        "  ┌──────────────┐",
        "  │ ждёт повтора │  файл без строки в MEMORY.md          1         +1",
        "  └───────┬──────┘",
        "          │ уверенность high, второй случай или твой approve,",
        "          │ и в MEMORY.md есть место",
        "          ▼",
        "  ┌──────────────┐",
        "  │ в индексе    │  строка в MEMORY.md                   2         +1",
        "  └───────┬──────┘",
        "          │ подтверждён в 2 сессиях за 30 дней или нарушен 2 раза",
        "          ▼",
        "  ┌──────────────┐",
        "  │ в скилле     │  правило в скилле или CLAUDE.md       2         +1",
        "  └──────────────┘",
        "  в стороне: снят 1 · убран чисткой 1 · отклонён тобой 1 · стал твоим 1",
        "  обратно в «ждёт»: MEMORY.md переполнен · снятый урок повторился",
        "",
    ]
    assert block[37:-1] == chart.memory_gauges(cfg, lessons, detailed=True)


def _arrows(line: str) -> list[int]:
    return [i for i, ch in enumerate(line) if ch in "┬│▼"]


def test_every_arrow_leaves_its_box_and_lands_in_the_same_column(cfg, folder, lessons):
    block = memory_map.block(cfg, lessons, None, FIRST, LAST, timezone.utc)
    bottoms = [i for i, line in enumerate(block) if line.lstrip().startswith("└") and "┬" in line]
    assert len(bottoms) == 3  # MEMORY.md and the first two steps of the path
    for i in bottoms:
        stem = _arrows(block[i + 1])[:len(_arrows(block[i]))]
        assert _arrows(block[i]) == stem
        heads = [j for j in range(i + 1, i + 4) if "▼" in block[j]]
        assert [k for k, ch in enumerate(block[heads[0]]) if ch == "▼"] == _arrows(block[i])


def test_the_box_is_closed_on_every_row(cfg, folder, lessons):
    block = memory_map.block(cfg, lessons, None, FIRST, LAST, timezone.utc)
    top = next(i for i, line in enumerate(block) if line.startswith("┌─ MEMORY.md"))
    assert {chart.columns(block[j]) for j in range(top, top + 3)} == {chart.columns(block[top])}
    assert block[top + 1].endswith(" │")


def test_projects_follow_one_another_with_a_blank_line(cfg, folder, lessons):
    other = cfg.projects_dir / sanitize(str(cfg.repos["toolbox"])) / "memory"
    other.mkdir(parents=True)
    write(other, "# Memory Index\n")
    lessons["kkkk1111"] = lesson(cfg, other, "kkkk1111", "staged", project_root=str(cfg.repos["toolbox"]))
    block = memory_map.block(cfg, lessons, None, FIRST, LAST, timezone.utc)
    heads = [i for i, line in enumerate(block) if line.startswith("🧠 ")]
    assert [block[i] for i in heads] == ["🧠 toolbox", "🧠 webapp"]
    assert block[heads[1] - 1] == ""


def test_an_unreadable_index_is_said_and_the_rest_is_drawn(cfg, folder, lessons):
    (folder / "MEMORY.md").write_bytes(b"\xff\xfe broken")
    block = memory_map.block(cfg, lessons, None, FIRST, LAST, timezone.utc)
    assert "   MEMORY.md не читается" in block
    assert block[-1] == "```"


def _readme_sample() -> list[str]:
    text = (Path(__file__).parent.parent / "README.md").read_text(encoding="utf-8")
    section = text.split("### In the weekly digest", 1)[1]
    return section.split("```text\n", 1)[1].split("```", 1)[0].rstrip("\n").split("\n")


def test_the_readme_shows_what_the_code_draws(cfg, folder):
    """The README's sample is the map of 18 notes (5 unread), 6 lessons in the index (2 with a pattern, 1 new this
    week), 2 new waiting, 3 moved into skills (1 this week), 1 tidied away and 1 rejected."""
    lines = ["# Memory Index", ""]
    for i in range(18):
        write(folder, "", f"note_{i}.md")
        lines.append(f"- [Note {i}](note_{i}.md) — x")
    found: dict[str, Lesson] = {}

    def add(i: int, status: str, **kw) -> None:
        item = lesson(cfg, folder, f"{i:08x}", status, **kw)
        found[item.id] = item
        if status in ("live", "staged"):
            (folder / item.file).write_text("x", encoding="utf-8")
        if status == "live":
            lines.append(f"- [{item.file}]({item.file}) — h")

    for i in range(6):
        add(i, "live", trigger="x" if i < 2 else "", **({"live_since": IN_WEEK, "created_at": IN_WEEK} if i == 5 else {}))
    for i, status, kw in ((10, "staged", {"created_at": IN_WEEK}), (11, "staged", {"created_at": IN_WEEK}),
                          (12, "promoted", {"updated_at": IN_WEEK}), (13, "promoted", {}), (14, "promoted", {}),
                          (15, "rejected", {}), (16, "removed", {})):
        add(i, status, **kw)
    (folder / "MEMORY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    reads = {"days": 30, "locations": [{"name": "webapp", "files": [
        {"name": f"note_{i}.md", "reads": 0 if i < 5 else 3} for i in range(18)]}]}
    i18n.use("en")
    try:
        block = memory_map.block(cfg, found, reads, FIRST, LAST, timezone.utc)
    finally:
        i18n.use("ru")
    assert _readme_sample() == block[3:-1]
