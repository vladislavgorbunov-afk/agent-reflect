"""The weekly digest's map of memory: what each project's MEMORY.md points to, and where its lessons stand.

Read-only. The counts come from the memory folder on disk and from the ledger, so the map is the memory
as it is when the digest is written; "this week" counts the lessons whose step falls on the digest's days.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, tzinfo
from pathlib import Path

from . import chart, i18n, memory, memory_reads
from .config import Config
from .consolidate import PROMOTION_MIN_SESSIONS, PROMOTION_MIN_VIOLATIONS, PROMOTION_WINDOW_DAYS
from .i18n import t
from .ledger import Lesson, parse_ts

# a markdown link to a .md file; one with no folder in it, (name.md) or (./name.md), is a file of the memory folder
_LINK = re.compile(r"\]\(([^()\s]+\.md)\)")
# statuses whose file agent-reflect keeps in the memory folder; every other file there is the user's
_KEPT = ("live", "staged", "retired")
_GAP = 4  # columns between the two columns under MEMORY.md


@dataclass(frozen=True)
class Picture:
    """One project's memory, counted."""
    project: str
    user_lines: int  # MEMORY.md lines that link to the user's files
    lesson_lines: int  # MEMORY.md lines that link to a live lesson
    user_files: int
    unread: int | None  # user files no agent opened within `days`; None when the reads were not counted
    days: int
    orphans: int  # user files no MEMORY.md line links to
    dead: int  # files MEMORY.md links to that are not in the folder
    hooked: int  # live lessons with a pattern for the hook: over a command, a path or an error text
    blocking: int  # live lessons the user made the hook refuse a matching command for (`guard block`)
    facts: int  # live lessons that are facts about the project, which go stale
    waiting: int
    live: int
    retired: int
    promoted: int
    removed: int
    rejected: int
    owned: int
    new_this_week: int
    live_this_week: int
    promoted_this_week: int


def _within(stamp: str | None, first: date, last: date, tz: tzinfo | None) -> bool:
    at = parse_ts(stamp)
    return at is not None and first <= at.astimezone(tz).date() <= last


def picture(cfg: Config, lessons: dict[str, Lesson], memory_dir: str, project: str, reads: dict | None,
            first: date, last: date, tz: tzinfo | None = None) -> Picture:
    """Count one memory folder. Raises memory.MemoryIndexError for a MEMORY.md that cannot be read."""
    folder = Path(memory_dir)
    text = memory.read_index(folder)
    mine = [lesson for lesson in lessons.values() if lesson.memory_dir == memory_dir]
    kept = {lesson.file for lesson in mine if lesson.status in _KEPT}
    live_files = {lesson.file for lesson in mine if lesson.status == "live"}
    user_lines = lesson_lines = 0
    linked: set[str] = set()
    for line in text.splitlines():
        targets = [target.removeprefix("./") for target in _LINK.findall(line)]
        if not targets:
            continue
        local = {target for target in targets if "/" not in target}
        linked.update(local)
        if live_files.intersection(local):
            lesson_lines += 1
        else:
            user_lines += 1
    try:
        on_disk = {p.name for p in folder.iterdir() if p.is_file() and p.name.endswith(".md")} - {"MEMORY.md"}
    except OSError:
        on_disk = set()
    user_files = on_disk - kept
    unread, days = None, memory_reads.WINDOW_DAYS
    if reads is not None:
        days = reads.get("days", days)
        name = memory_reads.location_name(cfg, folder.parent.name)
        found = next((loc for loc in reads.get("locations") or [] if loc.get("name") == name), None)
        unread = sum(1 for f in (found or {}).get("files") or [] if not f.get("reads"))
    status = [lesson.status for lesson in mine]
    return Picture(
        project=project, user_lines=user_lines, lesson_lines=lesson_lines, user_files=len(user_files),
        unread=unread, days=days, orphans=len(user_files - linked), dead=len(linked - on_disk),
        hooked=sum(1 for lesson in mine if lesson.status == "live" and lesson.patterns()),
        blocking=sum(1 for lesson in mine if lesson.status == "live" and lesson.block),
        facts=sum(1 for lesson in mine if lesson.status == "live" and lesson.kind == "fact"),
        waiting=status.count("staged"), live=status.count("live"), retired=status.count("retired"),
        promoted=status.count("promoted"), removed=status.count("removed"), rejected=status.count("rejected"),
        owned=status.count("user_owned"),
        new_this_week=sum(1 for lesson in mine if _within(lesson.created_at, first, last, tz)),
        live_this_week=sum(1 for lesson in mine if _within(lesson.live_since, first, last, tz)),
        promoted_this_week=sum(1 for lesson in mine
                               if lesson.status == "promoted" and _within(lesson.updated_at, first, last, tz)),
    )


def _pad(text: str, width: int) -> str:
    return text + " " * max(0, width - chart.columns(text))


def _structure(p: Picture, cap: int) -> list[str]:
    """MEMORY.md as a box, with an arrow from each of its two kinds of lines to the files they point to."""
    in_box = (i18n.count(p.user_lines, ("line of yours", "lines of yours"), ("твоя строка", "твоих строки", "твоих строк")),
              i18n.count(p.lesson_lines, ("lesson line", "lesson lines"), ("строка урока", "строки уроков", "строк уроков"))
              + t(f" of {cap}", f" из {cap}"))
    left = [i18n.count(p.user_files, ("file of yours", "files of yours"), ("твой файл", "твоих файла", "твоих файлов"))]
    if p.unread is not None:
        left.append(t(f"{p.unread} not opened in {p.days} days", f"{p.unread} не открывали за {p.days} дней"))
    if p.orphans:
        left.append(t(f"{p.orphans} with no line: agents do not see them",
                      f"{p.orphans} без строки: агенту не видны"))
    if p.dead:
        left.append(i18n.count(p.dead, ("link to a missing file", "links to missing files"),
                               ("ссылка на файл, которого нет", "ссылки на файлы, которых нет",
                                "ссылок на файлы, которых нет")))
    right = [t(f"{p.lesson_lines} auto_*.md in the index", f"{p.lesson_lines} auto_*.md в индексе"),
             t(f"{p.hooked} with a hook pattern", f"{p.hooked} с шаблоном для хука"),
             t(f"{p.live - p.hooked} without one", f"{p.live - p.hooked} без шаблона")]
    if p.blocking:
        right.append(t(f"of them blocking a command: {p.blocking}", f"из них блокируют команду: {p.blocking}"))
    if p.facts:
        right.append(t(f"of them facts: {p.facts}", f"из них фактов: {p.facts}"))
    width_left = max(chart.columns(text) for text in (in_box[0], *left)) + _GAP
    caption = "─ MEMORY.md · " + t("every session loads it whole", "каждая сессия загружает его целиком") + " "
    inner = max(width_left + max(chart.columns(text) for text in (in_box[1], *right)), chart.columns(caption))
    arrows = [3, width_left + 3]  # where the two arrows leave the box, counted from its left corner's right
    bottom = "".join("┬" if i in arrows else "─" for i in range(inner + 2))
    stem = _pad(" " * 4 + "│", width_left + 4) + "│"
    head = _pad(" " * 4 + "▼", width_left + 4) + "▼"
    out = ["┌" + caption + "─" * (inner + 2 - chart.columns(caption)) + "┐",
           "│ " + _pad(_pad(in_box[0], width_left) + in_box[1], inner) + " │",
           "└" + bottom + "┘", stem, head]
    for i in range(max(len(left), len(right))):
        row = _pad(left[i] if i < len(left) else "", width_left) + (right[i] if i < len(right) else "")
        out.append(("  " + row).rstrip())
    if p.waiting:
        out.append(" " * (width_left + 2) + f"+ {p.waiting} auto_*.md " + t("with no line: ", "без строки: ")
                   + i18n.plural(p.waiting, ("waits for", "wait for"), ("ждёт", "ждут", "ждут"))
                   + t(" a second case", " повтора"))
    if p.retired:
        out.append(" " * (width_left + 2) + f"+ {p.retired} auto_*.md " + t("with no line: ", "без строки: ")
                   + i18n.plural(p.retired, ("retired", "retired"), ("снят", "сняты", "сняты")))
    return out


def _path(p: Picture) -> list[str]:
    """The steps a lesson takes, each a box with how many lessons stand there now and came into it this week."""
    steps = [(t("waiting", "ждёт повтора"), t("a file with no line in MEMORY.md", "файл без строки в MEMORY.md"),
              p.waiting, p.new_this_week),
             (t("in the index", "в индексе"), t("a line in MEMORY.md", "строка в MEMORY.md"),
              p.live, p.live_this_week),
             (t("in a skill", "в скилле"), t("a rule in a skill or CLAUDE.md", "правило в скилле или CLAUDE.md"),
              p.promoted, p.promoted_this_week)]
    # what moves a lesson into each step, a line or two written on the arrow
    edges = [(t("a new lesson", "новый урок"),),
             (t("high confidence, a second case or your approve,", "уверенность high, второй случай или твой approve,"),
              t("and MEMORY.md has room", "и в MEMORY.md есть место")),
             (t(f"confirmed in {PROMOTION_MIN_SESSIONS} sessions within {PROMOTION_WINDOW_DAYS} days,"
                f" or broken {PROMOTION_MIN_VIOLATIONS} times",
                f"подтверждён в {PROMOTION_MIN_SESSIONS} сессиях за {PROMOTION_WINDOW_DAYS} дней"
                f" или нарушен {PROMOTION_MIN_VIOLATIONS} раза"),)]
    now_head, week_head = t("now", "сейчас"), t("this week", "за неделю")
    box = max(chart.columns(name) for name, *_ in steps) + 2
    mid = 2 + 1 + box // 2  # the column of the arrows: under the middle of the boxes
    desc = max(chart.columns(text) for _, text, *_ in steps)
    table = 2 + box + 2 + 2 + desc  # where the descriptions end
    now_w, week_w = chart.columns(now_head) + 2, chart.columns(week_head) + 2
    out = [_pad(t("lesson path", "путь урока"), table) + now_head.rjust(now_w) + week_head.rjust(week_w),
           "  " + t("the night's reading · corrections repeated over the week",
                    "ночной разбор · повторяющиеся поправки недели")]
    for (name, text, now, came), edge in zip(steps, edges):
        out += [*(" " * mid + "│ " + line for line in edge), " " * mid + "▼",
                "  ┌" + "─" * box + "┐",
                _pad("  │ " + _pad(name, box - 1) + "│  " + text, table) + str(now).rjust(now_w)
                + (f"+{came}" if came else "0").rjust(week_w)]
        out.append("  └" + "".join("┬" if 3 + i == mid else "─" for i in range(box)) + "┘")
    out[-1] = "  └" + "─" * box + "┘"  # nothing leaves the last step
    aside = [(t("retired", "снят"), p.retired), (t("tidied away", "убран чисткой"), p.removed),
             (t("rejected by you", "отклонён тобой"), p.rejected), (t("yours now", "стал твоим"), p.owned)]
    out.append("  " + t("aside: ", "в стороне: ") + " · ".join(f"{name} {n}" for name, n in aside))
    back = t("back to waiting: ", "обратно в «ждёт»: ")
    out += ["  " + back + t("· MEMORY.md over its limits, except a lesson that blocks a command",
                            "· MEMORY.md переполнен, кроме урока, который блокирует команду"),
            "  " + " " * chart.columns(back) + t("· a retired lesson happens again", "· снятый урок повторился")]
    return out


def projects(lessons: dict[str, Lesson]) -> list[tuple[str, str]]:
    """(memory dir, project name) of each project with a live or waiting lesson, as the gauges pick them."""
    found: dict[str, str] = {}
    for lesson in lessons.values():
        if lesson.status in ("live", "staged"):
            found.setdefault(lesson.memory_dir, Path(lesson.project_root).name or lesson.project_root)
    return sorted(found.items(), key=lambda item: (item[1], item[0]))


def block(cfg: Config, lessons: dict[str, Lesson], reads: dict | None, first: date, last: date,
          tz: tzinfo | None = None) -> list[str]:
    """The digest's memory section: per project the map of MEMORY.md, the lesson path and the gauges.

    Empty when no project has a live or waiting lesson. `reads` is the weekly scan of memory reads.
    """
    out: list[str] = []
    for memory_dir, project in projects(lessons):
        if out:
            out.append("")
        out.append(f"🧠 {project}")
        try:
            p = picture(cfg, lessons, memory_dir, project, reads, first, last, tz)
        except memory.MemoryIndexError:
            out.append("   MEMORY.md " + t("unreadable", "не читается"))
            continue
        out += [*_structure(p, cfg.auto_index_cap), "", *_path(p)]
    if not out:
        return []
    for line in chart.memory_gauges(cfg, lessons, detailed=True):
        out += ["", line] if line.startswith("🗃") else [line]
    return [t("## 🧠 Memory", "## 🧠 Память"), "", "```text", *out, "```"]
