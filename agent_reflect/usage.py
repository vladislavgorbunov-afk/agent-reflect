"""Do lessons reach the agent? Counts of the agent opening auto-lesson files, no model involved.

An index line is only a pointer: a lesson helps when the agent opens its file. While a
transcript is read, every `Read` of a `<memory>/auto_<id>.md` and every `Grep`/`Bash`
call that names `auto_<id>.md` counts as one read of that lesson. The counts are merged
into the ledger with the same once-only byte offsets as the health channel.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from agent_reflect.i18n import t
from agent_reflect.ledger import Lesson, State, parse_ts

_MEMORY_FILE_RE = re.compile(r"/memory/auto_([A-Za-z0-9]+)\.md$")
_MENTION_RE = re.compile(r"auto_([A-Za-z0-9]+)\.md")
_MENTION_TOOLS = ("Grep", "Bash")
# The digest says the index lines do not reach the agent only after this much evidence.
STALLED_AFTER_COUNTING = timedelta(days=14)  # since State.reads_since, not the install
STALLED_LIVE_FOR = timedelta(days=7)
STALLED_MIN_LESSONS = 3


def stalled_note() -> str:
    return t(
        "The index lines do not seem to reach the agent: live lessons have sat in MEMORY.md for a week or more, "
        "and the agent has opened none of them. The next experiment is a UserPromptSubmit hook that puts "
        "the lessons that fit the situation into the prompt by itself.",
        "Строки индекса, похоже, не доходят до агента: живые уроки висят в MEMORY.md неделю и дольше, "
        "а агент не открыл ни одного. Следующий эксперимент — хук UserPromptSubmit, который сам "
        "подставляет в запрос уроки, подходящие к ситуации.",
    )


@dataclass(frozen=True)
class LessonRead:
    lesson_id: str
    ts: str | None
    end: int = 0  # byte offset just past the entry in its transcript, so a re-read never counts it twice


def lesson_ids(tool: str, tool_input) -> list[str]:
    """The lessons one tool call reads: each id at most once per call."""
    if tool == "Read":
        path = tool_input.get("file_path") if isinstance(tool_input, dict) else None
        m = _MEMORY_FILE_RE.search(path) if isinstance(path, str) else None
        return [m.group(1)] if m else []
    if tool in _MENTION_TOOLS:
        text = tool_input if isinstance(tool_input, str) else json.dumps(tool_input, ensure_ascii=False)
        return list(dict.fromkeys(_MENTION_RE.findall(text)))
    return []


def merge(lessons: dict[str, Lesson], reads) -> None:
    """Add reads to the lessons they name; reads of files that are no lesson of ours are dropped."""
    for read in reads:
        lesson = lessons.get(read.lesson_id)
        if lesson is None:
            continue
        lesson.reads += 1
        if read.ts and (not lesson.last_read_at or read.ts > lesson.last_read_at):
            lesson.last_read_at = read.ts


def _live_since(lesson: Lesson) -> datetime | None:
    """When the lesson went live; one live before this was recorded dates from its creation."""
    return parse_ts(lesson.live_since or lesson.created_at)


def stats(state: State, now: datetime) -> dict:
    """All-time reads of the live lessons, and whether that says the index does not reach the agent."""
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    live = [lesson for lesson in state.lessons.values() if lesson.status == "live"]
    reads = sum(lesson.reads for lesson in live)
    counting = parse_ts(state.reads_since)
    settled = sum(1 for lesson in live
                  if (since := _live_since(lesson)) is not None and now - since >= STALLED_LIVE_FOR)
    stalled = (counting is not None and now - counting >= STALLED_AFTER_COUNTING
               and reads == 0 and settled >= STALLED_MIN_LESSONS)
    return {"reads": reads, "with_reads": sum(1 for lesson in live if lesson.reads), "live": len(live),
            "stalled": stalled}


def line(usage: dict) -> str:
    reads, read, live = usage.get("reads", 0), usage.get("with_reads", 0), usage.get("live", 0)
    return t(f"Lesson reads by the agent: {reads} (lessons read: {read} of {live} live)",
             f"Чтения уроков агентом: {reads} (уроков с чтениями: {read} из {live} живых)")
