"""What the weekly digest says about the lessons' upkeep and about the rules the agent keeps breaking.

Nothing here changes memory: it only proposes. A lesson nobody read, no hook showed and no session
repeated for a month can leave the index (`retire` keeps its file); a lesson broken while it was
in the index, with no trigger for the hook, is one the index alone does not deliver.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from .ledger import Lesson, State, parse_ts
from .promptkit import squash

STALE_AFTER = timedelta(days=30)


def _last_used(lesson: Lesson) -> datetime | None:
    """The last time anything touched the lesson: a read, a showing by the hook, a repeat or a violation."""
    stamps = [lesson.last_read_at, lesson.last_fired_at, lesson.last_violated_at,
              *(e.get("ts") for e in lesson.evidence if isinstance(e, dict))]
    times = [t for t in map(parse_ts, stamps) if t is not None]
    return max(times) if times else None


def lessons(state: State, now: datetime) -> list[dict]:
    """The live lessons worth a look: `stale` ones first, then `unhelped` ones, each in id order."""
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    stale, unhelped = [], []
    for lesson in sorted(state.lessons.values(), key=lambda item: item.id):
        if lesson.status != "live":
            continue
        since = parse_ts(lesson.live_since or lesson.created_at)
        used = _last_used(lesson)
        if since is not None and now - since >= STALE_AFTER and (used is None or now - used >= STALE_AFTER):
            stale.append({"lesson_id": lesson.id, "reason": "stale", "days": (now - since).days})
        elif lesson.violations and not lesson.trigger and not lesson.path_trigger:
            unhelped.append({"lesson_id": lesson.id, "reason": "unhelped", "violations": lesson.violations})
    return stale + unhelped


def broken_rules(records: list[dict], since: datetime | None) -> list[dict]:
    """The rules the user's corrections quoted as broken since `since`, most often broken first.

    `records` are the journal's; a rule is told from another by its words, whatever their case and spacing.
    """
    found: dict[str, dict] = {}
    for record in records:
        if record.get("type") != "verdict" or record.get("agent"):
            continue
        at = parse_ts(record.get("ts"))
        if since is not None and (at is None or at < since):
            continue
        for episode in record.get("episodes") or []:
            rule = squash(str(episode.get("rule") or "")) if isinstance(episode, dict) else ""
            if not rule:
                continue
            item = found.setdefault(rule.casefold(), {"rule": rule, "count": 0, "sessions": set(),
                                                      "said": str(episode.get("said") or "")})
            item["count"] += 1
            item["sessions"].add(str(record.get("session") or ""))
    ordered = sorted(found.values(), key=lambda item: (-item["count"], item["rule"].casefold()))
    return [{**item, "sessions": len(item["sessions"])} for item in ordered]
