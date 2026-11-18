"""Did moving a lesson into a skill help? The verdict a move gets once its window closed.

A lesson the weekly run moved into a skill or CLAUDE.md leaves memory. The nightly run keeps
watching its mistake (by its triggers, and by the model, which sees it marked as moved) and
records each session that made it again after the move. Once WINDOW has passed since the move
took effect (at once for a personal skill, at the merge for a pull request), the move gets one
verdict, preferring "confounded" when unsure:

- confounded: another move changed the same target in or around the window, so a change in
  the count cannot be put down to this one (checked first);
- not_helped: the mistake came back after the move;
- too_little_data: it did not, but it was in fewer than two sessions in the window before the
  move either, so no change could show;
- helped: it was in at least two sessions in the window before the move and in none after;
- not_merged: the move's pull request was closed unmerged: the rule is neither in memory nor in the skill.

A session counts after the move only when it started after the move took effect: one begun
before a merge never loaded the skill.

A comeback inside the window is reported at once ("watching"), without a verdict.
Nothing here writes memory or a skill: it only judges and proposes.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from agent_reflect.ledger import Lesson, State, parse_ts

# as long as the window promotion reads its evidence in (consolidate.PROMOTION_WINDOW_DAYS), so the
# evidence that moved a lesson is the "before" it is held against
WINDOW = timedelta(days=30)
MIN_BEFORE = 2
# "rolled_back" is no verdict on the move: `rollback` took its rule out of the skill (memory_archive)
VERDICTS = ("helped", "not_helped", "confounded", "too_little_data", "not_merged", "rolled_back")


def _iso(when: datetime) -> str:
    return when.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sessions(entries: list[dict], start: datetime, end: datetime, key: str = "ts") -> int:
    """Distinct sessions among `entries` whose time (`key`, else "ts") is in [start, end)."""
    found = set()
    for entry in entries:
        if not isinstance(entry, dict) or not entry.get("session"):
            continue
        ts = parse_ts(entry.get(key) or entry.get("ts"))
        if ts is not None and start <= ts < end:
            found.add(entry["session"])
    return len(found)


def _in_effect(lesson: Lesson, state: State) -> tuple[datetime | None, str]:
    """Since when the move works, or None while it does not yet; with "not_merged" for a closed PR."""
    if lesson.moved_since:
        return parse_ts(lesson.moved_since), ""
    moved = parse_ts(lesson.promoted_at)
    if not lesson.promoted_pr:
        return moved, ""
    pr = next((p for p in state.open_prs if isinstance(p, dict) and p.get("url") == lesson.promoted_pr), None)
    status = pr.get("status") if pr else ""
    if status == "merged":
        return parse_ts(pr.get("merged_at")) or moved, ""
    return None, "not_merged" if status == "closed" else ""


def _move_in_effect(move: dict, state: State) -> datetime | None:
    """Since when a recorded move changed its file: at once without a PR, at the merge with one."""
    if move.get("rolled_back"):
        return None
    if not move.get("pr_url"):
        return parse_ts(move.get("at"))
    pr = next((p for p in state.open_prs if isinstance(p, dict) and p.get("url") == move["pr_url"]), None)
    return parse_ts(pr.get("merged_at")) if pr and pr.get("status") == "merged" else None


def _confounded(lesson: Lesson, state: State, moved: datetime, since: datetime) -> bool:
    """Another move into the same target, by another run, in the window before the move or after it.

    The recorded moves count those of the user's notes too; the lessons cover moves made before
    moves were recorded.
    """
    for move in state.moves:
        if not isinstance(move, dict) or move.get("target") != lesson.promoted_to or move.get("at") == lesson.promoted_at:
            continue
        at = _move_in_effect(move, state)
        if at is not None and moved - WINDOW < at < since + WINDOW:
            return True
    for other in state.lessons.values():
        if other.id == lesson.id or other.promoted_to != lesson.promoted_to or other.promoted_at == lesson.promoted_at:
            continue
        if other.status != "promoted" or other.effect == "rolled_back":
            continue  # taken back: the file is as it was
        at = _in_effect(other, state)[0]
        if at is None:
            continue  # its PR is open or closed unmerged: it has not changed the file
        if at is not None and moved - WINDOW < at < since + WINDOW:
            return True
    return False


def check(state: State, now: datetime) -> list[dict]:
    """Judge every move whose window closed and report every comeback inside an open one.

    A verdict is given once: it is kept on the lesson (`effect`, `effect_at`) and the lesson is
    not judged again. Each item: {"lesson_id", "target", "verdict", "before", "after"}, and
    "days_left" for a move still watched.
    """
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    out = []
    for lesson in sorted(state.lessons.values(), key=lambda item: item.id):
        moved = parse_ts(lesson.promoted_at)
        if lesson.status != "promoted" or moved is None or lesson.effect:
            continue
        since, closed = _in_effect(lesson, state)
        before = _sessions(lesson.evidence, moved - WINDOW, moved)
        item = {"lesson_id": lesson.id, "target": lesson.promoted_to, "before": before}
        if closed:
            verdict, after = closed, 0
        elif since is None:
            continue  # the pull request is still open
        elif now < since + WINDOW:
            after = _sessions(lesson.after, since, now + timedelta(seconds=1), "started")
            if after:
                out.append({**item, "verdict": "watching", "after": after,
                            "days_left": (since + WINDOW - now).days})
            continue
        else:
            after = _sessions(lesson.after, since, since + WINDOW, "started")
            if _confounded(lesson, state, moved, since):
                verdict = "confounded"
            elif after:
                verdict = "not_helped"
            else:
                verdict = "too_little_data" if before < MIN_BEFORE else "helped"
        lesson.effect, lesson.effect_at = verdict, _iso(now)
        out.append({**item, "verdict": verdict, "after": after})
    return out
