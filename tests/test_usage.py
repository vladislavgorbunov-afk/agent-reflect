from __future__ import annotations

from datetime import datetime, timezone

from agent_reflect import usage
from agent_reflect.ledger import Lesson, State

NOW = datetime(2026, 10, 25, 4, 30, tzinfo=timezone.utc)


def _lesson(lid, *, status="live", live_since="2026-10-10T00:00:00Z", reads=0, created="2026-10-01T00:00:00Z"):
    return Lesson(id=lid, project_root="/p", memory_dir="/m", file=f"auto_{lid}.md", title=f"t {lid}", hook="h",
                  cue="c", lesson="l", why="w", confidence="high", repeats=1, evidence=[], status=status,
                  sha256="", created_at=created, updated_at=created, live_since=live_since, reads=reads)


def _state(*lessons, installed="2026-09-01T00:00:00Z", counting="2026-10-01T00:00:00Z"):
    return State(installed_at=installed, reads_since=counting, lessons={x.id: x for x in lessons})


def test_counts_reads_of_live_lessons_only():
    st = _state(_lesson("a", reads=3), _lesson("b"), _lesson("c", reads=5, status="staged"))
    stats = usage.stats(st, NOW)
    assert (stats["reads"], stats["with_reads"], stats["live"]) == (3, 1, 2)
    assert usage.line(stats) == "Чтения уроков агентом: 3 (уроков с чтениями: 1 из 2 живых)"
    assert stats["stalled"] is False


def test_stalled_after_two_weeks_with_three_week_old_live_lessons_and_no_reads():
    st = _state(_lesson("a"), _lesson("b"), _lesson("c"))
    assert usage.stats(st, NOW)["stalled"] is True
    assert "UserPromptSubmit" in usage.stalled_note()


def test_not_stalled_when_any_live_lesson_was_read():
    st = _state(_lesson("a", reads=1), _lesson("b"), _lesson("c"))
    assert usage.stats(st, NOW)["stalled"] is False


def test_not_stalled_within_fourteen_days_of_counting_reads():
    st = _state(_lesson("a"), _lesson("b"), _lesson("c"), counting="2026-10-12T00:00:00Z")
    assert usage.stats(st, NOW)["stalled"] is False
    st.reads_since = None  # never counted yet
    assert usage.stats(st, NOW)["stalled"] is False


def test_not_stalled_with_fewer_than_three_lessons_live_for_a_week():
    st = _state(_lesson("a"), _lesson("b"), _lesson("c", live_since="2026-10-20T00:00:00Z"))
    assert usage.stats(st, NOW)["stalled"] is False


def test_a_lesson_live_before_live_since_existed_dates_from_its_creation():
    st = _state(_lesson("a", live_since=None), _lesson("b", live_since=None),
                _lesson("c", live_since=None, created="2026-10-24T00:00:00Z"))
    assert usage.stats(st, NOW)["stalled"] is False
    st.lessons["c"].created_at = "2026-10-02T00:00:00Z"
    assert usage.stats(st, NOW)["stalled"] is True


def test_merge_adds_reads_to_known_lessons_and_keeps_the_latest_time():
    st = _state(_lesson("a"))
    reads = [usage.LessonRead("a", "2026-10-20T00:00:00Z"), usage.LessonRead("a", "2026-10-19T00:00:00Z"),
             usage.LessonRead("zzzz", "2026-10-20T00:00:00Z")]
    usage.merge(st.lessons, reads)
    assert (st.lessons["a"].reads, st.lessons["a"].last_read_at) == (2, "2026-10-20T00:00:00Z")
