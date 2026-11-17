from __future__ import annotations

import os
from dataclasses import replace
from datetime import datetime, timezone

from agent_reflect.ledger import (
    Lesson,
    State,
    distinct_sessions,
    load_state,
    normalize_cue,
    run_lock,
    save_state,
    snapshot_baseline,
)


def _lesson() -> Lesson:
    return Lesson(
        id="ab12cd34",
        project_root="/r",
        memory_dir="/r/memory",
        file="auto_ab12cd34.md",
        title="Заголовок",
        hook="Хук",
        cue="когда коммит блокирует хук",
        lesson="Текст",
        why="Потому что",
        confidence="high",
        repeats=2,
        evidence=[{"session": "s1", "ts": "2026-10-20T00:00:00Z", "text": "т"}],
        status="live",
        sha256="deadbeef",
        created_at="2026-10-20T00:00:00Z",
        updated_at="2026-10-20T01:00:00Z",
    )


def test_load_missing_returns_empty_state(tmp_path):
    assert load_state(tmp_path / "nope.json") == State()


def test_roundtrip_preserves_lessons(tmp_path):
    p = tmp_path / "sub" / "state.json"
    s = State(installed_at="2026-10-20T00:00:00Z")
    s.lessons["ab12cd34"] = _lesson()
    s.sessions["/x.jsonl"] = {"offset": 5}
    s.rejected_cues["/r"] = ["c"]
    s.event("note", detail="é")
    save_state(s, p)
    assert load_state(p) == s
    assert "é" in p.read_text(encoding="utf-8")
    assert s.events[0]["ts"].endswith("Z") and s.events[0]["kind"] == "note"


def test_save_is_atomic(tmp_path, monkeypatch):
    p = tmp_path / "state.json"
    save_state(State(installed_at="old"), p)
    before = p.read_text()

    def boom(*a, **k):
        raise OSError("boom")

    monkeypatch.setattr(os, "replace", boom)
    try:
        save_state(State(installed_at="new"), p)
    except OSError:
        pass
    assert p.read_text() == before


def test_run_lock_excludes_second_holder(tmp_path):
    p = tmp_path / "d" / "run.lock"
    with run_lock(p) as a:
        with run_lock(p) as b:
            assert a and not b
    with run_lock(p) as c:
        assert c


def test_snapshot_baseline_sets_offsets_to_size_once(tmp_path):
    f = tmp_path / "a.jsonl"
    f.write_bytes(b"12345")
    gone = tmp_path / "gone.jsonl"
    s = State()
    snapshot_baseline(s, [f, gone], "t1")
    assert s.installed_at == "t1"
    assert s.sessions == {str(f): {"offset": 5}}
    f.write_bytes(b"1234567890")
    snapshot_baseline(s, [f], "t2")
    assert s.installed_at == "t1"
    assert s.sessions[str(f)] == {"offset": 5}


def test_normalize_cue():
    assert normalize_cue("  Когда Коммит   блокирует хук. ") == "когда коммит блокирует хук"


NOW = datetime(2026, 10, 25, 4, 30, tzinfo=timezone.utc)


def _with_evidence(*entries: tuple) -> Lesson:
    return replace(_lesson(), evidence=[{"session": s, "ts": ts, "text": "т"} for s, ts in entries])


def test_distinct_sessions_counts_each_session_once_inside_the_window():
    lesson = _with_evidence(("s1", "2026-10-20T00:00:00Z"), ("s1", "2026-10-21T00:00:00Z"),
                            ("s2", "2026-09-26T05:00:00Z"), ("s3", "2026-09-20T00:00:00Z"))
    assert distinct_sessions(lesson, NOW) == 2  # s3 is 35 days old
    assert distinct_sessions(lesson, NOW, window_days=40) == 3


def test_distinct_sessions_ignores_empty_sessions_and_keeps_undated_evidence():
    lesson = _with_evidence(("", "2026-10-20T00:00:00Z"), (None, "2026-10-20T00:00:00Z"),
                            ("s1", None), ("s2", "garbage"), ("s3", "2026-10-24T10:00:00.123Z"))
    lesson.evidence.append({"text": "no session key"})
    assert distinct_sessions(lesson, NOW) == 3


def test_distinct_sessions_of_a_lesson_without_evidence_is_zero():
    assert distinct_sessions(_with_evidence(), NOW) == 0
