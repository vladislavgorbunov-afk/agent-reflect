from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from agent_reflect import journal
from agent_reflect.config import Config
from agent_reflect.ledger import Lesson, State
from conftest import write_jsonl

ISO = "%Y-%m-%dT%H:%M:%SZ"
NOW = datetime(2026, 10, 23, 3, 30, tzinfo=timezone.utc)
LOCAL = timezone(timedelta(hours=4))


@pytest.fixture
def cfg(tmp_path):
    return Config.under(tmp_path)


def _ago(days: int, now: datetime = NOW) -> str:
    return (now - timedelta(days=days)).strftime(ISO)


def test_append_then_load_round_trip(cfg):
    started, ended = NOW - timedelta(minutes=12), NOW - timedelta(minutes=2)
    result = {"segments_seen": 3, "llm_calls": 1, "outcomes": {"no_entry": 1}, "errors": 0,
              "verdicts": [{"session": "s1", "outcome": "no_entry"}]}
    event = {"ts": "2026-10-23T03:20:00Z", "kind": "lesson_live", "lesson_id": "a"}
    journal.append(cfg, journal.run_records(kind="reflect", trigger="schedule", started=started, ended=ended,
                                            result=result, events=[event]), NOW)

    run, verdict, ev = journal.load(cfg)

    assert (run["type"], verdict["type"], ev["type"]) == ("run", "verdict", "event")
    assert run["run"] == verdict["run"] == ev["run"] == "2026-10-23T03:18:00Z"
    assert run["no_signals"] == 0 and run["failed"] is None
    assert run["outcomes"] == {"no_entry": 1} and run["segments_seen"] == 3
    assert verdict["session"] == "s1" and ev["lesson_id"] == "a"


def test_run_records_default_missing_counters_to_zero(cfg):
    [run] = journal.run_records(kind="consolidate", trigger="manual", started=NOW, ended=NOW, result={},
                                events=[], failed="aborted", digest="digests/2026-W43.md")

    assert run == {
        "type": "run", "run": "2026-10-23T03:30:00Z", "kind": "consolidate", "trigger": "manual",
        "started_at": "2026-10-23T03:30:00Z", "ended_at": "2026-10-23T03:30:00Z",
        "segments_seen": 0, "no_signals": 0, "deferred": 0, "llm_calls": 0, "outcomes": {}, "errors": 0,
        "failed": "aborted", "digest": "digests/2026-W43.md",
    }


def test_run_records_carry_the_live_lessons_they_are_given(cfg):
    [run] = journal.run_records(kind="reflect", trigger="schedule", started=NOW, ended=NOW, result={}, events=[],
                                live={"/p": 2})
    [imported] = journal.run_records(kind="reflect", trigger="schedule", started=NOW, ended=NOW, result={},
                                     events=[])

    assert run["live"] == {"/p": 2}
    assert "live" not in imported  # a run read from the logs never counted them


def test_live_counts_are_the_live_lessons_of_each_project():
    def lesson(i, project, status):
        return Lesson(id=i, project_root=project, memory_dir="/m", file=f"auto_{i}.md", title="t", hook="h", cue="c",
                      lesson="l", why="w", confidence="high", repeats=1, evidence=[], status=status, sha256="",
                      created_at="", updated_at="")

    lessons = [lesson("a", "/p", "live"), lesson("b", "/p", "live"), lesson("c", "/p", "staged"),
               lesson("d", "/q", "live"), lesson("e", "/r", "rejected")]

    assert journal.live_counts(State(lessons={x.id: x for x in lessons})) == {"/p": 2, "/q": 1}
    assert journal.live_counts(State()) == {}


def test_run_records_name_the_run_by_its_utc_start(cfg):
    local = timezone(timedelta(hours=4))
    started = datetime(2026, 10, 23, 7, 30, tzinfo=local)

    [run] = journal.run_records(kind="reflect", trigger="schedule", started=started, ended=started,
                                result={}, events=[])

    assert run["run"] == run["started_at"] == "2026-10-23T03:30:00Z"


def test_append_drops_records_older_than_retention(cfg):
    old, new = _ago(journal.RETAIN_DAYS + 1), _ago(journal.RETAIN_DAYS - 1)
    journal.append(cfg, [
        {"type": "run", "run": "old", "ended_at": old},
        {"type": "run", "run": "new", "ended_at": new},
        {"type": "verdict", "run": "old", "ts": old},
        {"type": "verdict", "run": "new", "ts": new},
        {"type": "event", "run": "old", "ts": old, "kind": "lesson_live"},
        {"type": "event", "run": "new", "ts": new, "kind": "lesson_live"},
    ], NOW)

    assert [(r["type"], r["run"]) for r in journal.load(cfg)] == [
        ("run", "new"), ("verdict", "new"), ("event", "new")]


def test_append_prunes_what_is_already_in_the_file_as_it_ages(cfg):
    journal.append(cfg, [{"type": "run", "run": "first", "ended_at": _ago(0)}], NOW)
    later = NOW + timedelta(days=40)
    journal.append(cfg, [{"type": "run", "run": "second", "ended_at": _ago(0, later)}], later)
    assert [r["run"] for r in journal.load(cfg)] == ["first", "second"]

    much_later = NOW + timedelta(days=70)
    journal.append(cfg, [{"type": "run", "run": "third", "ended_at": _ago(0, much_later)}], much_later)
    assert [r["run"] for r in journal.load(cfg)] == ["second", "third"]


def test_verdict_without_ts_ages_by_its_run(cfg):
    journal.append(cfg, [
        {"type": "verdict", "run": _ago(journal.RETAIN_DAYS + 1), "session": "old"},
        {"type": "verdict", "run": _ago(1), "session": "new"},
        {"type": "verdict", "run": "not a time", "session": "undated"},
    ], NOW)

    assert [r["session"] for r in journal.load(cfg)] == ["new", "undated"]


def test_load_without_a_file_is_empty(cfg):
    assert journal.load(cfg) == []
    assert not cfg.journal_file.exists()


def test_load_skips_broken_lines(cfg):
    cfg.journal_file.parent.mkdir(parents=True)
    cfg.journal_file.write_text(
        '{"type": "run", "run": "a"}\n'
        '{"type": "run"\n'
        "not json\n"
        "[1, 2]\n"
        "\n"
        '{"type": "run", "run": "b"}\n', encoding="utf-8")

    assert [r["run"] for r in journal.load(cfg)] == ["a", "b"]


def test_append_rewrites_the_file_one_record_per_line(cfg):
    journal.append(cfg, [{"type": "run", "run": "a", "ended_at": _ago(0), "failed": "ошибка"}], NOW)

    [line] = cfg.journal_file.read_text(encoding="utf-8").splitlines()
    assert json.loads(line)["failed"] == "ошибка"
    assert not list(cfg.state_dir.glob("*.tmp"))


def test_records_with_line_separators_in_text_survive_a_rewrite(cfg):
    text = "a b c\x85d\x0be\x0cf\x1cg"  # str.splitlines() breaks on every one of these
    journal.append(cfg, [{"type": "run", "run": "a", "ended_at": _ago(0), "failed": text}], NOW)
    journal.append(cfg, [{"type": "run", "run": "b", "ended_at": _ago(0)}], NOW)  # rewrites from load

    assert [(r["run"], r.get("failed")) for r in journal.load(cfg)] == [("a", text), ("b", None)]


def test_events_added_survives_consolidate_clearing():
    first, second, fresh = ({"kind": k} for k in ("first", "second", "fresh"))
    state = State(events=[first, second])
    before, start = state.events, 1

    state.events = [fresh]  # consolidate clears the list it ran with

    assert journal.events_added(state, before, start) == [second, fresh]


def test_events_added_without_a_replaced_list_is_what_was_appended():
    first, second, appended = ({"kind": k} for k in ("first", "second", "appended"))
    state = State(events=[first, second])
    before, start = state.events, 1

    state.events.append(appended)

    assert journal.events_added(state, before, start) == [second, appended]


def _write_log(cfg, name: str, *lines: str) -> None:
    cfg.logs_dir.mkdir(parents=True, exist_ok=True)
    (cfg.logs_dir / name).write_text("\n".join(lines) + "\n", encoding="utf-8")


def _verdict_line(at: str, outcome: str = "no_entry", session: str = "s1", agent: str = "",
                  notes: str = "ничего нового", verdict: str = "NO_ENTRY") -> str:
    who = f"{session} {agent}" if agent else session
    return f"2026-10-23 {at},154 INFO verdict {verdict} → {outcome} · session {who} · {notes}"


def _done_line(at: str, result: dict, kind: str = "reflect") -> str:
    return f"2026-10-23 {at},854 INFO {kind} done: {json.dumps(result, ensure_ascii=False)}"


def test_import_turns_done_lines_into_runs(cfg):
    _write_log(cfg, "reflect.log",
               "2026-10-23 03:30:01,002 INFO something else entirely",
               _verdict_line("03:30:16", "no_entry", "s1", notes="нет сигналов"),
               _verdict_line("03:31:18", "repeat", "s2", verdict="REPEAT", notes="уже есть"),
               _done_line("03:31:43", {"segments_seen": 28, "llm_calls": 2,
                                       "outcomes": {"no_entry": 1, "repeat": 1}, "errors": 0}))

    run, first, second = journal.import_logs(cfg, LOCAL)

    assert run == {
        "type": "run", "run": "2026-10-22T23:30:16Z", "kind": "reflect", "trigger": "schedule",
        "started_at": "2026-10-22T23:30:16Z", "ended_at": "2026-10-22T23:31:43Z",
        "segments_seen": 28, "no_signals": 26, "deferred": 0, "llm_calls": 2,
        "outcomes": {"no_entry": 1, "repeat": 1}, "errors": 0, "failed": None, "digest": None,
    }
    assert first == {
        "type": "verdict", "run": "2026-10-22T23:30:16Z", "ts": "2026-10-22T23:30:16Z",
        "session": "s1", "agent": "", "transcript": None, "project": None, "title": None,
        "verdict": "NO_ENTRY", "outcome": "no_entry", "lesson_id": None, "file": None,
        "notes": "нет сигналов",
    }
    assert (second["run"], second["ts"], second["outcome"], second["verdict"], second["notes"]) == (
        "2026-10-22T23:30:16Z", "2026-10-22T23:31:18Z", "repeat", "REPEAT", "уже есть")


def test_import_counts_failed_calls_as_llm_error(cfg):
    _write_log(cfg, "reflect.log",
               _done_line("03:31:43", {"segments_seen": 4, "llm_calls": 3, "outcomes": {}, "errors": 3}),
               _done_line("11:00:00", {"segments_seen": 9, "llm_calls": 5,
                                       "outcomes": {"no_project": 4, "no_entry": 2, "llm_error": 1}, "errors": 2}))

    first, second = journal.import_logs(cfg, LOCAL)

    assert first["outcomes"] == {"llm_error": 3} and first["errors"] == 3
    assert second["outcomes"] == {"no_project": 4, "no_entry": 2, "llm_error": 3}  # 5 calls, 3 outcomes: 2 errors more
    assert second["no_signals"] == 0  # 9 seen - 5 called - 4 without a project


def test_import_without_a_verdict_line_starts_the_run_at_its_done_line(cfg):
    _write_log(cfg, "reflect.log", _done_line("09:15:00", {"segments_seen": 2, "llm_calls": 0, "outcomes": {}}))

    [run] = journal.import_logs(cfg, LOCAL)

    assert run["started_at"] == run["run"] == run["ended_at"] == "2026-10-23T05:15:00Z"
    assert run["trigger"] == "manual"  # 09:15 local, not the night run's 03:xx
    assert run["no_signals"] == 2 and run["errors"] == 0


@pytest.mark.parametrize("kind", ["reflect", "backfill"])
def test_import_leaves_the_verdicts_of_a_crashed_run_out_of_the_next(cfg, kind):
    _write_log(cfg, f"{kind}.log",
               _verdict_line("03:30:16", "repeat", "s1", verdict="REPEAT"),
               f"2026-10-23 03:30:20,001 ERROR {kind} crashed",
               "Traceback (most recent call last):",
               "RuntimeError: boom",
               _verdict_line("11:00:05", "no_entry", "s2"),
               _done_line("11:00:30", {"segments_seen": 1, "llm_calls": 1, "outcomes": {"no_entry": 1}, "errors": 0},
                          kind))

    run, verdict = journal.import_logs(cfg, LOCAL)

    assert (run["kind"], run["started_at"], run["trigger"]) == (kind, "2026-10-23T07:00:05Z", "manual")
    assert verdict["session"] == "s2"


def test_import_backfill_is_manual(cfg):
    _write_log(cfg, "backfill.log",
               "2026-10-23 03:10:00,001 INFO backfill: 12 transcripts reset since 2026-10-01",
               _verdict_line("03:30:16", "new_live", "s1", notes="урок"),
               _done_line("03:40:00", {"segments_seen": 3, "llm_calls": 1, "outcomes": {"new_live": 1}, "errors": 0},
                          kind="backfill"))

    run, verdict = journal.import_logs(cfg, LOCAL)

    assert (run["kind"], run["trigger"], run["run"]) == ("backfill", "manual", "2026-10-22T23:30:16Z")
    assert verdict["run"] == run["run"] and verdict["outcome"] == "new_live"


def test_import_uses_the_counters_a_newer_log_carries(cfg):
    _write_log(cfg, "reflect.log", _verdict_line("03:30:16", "no_entry"),
               _done_line("03:31:43", {"segments_seen": 28, "no_signals": 20, "deferred": 5, "llm_calls": 2,
                                       "outcomes": {"no_entry": 2}, "errors": 0,
                                       "verdicts": [{"session": "other", "outcome": "repeat"}]}))

    run, verdict = journal.import_logs(cfg, LOCAL)

    assert (run["no_signals"], run["deferred"]) == (20, 5)
    assert verdict["session"] == "s1"  # the lines of the log are the verdicts, not a `verdicts` key


def test_import_skips_dry_runs_and_garbage(cfg):
    _write_log(cfg, "reflect.log",
               "",
               "not a log line at all",
               _verdict_line("02:00:00", "no_entry", "dry"),
               _done_line("02:00:05", {"segments_seen": 5, "llm_calls": 0, "outcomes": {}, "errors": 0,
                                       "would_call": [1200, 3400]}),
               "2026-10-23 03:30:16,154 INFO verdict NO_ENTRY → no_entry · broken line with no session",
               _verdict_line("03:30:20", "no_entry", "s1"),
               "2026-10-23 03:30:30,154 ERROR reflect failed: boom",
               "2026-10-23 03:30:40,854 INFO reflect done: {not json}",
               "2026-10-23 03:30:41,854 INFO reflect done: [1, 2]",
               _verdict_line("03:31:00", "repeat", "s2"),
               _done_line("03:31:43", {"segments_seen": 3, "llm_calls": 1, "outcomes": {"repeat": 1}, "errors": 0}),
               _verdict_line("03:40:00", "no_entry", "unfinished"))  # a run that never wrote its done line

    run, verdict = journal.import_logs(cfg, LOCAL)

    assert run["started_at"] == "2026-10-22T23:31:00Z"  # what a skipped done line closed is not carried over
    assert verdict["session"] == "s2"


def test_import_names_a_session_by_its_transcript(cfg):
    write_jsonl(cfg.projects_dir / "-w-app" / "s1.jsonl", [{"type": "ai-title", "aiTitle": "Автотесты"}])
    write_jsonl(cfg.projects_dir / "-w-app" / "s1" / "subagents" / "agent-a1.jsonl", [{"type": "user"}])
    _write_log(cfg, "reflect.log",
               _verdict_line("03:30:16", session="s1"),
               _verdict_line("03:30:30", session="s1", agent="agent-a1"),
               _verdict_line("03:30:40", session="gone"),
               _verdict_line("03:30:50", session="sess*[1]", agent="agent-**"),
               _done_line("03:31:43", {"segments_seen": 4, "llm_calls": 4, "outcomes": {"no_entry": 4}}))

    _, main, sub, missing, odd = journal.import_logs(cfg, LOCAL)

    assert (main["title"], main["agent"]) == ("Автотесты", "")
    assert (sub["title"], sub["agent"]) == ("Автотесты (сабагент)", "agent-a1")
    assert missing["title"] is None and odd["title"] is None


def test_import_reads_both_logs_oldest_run_first(cfg):
    _write_log(cfg, "reflect.log", _done_line("11:00:00", {"segments_seen": 1, "llm_calls": 0, "outcomes": {}}))
    _write_log(cfg, "backfill.log", _done_line("09:00:00", {"segments_seen": 1, "llm_calls": 0, "outcomes": {}},
                                               kind="backfill"))

    assert [(r["kind"], r["started_at"]) for r in journal.import_logs(cfg, LOCAL)] == [
        ("backfill", "2026-10-23T05:00:00Z"), ("reflect", "2026-10-23T07:00:00Z")]


def test_import_without_logs_is_empty(cfg):
    assert journal.import_logs(cfg, LOCAL) == []

    cfg.logs_dir.mkdir(parents=True)
    (cfg.logs_dir / "reflect.log").write_bytes(b"\xff\xfe not utf-8 \xff\n")
    (cfg.logs_dir / "backfill.log").mkdir()  # unreadable as a file

    assert journal.import_logs(cfg, LOCAL) == []


def test_load_without_journal_reads_logs_and_append_keeps_them(cfg):
    _write_log(cfg, "reflect.log", _verdict_line("03:30:16", "no_entry"),
               _done_line("03:31:43", {"segments_seen": 4, "llm_calls": 1, "outcomes": {"no_entry": 1}}))
    imported = journal.import_logs(cfg, LOCAL)
    assert [r["type"] for r in imported] == ["run", "verdict"]

    assert journal.load(cfg, LOCAL) == imported
    assert not cfg.journal_file.exists()  # load never writes

    later = NOW + timedelta(hours=2)
    new = journal.run_records(kind="reflect", trigger="manual", started=later, ended=later, result={}, events=[])
    journal.append(cfg, new, later, LOCAL)

    assert journal.load(cfg, LOCAL) == [*imported, *new]
    assert cfg.journal_file.read_text(encoding="utf-8").startswith(json.dumps(imported[0], ensure_ascii=False))

    # the file exists now: the logs are no longer read
    (cfg.logs_dir / "reflect.log").write_text("", encoding="utf-8")
    journal.append(cfg, [], later, LOCAL)
    assert journal.load(cfg, LOCAL) == [*imported, *new]


def test_append_prunes_imported_records_like_any_other(cfg):
    _write_log(cfg, "reflect.log", _done_line("03:31:43", {"segments_seen": 1, "llm_calls": 0, "outcomes": {}}))

    journal.append(cfg, [], NOW + timedelta(days=journal.RETAIN_DAYS + 1), LOCAL)

    assert journal.load(cfg, LOCAL) == []
    assert cfg.journal_file.exists()
