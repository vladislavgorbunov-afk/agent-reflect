"""The health channel: hook failures and API errors are alerts for the user, never lessons."""
from __future__ import annotations

from datetime import datetime, timezone

import pytest
from conftest import write_jsonl

from agent_reflect import health
from agent_reflect.health import HealthEvent
from agent_reflect.transcript import read_segment

TS = "2026-10-20T10:00:00.000Z"
SID = "0a1b2c3d-1111-4222-8333-444455556666"
NOW = datetime(2026, 10, 21, 3, 0, tzinfo=timezone.utc)
GATE = '"${CLAUDE_PLUGIN_ROOT}"/templates/hooks/process/pre-commit-gate-check.sh'


def hook_failure(command=GATE, exit_code=127, stderr="line 1: jq: command not found\nmore",
                 stdout="", hook_event="PreToolUse", hook_name="PreToolUse:Bash", ts=TS, **extra):
    return {"type": "attachment", "timestamp": ts, "sessionId": SID,
            "attachment": {"type": "hook_non_blocking_error", "hookEvent": hook_event, "hookName": hook_name,
                           "command": command, "exitCode": exit_code, "stderr": stderr, "stdout": stdout,
                           "durationMs": 3, "toolUseID": "toolu_1", **extra}}


def api_error(**fields):
    return {"type": "system", "subtype": "api_error", "timestamp": TS, "sessionId": SID, "level": "error",
            **fields}


def event(key="PreToolUse|~/h/x.sh|1", kind="hook_failure", ts=TS, session=SID, detail="d", hook_name="h"):
    return HealthEvent(kind=kind, key=key, ts=ts, session_id=session, detail=detail, hook_name=hook_name)


# -- collection ------------------------------------------------------------------


def test_hook_failure_key_detail_and_session(tmp_path):
    p = tmp_path / "-w" / f"{SID}.jsonl"
    write_jsonl(p, [hook_failure(command="bash /Users/someone/hooks/x.sh --fast")])
    (ev,) = read_segment(p, 0).health
    assert ev.kind == "hook_failure"
    assert ev.key == "PreToolUse|bash ~/hooks/x.sh --fast|127"
    assert ev.detail == "line 1: jq: command not found"
    assert ev.ts == TS and ev.session_id == SID and ev.hook_name == "PreToolUse:Bash"


def test_hook_failure_detail_falls_back_to_stdout_and_is_scrubbed_and_clipped(tmp_path):
    p = tmp_path / "s.jsonl"
    token = "ghp_" + "f" * 36
    write_jsonl(p, [hook_failure(stderr="  \n", stdout=f"\n{token} " + "z" * 500 + "\nsecond")])
    (ev,) = read_segment(p, 0).health
    assert "ghp_" not in ev.detail and 0 < len(ev.detail) <= 200 and "second" not in ev.detail


@pytest.mark.parametrize("entry", [
    hook_failure(exit_code=0),
    hook_failure(exit_code="127"),
    {"type": "attachment", "timestamp": TS, "attachment": {"type": "hook_success", "exitCode": 1}},
    {"type": "system", "subtype": "compact_boundary", "timestamp": TS},
    {"type": "user", "timestamp": TS, "message": {"content": "hook_non_blocking_error exitCode 127"}},
])
def test_not_health_events(tmp_path, entry):
    p = tmp_path / "s.jsonl"
    write_jsonl(p, [entry])
    assert read_segment(p, 0).health == ()


def cli_api_error(attempt=10, most=10, status=None, message="Connection error.", **extra):
    """The CLI's api_error entry: a retry counter and an error object (hand-built)."""
    error = {"connection": {"code": "ECONNRESET", "isSSLError": False, "message": "socket hang up"},
             "formatted": "Connection error", "isNetworkDown": False, "message": message, "noResponse": True,
             "rateLimits": {}}
    if status is not None:
        error.update(status=status, requestId="req_1")
    retry = {} if attempt is None else {"retryAttempt": attempt, "maxRetries": most, "retryInMs": 500}
    return api_error(error=error, **retry, **extra)


def test_api_error_from_the_cli_shape(tmp_path):
    p = tmp_path / "s.jsonl"
    write_jsonl(p, [cli_api_error(status=529, message="Overloaded"), cli_api_error()])
    first, second = read_segment(p, 0).health
    assert first.kind == "api_error" and first.key == "api_error|529" and first.detail == "Overloaded"
    assert second.key == "api_error" and second.detail == "Connection error."


def test_only_an_exhausted_retry_is_an_api_error(tmp_path):
    p = tmp_path / "s.jsonl"
    write_jsonl(p, [cli_api_error(attempt=n, most=10, status=529) for n in range(1, 11)]
                + [cli_api_error(attempt=None, status=500)])
    keys = [e.key for e in read_segment(p, 0).health]
    assert keys == ["api_error|529", "api_error|500"]


def test_health_is_no_signal_and_never_rendered(tmp_path):
    p = tmp_path / "s.jsonl"
    write_jsonl(p, [hook_failure(), cli_api_error(status=500)])
    seg = read_segment(p, 0)
    assert seg.signals == () and len(seg.health) == 2
    assert "command not found" not in seg.render(5000)


def test_subagent_health_carries_the_parent_session(tmp_path):
    p = tmp_path / "-w" / SID / "subagents" / "agent-a1.jsonl"
    write_jsonl(p, [hook_failure()])
    (ev,) = read_segment(p, 0).health
    assert ev.session_id == SID


def test_health_events_carry_their_byte_end(tmp_path):
    p = tmp_path / "s.jsonl"
    write_jsonl(p, [hook_failure(), {"type": "user", "timestamp": TS, "message": {"content": "x"}}, hook_failure()])
    first, second = read_segment(p, 0).health
    lines = p.read_bytes().split(b"\n")
    assert first.end == len(lines[0]) + 1
    assert second.end == p.stat().st_size


# -- merge -----------------------------------------------------------------------


def test_merge_adds_counts_keeps_bounds_and_latest_detail():
    store: dict = {}
    added = health.merge(store, [
        event(ts="2026-10-20T10:00:00.000Z", detail="first", session="s1"),
        event(ts="2026-10-19T10:00:00.000Z", detail="older", session="s2"),
        event(ts="2026-10-21T10:00:00.000Z", detail="", session="s1"),
    ])
    assert added == {"PreToolUse|~/h/x.sh|1": 3}
    entry = store["PreToolUse|~/h/x.sh|1"]
    assert entry["count"] == 3 and entry["kind"] == "hook_failure"
    assert entry["first_ts"] == "2026-10-19T10:00:00.000Z" and entry["last_ts"] == "2026-10-21T10:00:00.000Z"
    assert entry["detail"] == "first" and entry["sessions"] == ["s1", "s2"] and entry["notified_at"] is None
    health.merge(store, [event(ts="2026-10-22T10:00:00.000Z", detail="newest")])
    assert store["PreToolUse|~/h/x.sh|1"]["count"] == 4 and store["PreToolUse|~/h/x.sh|1"]["detail"] == "newest"


def test_merge_keeps_at_most_twenty_sessions():
    store: dict = {}
    health.merge(store, [event(session=f"s{i}") for i in range(30)])
    assert store["PreToolUse|~/h/x.sh|1"]["sessions"] == [f"s{i}" for i in range(20)]


def test_ignore_matches_the_command_or_the_hook_name_of_hook_failures_only():
    store: dict = {}
    added = health.merge(store, [
        event(key="PreToolUse|~/.claude/plugins/dev-process-toolkit/x.sh|127"),
        event(key="PreToolUse|~/x.sh|127", hook_name="dev-process-toolkit:gate"),
        event(key="PreToolUse|~/mine.sh|1"),
        event(key="api_error|dev-process-toolkit", kind="api_error"),
    ], ignore=("dev-process-toolkit",))
    assert set(added) == {"PreToolUse|~/mine.sh|1", "api_error|dev-process-toolkit"} and set(store) == set(added)
    assert set(health.merge({}, [event(key="PreToolUse|~/x/dev-process-toolkit/y.sh|1")], ignore=())) == {
        "PreToolUse|~/x/dev-process-toolkit/y.sh|1"}


def test_ignore_matches_the_command_without_its_quotes(tmp_path):
    p = tmp_path / "s.jsonl"
    write_jsonl(p, [hook_failure(command='"/Users/someone/.claude/plugins/cache/dev-"process-toolkit/x.sh'),
                    hook_failure(command='"${CLAUDE_PLUGIN_ROOT}"/templates/hooks/process/x.sh')])
    store: dict = {}
    assert health.merge(store, read_segment(p, 0).health, ("dev-process-toolkit",)) == {
        'PreToolUse|"${CLAUDE_PLUGIN_ROOT}"/templates/hooks/process/x.sh|127': 1}
    assert health.merge({}, read_segment(p, 0).health, ("/templates/hooks/process/",)) == {
        'PreToolUse|"~/.claude/plugins/cache/dev-"process-toolkit/x.sh|127': 1}


# -- notification ----------------------------------------------------------------


def _store(n_keys=1, count=30, notified_at=None):
    store: dict = {}
    for i in range(n_keys):
        key = f"PreToolUse|\"$ROOT\"/hooks/hook{i}.sh --flag|127"
        health.merge(store, [event(key=key)] * count)
        store[key]["notified_at"] = notified_at
    return store


def test_due_needs_the_threshold_in_this_run_and_a_week_since_the_last_notice():
    store = _store()
    key = next(iter(store))
    assert health.due(store, {key: 20}, 20, NOW) == [key]
    assert health.due(store, {key: 19}, 20, NOW) == []
    store[key]["notified_at"] = "2026-10-18T03:00:00Z"
    assert health.due(store, {key: 20}, 20, NOW) == []
    store[key]["notified_at"] = "2026-10-13T03:00:00Z"
    assert health.due(store, {key: 20}, 20, NOW) == [key]


def test_notify_message_lists_three_keys_then_plus_n():
    store = _store(n_keys=5)
    keys = list(store)
    store["api_error|529"] = {"kind": "api_error", "count": 40, "first_ts": TS, "last_ts": TS, "sessions": [],
                              "detail": "Overloaded", "notified_at": None}
    message = health.notify_message(store, ["api_error|529", *keys])
    assert message.startswith("agent-reflect: тихие сбои — ")
    assert "хук hook0.sh падает (код 127), 30 раз" in message
    assert "529" in message and "40 раз" in message
    assert "hook2.sh" not in message and "+3" in message
    assert message.endswith("см. agent-reflect recent")


def test_entries_since_filters_by_window_and_ignore():
    store = _store(n_keys=1)
    health.merge(store, [event(key="PreToolUse|~/old.sh|1", ts="2026-09-01T00:00:00Z")])
    store["PreToolUse|~/dev-process-toolkit/x.sh|127"] = dict(store[next(iter(store))])
    since = datetime(2026, 10, 1, tzinfo=timezone.utc)
    keys = [e["key"] for e in health.entries_since(store, since, ignore=("dev-process-toolkit",))]
    assert keys == ['PreToolUse|"$ROOT"/hooks/hook0.sh --flag|127']


def test_describe_shows_kind_count_code_last_ts_and_detail():
    store = _store(n_keys=1)
    (entry,) = health.entries_since(store, None)
    line = health.describe(entry)
    assert "hook_failure" in line and "30" in line and "127" in line and TS in line and "hook0.sh" in line
