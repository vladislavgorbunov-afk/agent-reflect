from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone

import pytest
from conftest import write_jsonl

from agent_reflect import cli, journal
from agent_reflect.config import Config
from agent_reflect.ledger import Lesson, State, load_state, run_lock, save_state


@pytest.fixture
def cfg(tmp_path, monkeypatch):
    c = Config.under(tmp_path)
    monkeypatch.setattr(Config, "default", classmethod(lambda cls: c))
    return c


def _installed(cfg, **kw):
    save_state(State(installed_at="2026-10-01T00:00:00Z", **kw), cfg.state_file)


def test_cli_second_concurrent_run_exits_zero(cfg, monkeypatch):
    called = []
    monkeypatch.setattr(cli.reflect, "run", lambda *a, **k: called.append(1) or {})
    with run_lock(cfg.lock_file) as held:
        assert held
        assert cli.main(["reflect"]) == 0
    assert not called
    assert "already running" in (cfg.logs_dir / "reflect.log").read_text()


def test_backfill_resets_offsets_since_date(cfg, monkeypatch):
    old = cfg.projects_dir / "-p" / "old.jsonl"
    new = cfg.projects_dir / "-p" / "new.jsonl"
    write_jsonl(old, [{"type": "user"}])
    write_jsonl(new, [{"type": "user"}])
    ts = lambda s: datetime.strptime(s, "%Y-%m-%d").timestamp()
    os.utime(old, (ts("2026-09-01"), ts("2026-09-01")))
    os.utime(new, (ts("2026-10-10") + 3600, ts("2026-10-10") + 3600))
    _installed(cfg, sessions={str(old): {"offset": 5}, str(new): {"offset": 5}})
    seen = {}

    def fake_run(c, state, **kw):
        seen["sessions"] = dict(state.sessions)
        seen["max"] = kw["max_segments"]
        return {}

    monkeypatch.setattr(cli.reflect, "run", fake_run)
    assert cli.main(["backfill", "--since", "2026-10-10", "--max-segments", "3"]) == 0
    assert seen["sessions"][str(new)]["offset"] == 0
    assert seen["sessions"][str(old)]["offset"] == 5
    assert seen["max"] == 3
    assert load_state(cfg.state_file).sessions[str(new)]["offset"] == 0


def test_backfill_forgets_the_stored_cwd_so_the_project_is_resolved_again(cfg, monkeypatch):
    # a subagent recorded with its scratchpad cwd before the parent-project rule: re-reading
    # it must resolve the project afresh, not keep the stale cwd that made it no_project
    new = cfg.projects_dir / "-p" / "new.jsonl"
    write_jsonl(new, [{"type": "user"}])
    t = datetime.strptime("2026-10-10", "%Y-%m-%d").timestamp() + 3600
    os.utime(new, (t, t))
    _installed(cfg, sessions={str(new): {"offset": 5, "cwd": "/private/tmp/scratchpad"}})
    seen = {}
    monkeypatch.setattr(cli.reflect, "run", lambda c, state, **kw: seen.update(dict(state.sessions)) or {})
    assert cli.main(["backfill", "--since", "2026-10-10"]) == 0
    assert seen[str(new)] == {"offset": 0}


def test_backfill_resets_subagent_offsets_too(cfg, monkeypatch):
    agent = cfg.projects_dir / "-p" / "s1" / "subagents" / "agent-a1.jsonl"
    write_jsonl(agent, [{"type": "user"}])
    ts = datetime.strptime("2026-10-10", "%Y-%m-%d").timestamp() + 3600
    os.utime(agent, (ts, ts))
    _installed(cfg, sessions={str(agent): {"offset": 5}})
    seen = {}
    monkeypatch.setattr(cli.reflect, "run", lambda c, state, **kw: seen.update(state.sessions) or {})
    assert cli.main(["backfill", "--since", "2026-10-10"]) == 0
    assert seen[str(agent)]["offset"] == 0


def test_backfill_does_not_count_health_twice(cfg, monkeypatch):
    from agent_reflect import reflect

    cfg.claude_settings.parent.mkdir(parents=True, exist_ok=True)
    cfg.claude_settings.write_text(json.dumps({"autoDreamEnabled": False}))
    t = cfg.projects_dir / "-p" / "s1.jsonl"
    hook = {"type": "attachment", "timestamp": "2026-10-10T10:00:00Z", "sessionId": "s1",
            "attachment": {"type": "hook_non_blocking_error", "hookEvent": "Stop", "hookName": "Stop",
                           "command": "~/hooks/sound.sh", "exitCode": 1, "stderr": "no player", "stdout": ""}}
    write_jsonl(t, [hook] * 5)
    ts = datetime.strptime("2026-10-10", "%Y-%m-%d").timestamp() + 3600
    os.utime(t, (ts, ts))
    key = "Stop|~/hooks/sound.sh|1"
    _installed(cfg, sessions={str(t): {"offset": t.stat().st_size}}, subagent_baseline_done=True,
               health={key: {"kind": "hook_failure", "count": 5, "first_ts": "x", "last_ts": "x",
                             "sessions": ["s1"], "detail": "no player", "notified_at": None}})
    real = reflect.run

    def no_model(*a, **k):
        raise AssertionError("no model call expected")

    monkeypatch.setattr(cli.reflect, "run", lambda c, state, **kw: real(c, state, llm=no_model, **kw))
    monkeypatch.setattr(cli.digest, "notify", lambda *a, **k: None)
    assert cli.main(["backfill", "--since", "2026-10-10"]) == 0
    state = load_state(cfg.state_file)
    assert state.health[key]["count"] == 5
    assert state.sessions[str(t)]["offset"] == t.stat().st_size and state.health_offsets == {}


def test_backfill_bad_date_is_usage_error(cfg, capsys):
    assert cli.main(["backfill", "--since", "yesterday"]) == 2
    assert cli.main(["nonsense"]) == 2


def _lesson(i, status):
    return Lesson(id=i, project_root="/p", memory_dir="/m", file=f"auto_{i}.md", title="t", hook="h",
                  cue="c", lesson="l", why="w", confidence="high", repeats=1, evidence=[],
                  status=status, sha256="", created_at="", updated_at="")


def test_status_prints_counts(cfg, capsys):
    t = cfg.projects_dir / "-p" / "s.jsonl"
    write_jsonl(t, [{"type": "user"}])
    size = t.stat().st_size
    st = State(installed_at="2026-10-01T00:00:00Z", sessions={str(t): {"offset": size - 4}},
               lessons={"a": _lesson("a", "live"), "b": _lesson("b", "live"), "c": _lesson("c", "staged")},
               last_consolidate_at="2026-10-13T04:30:00Z", open_prs=[{"url": "https://x/pr/1"}])
    for n in range(7):
        st.event("e", n=n)
    save_state(st, cfg.state_file)
    assert cli.main(["status"]) == 0
    out = capsys.readouterr().out
    assert "2026-10-01T00:00:00Z" in out
    assert "tracked transcripts: 1, unread bytes: 4" in out
    assert "live=2" in out and "staged=1" in out
    assert "2026-10-13T04:30:00Z" in out
    assert "https://x/pr/1" in out
    assert '"n": 6' in out and '"n": 2' in out and '"n": 1' not in out


def test_rollback_list_and_restore(cfg, capsys):
    target = cfg.repo_dir.parent / "skill.md"
    target.write_text("new")
    saved = cfg.backups_dir / "20261018T043000" / "k" / "skill.md"
    saved.parent.mkdir(parents=True)
    saved.write_text("old")
    (saved.parent / "manifest.json").write_text(json.dumps([{"src": str(target), "saved": str(saved)}]))
    assert cli.main(["rollback", "--list"]) == 0
    assert capsys.readouterr().out.split() == ["20261018T043000"]
    assert cli.main(["rollback", "20261018T043000"]) == 0
    assert target.read_text() == "old"
    out = capsys.readouterr().out
    assert "20261018T043000" in out and str(target) in out
    assert cli.main(["rollback", "20990101T000000"]) == 1


def test_rollback_without_id_lists_and_restores_nothing(cfg, capsys):
    target = cfg.repo_dir.parent / "skill.md"
    target.write_text("new")
    saved = cfg.backups_dir / "20261018T043000" / "k" / "skill.md"
    saved.parent.mkdir(parents=True)
    saved.write_text("old")
    (saved.parent / "manifest.json").write_text(json.dumps([{"src": str(target), "saved": str(saved)}]))
    assert cli.main(["rollback"]) == 2
    captured = capsys.readouterr()
    assert "20261018T043000" in captured.out and "usage" in captured.err
    assert target.read_text() == "new"


@pytest.mark.parametrize("argv", [["rollback", "nope"], ["rollback", ".."], ["rollback", "../backups"],
                                  ["rollback", "--memory", ".."], ["rollback", "--memory", "2026-10-25"]])
def test_rollback_refuses_anything_but_a_run_id(cfg, capsys, argv):
    (cfg.backups_dir / "memory").mkdir(parents=True)
    assert cli.main(argv) == 2
    assert "not a run id" in capsys.readouterr().err


def test_rollback_locked_exits_zero_without_restoring(cfg):
    target = cfg.repo_dir.parent / "skill.md"
    target.write_text("new")
    saved = cfg.backups_dir / "20261018T043000" / "k" / "skill.md"
    saved.parent.mkdir(parents=True)
    saved.write_text("old")
    (saved.parent / "manifest.json").write_text(json.dumps([{"src": str(target), "saved": str(saved)}]))
    with run_lock(cfg.lock_file):
        assert cli.main(["rollback", "20261018T043000"]) == 0
    assert target.read_text() == "new"


def test_consolidate_writes_digest_and_notifies(cfg, monkeypatch, dubai):
    _installed(cfg)
    summary = {"new_lessons": [], "promoted": [{"target": "a"}], "prs": ["https://x/1"], "removed": [],
               "suggestions": [], "errors": [], "skipped_no_project": 0}
    monkeypatch.setattr(cli.consolidate, "run", lambda *a, **k: summary)
    sent = []
    monkeypatch.setattr(cli.digest, "notify", lambda m, p=None, **k: sent.append((m, p, k)))
    monkeypatch.setattr(cli, "_now", lambda: datetime(2026, 10, 26, 0, 30, tzinfo=timezone.utc))
    assert cli.main(["consolidate", "--force"]) == 0
    files = list(cfg.digests_dir.glob("*.md"))
    assert len(files) == 1
    assert sent == [("🛠 В скиллы 1 · 🔀 PR 1", files[0],
                     {"title": "📅 agent-reflect · 1–26 октября", "subtitle": "🧠 0 уроков → 🛠 1 скилл",
                      "group": "agent-reflect.week"})]


def test_consolidate_command_writes_chart_into_digest(cfg, monkeypatch):
    _installed(cfg)
    summary = {"new_lessons": [], "promoted": [{"target": "a"}], "prs": [], "removed": [], "suggestions": [],
               "errors": [], "skipped_no_project": 0}
    monkeypatch.setattr(cli.consolidate, "run", lambda *a, **k: summary)
    monkeypatch.setattr(cli.digest, "notify", lambda *a, **k: None)
    assert cli.main(["consolidate", "--force"]) == 0
    (path,) = cfg.digests_dir.glob("*.md")
    lines = path.read_text(encoding="utf-8").splitlines()
    chart = next(i for i, line in enumerate(lines) if line.startswith("## 📊 Неделя · "))
    assert lines[chart + 2] == "```text" and lines.index("## 🆕 Новые уроки · 0") > chart


@pytest.fixture
def dubai():
    """The machine's zone pinned to UTC+4, the one the digest reads local days in."""
    before = os.environ.get("TZ")
    os.environ["TZ"] = "Asia/Dubai"
    time.tzset()
    yield
    if before is None:
        os.environ.pop("TZ", None)
    else:
        os.environ["TZ"] = before
    time.tzset()


def test_consolidate_chart_runs_from_the_previous_consolidate_to_today(cfg, monkeypatch, dubai):
    # 02:00 on the 21st and 01:00 on the 28th, local: the days come from the local clock, not from UTC
    _installed(cfg, last_consolidate_at="2026-10-20T22:00:00Z")
    now = datetime(2026, 10, 27, 21, 0, tzinfo=timezone.utc)
    monkeypatch.setattr(cli, "_now", lambda: now)
    monkeypatch.setattr(cli.digest, "notify", lambda *a, **k: None)
    journal.append(cfg, [{"type": "run", "run": "r1", "kind": "reflect", "trigger": "schedule",
                          "started_at": "2026-10-22T10:00:00Z", "ended_at": "2026-10-22T10:05:00Z",
                          "outcomes": {"repeat": 2}}], now)

    def fake_run(c, state, **k):
        state.last_consolidate_at = "2026-10-27T21:00:00Z"  # consolidate moves it before the digest is made
        return {"new_lessons": [], "promoted": [{"target": "a"}], "prs": [], "removed": [], "suggestions": [],
                "errors": []}

    monkeypatch.setattr(cli.consolidate, "run", fake_run)
    assert cli.main(["consolidate", "--force"]) == 0
    (path,) = cfg.digests_dir.glob("*.md")
    text = path.read_text(encoding="utf-8")
    assert "## 📊 Неделя · 21–28 октября" in text
    rows = [line for line in text.splitlines() if line.startswith("   ") and line[3:5] in
            ("пн", "вт", "ср", "чт", "пт", "сб", "вс")]
    assert [line[3:8] for line in rows] == ["ср 21", "чт 22", "пт 23", "сб 24", "вс 25", "пн 26", "вт 27", "ср 28"]
    assert "   чт 22  🟦🟦  2" in rows


def test_digest_is_written_without_a_chart_the_journal_cannot_draw(cfg, monkeypatch):
    _installed(cfg)
    cfg.journal_file.mkdir(parents=True)  # a journal that cannot be read
    monkeypatch.setattr(cli.digest, "notify", lambda *a, **k: None)
    _consolidate_calls(monkeypatch, {"new_lessons": [], "promoted": [{"target": "a"}], "prs": [], "removed": [],
                                     "suggestions": [], "errors": []})
    assert cli.main(["consolidate", "--force"]) == 0
    (path,) = cfg.digests_dir.glob("*.md")
    text = path.read_text(encoding="utf-8")
    assert "## 🛠 Перенесено в скиллы · 1" in text and "Неделя" not in text
    assert "week chart not drawn" in (cfg.logs_dir / "consolidate.log").read_text()


def test_digest_is_written_without_a_chart_that_fails_to_draw(cfg, monkeypatch):
    _installed(cfg)
    monkeypatch.setattr(cli.digest, "notify", lambda *a, **k: None)

    def bug(*a, **k):
        raise TypeError("'NoneType' object is not subscriptable token=abcdef123456")

    monkeypatch.setattr(cli.digest, "week_block", bug)
    _consolidate_calls(monkeypatch, {"new_lessons": [], "promoted": [{"target": "a"}], "prs": [], "removed": [],
                                     "suggestions": [], "errors": []})
    assert cli.main(["consolidate", "--force"]) == 0
    (path,) = cfg.digests_dir.glob("*.md")
    text = path.read_text(encoding="utf-8")
    assert "## 🛠 Перенесено в скиллы · 1" in text and "Неделя" not in text
    log = (cfg.logs_dir / "consolidate.log").read_text()
    assert "week chart not drawn" in log and "NoneType" in log and "abcdef123456" not in log


def test_a_second_consolidate_in_the_week_keeps_the_first_digest(cfg, monkeypatch):
    _installed(cfg)
    monkeypatch.setattr(cli, "_now", lambda: RUN_AT)
    sent = []
    monkeypatch.setattr(cli.digest, "notify", lambda m, p=None, **k: sent.append(p))
    summary = {"new_lessons": [], "promoted": [{"target": "a"}], "prs": [], "removed": [], "suggestions": [],
               "errors": []}
    _consolidate_calls(monkeypatch, summary)
    assert cli.main(["consolidate", "--force"]) == 0
    assert cli.main(["consolidate", "--force"]) == 0
    first, second = cfg.digests_dir / "2026-W43.md", cfg.digests_dir / "2026-W43-2.md"
    assert {p.name for p in cfg.digests_dir.glob("*.md")} == {first.name, second.name}
    assert sent == [first, second]
    assert [r["digest"] for r in _of_type(cfg, "run")] == [str(first), str(second)]


def test_consolidate_empty_and_dry_run_write_nothing(cfg, monkeypatch, capsys):
    _installed(cfg)
    sent = []
    monkeypatch.setattr(cli.digest, "notify", lambda m, p=None, **k: sent.append(m))
    monkeypatch.setattr(cli.consolidate, "run", lambda *a, **k: {})
    assert cli.main(["consolidate"]) == 0
    monkeypatch.setattr(cli.consolidate, "run", lambda *a, **k: {"dry_run": True, "eligible": ["x"]})
    assert cli.main(["consolidate", "--dry-run"]) == 0
    assert json.loads(capsys.readouterr().out)["eligible"] == ["x"]
    assert not cfg.digests_dir.exists() and not sent


def test_crash_is_recorded_and_notifies_after_three(cfg, monkeypatch):
    _installed(cfg)

    def boom(*a, **k):
        raise RuntimeError("bad token=abcdef123456")

    monkeypatch.setattr(cli.reflect, "run", boom)
    sent = []
    monkeypatch.setattr(cli.digest, "notify", lambda m, p=None, **k: sent.append(m))
    for n in (1, 2, 3):
        assert cli.main(["reflect"]) == 1
        state = load_state(cfg.state_file)
        assert state.consecutive_failures == n
    assert state.events[-1]["kind"] == "run_crashed"
    assert "abcdef123456" not in state.events[-1]["error"]
    assert len(sent) == 1 and "3 раз подряд" in sent[0]
    assert "RuntimeError" in (cfg.logs_dir / "reflect.log").read_text()


def test_keyboard_interrupt_still_saves_state_and_reraises(cfg, monkeypatch):
    _installed(cfg)

    def interrupted(c, state, **k):
        state.event("wrote_memory")
        raise KeyboardInterrupt

    monkeypatch.setattr(cli.reflect, "run", interrupted)
    with pytest.raises(KeyboardInterrupt):
        cli.main(["reflect"])
    state = load_state(cfg.state_file)
    assert [e["kind"] for e in state.events] == ["wrote_memory", "run_crashed"]


def test_sigterm_handler_raises_systemexit_and_state_saved(cfg, monkeypatch):
    import signal
    _installed(cfg)
    before = signal.getsignal(signal.SIGTERM)

    def terminated(c, state, **k):
        state.event("wrote_memory")
        signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)

    monkeypatch.setattr(cli.reflect, "run", terminated)
    with pytest.raises(SystemExit) as info:
        cli.main(["reflect"])
    assert info.value.code == 143
    assert signal.getsignal(signal.SIGTERM) == before
    assert "wrote_memory" in [e["kind"] for e in load_state(cfg.state_file).events]


def test_traceback_in_log_is_scrubbed(cfg, monkeypatch):
    _installed(cfg)

    def boom(*a, **k):
        raise RuntimeError("password=hunter2hunter2")

    monkeypatch.setattr(cli.reflect, "run", boom)
    monkeypatch.setattr(cli.digest, "notify", lambda *a, **k: None)
    cli.main(["reflect"])
    assert "hunter2hunter2" not in (cfg.logs_dir / "reflect.log").read_text()


def test_consolidate_crashes_use_their_own_counter(cfg, monkeypatch):
    _installed(cfg)
    sent = []
    monkeypatch.setattr(cli.digest, "notify", lambda m, p=None, **k: sent.append(m))

    def boom(*a, **k):
        raise RuntimeError("x")

    monkeypatch.setattr(cli.consolidate, "run", boom)
    for _ in range(3):
        assert cli.main(["consolidate", "--force"]) == 1
    state = load_state(cfg.state_file)
    assert state.consolidate_failures == 3 and state.consecutive_failures == 0
    assert len(sent) == 1 and sent[0].startswith("agent-reflect: consolidate падает 3")
    monkeypatch.setattr(cli.consolidate, "run", lambda *a, **k: {})
    assert cli.main(["consolidate"]) == 0
    assert load_state(cfg.state_file).consolidate_failures == 0


def test_corrupt_state_notifies_and_is_left_untouched(cfg, monkeypatch):
    cfg.state_file.parent.mkdir(parents=True)
    cfg.state_file.write_text("{not json")
    called, sent = [], []
    monkeypatch.setattr(cli.reflect, "run", lambda *a, **k: called.append(1) or {})
    monkeypatch.setattr(cli.digest, "notify", lambda m, p=None, **k: sent.append(m))
    assert cli.main(["reflect"]) == 1
    assert cfg.state_file.read_text() == "{not json"
    assert not called
    assert sent == ["agent-reflect: файл состояния не читается, см. logs/reflect.log"]


def test_max_segments_must_be_positive(cfg):
    assert cli.main(["reflect", "--max-segments", "0"]) == 2
    assert cli.main(["backfill", "--since", "2026-10-01", "--max-segments", "-1"]) == 2


def test_install_failure_is_clean_exit_one(cfg, monkeypatch, capsys):
    import subprocess

    def failing(argv, **kw):
        if argv[1] == "bootstrap":
            raise subprocess.CalledProcessError(5, argv, stderr="Bootstrap failed: 5 token=abcdef123456")
        return None

    monkeypatch.setattr(cli.launchd.install, "__defaults__", (failing,))
    assert cli.main(["install"]) == 1
    err = capsys.readouterr().err
    assert "Bootstrap failed" in err and "abcdef123456" not in err


RECENT = "2026-10-20T00:00:00Z"


def _consolidate_calls(monkeypatch, result=None):
    calls = []
    monkeypatch.setattr(cli.consolidate, "run", lambda c, state, **k: calls.append(k) or (result or {}))
    return calls


def test_consolidate_tick_not_due_is_free(cfg, monkeypatch):
    _installed(cfg, last_consolidate_at=datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ"))
    before = cfg.state_file.read_bytes()
    calls = _consolidate_calls(monkeypatch)
    with run_lock(cfg.lock_file) as held:  # a nightly reflect is running
        assert held
        assert cli.main(["consolidate"]) == 0
    assert calls == [] and cfg.state_file.read_bytes() == before
    assert not (cfg.logs_dir / "consolidate.log").exists() and not cfg.journal_file.exists()


def test_due_consolidate_blocked_by_lock_runs_on_a_later_tick(cfg, monkeypatch):
    _installed(cfg)
    calls = _consolidate_calls(monkeypatch)
    with run_lock(cfg.lock_file):
        assert cli.main(["consolidate"]) == 0
    assert calls == []
    assert cli.main(["consolidate"]) == 0
    assert len(calls) == 1


def test_aborted_consolidate_counts_as_failure_and_names_the_reason(cfg, monkeypatch):
    _installed(cfg)
    sent = []
    monkeypatch.setattr(cli.digest, "notify", lambda m, p=None, **k: sent.append(m))
    _consolidate_calls(monkeypatch, {"aborted": "план не получен: claude timed out after 600s"})
    for n in (1, 2, 3):
        assert cli.main(["consolidate"]) == 1
        assert load_state(cfg.state_file).consolidate_failures == n
    assert not cfg.digests_dir.exists()
    assert len(sent) == 1 and sent[0].startswith("agent-reflect: consolidate падает 3")
    assert "claude timed out after 600s" in sent[0]
    assert "claude timed out" in (cfg.logs_dir / "consolidate.log").read_text()


def test_run_error_events_go_to_the_log_one_line_each(cfg, monkeypatch):
    _installed(cfg, events=[{"ts": "t0", "kind": "llm_error", "error": "OLD-ERROR"}])

    def fake_run(c, state, **k):
        state.event("llm_error", transcript="/t.jsonl", error="boom\nsecond line token=abcdef123456")
        state.event("lesson_new", lesson_id="x")
        return {}

    monkeypatch.setattr(cli.reflect, "run", fake_run)
    assert cli.main(["reflect"]) == 0
    lines = (cfg.logs_dir / "reflect.log").read_text().splitlines()
    errors = [ln for ln in lines if "llm_error" in ln]
    assert len(errors) == 1 and "boom" in errors[0] and "second line" in errors[0]
    assert "abcdef123456" not in errors[0]
    assert not any("OLD-ERROR" in ln or "lesson_new" in ln for ln in lines)


def test_consolidate_error_events_are_logged_even_after_events_are_cleared(cfg, monkeypatch):
    _installed(cfg)

    def fake_run(c, state, **k):
        state.event("git_failed", step="publish", error="rejected")
        state.events = []
        return {"new_lessons": [], "promoted": [], "prs": [], "removed": [], "suggestions": [],
                "errors": [{"kind": "git_failed"}]}

    monkeypatch.setattr(cli.consolidate, "run", fake_run)
    monkeypatch.setattr(cli.digest, "notify", lambda *a, **k: None)
    assert cli.main(["consolidate", "--force"]) == 0
    assert "git_failed" in (cfg.logs_dir / "consolidate.log").read_text()


@pytest.mark.parametrize("argv,module", [(["reflect", "--dry-run"], "reflect"),
                                         (["consolidate", "--dry-run"], "consolidate")])
def test_dry_run_never_saves_state(cfg, monkeypatch, capsys, argv, module):
    _installed(cfg)
    before = cfg.state_file.read_bytes()

    def mutating(c, state, **k):
        state.event("would_have_written")
        state.sessions["/t.jsonl"] = {"offset": 99}
        return {"dry_run": True}

    monkeypatch.setattr(getattr(cli, module), "run", mutating)
    assert cli.main(argv) == 0
    assert cfg.state_file.read_bytes() == before


def test_dry_run_crash_does_not_save_state(cfg, monkeypatch):
    _installed(cfg)
    before = cfg.state_file.read_bytes()

    def boom(c, state, **k):
        state.event("half_done")
        raise RuntimeError("x")

    monkeypatch.setattr(cli.reflect, "run", boom)
    assert cli.main(["reflect", "--dry-run"]) == 1
    assert cfg.state_file.read_bytes() == before and not cfg.journal_file.exists()


RECENT_NOW = datetime(2026, 10, 23, 5, 0, tzinfo=timezone.utc)


def _journal(cfg) -> None:
    """A scheduled night with one quiet decision, as the journal holds it."""
    night = "2026-10-22T23:30:00Z"
    write_jsonl(cfg.journal_file, [
        {"type": "run", "run": night, "kind": "reflect", "trigger": "schedule", "started_at": night,
         "ended_at": "2026-10-22T23:31:00Z", "segments_seen": 3, "no_signals": 2, "deferred": 0, "llm_calls": 1,
         "outcomes": {"no_entry": 1}, "errors": 0, "failed": None, "digest": None},
        {"type": "verdict", "run": night, "ts": night, "session": "s1", "agent": "", "transcript": "/t.jsonl",
         "project": "/p", "title": "Тихая сессия", "verdict": "NO_ENTRY", "outcome": "no_entry", "lesson_id": None,
         "file": None, "notes": "Ничего нового."},
    ])


def test_recent_command_reads_journal(cfg, capsys, monkeypatch):
    monkeypatch.setattr(cli, "_now", lambda: RECENT_NOW)
    _installed(cfg)
    _journal(cfg)
    before = cfg.state_file.read_bytes(), cfg.journal_file.read_bytes()

    assert cli.main(["recent"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("🌙 agent-reflect · ночь на ")
    assert "🗂 разобрано: 1 из 3" in out
    assert (cfg.state_file.read_bytes(), cfg.journal_file.read_bytes()) == before


@pytest.mark.parametrize("flag", ["-v", "--verbose"])
def test_recent_verbose_flag(cfg, capsys, monkeypatch, flag):
    monkeypatch.setattr(cli, "_now", lambda: RECENT_NOW)
    _journal(cfg)

    assert cli.main(["recent"]) == 0
    out = capsys.readouterr().out
    assert "  ⬜ … и ещё 1 раз «ничего» (с причинами: recent -v)" in out
    assert "Ничего нового" not in out
    assert cli.main(["recent", flag]) == 0
    assert "  ⬜ Тихая сессия — Ничего нового." in capsys.readouterr().out


def test_recent_hours_must_be_positive(cfg):
    assert cli.main(["recent", "--hours", "0"]) == 2


def test_recent_without_state_or_journal_writes_nothing(cfg, capsys):
    assert cli.main(["recent", "--hours", "6"]) == 0
    assert capsys.readouterr().out.startswith("🌙 agent-reflect · последние 6 ч\n")
    assert not cfg.journal_file.exists() and not cfg.state_file.exists()


RUN_AT = datetime(2026, 10, 23, 0, 30, tzinfo=timezone.utc)
RUN_ID = "2026-10-23T00:30:00Z"


def _verdict(**kw):
    return {"ts": "2026-10-23T00:31:00Z", "session": "s1", "agent": "", "transcript": "/t.jsonl", "project": "/p",
            "title": "Сессия", "verdict": "NEW_LESSON", "outcome": "new_live", "lesson_id": "x",
            "file": "auto_x.md", "notes": "Новое.", **kw}


def _result(**kw):
    return {"segments_seen": 3, "no_signals": 1, "deferred": 0, "llm_calls": 2, "errors": 0,
            "outcomes": {"new_live": 1, "no_entry": 1}, "verdicts": [_verdict()], **kw}


def _of_type(cfg, kind):
    assert cfg.journal_file.exists()
    return [r for r in journal.load(cfg) if r["type"] == kind]


def _done_line(cfg, cmd):
    return next(ln for ln in (cfg.logs_dir / f"{cmd}.log").read_text().splitlines() if f"{cmd} done" in ln)


def test_reflect_command_appends_run_verdicts_and_events(cfg, monkeypatch):
    _installed(cfg)
    monkeypatch.setattr(cli, "_now", lambda: RUN_AT)

    def fake_run(c, state, **k):
        state.event("lesson_new", lesson_id="x")
        return _result()

    monkeypatch.setattr(cli.reflect, "run", fake_run)
    assert cli.main(["reflect"]) == 0

    (run,) = _of_type(cfg, "run")
    assert (run["run"], run["kind"], run["failed"], run["digest"]) == (RUN_ID, "reflect", None, None)
    assert (run["segments_seen"], run["no_signals"], run["llm_calls"]) == (3, 1, 2)
    assert run["outcomes"] == {"new_live": 1, "no_entry": 1}
    (verdict,) = _of_type(cfg, "verdict")
    assert verdict["run"] == RUN_ID and verdict["lesson_id"] == "x"
    (event,) = _of_type(cfg, "event")
    assert event["run"] == RUN_ID and event["kind"] == "lesson_new"
    line = _done_line(cfg, "reflect")
    assert "segments_seen" in line and "verdicts" not in line


def test_runs_record_the_live_lessons_of_each_project_at_their_end(cfg, monkeypatch):
    lessons = {i: _lesson(i, status) for i, status in (("a", "live"), ("b", "live"), ("c", "staged"),
                                                       ("d", "rejected"))}
    lessons["e"] = Lesson(**{**vars(_lesson("e", "live")), "project_root": "/q"})
    _installed(cfg, lessons=lessons)
    monkeypatch.setattr(cli.digest, "notify", lambda *a, **k: None)

    def goes_live(c, state, **k):
        state.lessons["c"].status = "live"
        return _result()

    def crashes(c, state, **k):
        state.lessons["a"].status = "staged"
        raise RuntimeError("boom")

    monkeypatch.setattr(cli.reflect, "run", goes_live)
    assert cli.main(["reflect"]) == 0
    monkeypatch.setattr(cli.consolidate, "run", lambda c, state, **k: {})
    assert cli.main(["consolidate", "--force"]) == 0
    monkeypatch.setattr(cli.reflect, "run", crashes)
    assert cli.main(["backfill", "--since", "2026-10-01"]) == 1

    assert [(r["kind"], r["live"]) for r in _of_type(cfg, "run")] == [
        ("reflect", {"/p": 3, "/q": 1}), ("consolidate", {"/p": 3, "/q": 1}), ("backfill", {"/p": 2, "/q": 1})]


@pytest.mark.parametrize("parent,trigger", [(1, "schedule"), (4242, "manual")])
def test_scheduled_trigger_when_parent_is_launchd(cfg, monkeypatch, parent, trigger):
    _installed(cfg)
    monkeypatch.setattr(cli.os, "getppid", lambda: parent)
    assert cli._trigger() == trigger
    monkeypatch.setattr(cli.reflect, "run", lambda c, state, **k: _result())
    monkeypatch.setattr(cli.consolidate, "run", lambda c, state, **k: {})
    assert cli.main(["reflect"]) == 0
    assert cli.main(["consolidate", "--force"]) == 0
    assert [r["trigger"] for r in _of_type(cfg, "run")] == [trigger, trigger]


def test_backfill_is_journaled_as_manual_backfill(cfg, monkeypatch):
    _installed(cfg)
    monkeypatch.setattr(cli, "_now", lambda: RUN_AT)
    monkeypatch.setattr(cli.os, "getppid", lambda: 1)  # even a launchd parent does not make it scheduled

    def fake_run(c, state, **k):
        state.event("lesson_new", lesson_id="x")
        return _result()

    monkeypatch.setattr(cli.reflect, "run", fake_run)
    assert cli.main(["backfill", "--since", "2026-10-01"]) == 0

    (run,) = _of_type(cfg, "run")
    assert (run["run"], run["kind"], run["trigger"], run["failed"]) == (RUN_ID, "backfill", "manual", None)
    assert run["segments_seen"] == 3
    assert [v["run"] for v in _of_type(cfg, "verdict")] == [RUN_ID]
    assert [(e["run"], e["kind"]) for e in _of_type(cfg, "event")] == [(RUN_ID, "lesson_new")]
    assert "verdicts" not in _done_line(cfg, "backfill")


def test_the_first_journal_write_does_not_count_this_run_twice(cfg, monkeypatch):
    # with no journal yet the first append imports the logs, so the `done` line of this very run
    # must not be in the log by then
    _installed(cfg)
    cfg.logs_dir.mkdir(parents=True)
    (cfg.logs_dir / "reflect.log").write_text(
        '2026-10-22 03:31:43,154 INFO reflect done: {"segments_seen": 1, "llm_calls": 0, "errors": 0, '
        '"outcomes": {}}\n')
    monkeypatch.setattr(cli, "_now", lambda: RUN_AT)
    monkeypatch.setattr(cli.reflect, "run", lambda c, state, **k: _result())
    assert cli.main(["reflect"]) == 0

    runs = _of_type(cfg, "run")
    assert len(runs) == 2
    assert [r["run"] for r in runs].count(RUN_ID) == 1
    assert len(_of_type(cfg, "verdict")) == 1


def test_consolidate_abort_is_journaled_as_failed(cfg, monkeypatch):
    _installed(cfg)
    monkeypatch.setattr(cli, "_now", lambda: RUN_AT)
    monkeypatch.setattr(cli.digest, "notify", lambda *a, **k: None)
    _consolidate_calls(monkeypatch, {"aborted": "план не получен: claude timed out after 600s"})
    assert cli.main(["consolidate"]) == 1

    (run,) = _of_type(cfg, "run")
    assert (run["run"], run["kind"], run["digest"]) == (RUN_ID, "consolidate", None)
    assert "claude timed out after 600s" in run["failed"]


def test_consolidate_success_records_digest_path(cfg, monkeypatch):
    _installed(cfg)
    monkeypatch.setattr(cli.digest, "notify", lambda *a, **k: None)
    summary = {"new_lessons": [], "promoted": [{"target": "a"}], "prs": [], "removed": [], "suggestions": [],
               "errors": [{"kind": "git_failed"}], "skipped_no_project": 0}
    _consolidate_calls(monkeypatch, summary)
    assert cli.main(["consolidate", "--force"]) == 0

    (path,) = cfg.digests_dir.glob("*.md")
    (run,) = _of_type(cfg, "run")
    assert (run["kind"], run["digest"], run["failed"]) == ("consolidate", str(path), None)
    assert run["errors"] == 0  # the summary's `errors` lists what the digest tells, it is no counter


def test_consolidate_with_nothing_to_do_is_journaled_plain(cfg, monkeypatch):
    _installed(cfg)
    _consolidate_calls(monkeypatch, {})
    assert cli.main(["consolidate"]) == 0

    (run,) = _of_type(cfg, "run")
    assert (run["kind"], run["digest"], run["failed"]) == ("consolidate", None, None)


def test_consolidate_journals_events_although_it_replaces_them(cfg, monkeypatch):
    _installed(cfg, events=[{"ts": "t0", "kind": "old_event"}])
    monkeypatch.setattr(cli, "_now", lambda: RUN_AT)

    def fake_run(c, state, **k):
        state.event("git_failed", step="publish", error="rejected")
        state.events = [{"ts": "t1", "kind": "kept_event"}]
        return {}

    monkeypatch.setattr(cli.consolidate, "run", fake_run)
    assert cli.main(["consolidate"]) == 0
    assert [(e["run"], e["kind"]) for e in _of_type(cfg, "event")] == [
        (RUN_ID, "git_failed"), (RUN_ID, "kept_event")]


@pytest.mark.parametrize("argv,module,kind", [(["reflect"], "reflect", "reflect"),
                                              (["backfill", "--since", "2026-10-01"], "reflect", "backfill"),
                                              (["consolidate", "--force"], "consolidate", "consolidate")])
def test_crash_is_journaled(cfg, monkeypatch, argv, module, kind):
    _installed(cfg)
    monkeypatch.setattr(cli, "_now", lambda: RUN_AT)
    monkeypatch.setattr(cli.digest, "notify", lambda *a, **k: None)

    def boom(*a, **k):
        raise RuntimeError("boom token=abcdef123456")

    monkeypatch.setattr(getattr(cli, module), "run", boom)
    assert cli.main(argv) == 1

    (run,) = _of_type(cfg, "run")
    assert (run["run"], run["kind"], run["ended_at"]) == (RUN_ID, kind, RUN_ID)
    assert "boom" in run["failed"] and "abcdef123456" not in run["failed"]
    assert run["segments_seen"] == 0 and run["digest"] is None
    assert not _of_type(cfg, "verdict")


def test_crash_journals_the_events_added_before_it(cfg, monkeypatch):
    _installed(cfg, events=[{"ts": "t0", "kind": "old_event"}])
    monkeypatch.setattr(cli, "_now", lambda: RUN_AT)

    def half_done(c, state, **k):
        state.event("lesson_new", lesson_id="x")
        raise RuntimeError("boom")

    monkeypatch.setattr(cli.reflect, "run", half_done)
    assert cli.main(["reflect"]) == 1
    assert [(e["run"], e["kind"]) for e in _of_type(cfg, "event")] == [
        (RUN_ID, "lesson_new"), (RUN_ID, "run_crashed")]


def test_consolidate_crash_journals_events_although_it_replaced_them(cfg, monkeypatch):
    _installed(cfg, events=[{"ts": "t0", "kind": "old_event"}])
    monkeypatch.setattr(cli, "_now", lambda: RUN_AT)
    monkeypatch.setattr(cli.digest, "notify", lambda *a, **k: None)

    def half_done(c, state, **k):
        state.event("git_failed", step="publish", error="rejected")
        state.events = [{"ts": "t1", "kind": "kept_event"}]
        raise RuntimeError("boom")

    monkeypatch.setattr(cli.consolidate, "run", half_done)
    assert cli.main(["consolidate", "--force"]) == 1
    assert [(e["run"], e["kind"]) for e in _of_type(cfg, "event")] == [
        (RUN_ID, "git_failed"), (RUN_ID, "kept_event"), (RUN_ID, "run_crashed")]


@pytest.mark.parametrize("argv,module", [(["reflect", "--dry-run"], "reflect"),
                                         (["consolidate", "--dry-run"], "consolidate")])
def test_dry_run_writes_no_journal(cfg, monkeypatch, argv, module):
    _installed(cfg)
    monkeypatch.setattr(getattr(cli, module), "run", lambda c, state, **k: {"dry_run": True})
    assert cli.main(argv) == 0
    assert not cfg.journal_file.exists()


def test_dry_run_output_and_log_carry_no_verdicts(cfg, monkeypatch, capsys):
    _installed(cfg)
    monkeypatch.setattr(cli.reflect, "run", lambda c, state, **k: {"would_call": [10], "verdicts": []})
    assert cli.main(["reflect", "--dry-run"]) == 0
    assert json.loads(capsys.readouterr().out) == {"would_call": [10]}
    assert "verdicts" not in _done_line(cfg, "reflect")


def test_journal_write_failure_does_not_fail_the_run(cfg, monkeypatch):
    _installed(cfg)
    monkeypatch.setattr(cli.reflect, "run", lambda c, state, **k: _result())

    def broken(*a, **k):
        raise OSError("disk full token=abcdef123456")

    monkeypatch.setattr(cli.journal, "append", broken)
    assert cli.main(["reflect"]) == 0

    state = load_state(cfg.state_file)
    (event,) = [e for e in state.events if e["kind"] == "journal_failed"]
    assert "disk full" in event["error"] and "abcdef123456" not in event["error"]
    assert state.consecutive_failures == 0
    text = (cfg.logs_dir / "reflect.log").read_text()
    assert "disk full" in text and "abcdef123456" not in text and "reflect done" in text


@pytest.mark.parametrize("broken", ["live_counts", "run_records", "append"])
def test_a_bug_in_the_journal_does_not_fail_the_run(cfg, monkeypatch, broken):
    _installed(cfg)
    monkeypatch.setattr(cli.reflect, "run", lambda c, state, **k: _result())

    def bug(*a, **k):
        raise TypeError("unsupported operand token=abcdef123456")

    monkeypatch.setattr(cli.journal, broken, bug)
    assert cli.main(["reflect"]) == 0

    state = load_state(cfg.state_file)
    (event,) = [e for e in state.events if e["kind"] == "journal_failed"]
    assert "unsupported operand" in event["error"] and "abcdef123456" not in event["error"]
    assert state.consecutive_failures == 0
    text = (cfg.logs_dir / "reflect.log").read_text()
    assert "journal not written" in text and "abcdef123456" not in text and "reflect done" in text


def test_journal_that_cannot_be_encoded_does_not_fail_the_run(cfg, monkeypatch):
    # a lone surrogate in a title or note: json.dumps(ensure_ascii=False) writes it, utf-8 refuses it
    _installed(cfg)
    monkeypatch.setattr(cli.reflect, "run", lambda c, state, **k: _result(verdicts=[_verdict(title="a\ud800b")]))
    assert cli.main(["reflect"]) == 0

    state = load_state(cfg.state_file)
    assert [e["kind"] for e in state.events] == ["journal_failed"]
    assert state.consecutive_failures == 0
    assert not cfg.journal_file.exists() and not list(cfg.state_dir.glob("journal.jsonl.*"))
    assert "reflect done" in (cfg.logs_dir / "reflect.log").read_text()


def test_journal_encode_error_is_caught_wherever_it_is_raised(cfg, monkeypatch):
    _installed(cfg)
    monkeypatch.setattr(cli.reflect, "run", lambda c, state, **k: _result())

    def refuses(*a, **k):
        raise UnicodeEncodeError("utf-8", "\ud800", 0, 1, "surrogates not allowed")

    monkeypatch.setattr(cli.journal, "append", refuses)
    assert cli.main(["reflect"]) == 0
    assert [e["kind"] for e in load_state(cfg.state_file).events] == ["journal_failed"]


def test_journal_write_failure_does_not_hide_a_crash(cfg, monkeypatch):
    _installed(cfg)

    def boom(*a, **k):
        raise RuntimeError("boom")

    def broken(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(cli.reflect, "run", boom)
    monkeypatch.setattr(cli.journal, "append", broken)
    assert cli.main(["reflect"]) == 1
    state = load_state(cfg.state_file)
    assert [e["kind"] for e in state.events] == ["run_crashed", "journal_failed"]
    assert state.consecutive_failures == 1


def _scheduled(monkeypatch, clock: list, trigger: list, result: dict | None = None) -> list:
    """reflect at clock[0] started by trigger[0]; the notifications it sends end up in the list returned."""
    monkeypatch.setattr(cli, "_now", lambda: clock[0])
    monkeypatch.setattr(cli, "_trigger", lambda: trigger[0])
    monkeypatch.setattr(cli.reflect, "run", lambda *a, **k: dict(result or {}))
    sent = []
    monkeypatch.setattr(cli.digest, "notify", lambda m, p=None, **k: sent.append((m, p, k)))
    return sent


def test_the_nightly_reflect_writes_the_report_and_notifies(cfg, monkeypatch, dubai):
    _installed(cfg)
    sent = _scheduled(monkeypatch, [RUN_AT], ["schedule"],
                      {"segments_seen": 3, "no_signals": 2, "llm_calls": 1, "outcomes": {"no_entry": 1}})
    assert cli.main(["reflect"]) == 0
    path = cfg.reports_dir / "2026-10-23.md"  # 04:30 in Dubai
    text = path.read_text(encoding="utf-8")
    assert text.startswith("```text\n🌙 agent-reflect · ночь на 23 октября · 04:30 · ✅ без ошибок\n")
    assert "── 🗂 РАЗОБРАНО · 1 из 3 " in text and text.endswith("\n```\n")
    assert sent == [("✍️ Память 0 · 🚨 Требует тебя 0 · 🗂 Разобрано 1 из 3 · 🩺 Сбои 0", path,
                     {"title": "🌙 agent-reflect · ночь на 23 октября", "subtitle": "😴 тихая ночь · ✅ без ошибок",
                      "group": "agent-reflect.night"})]


def test_only_the_first_scheduled_reflect_of_the_night_reports(cfg, monkeypatch, dubai):
    _installed(cfg)
    clock, trigger = [RUN_AT], ["manual"]
    sent = _scheduled(monkeypatch, clock, trigger)
    assert cli.main(["reflect"]) == 0  # by hand: no report
    assert not cfg.reports_dir.exists() and not sent
    clock[0], trigger[0] = RUN_AT.replace(hour=1), "schedule"
    assert cli.main(["reflect"]) == 0  # the night's
    clock[0] = RUN_AT.replace(hour=6)
    assert cli.main(["reflect"]) == 0  # at login later that day
    assert [p for _, p, _ in sent] == [cfg.reports_dir / "2026-10-23.md"]
    clock[0] = RUN_AT.replace(day=24)
    assert cli.main(["reflect"]) == 0  # the next night
    assert [p.name for _, p, _ in sent] == ["2026-10-23.md", "2026-10-24.md"]


def test_a_dry_run_reports_nothing(cfg, monkeypatch, dubai, capsys):
    _installed(cfg)
    sent = _scheduled(monkeypatch, [RUN_AT], ["schedule"])
    assert cli.main(["reflect", "--dry-run"]) == 0
    assert not cfg.reports_dir.exists() and not sent


def test_a_broken_report_never_fails_the_run(cfg, monkeypatch, dubai):
    _installed(cfg)
    sent = _scheduled(monkeypatch, [RUN_AT], ["schedule"])

    def bug(*a, **k):
        raise RuntimeError("report bug token=abcdef123456")

    monkeypatch.setattr(cli.recent, "render", bug)
    assert cli.main(["reflect"]) == 0
    assert not sent and len(_of_type(cfg, "run")) == 1
    log = (cfg.logs_dir / "reflect.log").read_text()
    assert "night report not written" in log and "abcdef123456" not in log
