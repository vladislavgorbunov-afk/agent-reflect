"""Error triggers, blocking rules, verdicts on moves, fact lessons and the second look on another model.

- the second look at a new lesson runs on another model: two samples of one model share its biases;
- a lesson whose mistake shows only in the error text gets a trigger over that text;
- a lesson broken although the hook showed it can be made a blocking rule, by the user only;
- a lesson moved into a skill is still watched, and the move gets a verdict;
- a fact the session paid to discover is a lesson of its own kind, with how to check it again.
"""
from __future__ import annotations

from dataclasses import replace

from agent_reflect import reflect
from agent_reflect.config import Config
from agent_reflect.reflect import VERIFY_SCHEMA


# -- the second look runs on another model ---------------------------------------------------


def test_the_second_look_at_a_lesson_runs_on_the_verify_model_not_the_reflect_model(tmp_path):
    cfg = Config.under(tmp_path)
    assert cfg.verify_model != cfg.reflect_model
    seen = []

    def llm(cfg_, model, prompt, schema, *, effort):
        seen.append((model, schema is VERIFY_SCHEMA))
        return {"notes": "ok", "supported": True}

    verify = reflect._verifier(cfg, llm, {"verify_calls": 0, "errors": 0}, "segment")
    assert verify({"cue": "c", "lesson": "l", "why": "w", "evidence": "e"}, []) == (True, "ok")
    assert seen == [(cfg.verify_model, True)]


def test_the_verify_model_can_be_set_apart_from_the_reflect_model(tmp_path):
    cfg = replace(Config.under(tmp_path), verify_model="claude-haiku-5-5")
    seen = []

    def llm(cfg_, model, prompt, schema, *, effort):
        seen.append(model)
        return {"notes": "ok", "supported": False}

    verify = reflect._verifier(cfg, llm, {"verify_calls": 0, "errors": 0}, "segment")
    assert verify({"cue": "c", "lesson": "l", "why": "w", "evidence": "e"}, [])[0] is False
    assert seen == ["claude-haiku-5-5"]


# -- a trigger over the error text -----------------------------------------------------------

import json  # noqa: E402
import re  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
from pathlib import Path  # noqa: E402
from types import SimpleNamespace  # noqa: E402

import pytest  # noqa: E402

from agent_reflect import care, guard, guard_hook  # noqa: E402
from agent_reflect.ledger import Lesson, State  # noqa: E402
from agent_reflect.reflect import apply_verdict, trigger_hits, valid_error_trigger  # noqa: E402
from agent_reflect.transcript import read_segment  # noqa: E402
from conftest import write_jsonl  # noqa: E402

TS = "2026-10-19T10:00:00.000Z"
NOW = datetime(2026, 10, 20, 3, 0, 0, tzinfo=timezone.utc)
NOW_ISO = "2026-10-20T03:00:00Z"
REJECTED = "To github.com:x/y.git\n ! [rejected]        main -> main (fetch first)\nerror: failed to push some refs"
ERROR_TRIGGER = r"\[rejected\].*\(fetch first\)"


def _call(tool_id, command, session="s1"):
    return {"type": "assistant", "timestamp": TS, "sessionId": session, "cwd": "/proj",
            "message": {"content": [{"type": "tool_use", "id": tool_id, "name": "Bash",
                                     "input": {"command": command}}]}}


def _result(tool_id, text, error=False, session="s1"):
    return {"type": "user", "timestamp": TS, "sessionId": session, "cwd": "/proj",
            "message": {"content": [{"type": "tool_result", "tool_use_id": tool_id, "is_error": error,
                                     "content": text}]}}


def _push_segment(tmp_path, name="p.jsonl", session="s1", fixed=True):
    """`git push` is rejected twice; a pull with rebase before the push is the fix. No pattern over
    the command can tell them apart: the working command is a `git push` too."""
    entries = [
        _call("t1", "git push", session), _result("t1", REJECTED, error=True, session=session),
        _call("t2", "git push origin main", session), _result("t2", REJECTED, error=True, session=session),
    ]
    if fixed:
        entries += [_call("t3", "git pull --rebase && git push", session),
                    _result("t3", "To github.com:x/y.git\n   abc..def  main -> main", session=session)]
    path = tmp_path / name
    write_jsonl(path, entries)
    return read_segment(path, 0)


def test_an_error_trigger_must_match_a_failure_of_the_segment(tmp_path):
    seg = _push_segment(tmp_path)
    assert valid_error_trigger(seg, ERROR_TRIGGER) == (ERROR_TRIGGER, "")
    assert valid_error_trigger(seg, r"merge conflict")[1] == "matches no failing output of the segment"


def test_an_error_trigger_must_not_match_what_the_working_command_printed(tmp_path):
    assert valid_error_trigger(_push_segment(tmp_path), r"main -> main")[1] == "matches the output that worked"


@pytest.mark.parametrize("trigger", [r"error", r"failed", r".*", r"Exit code \d+", r"Traceback"])
def test_an_error_trigger_that_matches_any_failure_is_refused(tmp_path, trigger):
    assert valid_error_trigger(_push_segment(tmp_path), trigger)[1] == "matches any failure"


@pytest.fixture
def env(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    memory = tmp_path / "memory"
    memory.mkdir()
    (memory / "MEMORY.md").write_text("# Memory Index\n", encoding="utf-8")
    return SimpleNamespace(project=project, memory=memory, cfg=Config.under(tmp_path), state=State())


PUSH = {
    "verdict": "NEW", "kind": "tool", "title": "git push отклонён: сначала pull --rebase",
    "hook": "push отклонён (fetch first) → git pull --rebase, потом push",
    "cue": "push отклонён с пометкой fetch first", "lesson": "Сначала `git pull --rebase`, потом `git push`.",
    "why": "Дважды push отклонён.", "evidence": "[1] [rejected]",
    "refs": [{"ref": "1", "quote": "(fetch first)"}], "confidence": "medium", "error_trigger": ERROR_TRIGGER,
}


def _only(env):
    (lesson,) = env.state.lessons.values()
    return lesson


def test_a_tool_lesson_keeps_an_error_trigger_that_holds_and_drops_one_that_does_not(env, tmp_path):
    seg = _push_segment(tmp_path)
    assert apply_verdict(env.state, env.cfg, seg, env.project, env.memory, PUSH, NOW_ISO) == "new_staged"
    assert _only(env).error_trigger == ERROR_TRIGGER
    env.state.lessons.clear()
    bad = {**PUSH, "error_trigger": "error", "cue": "другой"}
    assert apply_verdict(env.state, env.cfg, seg, env.project, env.memory, bad, NOW_ISO) == "new_staged"
    assert _only(env).error_trigger == ""
    assert any(e["kind"] == "trigger_dropped" and e.get("on") == "error" for e in env.state.events)


def test_a_failure_whose_text_matches_a_lessons_error_trigger_is_a_repeat_without_the_model(env, tmp_path):
    apply_verdict(env.state, env.cfg, _push_segment(tmp_path), env.project, env.memory, PUSH, NOW_ISO)
    lesson = _only(env)
    again = _push_segment(tmp_path, "q.jsonl", session="s2", fixed=False)
    hits = trigger_hits(env.state, env.cfg, again, env.project, NOW_ISO)
    assert [(h["verdict"], h["lesson_id"], h["outcome"]) for h in hits] == [("REPEAT", lesson.id, "repeat")]
    assert env.state.lessons[lesson.id].repeats == 2


def _lesson(lesson_id="e1", status="live", **over):
    base = dict(id=lesson_id, project_root="/Users/u/git/app", memory_dir="/m", file=f"auto_{lesson_id}.md",
                title="push", hook="h", cue="c", lesson="Сначала `git pull --rebase`.", why="w",
                confidence="medium", repeats=2, evidence=[], status=status, sha256="",
                created_at="2026-10-20T00:00:00Z", updated_at="2026-10-20T00:00:00Z")
    return Lesson(**{**base, **over})


def test_the_rules_carry_an_error_trigger_of_a_trusted_lesson():
    state = State(lessons={"e1": _lesson(error_trigger=ERROR_TRIGGER), "e2": _lesson("e2", status="staged",
                                                                                         error_trigger=ERROR_TRIGGER)})
    assert [(r["id"], r["on"], r["pattern"]) for r in guard.rules(state)] == [("e1", "error", ERROR_TRIGGER)]


def _hook(tmp_path, payload, rules):
    guard_dir = tmp_path / "guard"
    guard_dir.mkdir(exist_ok=True)
    (guard_dir / guard.RULES_NAME).write_text(json.dumps({"rules": rules}), encoding="utf-8")
    proc = subprocess.run([sys.executable, guard_hook.__file__], input=json.dumps(payload), text=True,
                          capture_output=True, env={"AGENT_REFLECT_GUARD_DIR": str(guard_dir), "PATH": "/usr/bin:/bin"})
    fires = guard_dir / guard.FIRES_NAME
    return proc, (fires.read_text(encoding="utf-8").splitlines() if fires.exists() else [])


ERROR_RULES = guard.rules(State(lessons={"e1": _lesson(error_trigger=ERROR_TRIGGER)}))
TRANSCRIPT = "/h/.claude/projects/-Users-u-git-app/s1.jsonl"


def test_the_hook_shows_the_lesson_after_a_failure_whose_error_matches(tmp_path):
    payload = {"session_id": "s1", "transcript_path": TRANSCRIPT, "cwd": "/x", "hook_event_name": "PostToolUseFailure",
               "tool_name": "Bash", "tool_input": {"command": "git push"}, "error": REJECTED}
    proc, fires = _hook(tmp_path, payload, ERROR_RULES)
    out = json.loads(proc.stdout)["hookSpecificOutput"]
    assert out["hookEventName"] == "PostToolUseFailure"
    assert "уже встречалась" in out["additionalContext"] and "git pull --rebase" in out["additionalContext"]
    assert [json.loads(line)["on"] for line in fires] == ["error"]


def _after_success(command, stderr):
    return {"session_id": "s1", "transcript_path": TRANSCRIPT, "cwd": "/x", "hook_event_name": "PostToolUse",
            "tool_name": "Bash", "tool_input": {"command": command},
            "tool_response": {"stdout": "", "stderr": stderr, "interrupted": False}}


def test_the_hook_reads_the_output_of_a_command_that_failed_quietly(tmp_path):
    proc, _ = _hook(tmp_path, _after_success("git push || true", REJECTED + "\nExit code 1"), ERROR_RULES)
    assert json.loads(proc.stdout)["hookSpecificOutput"]["hookEventName"] == "PostToolUse"


def test_the_output_of_a_command_that_worked_is_not_read_for_errors(tmp_path):
    # a test run whose fixtures print the error is no failure: nor does the night count it as one
    proc, fires = _hook(tmp_path, _after_success("make check", REJECTED + "\n1155 passed"), ERROR_RULES)
    assert (proc.stdout, fires) == ("", [])


def test_the_hooks_failure_lines_are_the_nightly_ones():
    from agent_reflect import transcript

    assert guard_hook._SILENT_RE.pattern == transcript._SILENT_RE.pattern
    assert guard_hook._SILENT_RE.flags == transcript._SILENT_RE.flags


def test_the_hook_shows_an_error_lesson_once_a_session(tmp_path):
    payload = {"session_id": "s1", "transcript_path": TRANSCRIPT, "cwd": "/x", "hook_event_name": "PostToolUseFailure",
               "tool_name": "Bash", "tool_input": {"command": "git push"}, "error": REJECTED}
    _hook(tmp_path, payload, ERROR_RULES)
    proc, fires = _hook(tmp_path, payload, ERROR_RULES)
    assert proc.stdout == "" and len(fires) == 1


def test_an_error_rule_never_fires_before_the_command_runs(tmp_path):
    payload = {"session_id": "s1", "transcript_path": TRANSCRIPT, "cwd": "/x", "hook_event_name": "PreToolUse",
               "tool_name": "Bash", "tool_input": {"command": "echo ' ! [rejected] main -> main (fetch first)'"}}
    proc, fires = _hook(tmp_path, payload, ERROR_RULES)
    assert (proc.stdout, fires) == ("", [])


def test_the_hook_goes_in_after_bash_and_after_a_failed_bash_and_comes_out_of_both(tmp_path):
    cfg = Config.under(tmp_path)
    assert guard.install_hook(cfg) is True and guard.install_hook(cfg) is False
    hooks = json.loads(cfg.claude_settings.read_text(encoding="utf-8"))["hooks"]
    assert sorted(hooks) == ["PostToolUse", "PostToolUseFailure", "PreToolUse", "SessionStart"]
    for event in ("PostToolUse", "PostToolUseFailure"):
        (entry,) = hooks[event]
        assert entry["matcher"] == "^Bash$" and guard.HOOK_MARK in entry["hooks"][0]["command"]
    assert guard.uninstall_hook(cfg) is True and guard.uninstall_hook(cfg) is False
    assert json.loads(cfg.claude_settings.read_text(encoding="utf-8")) == {}


def test_a_broken_lesson_with_only_an_error_trigger_is_not_called_unhelped():
    state = State(lessons={"e1": _lesson(violations=2, error_trigger=ERROR_TRIGGER, live_since=NOW_ISO)})
    assert care.lessons(state, NOW) == []


def test_an_error_trigger_typed_by_hand_is_checked():
    assert guard.check_trigger(ERROR_TRIGGER, "error") == ""
    assert guard.check_trigger(r"fatal", "error") == "matches any failure"
    assert guard.check_trigger(r"Permission denied", "error") == "matches any failure"


def test_guard_set_error_writes_the_error_trigger_and_lists_it(tmp_path, monkeypatch, capsys):
    from agent_reflect.cli import main
    from agent_reflect.ledger import load_state, save_state

    cfg = Config.under(tmp_path)
    monkeypatch.setattr(Config, "default", classmethod(lambda cls: cfg))
    save_state(State(lessons={"e1": _lesson()}), cfg.state_file)
    assert main(["guard", "set", "--error", "e1", ERROR_TRIGGER]) == 0
    lesson = load_state(cfg.state_file).lessons["e1"]
    assert (lesson.trigger, lesson.error_trigger) == ("", ERROR_TRIGGER)
    assert main(["guard", "set", "--error", "e1", "failed"]) == 2
    assert main(["guard", "set", "--error", "--path", "e1", "/x/"]) == 2
    capsys.readouterr()
    assert main(["guard", "test", "--error", REJECTED]) == 0 and capsys.readouterr().out.startswith("e1  ")
    assert main(["guard", "list"]) == 0 and f"ошибка /{ERROR_TRIGGER}/" in capsys.readouterr().out
    assert main(["guard", "unset", "--error", "e1"]) == 0 and load_state(cfg.state_file).lessons["e1"].error_trigger == ""


# -- a lesson broken although the hook showed it can be made blocking, by the user only -----------

ECHO = r"\becho\s+=+(\s|;|&|$)"


def _rules_of(**lessons):
    return guard.rules(State(lessons={k: _lesson(k, **v) for k, v in lessons.items()}))


def test_only_a_blocking_lessons_command_rule_says_block():
    rules = _rules_of(b1=dict(trigger=ECHO, block=True, error_trigger=ERROR_TRIGGER), b2=dict(trigger=ECHO))
    assert [(r["id"], r["on"], r.get("block", False)) for r in rules] == [
        ("b1", "command", True), ("b1", "error", False), ("b2", "command", False)]


def _pre(command):
    return {"session_id": "s1", "transcript_path": TRANSCRIPT, "cwd": "/x", "hook_event_name": "PreToolUse",
            "tool_name": "Bash", "tool_input": {"command": command}}


def test_the_hook_refuses_a_command_that_matches_a_blocking_lesson(tmp_path):
    proc, fires = _hook(tmp_path, _pre("echo ===== done"), _rules_of(b1=dict(trigger=ECHO, block=True)))
    out = json.loads(proc.stdout)["hookSpecificOutput"]
    assert out["permissionDecision"] == "deny" and "additionalContext" not in out
    assert "отклонена" in out["permissionDecisionReason"] and "guard unblock b1" in out["permissionDecisionReason"]
    assert [json.loads(line).get("blocked") for line in fires] == [True]


def test_a_lesson_that_is_not_blocking_only_adds_context(tmp_path):
    proc, fires = _hook(tmp_path, _pre("echo ===== done"), _rules_of(b2=dict(trigger=ECHO)))
    out = json.loads(proc.stdout)["hookSpecificOutput"]
    assert "permissionDecision" not in out and "уже подводила" in out["additionalContext"]
    assert [json.loads(line).get("blocked") for line in fires] == [None]


def test_guard_block_needs_a_trusted_lesson_with_a_command_trigger_and_unset_lifts_it(tmp_path, monkeypatch, capsys):
    from agent_reflect.cli import main
    from agent_reflect.ledger import load_state, save_state

    cfg = Config.under(tmp_path)
    monkeypatch.setattr(Config, "default", classmethod(lambda cls: cfg))
    save_state(State(lessons={"b1": _lesson("b1", trigger=ECHO), "e1": _lesson(error_trigger=ERROR_TRIGGER),
                              "s1": _lesson("s1", status="staged", trigger=ECHO)}), cfg.state_file)
    assert main(["guard", "block", "e1"]) == 2 and main(["guard", "block", "s1"]) == 2
    assert main(["guard", "block", "nope"]) == 2
    assert main(["guard", "block", "b1"]) == 0 and load_state(cfg.state_file).lessons["b1"].block is True
    rules = json.loads((guard.guard_dir(cfg) / guard.RULES_NAME).read_text(encoding="utf-8"))["rules"]
    assert [r["id"] for r in rules if r.get("block")] == ["b1"]
    capsys.readouterr()
    assert main(["guard", "list"]) == 0 and "блокирует" in capsys.readouterr().out
    assert main(["guard", "unblock", "b1"]) == 0 and load_state(cfg.state_file).lessons["b1"].block is False
    assert main(["guard", "block", "b1"]) == 0 and main(["guard", "unset", "b1"]) == 0
    assert load_state(cfg.state_file).lessons["b1"].block is False


def _broken(*sessions, after_showing=True):
    return [{"session": s, "ts": NOW_ISO, "text": "[1] x", "violated": True, "after_showing": after_showing}
            for s in sessions]


def test_a_lesson_broken_twice_after_the_hook_had_shown_it_is_proposed_for_blocking():
    state = State(lessons={
        "i1": _lesson("i1", trigger=ECHO, violations=2, fires=3, live_since=NOW_ISO, evidence=_broken("s1", "s2")),
        "i2": _lesson("i2", trigger=ECHO, violations=2, fires=3, block=True, live_since=NOW_ISO,
                      evidence=_broken("s1", "s2")),
        "i3": _lesson("i3", trigger=ECHO, violations=2, fires=3, live_since=NOW_ISO,
                      evidence=_broken("s1", "s2", after_showing=False)),
        "i4": _lesson("i4", trigger=ECHO, violations=2, fires=3, live_since=NOW_ISO,
                      evidence=_broken("s1") + _broken("s2", after_showing=False)),
    })
    assert care.lessons(state, NOW) == [{"lesson_id": "i1", "reason": "ignored", "violations": 2, "fires": 3}]


def _echo_session(tmp_path, session, *calls):
    """Echo commands of one session, each failing: (call time, result time)."""
    entries = []
    for n, (call, result) in enumerate(calls, 1):
        entries += [{**_call(f"t{n}", "echo ===== x", session), "timestamp": call},
                    {**_result(f"t{n}", "boom\nExit code 1", error=True, session=session), "timestamp": result}]
    path = tmp_path / f"{session}.jsonl"
    write_jsonl(path, entries)
    return read_segment(path, 0)


def _live_echo(env):
    from agent_reflect.memory import write_lesson

    lesson = _lesson("v1", project_root=str(env.project), memory_dir=str(env.memory), trigger=ECHO,
                     live_since="2026-10-01T00:00:00Z")
    lesson.sha256 = write_lesson(env.memory, lesson)
    env.state.lessons["v1"] = lesson
    return lesson


def test_a_failure_after_an_earlier_showing_counts_as_read_and_ignored(env, tmp_path):
    lesson = _live_echo(env)
    # the hook fired at 10:00 (first command) and at 10:05 (second): the second ran after the first's result
    lesson.fire_times = {"s1": ["2026-10-19T10:00:00Z", "2026-10-19T10:05:00Z"]}
    seg = _echo_session(tmp_path, "s1", ("2026-10-19T10:00:00.000Z", "2026-10-19T10:00:02.000Z"),
                        ("2026-10-19T10:05:00.000Z", "2026-10-19T10:05:02.000Z"))
    trigger_hits(env.state, env.cfg, seg, env.project, NOW_ISO)
    (evidence,) = env.state.lessons["v1"].evidence
    assert evidence["after_showing"] is True and evidence["text"].startswith("[2] ")


def test_the_failure_the_hook_fired_on_first_is_not_ignoring_it(env, tmp_path):
    lesson = _live_echo(env)
    lesson.fire_times = {"s1": ["2026-10-19T10:00:00Z", "2026-10-19T10:05:00Z"]}
    # only the first command failed; the second, after the showing, worked
    seg = _echo_session(tmp_path, "s1", ("2026-10-19T10:00:00.000Z", "2026-10-19T10:00:02.000Z"))
    trigger_hits(env.state, env.cfg, seg, env.project, NOW_ISO)
    assert [e.get("after_showing") for e in env.state.lessons["v1"].evidence] == [None]


def test_a_later_segment_marks_a_counted_session_as_ignored(env, tmp_path):
    lesson = _live_echo(env)
    lesson.fire_times = {"s1": ["2026-10-19T10:00:00Z", "2026-10-19T10:05:00Z"]}
    first = _echo_session(tmp_path, "s1", ("2026-10-19T10:00:00.000Z", "2026-10-19T10:00:02.000Z"))
    trigger_hits(env.state, env.cfg, first, env.project, NOW_ISO)
    later = _echo_session(tmp_path, "s1", ("2026-10-19T10:05:00.000Z", "2026-10-19T10:05:02.000Z"))
    assert trigger_hits(env.state, env.cfg, later, env.project, NOW_ISO) == []
    assert [e.get("after_showing") for e in env.state.lessons["v1"].evidence] == [True]


def test_the_night_keeps_when_the_hook_showed_a_command_lesson(tmp_path):
    cfg = Config.under(tmp_path)
    state = State(lessons={"b1": _lesson("b1", trigger=ECHO)})
    log = guard.guard_dir(cfg) / guard.FIRES_NAME
    log.parent.mkdir(parents=True)
    log.write_text("".join(json.dumps(e) + "\n" for e in (
        {"ts": NOW_ISO, "lesson_id": "b1", "session": "s1", "on": "command"},
        {"ts": NOW_ISO, "lesson_id": "b1", "session": "s1", "on": "command"},
        {"ts": NOW_ISO, "lesson_id": "b1", "session": "s2", "on": "error"},
        {"ts": NOW_ISO, "lesson_id": "b1", "session": "s3", "on": "command", "blocked": True})), encoding="utf-8")
    assert guard.merge_fires(cfg, state) == 4
    # the times the hook showed it as a command was run, per session; an error showing is no command's
    assert state.lessons["b1"].fire_times == {"s1": [NOW_ISO, NOW_ISO], "s3": [NOW_ISO]}
    assert [(e["kind"], e["lesson_id"], e["session"]) for e in state.events] == [("lesson_blocked", "b1", "s3")]


def test_the_digest_proposes_guard_block_for_an_ignored_lesson():
    from agent_reflect import digest

    lessons = {"i1": _lesson("i1", title="echo без кавычек", trigger=ECHO)}
    line = digest._care_line({"lesson_id": "i1", "reason": "ignored", "violations": 2, "fires": 3}, lessons)
    assert "`agent-reflect guard block i1`" in line and "показывал" in line


# -- a lesson moved into a skill is still watched, and the move gets a verdict -------------------

from datetime import timedelta  # noqa: E402

from agent_reflect import chart, effect  # noqa: E402
from agent_reflect.reflect import _auto_line, prompt_lessons  # noqa: E402

DAY = timedelta(days=1)


def _iso(when):
    return when.strftime("%Y-%m-%dT%H:%M:%SZ")


def _moved(lesson_id="p1", *, at, before=2, after=(), target="personal:alpha", pr="", **over):
    """A lesson moved into `target` at `at`: `before` sessions made its mistake in the days before, `after` (days
    after the move) the sessions that made it since."""
    evidence = [{"session": f"b{n}", "ts": _iso(at - (n + 1) * DAY), "text": "[1] x"} for n in range(before)]
    comebacks = [{"session": f"a{d}", "ts": _iso(at + d * DAY), "text": "[1] x", "how": "model"} for d in after]
    over.setdefault("moved_since", None if pr else _iso(at))
    return _lesson(lesson_id, **{"status": "promoted", **over}, promoted_at=_iso(at), promoted_to=target,
                   promoted_pr=pr, evidence=evidence, after=comebacks)


def _verdicts(state, now=NOW):
    return {item["lesson_id"]: item["verdict"] for item in effect.check(state, now)}


def test_a_move_whose_mistake_never_came_back_helped_and_gets_its_verdict_once():
    state = State(lessons={"p1": _moved(at=NOW - effect.WINDOW - DAY)})
    assert _verdicts(state) == {"p1": "helped"}
    assert state.lessons["p1"].effect == "helped" and state.lessons["p1"].effect_at == NOW_ISO
    assert effect.check(state, NOW) == []


def test_a_move_whose_mistake_came_back_did_not_help():
    state = State(lessons={"p1": _moved(at=NOW - effect.WINDOW - DAY, after=(3,))})
    (item,) = effect.check(state, NOW)
    assert (item["verdict"], item["before"], item["after"]) == ("not_helped", 2, 1)


def test_another_move_into_the_same_target_in_the_window_confounds_the_verdict():
    at = NOW - effect.WINDOW - DAY
    state = State(lessons={"p1": _moved(at=at), "p2": _moved("p2", at=at + 10 * DAY),
                           "p3": _moved("p3", at=at, target="personal:beta")})
    assert _verdicts(state)["p1"] == "confounded" and _verdicts(State(lessons={"p3": state.lessons["p3"]})) == {}
    assert state.lessons["p3"].effect == "helped"


def test_a_move_with_one_session_before_it_has_too_little_data():
    assert _verdicts(State(lessons={"p1": _moved(at=NOW - effect.WINDOW - DAY, before=1)})) == {"p1": "too_little_data"}


def test_a_move_through_a_pull_request_counts_from_the_merge():
    at = NOW - effect.WINDOW - DAY
    url = "https://github.com/o/r/pull/1"
    lesson = _moved(at=at, pr=url, after=(2,))
    state = State(lessons={"p1": lesson}, open_prs=[{"url": url, "status": "open"}])
    assert effect.check(state, NOW) == [] and lesson.effect == ""
    state.open_prs[0].update(status="merged", merged_at=_iso(at + 5 * DAY))
    assert effect.check(state, NOW) == [] and lesson.effect == ""  # its window is still open
    assert _verdicts(state, NOW + 5 * DAY) == {"p1": "helped"}  # the comeback came before the merge


def test_a_move_whose_pull_request_was_closed_unmerged_says_so():
    url = "https://github.com/o/r/pull/1"
    state = State(lessons={"p1": _moved(at=NOW - 3 * DAY, pr=url)}, open_prs=[{"url": url, "status": "closed"}])
    assert _verdicts(state) == {"p1": "not_merged"}


def test_a_comeback_inside_the_window_is_reported_at_once_without_a_verdict():
    state = State(lessons={"p1": _moved(at=NOW - 5 * DAY, after=(1, 2))})
    (item,) = effect.check(state, NOW)
    assert (item["verdict"], item["after"], item["days_left"]) == ("watching", 2, effect.WINDOW.days - 5)
    assert state.lessons["p1"].effect == ""


def test_a_lesson_moved_before_moves_were_recorded_is_not_judged():
    long_ago = NOW - 90 * DAY
    old = _lesson("p1", status="promoted", created_at=_iso(long_ago),
                  evidence=[{"session": f"b{n}", "ts": _iso(long_ago + n * DAY)} for n in range(2)])
    state = State(lessons={"p1": old})
    assert effect.check(state, NOW) == [] and old.effect == ""


def test_the_reflect_prompt_lists_a_moved_lesson_with_its_mark():
    moved = _moved(at=datetime(2026, 10, 1, tzinfo=timezone.utc))
    old = _lesson("p0", status="promoted")
    state = State(lessons={"p1": moved, "p0": old, "r1": _lesson("r1", status="removed")})
    assert [lesson.id for lesson in prompt_lessons(state, Path(moved.project_root))] == ["p1"]
    assert _auto_line(moved).startswith("p1 [перенесён в скилл 2026-10-01T00:00:00Z]: ")


def _promoted_env(env, tmp_path, at="2026-10-01T00:00:00Z", **over):
    over.setdefault("moved_since", at)
    lesson = _lesson("p1", status="promoted", project_root=str(env.project), memory_dir=str(env.memory),
                     promoted_at=at, promoted_to="personal:alpha", **over)
    env.state.lessons["p1"] = lesson
    return lesson


@pytest.mark.parametrize("verdict", ["REPEAT", "LESSON_VIOLATED"])
def test_a_moved_lessons_mistake_named_by_the_model_is_recorded_after_the_move_without_a_file(env, tmp_path, verdict):
    lesson = _promoted_env(env, tmp_path)
    seg = _push_segment(tmp_path)
    entry = {"verdict": verdict, "lesson_id": "p1", "evidence": "[1] [rejected]"}
    assert apply_verdict(env.state, env.cfg, seg, env.project, env.memory, entry, NOW_ISO) == "after_promotion"
    assert apply_verdict(env.state, env.cfg, seg, env.project, env.memory, entry, NOW_ISO) == "after_promotion"
    assert [(a["session"], a["how"]) for a in lesson.after] == [("s1", "model")]
    assert lesson.status == "promoted" and sorted(p.name for p in env.memory.iterdir()) == ["MEMORY.md"]


def test_a_session_that_started_before_the_move_says_nothing_about_it(env, tmp_path):
    lesson = _promoted_env(env, tmp_path, at="2026-10-25T00:00:00Z")
    entry = {"verdict": "REPEAT", "lesson_id": "p1", "evidence": "[1] [rejected]"}
    assert apply_verdict(env.state, env.cfg, _push_segment(tmp_path), env.project, env.memory, entry,
                         NOW_ISO) == "no_entry"
    assert lesson.after == [] and [e["kind"] for e in env.state.events] == ["after_promotion_skipped"]


def test_a_moved_lessons_trigger_still_finds_its_mistake(env, tmp_path):
    lesson = _promoted_env(env, tmp_path, error_trigger=ERROR_TRIGGER)
    hits = trigger_hits(env.state, env.cfg, _push_segment(tmp_path, fixed=False), env.project, NOW_ISO)
    assert [(h["lesson_id"], h["outcome"]) for h in hits] == [("p1", "after_promotion")]
    assert [(a["session"], a["how"]) for a in lesson.after] == [("s1", "trigger")]
    assert trigger_hits(env.state, env.cfg, _push_segment(tmp_path, fixed=False), env.project, NOW_ISO) == []


def test_a_mistake_after_a_move_is_drawn_as_a_repeat():
    assert chart._SQUARE_OF["after_promotion"] == chart._SQUARE_OF["repeat"]


def test_the_digest_gives_each_verdict_in_words():
    from agent_reflect import digest

    lessons = {"p1": _moved(at=NOW, title="push отклонён")}
    words = {"helped": "помог", "not_helped": "не помог", "confounded": "не понять", "too_little_data": "мало данных",
             "not_merged": "без мержа", "watching": "повторилась"}
    for verdict, word in words.items():
        item = {"lesson_id": "p1", "verdict": verdict, "target": "personal:alpha", "before": 2, "after": 1,
                "days_left": 9}
        line = digest._effect_line(item, lessons)
        assert word in line and "push отклонён" in line and "personal:alpha" in line, (verdict, line)
    text = digest.render({"effects": [{"lesson_id": "p1", "verdict": "helped", "target": "personal:alpha",
                                       "before": 2, "after": 0}]}, lessons, "2026-W43")
    assert "Помогли ли переносы" in text


def test_the_reflect_prompt_explains_the_moved_mark():
    from agent_reflect.promptkit import load_template

    text = load_template("reflect.md").template
    assert "[перенесён в скилл <время>]" in text and "помог ли перенос" in text


# -- a fact the session paid to discover is a lesson of its own kind, with how to check it again ----

from agent_reflect.memory import render_lesson  # noqa: E402
from agent_reflect.reflect import build_verify_prompt  # noqa: E402

FACT = {
    "verdict": "NEW", "kind": "fact", "title": "push в main требует rebase",
    "hook": "main уходит вперёд за время работы → git pull --rebase перед push",
    "cue": "собираюсь пушить в main", "lesson": "В этом репо main двигают часто: перед `git push` — `git pull --rebase`.",
    "why": "Дважды push отклонён (fetch first).", "evidence": "[1] [rejected]",
    "refs": [{"ref": "1", "quote": "(fetch first)"}], "confidence": "medium",
    "recheck": "`git fetch && git status -sb` — показывает, отстаёт ли ветка",
}


def test_a_fact_lesson_keeps_how_to_recheck_it(env, tmp_path):
    assert apply_verdict(env.state, env.cfg, _push_segment(tmp_path), env.project, env.memory, FACT,
                         NOW_ISO) == "new_staged"
    lesson = _only(env)
    assert (lesson.kind, lesson.recheck) == ("fact", FACT["recheck"])
    text = (env.memory / lesson.file).read_text(encoding="utf-8")
    assert "**Как перепроверить:** `git fetch && git status -sb`" in text and "type: project" in text


def test_a_fact_without_a_way_to_recheck_it_is_not_written(env, tmp_path):
    bare = {k: v for k, v in FACT.items() if k != "recheck"}
    assert apply_verdict(env.state, env.cfg, _push_segment(tmp_path), env.project, env.memory, bare,
                         NOW_ISO) == "rejected_not_concrete"
    assert env.state.lessons == {}


def test_a_fact_without_quotes_is_not_written(env, tmp_path):
    bare = {k: v for k, v in FACT.items() if k != "refs"}
    assert apply_verdict(env.state, env.cfg, _push_segment(tmp_path), env.project, env.memory, bare,
                         NOW_ISO) == "rejected_bad_quote"


def test_a_tool_lesson_has_no_recheck_line():
    text = render_lesson(_lesson(recheck="x"))
    assert "Как перепроверить" not in text and "type: feedback" in text


def test_the_second_look_at_a_fact_sees_how_to_recheck_it():
    prompt = build_verify_prompt({"cue": "c", "lesson": "l", "why": "w", "kind": "fact", "recheck": "git status"}, [])
    assert "recheck: git status" in prompt and "kind: fact" in prompt
    assert "recheck:" not in build_verify_prompt({"cue": "c", "lesson": "l", "why": "w", "kind": "tool"}, [])


def test_a_fact_unconfirmed_for_a_month_is_proposed_for_a_recheck():
    old = _iso(NOW - 40 * DAY)
    fresh = _iso(NOW - 3 * DAY)
    state = State(lessons={
        "f1": _lesson("f1", kind="fact", recheck="git status", live_since=old, created_at=old,
                      evidence=[{"session": "s", "ts": old}], last_read_at=fresh),
        "f2": _lesson("f2", kind="fact", recheck="git status", live_since=old, created_at=old,
                      evidence=[{"session": "s", "ts": fresh}], last_read_at=fresh),
        "t1": _lesson("t1", live_since=old, created_at=old, evidence=[{"session": "s", "ts": old}],
                      last_read_at=fresh),
    })
    assert care.lessons(state, NOW) == [{"lesson_id": "f1", "reason": "recheck", "days": 40}]


def test_the_digest_says_how_to_recheck_a_fact():
    from agent_reflect import digest

    lessons = {"f1": _lesson("f1", title="main двигают часто", kind="fact", recheck="`git status -sb`")}
    line = digest._care_line({"lesson_id": "f1", "reason": "recheck", "days": 40}, lessons)
    assert "`git status -sb`" in line and "40 дней" in line and "agent-reflect retire f1" in line


def test_the_prompts_describe_the_fact_kind():
    from agent_reflect.promptkit import load_template

    assert "kind: \"fact\"" in load_template("reflect.md").template
    assert "`recheck`" in load_template("reflect.md").template
    assert "fact" in load_template("verify.md").template


def test_blank_lines_do_not_count_against_the_growth_of_an_always_loaded_file():
    from agent_reflect.edits import growth_error

    assert growth_error("Keep commits small.", "Keep commits small.\n\nRun the gate first.\n") is None
    assert "adds 2 lines" in growth_error("Keep commits small.", "Keep commits small.\nRun the gate.\nPush.")


def test_a_move_with_its_verdict_is_no_longer_watched(env, tmp_path):
    lesson = _promoted_env(env, tmp_path, error_trigger=ERROR_TRIGGER, effect="helped")
    assert prompt_lessons(env.state, env.project) == []
    assert trigger_hits(env.state, env.cfg, _push_segment(tmp_path, fixed=False), env.project, NOW_ISO) == []
    assert lesson.after == []


# -- review fixes --------------------------------------------------------------------------

def test_the_night_counts_an_error_trigger_when_no_lesson_has_a_command_trigger(tmp_path):
    from agent_reflect.projects import memory_dir_for, sanitize
    from test_seams import SESSION, _git_repo, _write, tool_use

    cfg = Config.under(tmp_path)
    cfg.claude_settings.parent.mkdir(parents=True, exist_ok=True)
    cfg.claude_settings.write_text(json.dumps({"autoDreamEnabled": False}))
    root = _git_repo(tmp_path / "work" / "alpha")
    failed = {**tool_use(str(root), 1, "toolu_1", "git push")}
    failed.update(type="user", uuid="u1", message={"role": "user", "content": [
        {"type": "tool_result", "tool_use_id": "toolu_1", "is_error": True, "content": REJECTED}]})
    _write(cfg.projects_dir / sanitize(str(root)) / f"{SESSION}.jsonl",
           [tool_use(str(root), 0, "toolu_1", "git push"), failed])
    moved = _lesson("p1", status="promoted", project_root=str(root),
                    memory_dir=str(memory_dir_for(cfg.projects_dir, root)), promoted_at="2026-10-01T00:00:00Z",
                    moved_since="2026-10-01T00:00:00Z", promoted_to="personal:alpha", error_trigger=ERROR_TRIGGER)
    state = State(installed_at="2026-10-01T00:00:00Z", lessons={"p1": moved})

    result = reflect.run(cfg, state, now=datetime(2026, 10, 20, 3, 0, tzinfo=timezone.utc),
                         llm=lambda *a, **k: {"verdict": "NO_ENTRY", "notes": "n"})
    assert result["outcomes"]["after_promotion"] == 1 and [a["how"] for a in moved.after] == ["trigger"]


def test_a_blocking_rule_is_never_crowded_out_by_others(tmp_path):
    rules = _rules_of(a1=dict(trigger=ECHO), a2=dict(trigger=ECHO), a3=dict(trigger=ECHO),
                      z9=dict(trigger=ECHO, block=True))
    proc, _ = _hook(tmp_path, _pre("echo ===== done"), rules)
    assert json.loads(proc.stdout)["hookSpecificOutput"]["permissionDecision"] == "deny"


def test_a_rule_shown_once_a_session_is_not_hidden_by_rules_already_shown(tmp_path):
    rules = _rules_of(**{f"e{n}": dict(error_trigger=ERROR_TRIGGER) for n in range(1, 5)})
    payload = {"session_id": "s1", "transcript_path": TRANSCRIPT, "cwd": "/x", "hook_event_name": "PostToolUseFailure",
               "tool_name": "Bash", "tool_input": {"command": "git push"}, "error": REJECTED}
    _hook(tmp_path, payload, rules)
    proc, fires = _hook(tmp_path, payload, rules)
    assert "auto_e4.md" in json.loads(proc.stdout)["hookSpecificOutput"]["additionalContext"]
    assert sorted(json.loads(line)["lesson_id"] for line in fires) == ["e1", "e2", "e3", "e4"]


def test_the_output_of_a_command_that_only_reads_is_no_error_of_its_own(tmp_path):
    def after(command):
        return {"session_id": "s1", "transcript_path": TRANSCRIPT, "cwd": "/x", "hook_event_name": "PostToolUse",
                "tool_name": "Bash", "tool_input": {"command": command},
                "tool_response": {"stdout": REJECTED, "stderr": ""}}
    for command in ('grep -rn "fetch first" .', "cat auto_e1.md", "git log -3", "sed -n 1,9p x"):
        # what a reader prints may hold a failure line too: it is still what it read
        payload = after(command)
        payload["tool_response"]["stdout"] = REJECTED + "\nExit code 1"
        assert _hook(tmp_path, payload, ERROR_RULES)[0].stdout == "", command
        assert _hook(tmp_path, after(command), ERROR_RULES)[0].stdout == "", command
    # a failure the pipe hid from the tool still shows in what it printed
    quiet = after("git push 2>&1 | tail -3")
    quiet["tool_response"]["stdout"] = REJECTED + "\nExit code 1"
    assert _hook(tmp_path, quiet, ERROR_RULES)[0].stdout != ""


def test_the_hooks_readers_are_the_nightly_readers():
    from agent_reflect import transcript

    assert (guard_hook._READERS, guard_hook._GIT_READERS) == (transcript._READERS, transcript._GIT_READERS)


def test_a_comeback_after_the_move_is_not_too_little_data():
    state = State(lessons={"p1": _moved(at=NOW - effect.WINDOW - DAY, before=1, after=(3,))})
    assert _verdicts(state) == {"p1": "not_helped"}


def test_a_session_started_before_the_merge_is_not_after_the_move():
    at = NOW - effect.WINDOW - 10 * DAY
    url = "https://github.com/o/r/pull/1"
    lesson = _moved(at=at, pr=url)
    lesson.after = [{"session": "a1", "ts": _iso(at + 6 * DAY), "started": _iso(at + 2 * DAY), "how": "model"}]
    state = State(lessons={"p1": lesson}, open_prs=[{"url": url, "status": "merged", "merged_at": _iso(at + 5 * DAY)}])
    assert _verdicts(state) == {"p1": "helped"}


def test_a_moved_lessons_comeback_keeps_when_its_session_started(env, tmp_path):
    lesson = _promoted_env(env, tmp_path)
    entry = {"verdict": "REPEAT", "lesson_id": "p1", "evidence": "[1] [rejected]"}
    apply_verdict(env.state, env.cfg, _push_segment(tmp_path), env.project, env.memory, entry, NOW_ISO)
    assert lesson.after[0]["started"] == "2026-10-19T10:00:00.000Z"


def test_a_fact_the_user_confirmed_leaves_the_recheck_list(tmp_path, monkeypatch, capsys):
    from agent_reflect.cli import main
    from agent_reflect.ledger import load_state, save_state

    cfg = Config.under(tmp_path)
    monkeypatch.setattr(Config, "default", classmethod(lambda cls: cfg))
    old = "2026-08-01T00:00:00Z"
    save_state(State(lessons={"f1": _lesson("f1", kind="fact", recheck="git status", live_since=old,
                                            created_at=old, last_read_at=NOW_ISO),
                              "t1": _lesson("t1")}), cfg.state_file)
    assert [c["reason"] for c in care.lessons(load_state(cfg.state_file), NOW)] == ["recheck"]
    assert main(["confirm", "t1"]) == 2 and main(["confirm", "nope"]) == 2
    assert main(["confirm", "f1"]) == 0
    assert care.lessons(load_state(cfg.state_file), datetime.now(timezone.utc)) == []


def test_the_digest_says_how_to_confirm_a_fact():
    from agent_reflect import digest

    lessons = {"f1": _lesson("f1", title="t", kind="fact", recheck="x")}
    assert "`agent-reflect confirm f1`" in digest._care_line({"lesson_id": "f1", "reason": "recheck", "days": 40},
                                                             lessons)


def test_the_reads_line_counts_hook_showings_of_any_trigger():
    from agent_reflect import digest

    lessons = {"e1": _lesson(error_trigger=ERROR_TRIGGER, fires=4)}
    assert "показан хуком: 4 раза" in "\n".join(digest._usage_lines({"reads": 0, "with_reads": 0, "live": 1}, lessons))



# -- review fixes, second round ----------------------------------------------------------

def test_a_lesson_shown_for_its_file_is_still_shown_for_its_error(tmp_path):
    rules = _rules_of(x1=dict(path_trigger=r"/lib/.*\.g\.dart$", error_trigger=ERROR_TRIGGER))
    edit = {"session_id": "s1", "transcript_path": TRANSCRIPT, "cwd": "/x", "hook_event_name": "PreToolUse",
            "tool_name": "Edit", "tool_input": {"file_path": "/Users/u/git/app/lib/a.g.dart"}}
    failure = {"session_id": "s1", "transcript_path": TRANSCRIPT, "cwd": "/x",
               "hook_event_name": "PostToolUseFailure", "tool_name": "Bash", "tool_input": {"command": "git push"},
               "error": REJECTED}
    _hook(tmp_path, edit, rules)
    proc, fires = _hook(tmp_path, failure, rules)
    assert "auto_x1.md" in json.loads(proc.stdout)["hookSpecificOutput"]["additionalContext"]
    assert [json.loads(line)["on"] for line in fires] == ["path", "error"]


def test_the_after_bash_hook_starts_python_only_while_there_are_error_rules(tmp_path):
    cfg = Config.under(tmp_path)
    guard.install_hook(cfg)
    hooks = json.loads(cfg.claude_settings.read_text(encoding="utf-8"))["hooks"]
    marker = guard.guard_dir(cfg) / guard.ERROR_RULES_MARK
    for event in ("PostToolUse", "PostToolUseFailure"):
        assert hooks[event][0]["hooks"][0]["command"].startswith(f'test -e "{marker}" && ')
    assert "test -e" not in hooks["PreToolUse"][0]["hooks"][0]["command"]
    guard.sync(cfg, State(lessons={"e1": _lesson(error_trigger=ERROR_TRIGGER)}))
    assert marker.exists()
    guard.sync(cfg, State(lessons={"e1": _lesson(trigger=ECHO)}))
    assert not marker.exists()


def test_a_closed_or_rolled_back_move_does_not_confound_another():
    at = NOW - effect.WINDOW - DAY
    url = "https://github.com/o/r/pull/9"
    state = State(lessons={"p1": _moved(at=at), "p2": _moved("p2", at=at + 10 * DAY, pr=url),
                           "p3": _moved("p3", at=at + 3 * DAY, status="live")},
                  open_prs=[{"url": url, "status": "closed"}])
    assert _verdicts(state)["p1"] == "helped"


def test_an_error_pattern_hit_shows_the_command_and_what_matched(env, tmp_path):
    apply_verdict(env.state, env.cfg, _push_segment(tmp_path), env.project, env.memory, PUSH, NOW_ISO)
    noisy = REJECTED.replace("To github.com:x/y.git", "x" * 300)
    entries = [_call("t1", "git push origin main", "s2"), _result("t1", noisy, error=True, session="s2")]
    path = tmp_path / "noisy.jsonl"
    write_jsonl(path, entries)
    (hit,) = trigger_hits(env.state, env.cfg, read_segment(path, 0), env.project, NOW_ISO)
    assert "git push origin main" in hit["notes"] and "[rejected]" in hit["notes"]


# -- the reports say what changed --------------------------------------------------------

def _card(lesson, *events):
    from agent_reflect import recent

    card = recent._lesson_card(lesson.id, lesson, [dict(e) for e in events], NOW, timezone.utc)
    return "\n".join([card.head, *card.body, *card.foot])


def test_the_night_card_says_a_moved_lessons_mistake_came_back():
    lesson = _moved(at=NOW - 5 * DAY, after=(1, 2), title="push отклонён")
    text = _card(lesson, {"kind": "lesson_after_promotion", "lesson_id": lesson.id, "session": "a2", "counted": True})
    assert "ошибка перенесённого урока повторилась" in text and "personal:alpha" in text
    assert "после переноса: 2" in text and "auto_p1.md" not in text and text.endswith("personal:alpha")


def test_the_night_card_says_the_hook_refused_a_command():
    lesson = _lesson("b1", trigger=ECHO, block=True, title="echo без кавычек")
    text = _card(lesson, {"kind": "lesson_blocked", "lesson_id": "b1", "session": "s3"})
    assert "хук отклонил команду" in text and "блокирует" in text


def test_the_night_card_shows_the_hooks_patterns_and_how_to_recheck_a_fact():
    lesson = _lesson("f1", kind="fact", recheck="`git status -sb`", error_trigger=ERROR_TRIGGER,
                     path_trigger="/lib/", status="staged")
    text = _card(lesson, {"kind": "lesson_new", "lesson_id": "f1"})
    assert f"ошибка /{ERROR_TRIGGER}/" in text and "файл //lib//" in text
    assert "как перепроверить: `git status -sb`" in text


def test_the_digest_lesson_line_marks_a_fact_and_its_hook():
    from agent_reflect import digest

    lessons = {"f1": _lesson("f1", title="t", kind="fact", recheck="`git status`", error_trigger=ERROR_TRIGGER)}
    line = digest._lesson_line("f1", lessons)
    assert "факт" in line and "Перепроверить: `git status`" in line and f"ошибка /{ERROR_TRIGGER}/" in line


def test_the_new_report_lines_speak_english():
    from agent_reflect import digest, i18n

    i18n.use("en")
    try:
        moved = _moved(at=NOW - 5 * DAY, after=(1,), title="push", lesson="Pull first.", cue="push rejected")
        fact = _lesson("f1", title="main", lesson="Rebase first.", cue="push", kind="fact", recheck="git status",
                       error_trigger=ERROR_TRIGGER, path_trigger="/lib/", trigger=ECHO, block=True, why="w")
        texts = [_card(moved, {"kind": "lesson_after_promotion", "lesson_id": "p1", "counted": True}),
                 _card(fact, {"kind": "lesson_blocked", "lesson_id": "f1"}),
                 digest._lesson_line("f1", {"f1": fact})]
    finally:
        i18n.use("ru")
    assert [line for text in texts for line in text.splitlines() if re.search("[а-яё]", line, re.I)] == []



# -- review fixes, third round ---------------------------------------------------------------

def test_only_the_user_lifts_a_block_the_agent_is_refused(tmp_path, monkeypatch, capsys):
    from agent_reflect.cli import main
    from agent_reflect.ledger import load_state, save_state

    cfg = Config.under(tmp_path)
    monkeypatch.setattr(Config, "default", classmethod(lambda cls: cfg))
    save_state(State(lessons={"b1": _lesson("b1", trigger=ECHO, block=True), "a1": _lesson("a1", trigger=ECHO)}),
               cfg.state_file)
    monkeypatch.setenv("CLAUDECODE", "1")
    for argv in (["guard", "unblock", "b1"], ["guard", "block", "a1"], ["guard", "unset", "b1"],
                 ["guard", "set", "--error", "b1", ERROR_TRIGGER], ["retire", "b1"], ["guard", "off"], ["uninstall"]):
        assert main(argv) == 2, argv
        assert "в своём терминале" in capsys.readouterr().err
    after = load_state(cfg.state_file).lessons
    assert (after["b1"].block, after["b1"].status, after["b1"].error_trigger, after["a1"].block) == (
        True, "live", "", False)
    assert main(["guard", "set", "--error", "a1", ERROR_TRIGGER]) == 0  # a lesson that blocks nothing is fair game
    monkeypatch.delenv("CLAUDECODE")
    assert main(["guard", "unblock", "b1"]) == 0 and load_state(cfg.state_file).lessons["b1"].block is False


def test_the_agent_may_turn_the_hook_off_while_nothing_blocks(tmp_path, monkeypatch):
    from agent_reflect.cli import main
    from agent_reflect.ledger import save_state

    cfg = Config.under(tmp_path)
    monkeypatch.setattr(Config, "default", classmethod(lambda cls: cfg))
    save_state(State(lessons={"a1": _lesson("a1", trigger=ECHO)}), cfg.state_file)
    monkeypatch.setenv("CLAUDECODE", "1")
    assert main(["guard", "off"]) == 0


def test_a_vague_fact_is_not_written(env, tmp_path):
    vague = {**FACT, "lesson": "В этом проекте сборка устроена иначе, учитывай это."}
    assert apply_verdict(env.state, env.cfg, _push_segment(tmp_path), env.project, env.memory, vague,
                         NOW_ISO) == "rejected_not_concrete"


def test_a_move_whose_pr_is_open_is_watched_from_when_it_left_memory(env, tmp_path):
    # its comebacks are kept while the PR waits; the verdict counts only those after the merge
    lesson = _promoted_env(env, tmp_path, moved_since=None, promoted_pr="https://github.com/o/r/pull/1",
                           error_trigger=ERROR_TRIGGER)
    assert prompt_lessons(env.state, env.project) == [lesson]
    hits = trigger_hits(env.state, env.cfg, _push_segment(tmp_path, fixed=False), env.project, NOW_ISO)
    assert [h["outcome"] for h in hits] == ["after_promotion"] and len(lesson.after) == 1


def test_the_moved_mark_and_the_comeback_count_from_when_the_lesson_left_memory(env, tmp_path):
    lesson = _promoted_env(env, tmp_path, at="2026-10-19T12:00:00Z", moved_since="2026-10-20T00:00:00Z",
                           promoted_pr="https://github.com/o/r/pull/1")
    assert _auto_line(lesson).startswith("p1 [перенесён в скилл 2026-10-19T12:00:00Z]: ")
    entry = {"verdict": "REPEAT", "lesson_id": "p1", "evidence": "[1] [rejected]"}
    # the session started at 10:00, before the lesson left memory at 12:00: it had the lesson there
    assert apply_verdict(env.state, env.cfg, _push_segment(tmp_path), env.project, env.memory, entry,
                         NOW_ISO) == "no_entry"
    assert apply_verdict(env.state, env.cfg, _push_segment(tmp_path), env.project, env.memory, entry,
                         NOW_ISO) == "no_entry"
    assert lesson.after == [] and [e["kind"] for e in env.state.events] == ["after_promotion_skipped"]


def test_a_merge_puts_its_moves_into_effect(tmp_path):
    from agent_reflect.consolidate import sync_merged_prs

    cfg = Config.under(tmp_path)
    url = "https://github.com/o/r/pull/1"
    moved = _lesson("p1", status="promoted", promoted_at="2026-10-01T00:00:00Z", promoted_pr=url)
    state = State(lessons={"p1": moved},
                  open_prs=[{"repo": "webapp", "url": url, "branch": "b", "status": "open", "cursor_sync": []}])

    class Git:
        def gh_token(self):
            return "tok"

        def pr_state(self, repo, url_, token):
            return "MERGED"

        def pr_merged_at(self, repo, url_, token):
            return "2026-10-05T09:00:00Z"

    sync_merged_prs(cfg, state, git=Git(), now_iso=NOW_ISO)
    assert moved.moved_since == "2026-10-05T09:00:00Z"


def test_an_open_pr_or_a_rolled_back_move_does_not_confound_another():
    at = NOW - effect.WINDOW - DAY
    url = "https://github.com/o/r/pull/9"
    state = State(lessons={"p1": _moved(at=at), "p2": _moved("p2", at=at + 10 * DAY, pr=url),
                           "p3": _moved("p3", at=at + 3 * DAY, effect="rolled_back")},
                  open_prs=[{"url": url, "status": "open"}])
    assert _verdicts(state)["p1"] == "helped"


def test_a_rollback_of_a_personal_skill_ends_the_watch_on_its_moves(tmp_path):
    from agent_reflect import memory_archive, skills

    cfg = Config.under(tmp_path)
    run_id = "20261025T043000"
    for lesson_id, target, pr in (("p1", "personal:alpha", None), ("p2", "repo:webapp:CLAUDE.md", "u1")):
        memory_archive.append_journal(cfg, {"ts": "2026-10-25T04:30:00Z", "run_id": run_id, "kind": "retire",
                                            "lesson_id": lesson_id, "target": target, "pr_url": pr})
    run_at = datetime(2026, 10, 25, 4, 30, tzinfo=timezone.utc)
    state = State(lessons={"p1": _moved(at=run_at),
                           "p2": _moved("p2", at=run_at, pr="u1", target="repo:webapp:CLAUDE.md")})
    ended = memory_archive.end_rolled_back_moves(cfg, state, run_id, {skills.backup_folder(cfg, run_id, "personal:alpha")},
                                                 NOW_ISO)
    assert ended == ["p1"]
    assert (state.lessons["p1"].effect, state.lessons["p2"].effect) == ("rolled_back", "")
    # p1's rule is gone from the skill; p2 is still watched while its PR waits
    assert prompt_lessons(state, Path(state.lessons["p1"].project_root)) == [state.lessons["p2"]]


def test_the_patterns_of_a_lesson_are_listed_in_one_place():
    lesson = _lesson(trigger="a", path_trigger="/b/", error_trigger="c")
    assert lesson.patterns() == [("command", "a"), ("path", "/b/"), ("error", "c")]
    assert _lesson().patterns() == []


def test_the_agent_may_not_turn_the_hook_off_when_the_state_cannot_be_read(tmp_path, monkeypatch):
    from agent_reflect.cli import main

    cfg = Config.under(tmp_path)
    monkeypatch.setattr(Config, "default", classmethod(lambda cls: cfg))
    cfg.state_file.parent.mkdir(parents=True, exist_ok=True)
    cfg.state_file.write_text("{ not json", encoding="utf-8")
    monkeypatch.setenv("CLAUDECODE", "1")
    assert main(["guard", "off"]) == 2 and main(["uninstall"]) == 2



# -- review fixes, fourth round ---------------------------------------------------------------

def test_a_block_left_on_a_lesson_out_of_the_hooks_rules_does_not_bind_the_agent(tmp_path, monkeypatch):
    from agent_reflect.cli import main
    from agent_reflect.ledger import save_state

    cfg = Config.under(tmp_path)
    monkeypatch.setattr(Config, "default", classmethod(lambda cls: cfg))
    save_state(State(lessons={"b1": _lesson("b1", trigger=ECHO, block=True, status="retired")}), cfg.state_file)
    monkeypatch.setenv("CLAUDECODE", "1")
    assert main(["guard", "off"]) == 0


def test_the_refusal_says_why(tmp_path, monkeypatch, capsys):
    from agent_reflect.cli import main

    cfg = Config.under(tmp_path)
    monkeypatch.setattr(Config, "default", classmethod(lambda cls: cfg))
    monkeypatch.setenv("CLAUDECODE", "1")
    assert main(["guard", "block", "x"]) == 2 and "CLAUDECODE=1" in capsys.readouterr().err


def test_the_rollback_line_speaks_the_runs_language(tmp_path, monkeypatch, capsys):
    from agent_reflect import cli, memory_archive

    cfg = Config.under(tmp_path)
    from agent_reflect.ledger import save_state
    save_state(State(lessons={"p1": _moved(at=NOW)}), cfg.state_file)
    monkeypatch.setattr(memory_archive, "end_rolled_back_moves", lambda *a: ["p1"])
    cli._lift_holds(cfg, "20261025T043000", set())
    assert "перенос p1 отменён" in capsys.readouterr().out


def test_a_rollback_ends_every_later_move_into_the_restored_file(tmp_path):
    from agent_reflect import memory_archive, skills

    cfg = Config.under(tmp_path)
    run_id = "20261025T043000"
    run_at = datetime(2026, 10, 25, 4, 30, tzinfo=timezone.utc)
    state = State(lessons={"p1": _moved(at=run_at), "p2": _moved("p2", at=run_at + 7 * DAY),
                           "p0": _moved("p0", at=run_at - 7 * DAY),
                           "q1": _moved("q1", at=run_at, target="personal:beta")},
                  moves=[{"target": "personal:alpha", "at": _iso(run_at), "pr_url": "", "run_id": run_id},
                         {"target": "personal:beta", "at": _iso(run_at), "pr_url": "", "run_id": run_id}])
    ended = memory_archive.end_rolled_back_moves(cfg, state, run_id, {skills.backup_folder(cfg, run_id, "personal:alpha")},
                                                 NOW_ISO)
    assert sorted(ended) == ["p1", "p2"]
    assert [m.get("rolled_back") for m in state.moves] == [True, None]


def test_a_move_of_only_user_notes_into_the_same_file_confounds(tmp_path):
    at = NOW - effect.WINDOW - DAY
    state = State(lessons={"p1": _moved(at=at)},
                  moves=[{"target": "personal:alpha", "at": _iso(at), "pr_url": "", "run_id": "r1"},
                         {"target": "personal:alpha", "at": _iso(at + 9 * DAY), "pr_url": "", "run_id": "r2"}])
    assert _verdicts(state) == {"p1": "confounded"}
    state = State(lessons={"p1": _moved(at=at)},
                  moves=[{"target": "personal:alpha", "at": _iso(at + 9 * DAY), "pr_url": "", "run_id": "r2",
                          "rolled_back": True}])
    assert _verdicts(state) == {"p1": "helped"}


def test_the_night_card_of_a_pr_move_says_it_waits_for_the_merge():
    lesson = _moved(at=NOW - 3 * DAY, pr="https://github.com/o/r/pull/1")
    text = _card(lesson, {"kind": "lesson_after_promotion", "lesson_id": "p1", "counted": True})
    assert "ждёт мержа PR" in text and text.endswith("personal:alpha")
    from agent_reflect import recent

    old = _lesson("o1", status="promoted")  # moved before moves were recorded: no target to name
    card = recent._lesson_card("o1", old, [{"kind": "lesson_after_promotion", "lesson_id": "o1"}], NOW, timezone.utc)
    assert card.foot == ["скилл"]


def test_rolled_back_is_a_verdict():
    assert "rolled_back" in effect.VERDICTS



# -- review fixes, fifth round ----------------------------------------------------------------

def test_commands_called_together_are_no_reading_of_each_others_showing(env, tmp_path):
    lesson = _live_echo(env)
    # two echo commands in one message: the hook fired for both in the same second, before either result
    lesson.fire_times = {"s1": ["2026-10-19T10:00:00Z", "2026-10-19T10:00:00Z"]}
    seg = _echo_session(tmp_path, "s1", ("2026-10-19T10:00:00.100Z", "2026-10-19T10:00:03.000Z"),
                        ("2026-10-19T10:00:00.100Z", "2026-10-19T10:00:03.500Z"))
    trigger_hits(env.state, env.cfg, seg, env.project, NOW_ISO)
    assert [e.get("after_showing") for e in env.state.lessons["v1"].evidence] == [None]


def test_the_hook_keeps_the_latest_times_and_sessions_only(tmp_path):
    cfg = Config.under(tmp_path)
    state = State(lessons={"b1": _lesson("b1", trigger=ECHO)})
    log = guard.guard_dir(cfg) / guard.FIRES_NAME
    log.parent.mkdir(parents=True)
    fires = [{"ts": f"2026-10-19T10:{m:02d}:00Z", "lesson_id": "b1", "session": "s0", "on": "command"}
             for m in range(guard.FIRE_TIMES_MAX + 3)]
    fires += [{"ts": NOW_ISO, "lesson_id": "b1", "session": f"x{n}", "on": "command"}
              for n in range(guard.FIRED_SESSIONS_MAX)]
    fires.append({"ts": NOW_ISO, "lesson_id": "b1", "session": "x0", "on": "command"})  # an old session again
    log.write_text("".join(json.dumps(e) + "\n" for e in fires), encoding="utf-8")
    guard.merge_fires(cfg, state)
    times = state.lessons["b1"].fire_times
    assert len(times) == guard.FIRED_SESSIONS_MAX and "s0" not in times  # the oldest session went first
    assert list(times)[-1] == "x0" and times["x0"] == [NOW_ISO, NOW_ISO]
    state = State(lessons={"b1": _lesson("b1", trigger=ECHO)})
    state.guard_offset = 0
    log.write_text("".join(json.dumps(e) + "\n" for e in fires[:guard.FIRE_TIMES_MAX + 3]), encoding="utf-8")
    guard.merge_fires(cfg, state)
    assert state.lessons["b1"].fire_times["s0"][0] == "2026-10-19T10:03:00Z"


def test_the_weekly_plan_is_told_what_a_blocking_lesson_is():
    from agent_reflect.promptkit import load_template

    text = load_template("consolidate_plan.md").template
    assert "«блокирует: да» не удаляй и не склеивай" in text


def test_the_night_card_of_a_rolled_back_move_says_so():
    lesson = _moved(at=NOW - 3 * DAY, effect="rolled_back")
    assert "перенос отменён" in _card(lesson, {"kind": "lesson_after_promotion", "lesson_id": "p1"})


def test_a_merge_does_not_carry_the_absorbed_lessons_hook_readings(tmp_path):
    from agent_reflect import consolidate

    keep = {"session": "k", "ts": NOW_ISO, "violated": True, "after_showing": True}
    gone = {"session": "g", "ts": NOW_ISO, "violated": True, "after_showing": True}
    assert consolidate.absorbed_evidence([gone]) == [{"session": "g", "ts": NOW_ISO, "violated": True}]
    assert keep["after_showing"] is True


def test_the_agent_may_change_a_lesson_whose_block_no_longer_holds(tmp_path, monkeypatch):
    from agent_reflect.cli import main
    from agent_reflect.ledger import save_state

    cfg = Config.under(tmp_path)
    monkeypatch.setattr(Config, "default", classmethod(lambda cls: cfg))
    save_state(State(lessons={"b1": _lesson("b1", trigger=ECHO, block=True, status="staged")}), cfg.state_file)
    monkeypatch.setenv("CLAUDECODE", "1")
    assert main(["guard", "unset", "b1"]) == 0
