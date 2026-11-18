"""The Bash hook that shows a lesson when a command matches its trigger, and what feeds it."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from agent_reflect import guard, guard_hook
from agent_reflect.cli import main
from agent_reflect.config import Config
from agent_reflect.ledger import Lesson, State, load_state, save_state

PROJECT = "/Users/u/git/app"
TRIGGER = r"\becho\s+=+"


def lesson(lesson_id="ab12cd34", status="live", trigger=TRIGGER, project=PROJECT, **over) -> Lesson:
    base = dict(id=lesson_id, project_root=project, memory_dir="/m", file=f"auto_{lesson_id}.md", title="echo =====",
                hook="h", cue="c", lesson="Пиши `echo \"=====\"` в кавычках.", why="w", confidence="medium",
                repeats=2, evidence=[], status=status, sha256="", created_at="2026-10-20T00:00:00Z",
                updated_at="2026-10-20T00:00:00Z", trigger=trigger)
    return Lesson(**{**base, **over})


@pytest.fixture
def cfg(tmp_path):
    return Config.under(tmp_path)


def test_rules_hold_the_trusted_lessons_with_a_trigger_that_compiles():
    state = State(lessons={l.id: l for l in (
        lesson("a1"), lesson("a2", status="staged"), lesson("a3", trigger=""), lesson("a4", trigger="("),
        lesson("a5", status="promoted"))})
    rules = guard.rules(state)
    assert [r["id"] for r in rules] == ["a1", "a5"]
    assert rules[0] == {"id": "a1", "file": "auto_a1.md", "project": PROJECT, "slug": "-Users-u-git-app",
                        "on": "command", "pattern": TRIGGER, "text": "Пиши `echo \"=====\"` в кавычках."}


def test_sync_writes_the_rules_once_and_again_only_when_they_change(cfg):
    state = State(lessons={"a1": lesson("a1")})
    assert guard.sync(cfg, state) is True and guard.sync(cfg, state) is False
    path = guard.guard_dir(cfg) / guard.RULES_NAME
    assert json.loads(path.read_text(encoding="utf-8"))["rules"][0]["id"] == "a1"
    state.lessons["a1"].trigger = ""
    assert guard.sync(cfg, state) is True and json.loads(path.read_text(encoding="utf-8")) == {"rules": []}


def _hook(tmp_path, payload, rules=None):
    """Run the hook as Claude Code does: JSON on stdin, its own process."""
    guard_dir = tmp_path / "guard"
    guard_dir.mkdir(exist_ok=True)
    if rules is not None:
        (guard_dir / guard.RULES_NAME).write_text(json.dumps({"rules": rules}), encoding="utf-8")
    proc = subprocess.run([sys.executable, guard_hook.__file__], input=json.dumps(payload), text=True,
                          capture_output=True, env={"AGENT_REFLECT_GUARD_DIR": str(guard_dir), "PATH": "/usr/bin:/bin"})
    fires = guard_dir / guard.FIRES_NAME
    return proc, (fires.read_text(encoding="utf-8").splitlines() if fires.exists() else [])


def _payload(command, transcript=f"/h/.claude/projects/-Users-u-git-app/s1.jsonl", cwd="/elsewhere", tool="Bash"):
    return {"session_id": "s1", "transcript_path": transcript, "cwd": cwd, "hook_event_name": "PreToolUse",
            "tool_name": tool, "tool_input": {"command": command}}


RULES = guard.rules(State(lessons={"a1": lesson("a1")}))


def test_the_hook_shows_the_lesson_beside_a_matching_command_and_logs_the_fire(tmp_path):
    proc, fires = _hook(tmp_path, _payload("git status; echo =====; git log"), RULES)
    assert proc.returncode == 0 and proc.stderr == ""
    out = json.loads(proc.stdout)["hookSpecificOutput"]
    assert out["hookEventName"] == "PreToolUse" and "permissionDecision" not in out
    assert "Пиши `echo \"=====\"` в кавычках." in out["additionalContext"] and "auto_a1.md" in out["additionalContext"]
    assert [json.loads(line)["lesson_id"] for line in fires] == ["a1"]
    assert json.loads(fires[0])["session"] == "s1"


@pytest.mark.parametrize("payload", [
    _payload('echo "====="'),                                                   # the working form
    _payload("echo =====", transcript="/h/.claude/projects/-other/s1.jsonl"),  # another project
    _payload("echo =====", tool="Read"),                                        # not a shell command
    {"tool_name": "Bash", "tool_input": {}},                                     # no command at all
])
def test_the_hook_says_nothing_when_no_rule_of_the_project_matches(tmp_path, payload):
    proc, fires = _hook(tmp_path, payload, RULES)
    assert (proc.returncode, proc.stdout, fires) == (0, "", [])


def test_the_hook_knows_the_project_by_its_cwd_too(tmp_path):
    proc, _ = _hook(tmp_path, _payload("echo =====", transcript="", cwd=PROJECT + "/lib"), RULES)
    assert "additionalContext" in proc.stdout


def test_a_subagent_of_the_project_is_under_the_same_rules(tmp_path):
    sub = "/h/.claude/projects/-Users-u-git-app/s1/subagents/agent-a1.jsonl"
    proc, fires = _hook(tmp_path, _payload("echo =====", transcript=sub), RULES)
    assert "additionalContext" in proc.stdout and len(fires) == 1


@pytest.mark.parametrize("rules_text", [None, "мусор", '{"rules": [{"pattern": "("}, 7]}'])
def test_the_hook_never_fails_a_command_over_its_own_files(tmp_path, rules_text):
    guard_dir = tmp_path / "guard"
    guard_dir.mkdir()
    if rules_text is not None:
        (guard_dir / guard.RULES_NAME).write_text(rules_text, encoding="utf-8")
    proc, _ = _hook(tmp_path, _payload("echo ====="))
    assert (proc.returncode, proc.stdout, proc.stderr) == (0, "", "")
    proc = subprocess.run([sys.executable, guard_hook.__file__], input="не json", text=True, capture_output=True,
                          env={"AGENT_REFLECT_GUARD_DIR": str(guard_dir), "PATH": "/usr/bin:/bin"})
    assert (proc.returncode, proc.stdout) == (0, "")


def test_fires_are_counted_into_the_lessons_once(cfg):
    state = State(lessons={"a1": lesson("a1")})
    log = guard.guard_dir(cfg) / guard.FIRES_NAME
    log.parent.mkdir(parents=True)
    log.write_text('{"ts": "2026-11-01T10:00:00Z", "lesson_id": "a1"}\n{"ts": "2026-11-02T10:00:00Z", "lesson_id": "zz"}\n'
                   'мусор\n{"ts": "2026-11-03T10:00:00Z", "lesson_id": "a1"}\n{"ts": "2026-11-04', encoding="utf-8")
    assert guard.merge_fires(cfg, state) == 2
    assert (state.lessons["a1"].fires, state.lessons["a1"].last_fired_at) == (2, "2026-11-03T10:00:00Z")
    assert guard.merge_fires(cfg, state) == 0 and state.lessons["a1"].fires == 2
    with log.open("a", encoding="utf-8") as fh:
        fh.write('T10:00:00Z", "lesson_id": "a1"}\n')
    assert guard.merge_fires(cfg, state) == 1 and state.lessons["a1"].fires == 3
    log.write_text("", encoding="utf-8")  # the log was cleared: counting starts over, nothing is lost
    assert guard.merge_fires(cfg, state) == 0 and state.guard_offset == 0


def test_a_trigger_typed_by_hand_is_checked():
    assert guard.check_trigger(TRIGGER) == ""
    assert guard.check_trigger("") == "empty" and guard.check_trigger(".*") == "matches any command"
    assert guard.check_trigger("(").startswith("not a regex") and guard.check_trigger("x" * 201) == "too long"


def _settings(cfg):
    return json.loads(cfg.claude_settings.read_text(encoding="utf-8"))


def test_the_hook_goes_into_settings_beside_the_users_own_and_comes_out_alone(cfg):
    cfg.claude_settings.parent.mkdir(parents=True)
    mine = {"matcher": "^(AskUserQuestion)$", "hooks": [{"type": "command", "command": "python3 nani.py"}]}
    cfg.claude_settings.write_text(json.dumps({"env": {"A": "1"}, "hooks": {"PreToolUse": [mine], "Stop": []}}))
    assert guard.install_hook(cfg) is True and guard.install_hook(cfg) is False
    entries = _settings(cfg)["hooks"]["PreToolUse"]
    assert entries[0] == mine and entries[1]["matcher"] == guard.HOOK_MATCHER
    assert guard.HOOK_MARK in entries[1]["hooks"][0]["command"] and str(cfg.repo_dir) in entries[1]["hooks"][0]["command"]
    assert _settings(cfg)["env"] == {"A": "1"}
    assert guard.uninstall_hook(cfg) is True and guard.uninstall_hook(cfg) is False
    assert _settings(cfg) == {"env": {"A": "1"}, "hooks": {"PreToolUse": [mine], "Stop": []}}


def test_settings_without_hooks_get_them_and_lose_them_again(cfg):
    assert guard.install_hook(cfg) is True
    assert list(_settings(cfg)["hooks"]) == ["PreToolUse", "PostToolUse", "PostToolUseFailure", "SessionStart"]
    assert guard.uninstall_hook(cfg) is True and _settings(cfg) == {}


def test_the_session_start_hook_goes_in_beside_the_users_own_and_comes_out_alone(cfg):
    cfg.claude_settings.parent.mkdir(parents=True)
    mine = {"hooks": [{"type": "command", "command": "afplay start.aiff"}]}
    cfg.claude_settings.write_text(json.dumps({"hooks": {"SessionStart": [mine]}}))
    assert guard.install_hook(cfg) is True and guard.install_hook(cfg) is False
    entries = _settings(cfg)["hooks"]["SessionStart"]
    assert entries[0] == mine and "matcher" not in entries[1]  # every start: a new session, a resume, a /clear
    (hook,) = entries[1]["hooks"]
    assert "agent_reflect/session_hook.py" in hook["command"] and str(cfg.repo_dir) in hook["command"]
    assert guard.uninstall_hook(cfg) is True
    assert _settings(cfg) == {"hooks": {"SessionStart": [mine]}}


def test_a_moved_repository_replaces_its_old_hook_entry(cfg, tmp_path):
    guard.install_hook(cfg)
    moved = Config.under(tmp_path / "elsewhere")
    moved = type(moved)(**{**moved.__dict__, "claude_settings": cfg.claude_settings})
    assert guard.install_hook(moved) is True
    (entry,) = _settings(cfg)["hooks"]["PreToolUse"]
    assert str(moved.repo_dir) in entry["hooks"][0]["command"]


def test_guard_set_writes_the_trigger_and_the_rules_and_refuses_a_bad_one(cfg, monkeypatch, capsys):
    monkeypatch.setattr(Config, "default", classmethod(lambda cls: cfg))
    save_state(State(lessons={"a1": lesson("a1", trigger="")}), cfg.state_file)
    assert main(["guard", "set", "a1", TRIGGER]) == 0
    assert load_state(cfg.state_file).lessons["a1"].trigger == TRIGGER
    rules = json.loads((guard.guard_dir(cfg) / guard.RULES_NAME).read_text(encoding="utf-8"))["rules"]
    assert [r["id"] for r in rules] == ["a1"]
    assert main(["guard", "set", "a1", ".*"]) == 2 and main(["guard", "set", "nope", TRIGGER]) == 2
    assert load_state(cfg.state_file).lessons["a1"].trigger == TRIGGER
    capsys.readouterr()
    assert main(["guard", "test", "echo", "====="]) == 0 and capsys.readouterr().out.startswith("a1  ")
    assert main(["guard", "list"]) == 0 and "a1  показывается, сработал: 0" in capsys.readouterr().out
    assert main(["guard", "unset", "a1"]) == 0 and load_state(cfg.state_file).lessons["a1"].trigger == ""
