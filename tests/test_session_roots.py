from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

from agent_reflect import guard, session_hook, session_roots
from agent_reflect.config import Config
from agent_reflect.projects import resolve_project_root

# the hook's git and the fixture's must not read the developer's own git config
GIT_ENV = {**os.environ, "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1"}


def _git(*args, cwd):
    subprocess.run(["git", *args], cwd=cwd, env=GIT_ENV, check=True, capture_output=True, text=True)


@pytest.fixture
def repo(tmp_path):
    """A main checkout and a linked worktree of it, the way a worker's review checkout is made."""
    main = tmp_path / "main"
    main.mkdir()
    _git("init", "-q", "-b", "main", cwd=main)
    _git("-c", "user.name=t", "-c", "user.email=t@example.com", "commit", "-q", "--allow-empty", "-m", "init",
         cwd=main)
    worktree = tmp_path / "scratchpad" / "wt-review-7"
    _git("worktree", "add", "-q", "--detach", str(worktree), cwd=main)
    return main.resolve(), worktree


def _start(hook_dir, payload):
    text = payload if isinstance(payload, str) else json.dumps(payload)
    return subprocess.run([sys.executable, session_hook.__file__], input=text, text=True, capture_output=True,
                          env={**GIT_ENV, "AGENT_REFLECT_GUARD_DIR": str(hook_dir)}, timeout=30)


def _payload(cwd, transcript="/t/s1.jsonl"):
    return {"session_id": "s1", "transcript_path": transcript, "cwd": str(cwd),
            "hook_event_name": "SessionStart", "source": "startup"}


def _records(hook_dir):
    path = hook_dir / session_roots.ROOTS_NAME
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []


def test_a_session_started_in_a_worktree_records_its_main_repository(tmp_path, repo):
    main, worktree = repo
    proc = _start(tmp_path / "hooks", _payload(worktree))
    assert proc.returncode == 0 and proc.stdout == ""  # SessionStart output would land in the session's context
    (record,) = _records(tmp_path / "hooks")
    assert (record["transcript"], record["root"]) == ("/t/s1.jsonl", str(main))
    assert record["root"] == str(resolve_project_root(str(worktree)))  # what the nightly run sees while it exists


def test_a_session_in_the_main_checkout_or_outside_git_records_nothing(tmp_path, repo):
    main, _ = repo
    outside = tmp_path / "notes"
    outside.mkdir()
    for cwd in (main, outside, tmp_path / "missing"):
        proc = _start(tmp_path / "hooks", _payload(cwd))
        assert proc.returncode == 0 and proc.stdout == ""
    assert _records(tmp_path / "hooks") == []


@pytest.mark.parametrize("payload", ["not json", "[]", json.dumps({"cwd": "/x"}), json.dumps({"transcript_path": "/t"})])
def test_a_payload_it_cannot_use_is_passed_over_quietly(tmp_path, payload):
    proc = _start(tmp_path / "hooks", payload)
    assert (proc.returncode, proc.stdout, proc.stderr) == (0, "", "")
    assert not (tmp_path / "hooks").exists()


def test_the_hook_writes_where_the_nightly_run_reads():
    assert session_hook.ROOTS_NAME == session_roots.ROOTS_NAME
    assert os.path.expanduser("~/.claude/agent-reflect") == session_hook.DEFAULT_DIR


@pytest.fixture
def cfg(tmp_path):
    return Config.under(tmp_path)


def _write(cfg, *records, name=session_roots.ROOTS_NAME):
    path = guard.guard_dir(cfg) / name
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        for record in records:
            fh.write(record if isinstance(record, str) else json.dumps(record) + "\n")


def test_the_latest_record_of_a_transcript_wins_and_a_broken_line_is_skipped(cfg):
    _write(cfg, {"transcript": "/t/a.jsonl", "root": "/r/old"}, "{broken\n", ["not", "a", "record"],
           {"transcript": "/t/b.jsonl"}, {"transcript": "/t/a.jsonl", "root": "/r/new"})
    _write(cfg, json.dumps(["x"]) + "\n")
    assert session_roots.load(cfg) == {"/t/a.jsonl": "/r/new"}


def test_no_file_is_no_records(cfg):
    assert session_roots.load(cfg) == {}


def test_a_full_file_is_rotated_and_both_halves_are_still_read(cfg):
    _write(cfg, {"transcript": "/t/old.jsonl", "root": "/r/1"}, {"transcript": "/t/both.jsonl", "root": "/r/1"})
    assert session_roots.rotate(cfg, limit=10**6) is False  # small: stays where it is
    assert session_roots.rotate(cfg, limit=10) is True
    assert not (guard.guard_dir(cfg) / session_roots.ROOTS_NAME).exists()
    _write(cfg, {"transcript": "/t/both.jsonl", "root": "/r/2"})  # the hook appends to a fresh file
    assert session_roots.load(cfg) == {"/t/old.jsonl": "/r/1", "/t/both.jsonl": "/r/2"}
    # the next rotation drops the oldest half: transcripts that old are gone from Claude Code too
    assert session_roots.rotate(cfg, limit=10) is True
    assert session_roots.load(cfg) == {"/t/both.jsonl": "/r/2"}
