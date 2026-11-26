from __future__ import annotations

import fcntl
import json
import os
import subprocess
import sys
import threading

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
    (main / "e2e").mkdir(parents=True)
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


def _payload(cwd, source="startup", session="s1"):
    return {"session_id": session, "transcript_path": f"/t/{session}.jsonl", "cwd": str(cwd),
            "hook_event_name": "SessionStart", "source": source}


def _records(hook_dir):
    path = hook_dir / session_roots.ROOTS_NAME
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []


def test_a_session_started_in_a_worktree_records_its_main_repository(tmp_path, repo):
    main, worktree = repo
    proc = _start(tmp_path / "hooks", _payload(worktree))
    assert proc.returncode == 0 and proc.stdout == ""  # SessionStart output would land in the session's context
    (record,) = _records(tmp_path / "hooks")
    assert (record["session"], record["transcript"], record["root"]) == ("s1", "/t/s1.jsonl", str(main))
    assert record["root"] == str(resolve_project_root(str(worktree)))  # what the nightly run sees while it exists


def test_a_cleared_session_is_a_new_session_and_is_recorded_too(tmp_path, repo):
    _, worktree = repo
    _start(tmp_path / "hooks", _payload(worktree, source="clear", session="s2"))
    assert [r["session"] for r in _records(tmp_path / "hooks")] == ["s2"]


@pytest.mark.parametrize("source", ["resume", "compact"])
def test_a_resumed_or_compacted_session_is_not_recorded_again(tmp_path, repo, source):
    # the agent may have moved to another repository's worktree since it started
    _, worktree = repo
    _start(tmp_path / "hooks", _payload(worktree, source=source))
    assert _records(tmp_path / "hooks") == []


def test_a_session_in_the_main_checkout_its_subfolder_or_outside_git_records_nothing(tmp_path, repo):
    main, _ = repo
    outside = tmp_path / "notes"
    outside.mkdir()
    for cwd in (main, main / "e2e", outside, tmp_path / "missing"):
        proc = _start(tmp_path / "hooks", _payload(cwd))
        assert proc.returncode == 0 and proc.stdout == ""
    assert _records(tmp_path / "hooks") == []


@pytest.mark.parametrize("payload", ["not json", "[]", json.dumps({"cwd": "/x", "session_id": "s"}),
                                     json.dumps({"session_id": "s", "transcript_path": "/t"}),
                                     json.dumps({"cwd": "/x", "transcript_path": "/t"})])
def test_a_payload_it_cannot_use_is_passed_over_quietly(tmp_path, payload):
    proc = _start(tmp_path / "hooks", payload)
    assert (proc.returncode, proc.stdout, proc.stderr) == (0, "", "")
    assert not (tmp_path / "hooks").exists()


def test_the_hook_writes_where_the_nightly_run_reads():
    assert session_hook.ROOTS_NAME == session_roots.ROOTS_NAME
    assert session_hook.LOCK_NAME == session_roots.LOCK_NAME
    assert os.path.expanduser("~/.claude/agent-reflect") == session_hook.DEFAULT_DIR


@pytest.fixture
def cfg(tmp_path):
    return Config.under(tmp_path)


def _write(cfg, *records):
    path = guard.guard_dir(cfg) / session_roots.ROOTS_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        for record in records:
            fh.write(record if isinstance(record, str) else json.dumps(record) + "\n")


def test_the_first_record_of_a_session_wins_and_a_broken_line_is_skipped(cfg):
    _write(cfg, {"session": "a", "transcript": "/t/a.jsonl", "root": "/r/start"}, "{broken\n", ["not", "a", "record"],
           {"session": "b", "transcript": "/t/b.jsonl"}, {"session": "a", "transcript": "/t/a.jsonl", "root": "/r/later"},
           json.dumps(["x"]) + "\n")
    assert session_roots.load(cfg) == {"a": "/r/start"}


def test_a_record_without_a_session_is_keyed_by_its_transcript_name(cfg):
    _write(cfg, {"transcript": "/t/abc-123.jsonl", "root": "/r/1"})
    assert session_roots.load(cfg) == {"abc-123": "/r/1"}


def test_no_file_is_no_records(cfg):
    assert session_roots.load(cfg) == {}


def test_a_full_log_keeps_the_first_record_of_every_session_whose_transcript_is_still_there(cfg, tmp_path):
    live = tmp_path / "live.jsonl"
    live.write_text("{}\n")
    _write(cfg, {"session": "live", "transcript": str(live), "root": "/r/1"},
           {"session": "gone", "transcript": str(tmp_path / "gone.jsonl"), "root": "/r/2"},
           {"session": "live", "transcript": str(live), "root": "/r/3"}, "{broken\n")
    assert session_roots.compact(cfg, limit=10**6) is False  # small: left as it is
    assert session_roots.compact(cfg, limit=10) is True
    assert session_roots.load(cfg) == {"live": "/r/1"}
    assert len(_records(guard.guard_dir(cfg))) == 1
    assert session_roots.compact(cfg, limit=10) is False  # nothing left to drop: not rewritten


def test_compacting_no_log_is_a_no_op(cfg):
    assert session_roots.compact(cfg, limit=10) is False
    assert not (guard.guard_dir(cfg) / session_roots.ROOTS_NAME).exists()


def test_the_hook_waits_for_the_lock_the_compaction_holds(tmp_path, repo):
    _, worktree = repo
    hook_dir = tmp_path / "hooks"
    hook_dir.mkdir()
    with open(hook_dir / session_roots.LOCK_NAME, "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        with pytest.raises(subprocess.TimeoutExpired):  # killed while it waits, before it writes
            subprocess.run([sys.executable, session_hook.__file__], input=json.dumps(_payload(worktree)),
                           text=True, capture_output=True, timeout=2,
                           env={**GIT_ENV, "AGENT_REFLECT_GUARD_DIR": str(hook_dir)})
    assert _records(hook_dir) == []
    _start(hook_dir, _payload(worktree))
    assert len(_records(hook_dir)) == 1


def test_the_compaction_waits_for_the_lock_the_hook_holds(cfg):
    _write(cfg, {"session": "gone", "transcript": "/nowhere/gone.jsonl", "root": "/r/1"})
    done = []
    with open(guard.guard_dir(cfg) / session_roots.LOCK_NAME, "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        worker = threading.Thread(target=lambda: done.append(session_roots.compact(cfg, limit=10)))
        worker.start()
        worker.join(timeout=1)
        assert worker.is_alive() and done == []
    worker.join(timeout=10)
    assert done == [True] and session_roots.load(cfg) == {}
