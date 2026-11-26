#!/usr/bin/env python3
"""Session-start hook: notes the repository a session started in, while its folder still exists.

The nightly run gives a session to the project of the folder it started in, and asks git whose
checkout that folder is. A worker started in a linked worktree (a review checkout in a scratchpad,
say) loses that answer when the worktree is removed before the night: git has nothing to look at,
and the session went unread as no project. This hook asks git once when a session starts and, for a
session in a linked worktree, appends `{"session", "transcript", "cwd", "root"}` to the
session-roots log the nightly run reads (`agent_reflect/session_roots.py`).

Only a start is noted: a new session, or a /clear, which begins a new session id and transcript. A
resume or a compaction is the same session going on, and the agent may have moved to another
repository's worktree since it started, so writing then would name the wrong project. Nothing is
written for a folder that is not a linked worktree (the main checkout, a subfolder of it, a folder
outside git): that folder will answer the night as well as it answers now.

Stands alone on purpose (stdlib only, no import of the package), as the guard hook does: it runs
under /usr/bin/python3 before every session. It prints nothing, since what a SessionStart hook prints
goes into the session's context, and any failure here is silent. The append holds the same lock the
nightly compaction takes, so a line written while the log is being rewritten is not lost.
"""
from __future__ import annotations

import fcntl
import json
import os
import subprocess
import sys
import time

DEFAULT_DIR = os.path.expanduser("~/.claude/agent-reflect")
ROOTS_NAME = "session-roots.jsonl"
LOCK_NAME = "session-roots.lock"
GIT_TIMEOUT_S = 3  # the hook itself is stopped at 5
CONTINUED = ("resume", "compact")  # the same session going on: its start is already noted


def linked_worktree_root(cwd: str):
    """The main checkout when `cwd` is inside a linked worktree, else None (as projects._git_main_root)."""
    try:
        proc = subprocess.run(["git", "-C", cwd, "rev-parse", "--path-format=absolute", "--git-dir",
                               "--git-common-dir"],
                              capture_output=True, text=True, check=False, timeout=GIT_TIMEOUT_S)
    except (OSError, subprocess.SubprocessError):
        return None
    lines = (proc.stdout or "").strip().split("\n")
    if proc.returncode != 0 or len(lines) != 2:
        return None
    git_dir, common = (line.strip() for line in lines)
    # in the main checkout (and any subfolder of it) the two are the same .git
    if git_dir == common or not common.endswith("/.git") or len(common) <= len("/.git"):
        return None
    return common[: -len("/.git")]


def main() -> int:
    try:
        data = json.loads(sys.stdin.read())
        session, transcript, cwd = data.get("session_id"), data.get("transcript_path"), data.get("cwd")
        if not all(isinstance(v, str) and v for v in (session, transcript, cwd)):
            return 0
        if data.get("source") in CONTINUED:
            return 0
        root = linked_worktree_root(cwd)
        if root is None:
            return 0
        hook_dir = os.environ.get("AGENT_REFLECT_GUARD_DIR") or DEFAULT_DIR
        os.makedirs(hook_dir, exist_ok=True)
        record = {"session": session, "transcript": transcript, "cwd": cwd, "root": root,
                  "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        with open(os.path.join(hook_dir, LOCK_NAME), "a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            with open(os.path.join(hook_dir, ROOTS_NAME), "a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
