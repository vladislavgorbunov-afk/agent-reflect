#!/usr/bin/env python3
"""Session-start hook: notes the repository a session started in, while its folder still exists.

The nightly run gives a session to the project of the folder it started in, and asks git whose
checkout that folder is. A worker started in a linked worktree (a review checkout in a scratchpad,
say) loses that answer when the worktree is removed before the night: git has nothing to look at,
and the session went unread as no project. Claude Code starts this hook once per session start
(SessionStart: a new session, a resume, a /clear, a compaction). It asks git once and appends
`{"transcript", "cwd", "root"}` to the session-roots log the nightly run reads
(`agent_reflect/session_roots.py`). Nothing is written for a session whose folder is the checkout
itself or is outside git: the folder will answer the night as well as it answers now.

Stands alone on purpose (stdlib only, no import of the package), as the guard hook does: it runs
under /usr/bin/python3 before every session. It prints nothing, since what a SessionStart hook prints
goes into the session's context, and any failure here is silent.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time

DEFAULT_DIR = os.path.expanduser("~/.claude/agent-reflect")
ROOTS_NAME = "session-roots.jsonl"
GIT_TIMEOUT_S = 3  # the hook itself is stopped at 5


def main_root(cwd: str):
    """The main checkout of the repository `cwd` is in, or None (as projects._git_main_root)."""
    try:
        proc = subprocess.run(["git", "-C", cwd, "rev-parse", "--path-format=absolute", "--git-common-dir"],
                              capture_output=True, text=True, check=False, timeout=GIT_TIMEOUT_S)
    except (OSError, subprocess.SubprocessError):
        return None
    out = (proc.stdout or "").strip()
    if proc.returncode != 0 or not out.endswith("/.git") or len(out) <= len("/.git"):
        return None
    return out[: -len("/.git")]


def main() -> int:
    try:
        data = json.loads(sys.stdin.read())
        transcript, cwd = data.get("transcript_path"), data.get("cwd")
        if not (isinstance(transcript, str) and transcript and isinstance(cwd, str) and cwd):
            return 0
        root = main_root(cwd)
        if root is None or os.path.realpath(cwd) == os.path.realpath(root):
            return 0
        hook_dir = os.environ.get("AGENT_REFLECT_GUARD_DIR") or DEFAULT_DIR
        os.makedirs(hook_dir, exist_ok=True)
        record = {"transcript": transcript, "cwd": cwd, "root": root,
                  "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        # one short append per line: lines from sessions starting at once do not interleave
        with open(os.path.join(hook_dir, ROOTS_NAME), "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
