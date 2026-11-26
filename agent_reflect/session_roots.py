"""Where each session started: the log the session-start hook writes, read by the nightly run.

A session started in a linked worktree is noted with its main repository (`session_hook.py`),
keyed by its session id, which is also its transcript's file name. The nightly run asks git first;
this log answers only when the folder is gone by then. The first record of a session wins: that is
where it started.

The log is compacted once it is large: the first record of every session whose transcript is still
on disk is kept, and the rest go. Claude Code removes old transcripts itself, so what stays is
bounded by what can still be read, and a record is never dropped while its session can be. The
rewrite holds the lock the hook appends under, so a line written meanwhile is not lost.
"""
from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path

from agent_reflect.config import Config
from agent_reflect.guard import guard_dir
from agent_reflect.session_hook import LOCK_NAME, ROOTS_NAME

COMPACT_BYTES = 262_144  # a few years of worktree sessions


def _parse(text: str) -> list[tuple[str, dict]]:
    """(session key, record) of every usable line, in order."""
    out = []
    for line in text.splitlines():
        try:
            record = json.loads(line)
        except ValueError:
            continue
        if not isinstance(record, dict):
            continue
        root, session, transcript = record.get("root"), record.get("session"), record.get("transcript")
        if not (isinstance(root, str) and root):
            continue
        if not (isinstance(session, str) and session):
            # the transcript's file name is its session id
            session = Path(transcript).stem if isinstance(transcript, str) and transcript else ""
        if session:
            out.append((session, record))
    return out


def load(cfg: Config) -> dict[str, str]:
    """Session id → main repository root; the first record of a session wins."""
    try:
        text = (guard_dir(cfg) / ROOTS_NAME).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return {}
    roots: dict[str, str] = {}
    for session, record in _parse(text):
        roots.setdefault(session, record["root"])
    return roots


def compact(cfg: Config, limit: int | None = None, exists=os.path.exists) -> bool:
    """Rewrite a log over `limit` bytes (COMPACT_BYTES by default) to the first record of each
    session whose transcript is still there; True when it was rewritten."""
    limit = COMPACT_BYTES if limit is None else limit
    path = guard_dir(cfg) / ROOTS_NAME
    try:
        if path.stat().st_size <= limit:
            return False
        with open(path.with_name(LOCK_NAME), "a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            text = path.read_text(encoding="utf-8", errors="replace")
            kept, seen = [], set()
            for session, record in _parse(text):
                transcript = record.get("transcript")
                if session in seen or not (isinstance(transcript, str) and exists(transcript)):
                    continue
                seen.add(session)
                kept.append(record)
            if len(kept) == len([line for line in text.splitlines() if line.strip()]):
                return False
            tmp = path.with_name(path.name + ".tmp")
            tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in kept), encoding="utf-8")
            os.replace(tmp, path)
    except OSError:
        return False
    return True
