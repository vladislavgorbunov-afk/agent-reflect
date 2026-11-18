"""Where each session started: the log the session-start hook writes, read by the nightly run.

A session started in a linked worktree is noted with its main repository (`session_hook.py`).
The nightly run asks git first; this log answers only when the folder is gone by then.
The log is rotated once it is large: the hook keeps appending to the fresh file, and the run reads
the rotated half too, so a line written while the file is renamed lands in a half that is read.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from agent_reflect.config import Config
from agent_reflect.guard import guard_dir
from agent_reflect.session_hook import ROOTS_NAME

ROTATE_BYTES = 262_144  # about a thousand session starts in worktrees


def _paths(cfg: Config) -> tuple[Path, Path]:
    path = guard_dir(cfg) / ROOTS_NAME
    return path.with_name(path.name + ".1"), path


def load(cfg: Config) -> dict[str, str]:
    """Transcript path → main repository root; the latest line of a transcript wins."""
    roots: dict[str, str] = {}
    for path in _paths(cfg):  # the rotated half first: it is older
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for line in text.splitlines():
            try:
                record = json.loads(line)
            except ValueError:
                continue
            if not isinstance(record, dict):
                continue
            transcript, root = record.get("transcript"), record.get("root")
            if isinstance(transcript, str) and transcript and isinstance(root, str) and root:
                roots[transcript] = root
    return roots


def rotate(cfg: Config, limit: int = ROTATE_BYTES) -> bool:
    """Move a log over `limit` bytes aside, dropping the half moved aside before; True when it moved."""
    rotated, path = _paths(cfg)
    try:
        if path.stat().st_size <= limit:
            return False
        os.replace(path, rotated)
    except OSError:
        return False
    return True
