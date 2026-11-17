from __future__ import annotations

import re
import subprocess
from pathlib import Path

_WORKTREE_MARKER = "/.claude/worktrees/"
GIT_TIMEOUT_S = 30


def sanitize(path: str) -> str:
    """Claude Code's project-folder slug for an absolute path."""
    return re.sub(r"[^A-Za-z0-9]", "-", path)


def _git_main_root(cwd: str, run) -> Path | None:
    try:
        proc = run(
            ["git", "-C", cwd, "rev-parse", "--path-format=absolute", "--git-common-dir"],
            capture_output=True,
            text=True,
            check=False,
            timeout=GIT_TIMEOUT_S,
        )
    except (OSError, subprocess.SubprocessError):  # TimeoutExpired is a SubprocessError
        return None
    if getattr(proc, "returncode", 1) != 0:
        return None
    out = (getattr(proc, "stdout", "") or "").strip()
    if not out.endswith("/.git"):
        return None
    return Path(out[: -len("/.git")]) if len(out) > len("/.git") else None


def resolve_project_root(cwd: str | None, run=subprocess.run) -> Path | None:
    """Main repository root for a session's cwd, or None. Never raises."""
    if not cwd:
        return None
    root = _git_main_root(cwd, run)
    if root is not None:
        return root
    head, marker, _ = cwd.partition(_WORKTREE_MARKER)
    if marker and head:
        return Path(head)
    return None


def memory_dir_for(projects_dir: Path, root: Path) -> Path:
    return projects_dir / sanitize(str(root)) / "memory"
