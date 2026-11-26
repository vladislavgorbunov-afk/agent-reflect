from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from agent_reflect.scrub import scrub

_TIMEOUT = 120
_STDERR_LIMIT = 300
_STATES = {"OPEN", "MERGED", "CLOSED"}


class GitError(Exception):
    def __init__(self, message: str, returncode: int | None = None):
        super().__init__(message)
        self.returncode = returncode  # None: the command timed out or never started


def _exec(argv: list[str], run, *, cwd: Path | None = None, env: dict | None = None,
          input: str | None = None, errors: str | None = None) -> str:
    """Run one command; return stdout. Failures raise GitError carrying the
    command name and a short scrubbed stderr (never the full argv).

    errors="replace" reads bytes that are not UTF-8 as U+FFFD instead of failing."""
    name = " ".join(argv[:2]) if argv[0] == "gh" else argv[0]
    kwargs: dict = {"capture_output": True, "text": True, "timeout": _TIMEOUT, "check": False}
    if cwd is not None:
        kwargs["cwd"] = cwd
    if env is not None:
        kwargs["env"] = env
    if input is not None:
        kwargs["input"] = input
    if errors is not None:
        kwargs["errors"] = errors
    try:
        proc = run(argv, **kwargs)
    except subprocess.TimeoutExpired:
        raise GitError(f"{name} timed out after {_TIMEOUT}s") from None
    except OSError as exc:
        raise GitError(scrub(f"{name} could not start: {exc}")[:_STDERR_LIMIT]) from None
    if proc.returncode != 0:
        err = scrub((proc.stderr or "").strip())[:_STDERR_LIMIT]
        raise GitError(f"{name} failed ({proc.returncode}): {err}", proc.returncode)
    return proc.stdout or ""


def _git(repo: Path, args: list[str], run, *, errors: str | None = None) -> str:
    return _exec(["git", "-C", str(repo), *args], run, errors=errors)


def _gh_env(token: str) -> dict:
    return {**os.environ, "GH_TOKEN": token}


def gh_token(run=subprocess.run) -> str:
    out = _exec(["git", "credential", "fill"], run, input="protocol=https\nhost=github.com\n\n")
    for line in out.splitlines():
        if line.startswith("password="):
            token = line[len("password="):]
            if token:
                return token
    raise GitError("git credential fill returned no password")


def prepare_worktree(repo: Path, branch: str, dest: Path, run=subprocess.run) -> None:
    _git(repo, ["fetch", "origin", "main"], run)
    _git(repo, ["worktree", "add", "-b", branch, str(dest), "origin/main"], run)


def commit_and_push(worktree: Path, paths: list[str], message: str, branch: str,
                    run=subprocess.run) -> str:
    """Commit the paths and push the branch; returns the pushed commit."""
    _git(worktree, ["add", "--", *paths], run)
    _git(worktree, ["commit", "-m", message], run)
    sha = _git(worktree, ["rev-parse", "HEAD"], run).strip()
    _git(worktree, ["push", "-u", "origin", branch], run)
    return sha


def commits_since(worktree: Path, sha: str, run=subprocess.run) -> int:
    """Commits on the worktree's HEAD after `sha`: work done there after the run pushed."""
    return int(_git(worktree, ["rev-list", "--count", f"{sha}..HEAD"], run).strip())


def prune_worktrees(repo: Path, run=subprocess.run) -> None:
    """Drop git's records of worktrees whose folders are gone."""
    _git(repo, ["worktree", "prune"], run)


def create_draft_pr(worktree: Path, branch: str, title: str, body: str, token: str,
                    run=subprocess.run) -> str:
    out = _exec(
        ["gh", "pr", "create", "--draft", "--base", "main", "--head", branch,
         "--title", title, "--body", body],
        run, cwd=worktree, env=_gh_env(token),
    )
    lines = out.strip().splitlines()
    if not lines:
        raise GitError("gh pr create returned no URL")
    return lines[-1].strip()


def pr_state(repo: Path, url: str, token: str, run=subprocess.run) -> str:
    out = _exec(["gh", "pr", "view", url, "--json", "state"], run, cwd=repo, env=_gh_env(token))
    try:
        state = json.loads(out).get("state")
    except (ValueError, AttributeError):
        raise GitError("gh pr view returned unparseable output") from None
    if state not in _STATES:
        raise GitError(f"gh pr view returned unexpected state: {str(state)[:40]}")
    return state


def pr_merged_at(repo: Path, url: str, token: str, run=subprocess.run) -> str | None:
    """When the PR was merged, as YYYY-MM-DDTHH:MM:SSZ, or None when GitHub gives no time."""
    out = _exec(["gh", "pr", "view", url, "--json", "mergedAt"], run, cwd=repo, env=_gh_env(token))
    try:
        value = json.loads(out).get("mergedAt")
    except (ValueError, AttributeError):
        raise GitError("gh pr view returned unparseable output") from None
    if not isinstance(value, str) or not value:
        return None
    try:
        when = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    return when.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def file_at_origin_main(repo: Path, rel_path: str, run=subprocess.run) -> str:
    _git(repo, ["fetch", "origin", "main"], run)
    return _git(repo, ["show", f"origin/main:{rel_path}"], run)


def origin_main_sha(repo: Path, run=subprocess.run) -> str:
    """origin/main after a fetch: the commit a weekly scan reads up to."""
    _git(repo, ["fetch", "origin", "main"], run)
    return _git(repo, ["rev-parse", "origin/main"], run).strip()


def is_ancestor(repo: Path, sha: str, head: str, run=subprocess.run) -> bool:
    """Whether `sha` is in `head`'s history; asked right after `head` was read from the same repo.

    Exit 1: a history rewrite left `sha` behind. Exit 128: git does not know `sha` at all
    (pruned since, or never there). Neither ever heals, so both are False. Any other
    failure raises GitError.
    """
    try:
        _git(repo, ["merge-base", "--is-ancestor", sha, head], run)
    except GitError as exc:
        if exc.returncode in (1, 128):
            return False
        raise
    return True


def commits_touching(repo: Path, since: str, until: str, paths: list[str],
                     run=subprocess.run) -> list[tuple[str, str, str]]:
    """(sha, date, subject) of the non-merge commits in since..until that touch `paths`, oldest first.

    A subject in bytes that are not UTF-8 reads with U+FFFD in their place."""
    out = _git(repo, ["log", "--reverse", "--no-merges", "--date=short", "--format=%H%x1f%ad%x1f%s",
                      f"{since}..{until}", "--", *paths], run, errors="replace")
    rows = []
    for line in out.splitlines():
        parts = line.split("\x1f")
        if len(parts) == 3:
            rows.append((parts[0], parts[1], parts[2]))
    return rows


def commit_diff(repo: Path, sha: str, paths: list[str], run=subprocess.run) -> str:
    """What one commit changed in `paths`, as a unified diff; bytes that are not UTF-8 read as U+FFFD."""
    return _git(repo, ["show", "--format=", "--unified=3", sha, "--", *paths], run, errors="replace")


def remove_worktree(repo: Path, dest: Path, run=subprocess.run, force: bool = True) -> None:
    """Without force git refuses a worktree with modified or untracked files. Ignored files go with
    it either way, and its commits stay on its branch."""
    _git(repo, ["worktree", "remove", *(["--force"] if force else []), str(dest)], run)
