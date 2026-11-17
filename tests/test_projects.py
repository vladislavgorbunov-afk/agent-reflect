from __future__ import annotations

import subprocess
from pathlib import Path

from agent_reflect.projects import memory_dir_for, resolve_project_root, sanitize


def _git(cwd: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(cwd), *args], check=True, capture_output=True)


def test_sanitize_matches_claude_slugs():
    assert sanitize("/Users/me/git/webapp") == "-Users-me-git-webapp"
    assert (
        sanitize("/Users/me/git/webapp/.claude/worktrees/competent-moser")
        == "-Users-me-git-webapp--claude-worktrees-competent-moser"
    )


def test_resolve_main_repo_from_worktree(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.name", "t")
    _git(repo, "config", "user.email", "t@example.com")
    (repo / "a.txt").write_text("a")
    _git(repo, "add", "a.txt")
    _git(repo, "commit", "-q", "-m", "init")
    worktree = repo / ".claude" / "worktrees" / "wt"
    _git(repo, "worktree", "add", "-q", str(worktree), "-b", "wt")
    assert resolve_project_root(str(worktree)) == repo.resolve()
    assert resolve_project_root(str(repo)) == repo.resolve()


def test_resolve_deleted_worktree_by_path_rule():
    assert resolve_project_root("/x/repo/.claude/worktrees/gone") == Path("/x/repo")


def test_resolve_non_git_returns_none(tmp_path):
    assert resolve_project_root(str(tmp_path)) is None
    assert resolve_project_root(None) is None


def test_resolve_survives_missing_git():
    def boom(*a, **k):
        raise FileNotFoundError("git")

    assert resolve_project_root("/x/repo/.claude/worktrees/gone", run=boom) == Path("/x/repo")
    assert resolve_project_root("/x/plain", run=boom) is None


def test_memory_dir_for():
    assert memory_dir_for(Path("/p"), Path("/Users/me/git/webapp")) == Path(
        "/p/-Users-me-git-webapp/memory"
    )


def test_resolve_git_call_times_out_and_a_timeout_falls_back():
    seen = []

    def hung(argv, **kw):
        seen.append(kw.get("timeout"))
        raise subprocess.TimeoutExpired(argv, kw.get("timeout"))

    assert resolve_project_root("/x/repo/.claude/worktrees/gone", run=hung) == Path("/x/repo")
    assert resolve_project_root("/x/plain", run=hung) is None
    assert seen == [30, 30]
