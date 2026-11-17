from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from agent_reflect import gitops
from agent_reflect.gitops import GitError


class FakeRun:
    def __init__(self, stdout: str = "", returncode: int = 0, stderr: str = ""):
        self.calls: list[tuple[list[str], dict]] = []
        self._result = (stdout, returncode, stderr)

    def __call__(self, argv, **kwargs):
        self.calls.append((list(argv), kwargs))
        out, rc, err = self._result
        return subprocess.CompletedProcess(argv, rc, stdout=out, stderr=err)


def test_gh_token_parsed_from_credential_fill():
    run = FakeRun("protocol=https\nhost=github.com\nusername=x\npassword=tok123\n")
    assert gitops.gh_token(run=run) == "tok123"
    argv, kw = run.calls[0]
    assert argv == ["git", "credential", "fill"]
    assert kw["input"] == "protocol=https\nhost=github.com\n\n"
    assert kw["capture_output"] is True and kw["text"] is True
    assert kw["check"] is False and kw["timeout"] == 120


def test_gh_token_missing_password_raises():
    with pytest.raises(GitError):
        gitops.gh_token(run=FakeRun("protocol=https\nhost=github.com\n"))


def test_create_draft_pr_token_only_in_env(tmp_path):
    run = FakeRun("https://github.com/o/r/pull/7\n")
    url = gitops.create_draft_pr(tmp_path, "docs/x", "T", "B", "tok123", run=run)
    assert url == "https://github.com/o/r/pull/7"
    argv, kw = run.calls[0]
    assert argv[:3] == ["gh", "pr", "create"]
    assert "--draft" in argv
    assert argv[argv.index("--base") + 1] == "main"
    assert argv[argv.index("--head") + 1] == "docs/x"
    assert argv[argv.index("--title") + 1] == "T"
    assert argv[argv.index("--body") + 1] == "B"
    assert not any("tok123" in a for a in argv)
    assert kw["env"]["GH_TOKEN"] == "tok123"
    assert kw["cwd"] == tmp_path and kw["timeout"] == 120 and kw["check"] is False


@pytest.mark.parametrize("state", ["OPEN", "MERGED", "CLOSED"])
def test_pr_state_parses_json(tmp_path, state):
    run = FakeRun('{"state": "%s"}\n' % state)
    assert gitops.pr_state(tmp_path, "https://github.com/o/r/pull/7", "tok123", run=run) == state
    argv, kw = run.calls[0]
    assert argv == ["gh", "pr", "view", "https://github.com/o/r/pull/7", "--json", "state"]
    assert kw["env"]["GH_TOKEN"] == "tok123"


@pytest.mark.parametrize("out", ['{"state": "DRAFT"}', "not json", '{"x": 1}'])
def test_pr_state_unknown_raises(tmp_path, out):
    with pytest.raises(GitError):
        gitops.pr_state(tmp_path, "u", "tok123", run=FakeRun(out))


@pytest.mark.parametrize("code, expected", [(0, True), (1, False), (128, False)])
def test_is_ancestor_reads_the_exit_code(tmp_path, code, expected):
    run = FakeRun(returncode=code, stderr="fatal: Not a valid commit name")
    assert gitops.is_ancestor(tmp_path, "a" * 40, "c" * 40, run=run) is expected
    argv, _ = run.calls[0]
    assert argv == ["git", "-C", str(tmp_path), "merge-base", "--is-ancestor", "a" * 40, "c" * 40]


def test_is_ancestor_raises_on_an_exit_it_does_not_know(tmp_path):
    with pytest.raises(GitError, match=r"\(129\)"):
        gitops.is_ancestor(tmp_path, "a" * 40, "c" * 40, run=FakeRun(returncode=129, stderr="usage: git merge-base"))


def test_nonzero_exit_raises_scrubbed_git_error(tmp_path):
    secret = "ghp_" + "a" * 36
    run = FakeRun(returncode=1, stderr=f"fatal: bad token={secret}")
    with pytest.raises(GitError) as e:
        gitops.file_at_origin_main(tmp_path, "a.md", run=run)
    assert secret not in str(e.value)
    assert "git" in str(e.value)


def test_timeout_and_missing_binary_raise_git_error(tmp_path):
    def boom(argv, **kw):
        raise subprocess.TimeoutExpired(argv, 120)

    with pytest.raises(GitError):
        gitops.remove_worktree(tmp_path, tmp_path / "w", run=boom)

    def missing(argv, **kw):
        raise FileNotFoundError("gh")

    with pytest.raises(GitError):
        gitops.pr_state(tmp_path, "u", "t", run=missing)


def _git(cwd: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True, text=True
    ).stdout


def test_worktree_commit_push_against_local_bare_remote(tmp_path):
    bare = tmp_path / "origin.git"
    _git(tmp_path, "init", "--bare", "-b", "main", str(bare))
    clone = tmp_path / "clone"
    _git(tmp_path, "clone", str(bare), str(clone))
    _git(clone, "config", "user.name", "T")
    _git(clone, "config", "user.email", "t@example.com")
    (clone / "a.md").write_text("one\n")
    _git(clone, "add", "a.md")
    _git(clone, "commit", "-m", "init")
    _git(clone, "push", "-u", "origin", "HEAD:main")
    (clone / "wip.txt").write_text("user work in progress\n")  # untracked, must survive

    head_before = _git(clone, "rev-parse", "HEAD")
    status_before = _git(clone, "status", "--porcelain")

    wt = tmp_path / "wt"
    gitops.prepare_worktree(clone, "docs/agent-reflect-2026-10-20", wt)
    assert _git(clone, "rev-parse", "HEAD") == head_before
    assert _git(clone, "status", "--porcelain") == status_before

    (wt / "a.md").write_text("two\n")
    gitops.commit_and_push(wt, ["a.md"], "docs: change", "docs/agent-reflect-2026-10-20")

    assert "docs/agent-reflect-2026-10-20" in _git(bare, "branch", "--list")
    assert _git(bare, "show", "docs/agent-reflect-2026-10-20:a.md") == "two\n"
    assert _git(clone, "rev-parse", "HEAD") == head_before
    assert _git(clone, "status", "--porcelain") == status_before
    assert (clone / "a.md").read_text() == "one\n"

    # file_at_origin_main reads main from the remote after a fetch
    (clone / "b.md").write_text("b\n")
    assert gitops.file_at_origin_main(clone, "a.md") == "one\n"

    gitops.remove_worktree(clone, wt)
    assert not wt.exists()
    assert str(wt) not in _git(clone, "worktree", "list")


def test_commit_and_push_command_sequence(tmp_path):
    run = FakeRun()
    gitops.commit_and_push(tmp_path, ["a.md", "b.md"], "msg", "br", run=run)
    argvs = [c[0] for c in run.calls]
    assert argvs == [
        ["git", "-C", str(tmp_path), "add", "--", "a.md", "b.md"],
        ["git", "-C", str(tmp_path), "commit", "-m", "msg"],
        ["git", "-C", str(tmp_path), "push", "-u", "origin", "br"],
    ]
