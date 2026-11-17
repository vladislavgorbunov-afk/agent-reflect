"""Weekly: test-writing rules a project changed that dev-process-toolkit could carry.

A project works out how its tests are written in a few files of its own (an e2e
skill, a selector doc, a testing spec). Some of those rules hold for every
project the toolkit serves; most are about one project's runner or stand.
Telling them apart is a judgement, so this step only proposes: it reads what
changed in the watched files on origin/main since the last scan and lists the
candidates in the weekly digest. Carrying one over stays a person's call.

The digest of a week is rewritten by a second run in that week, so every
candidate also goes into an append-only file (Config.upstream_candidates_file).

The first scan of a repo is a baseline: what was there before it began is not news.
A mark that origin/main's history no longer reaches (a rewrite) starts a new baseline.
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

from agent_reflect import gitops
from agent_reflect.config import Config
from agent_reflect.digest import week_tag
from agent_reflect.gitops import GitError
from agent_reflect.ledger import State
from agent_reflect.llm import LlmError
from agent_reflect.promptkit import ERROR_MAX, DataTags, error_text, load_template, model_text, nonce, squash
from agent_reflect.scrub import scrub

# Commits read per repo per scan; the mark stops at the last one read, the rest wait a week.
COMMITS_MAX = 25
DIFF_MAX = 6000
DIFF_CUT = f"[… diff cut at {DIFF_MAX} chars]"
TEXT_MAX = 300
SHA_MIN = 7
ISO = "%Y-%m-%dT%H:%M:%SZ"
# Where in dev-process-toolkit a carried rule can go; the prompt offers the same list.
PLACES = ("skills/tdd-write-test", "skills/tdd-implement", "skills/debug", "docs/patterns.md",
          "templates/spec-templates/testing-spec.md.template", "docs/verification-skills.md")

_STR = {"type": "string"}
UPSTREAM_SCHEMA: dict = {
    "type": "object",
    "properties": {"candidates": {"type": "array", "items": {
        "type": "object",
        "properties": {"sha": _STR, "rule": _STR, "why": _STR, "place": {"type": "string", "enum": list(PLACES)}},
        "required": ["sha", "rule", "why", "place"],
        "additionalProperties": False,
    }}},
    "required": ["candidates"],
    "additionalProperties": False,
}

_TAGS = DataTags(("commits",))


def build_prompt(commits: list[dict]) -> str:
    blocks = "\n\n".join(
        f"репозиторий: {c['repo']} | sha: {c['sha']} | дата: {c['date']} | заголовок: {_TAGS.defuse(c['subject'])}\n"
        f"{_TAGS.defuse(c['diff'])}"
        for c in commits
    )
    return load_template("upstream_candidates.md").substitute(nonce=nonce(), cut=DIFF_CUT, commits=blocks)


def _cut(diff: str) -> str:
    return diff if len(diff) <= DIFF_MAX else f"{diff[:DIFF_MAX]}\n{DIFF_CUT}"


def _line(value) -> str:
    """A model-written one-liner, safe to store and show in the digest."""
    return squash(model_text(value))


def _invalid(state: State, reason: str) -> None:
    state.event("plan_invalid", item="upstream", reason=scrub(reason)[:ERROR_MAX])


def _candidates(state: State, answer, commits: list[dict]) -> list[dict] | None:
    """The answer's candidates, each with its commit's full sha; None when the answer itself is malformed."""
    items = answer.get("candidates") if isinstance(answer, dict) else None
    if not isinstance(items, list):
        _invalid(state, "candidates is not a list")
        return None
    out = []
    for item in items:
        if not isinstance(item, dict):
            _invalid(state, "a candidate is not an object")
            continue
        sha = model_text(item.get("sha"))
        commit = next((c for c in commits if len(sha) >= SHA_MIN and c["sha"].startswith(sha)), None)
        if commit is None:
            _invalid(state, f"candidate names a commit that was not shown: {sha[:12]}")
            continue
        rule = _line(item.get("rule"))[:TEXT_MAX]
        if not rule:
            _invalid(state, f"candidate for {sha[:SHA_MIN]} has no rule")
            continue
        place = _line(item.get("place"))
        out.append({"repo": commit["repo"], "sha": commit["sha"], "date": commit["date"],
                    "subject": squash(scrub(commit["subject"]))[:TEXT_MAX], "rule": rule,
                    "why": _line(item.get("why"))[:TEXT_MAX],
                    "place": place if place in PLACES else ""})  # an unknown place: the person picks one
    return out


def _record(cfg: Config, candidates: list[dict], now: datetime) -> None:
    """Append the candidates whose commit the file does not hold yet, one JSON line each.

    The file is never rewritten. Raises OSError.
    """
    path = cfg.upstream_candidates_file
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        old = path.read_bytes()
    except FileNotFoundError:
        old = b""
    seen = set()
    for line in old.decode("utf-8", "replace").splitlines():
        try:
            entry = json.loads(line)
        except ValueError:
            continue  # a line cut short by a crash
        if isinstance(entry, dict):
            seen.add((entry.get("repo"), entry.get("sha")))
    stamp = {"ts": now.strftime(ISO), "week": week_tag(now)}
    lines = [json.dumps({**stamp, **c}, ensure_ascii=False) for c in candidates if (c["repo"], c["sha"]) not in seen]
    if not lines:
        return
    head = "\n" if old and not old.endswith(b"\n") else ""  # never glue onto a line cut short
    with path.open("a", encoding="utf-8") as fh:
        fh.write(head + "".join(f"{line}\n" for line in lines))
        fh.flush()
        os.fsync(fh.fileno())


def step(cfg: Config, state: State, *, now: datetime, llm, git=gitops) -> list[dict]:
    """The week's candidates. Moves the marks only once the commits were judged and recorded."""
    if not cfg.upstream_sources:
        return []
    commits: list[dict] = []
    marks: dict[str, str] = {}
    for name, paths in cfg.upstream_sources:
        repo = cfg.repos.get(name)
        if repo is None or not Path(repo).is_dir():
            continue
        try:
            head = git.origin_main_sha(repo)
            since = state.upstream_synced.get(name)
            if not since:
                marks[name] = head
                continue
            if since == head:
                continue
            if not git.is_ancestor(repo, since, head):
                # since..head would fail or read rewritten copies every week: start over from here
                state.event("upstream_rebaselined", repo=name, old=since, new=head)
                marks[name] = head
                continue
            rows = git.commits_touching(repo, since, head, list(paths))
            mark = head
            if len(rows) > COMMITS_MAX:
                rows = rows[:COMMITS_MAX]
                mark = rows[-1][0]
            found = [{"repo": name, "sha": sha, "date": date, "subject": subject,
                      "diff": _cut(git.commit_diff(repo, sha, list(paths)))}
                     for sha, date, subject in rows]
        except GitError as exc:
            state.event("git_failed", repo=name, step="upstream", error=error_text(exc))
            continue
        commits += found
        marks[name] = mark
    candidates: list[dict] = []
    if commits:
        try:
            answer = llm(cfg, cfg.consolidate_model, build_prompt(commits), UPSTREAM_SCHEMA,
                         effort=cfg.consolidate_effort)
        except LlmError as exc:
            state.event("llm_error", step="upstream", error=error_text(exc))
            return []  # no mark moves: next week reads the same commits again
        judged = _candidates(state, answer, commits)
        if judged is None:
            return []  # a malformed answer is a model failure too
        candidates = judged
    shown = [{**c, "sha": c["sha"][:SHA_MIN]} for c in candidates]
    if candidates:
        try:
            _record(cfg, candidates, now)
        except (OSError, UnicodeEncodeError) as exc:
            state.event("upstream_failed", step="record", file=str(cfg.upstream_candidates_file),
                        error=error_text(exc))
            return shown  # in this week's digest; the marks stay, so next week records them
    state.upstream_synced.update(marks)
    return shown
