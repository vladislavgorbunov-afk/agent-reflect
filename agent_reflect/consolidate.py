from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

from agent_reflect import (
    care, clusters, cursor_index, gitops, health, i18n, journal, memory_archive, memory_reads, upstream, usage,
)
from agent_reflect.edits import INCIDENT_LOG, SKILL_LARGE_CHARS, incident_error, size_error, unified_diff
from agent_reflect.config import RUN_ID_FORMAT, Config
from agent_reflect.gitops import GitError
from agent_reflect.ledger import Lesson, State, distinct_sessions, parse_ts
from agent_reflect.llm import LlmError, run_structured
from agent_reflect.memory import (
    MemoryIndexError,
    MemoryWriteError,
    autodream_disabled,
    go_live_if_room,
    managed_projects,
    rebalance,
    remove_lesson,
    sync_ownership,
    write_lesson,
)
from agent_reflect.projects import memory_dir_for
from agent_reflect.promptkit import ERROR_MAX, DataTags, error_text, load_template, model_text, nonce, squash
from agent_reflect.scrub import scrub
from agent_reflect.skills import EditError, Target, apply_edit_text, apply_personal, registry, write_atomic

ERROR_KINDS = frozenset({
    "llm_error", "apply_crashed", "memory_write_failed", "bad_verdict", "run_crashed",
    "autodream_not_disabled", "segment_unreadable", "index_unreadable", "edit_failed",
    "git_failed", "plan_invalid", "edit_skipped", "pr_closed", "cursor_copy_diverged", "prepare_failed",
    "transcript_failed", "archive_file_skipped", "upstream_failed", "journal_failed",
    "memory_reads_failed", "care_failed", "clusters_failed", "guard_rules_unwritten", "guard_fires_unread",
})
# Worth the user's eye in the digest, but nothing failed.
ADVISORY_KINDS = frozenset({"skill_large", "upstream_rebaselined"})
LIVE = ("live", "staged")
ISO = "%Y-%m-%dT%H:%M:%SZ"
# launchd fires late after sleep, so a run counts as due one day before the full interval.
DUE_SLACK = timedelta(days=1)
# A skill takes a lesson only after it showed up in this many sessions within the window.
PROMOTION_MIN_SESSIONS = 2
PROMOTION_WINDOW_DAYS = 30
# Violations of a live lesson that make it eligible on their own.
PROMOTION_MIN_VIOLATIONS = 2
# One plan may retire (merge away or remove) at most this share of the live and staged
# auto-lessons, rounded up and at least one; items beyond it are dropped in plan order.
REMOVAL_CAP_PERCENT = 25
COMMIT_SUBJECT = "docs(skills): Apply agent-reflect lessons"
CO_AUTHORED_BY = "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
PR_FOOTER = "🤖 Generated with [Claude Code](https://claude.com/claude-code)"
# The PR is for the team and must be English; a model line in another script is replaced.
FALLBACK_SUMMARY = "Add a rule learned from past agent sessions."
FALLBACK_WHY = "The same agent mistake came up in past sessions."
_CYRILLIC = re.compile("[Ѐ-ӿ]")
REDACTED = "[REDACTED]"
TEXT_MAX = 300
USER_TEXT_MAX = 4000
_WRITE_ERRORS = (MemoryWriteError, MemoryIndexError, OSError, UnicodeEncodeError)


def _object(properties: dict) -> dict:
    return {"type": "object", "properties": properties, "required": list(properties),
            "additionalProperties": False}


_STR = {"type": "string"}
_STRS = {"type": "array", "items": _STR}

PLAN_SCHEMA: dict = _object({
    "promotions": {"type": "array", "items": _object(
        {"target": _STR, "lesson_ids": _STRS, "user_rule_files": _STRS, "reason": _STR})},
    "merges": {"type": "array", "items": _object({"keep": _STR, "absorb": _STRS, "reason": _STR})},
    "removals": {"type": "array", "items": _object({"lesson_id": _STR, "reason": _STR})},
    "user_entry_suggestions": {"type": "array", "items": _object({"file": _STR, "suggestion": _STR})},
})
EDIT_SCHEMA: dict = _object({
    "skip": {"type": "boolean"}, "reason": _STR, "old_string": _STR, "new_string": _STR, "summary": _STR,
    "why_en": _STR,
})

_TAGS = DataTags(("auto_lessons", "user_rules", "targets", "target_file", "lessons"))


def is_due(state: State, now: datetime, interval_days: int) -> bool:
    """Due once `interval_days` (less a day of slack) passed since the last run, or since install."""
    since = state.last_consolidate_at or state.installed_at
    if not since:
        return True
    try:
        then = datetime.strptime(since, ISO).replace(tzinfo=timezone.utc)
    except ValueError:
        return True
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    return now - then >= timedelta(days=interval_days) - DUE_SLACK


def eligible(state: State, now: datetime) -> tuple[list[Lesson], dict[str, list[dict]]]:
    """Lessons confirmed enough to promote, and user entries the agent broke again.

    A lesson needs evidence from PROMOTION_MIN_SESSIONS distinct sessions within the
    last PROMOTION_WINDOW_DAYS, even at high confidence: one session proves nothing.
    A lesson the agent kept breaking while it sat in memory is promoted whatever its
    sessions: memory did not hold it, so it belongs in a skill.
    """
    lessons = [lesson for lesson in state.lessons.values()
               if lesson.status in LIVE and not lesson.rollback_hold  # its skill edit from before a rollback stands
               and (lesson.violations >= PROMOTION_MIN_VIOLATIONS
                    or ((lesson.repeats >= 2 or lesson.confidence == "high")
                        and distinct_sessions(lesson, now, PROMOTION_WINDOW_DAYS) >= PROMOTION_MIN_SESSIONS))]
    files = {path: list(evidence) for path, evidence in state.user_rule_repeats.items() if evidence}
    return lessons, files


# -- files we may write ---------------------------------------------------------

def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _plain_file_bytes(path: Path) -> bytes | None:
    """The bytes of a plain file reached without any symlink on the way, else None. Never raises."""
    try:
        if path.resolve() != path.absolute() or not path.is_file():
            return None
        return path.read_bytes()
    except OSError:
        return None


def _inside_worktree(dest: Path, rel: str) -> Path | None:
    """dest/rel when it is a plain file and nothing under dest on the way is a symlink."""
    path = dest / rel
    try:
        if path.resolve() != dest.resolve() / rel or not path.is_file():
            return None
    except OSError:
        return None
    return path


# -- merged PRs ------------------------------------------------------------------

def sync_merged_prs(cfg: Config, state: State, git=gitops, token: str | None = None) -> None:
    """Close the loop on earlier PRs: a merge copies the merged skills into their Cursor copies."""
    pending = [pr for pr in state.open_prs if pr.get("status") == "open"]
    if not pending:
        return
    if token is None:
        try:
            token = git.gh_token()
        except GitError as exc:
            state.event("git_failed", step="gh_token", error=error_text(exc))
            return
    for pr in pending:
        try:
            _sync_one(cfg, state, git, token, pr)
        except Exception as exc:  # one bad record must not stop the others or the run
            state.event("git_failed", step="pr_state", url=str(pr.get("url")), error=error_text(exc))


def _sync_one(cfg: Config, state: State, git, token: str, pr: dict) -> None:
    repo = cfg.repos.get(pr.get("repo"))
    if repo is None:
        state.event("git_failed", step="pr_state", url=pr.get("url"), error="unknown repo")
        return
    try:
        pr_status = git.pr_state(repo, pr["url"], token)
    except GitError as exc:
        state.event("git_failed", step="pr_state", url=pr["url"], error=error_text(exc))
        return
    if pr_status == "CLOSED":
        pr["status"] = "closed"
        state.event("pr_closed", repo=pr["repo"], url=pr["url"])
    elif pr_status == "MERGED" and _copy_merged(state, git, repo, pr):
        pr["status"] = "merged"


def _copy_merged(state: State, git, repo: Path, pr: dict) -> bool:
    """Copy every merged skill into its Cursor copy; False keeps the PR open for a retry.

    A copy is written only when it is a plain file reached without a symlink and its
    bytes are still the ones recorded when the PR was opened (or last synced): a
    locally edited copy is the user's and is never overwritten.
    """
    done = True
    for pair in pr.get("cursor_sync", []):
        dst = Path(pair["dst"])
        if not dst.exists() and not dst.is_symlink():
            continue  # the user removed the copy: not ours to recreate
        current = _plain_file_bytes(dst)
        if current is None:
            state.event("cursor_copy_diverged", url=pr["url"], dst=str(dst),
                        reason="the copy is not a plain file or its path goes through a symlink")
            continue
        if pair.get("sha256") != _sha(current):
            state.event("cursor_copy_diverged", url=pr["url"], dst=str(dst),
                        reason="the copy changed after the PR was opened; left as it is")
            continue
        try:
            text = git.file_at_origin_main(repo, pair["rel"])
            write_atomic(dst, text)
        except GitError as exc:
            state.event("git_failed", step="file_at_origin_main", url=pr["url"], error=error_text(exc))
            done = False
            continue
        except (OSError, UnicodeEncodeError) as exc:
            state.event("edit_failed", target=str(dst), error=error_text(exc))
            done = False
            continue
        pair["sha256"] = _sha(text.encode("utf-8"))
    return done


# -- prompts ---------------------------------------------------------------------

def build_plan_prompt(cfg: Config, auto: list[Lesson], eligible_ids: set[str],
                      user_texts: dict[str, str], user_files: dict[str, list[dict]],
                      targets: list[Target]) -> str:
    lessons = (
        f"{lesson.id} | проект {lesson.project_root} | {lesson.status} | {lesson.confidence} | "
        f"повторов: {lesson.repeats} | к переносу: {'да' if lesson.id in eligible_ids else 'нет'} | "
        f"Когда: {lesson.cue} → {lesson.lesson}"
        for lesson in auto
    )
    rules = "\n\n".join(
        f"файл: {path} | повторов: {len(user_files[path])}\n{_TAGS.defuse(text)}\n"
        f"опоры: {_TAGS.defuse(squash('; '.join(e.get('text', '') for e in user_files[path])))}"
        for path, text in user_texts.items()
    )
    return load_template("consolidate_plan.md").substitute(
        nonce=nonce(),
        auto_lessons=_TAGS.one_per_line(lessons),
        user_rules=rules or "—",
        targets=_TAGS.one_per_line(_target_line(cfg, t) for t in targets),
    )


def build_edit_prompt(target: Target, text: str, lessons: list[Lesson],
                      user_texts: list[tuple[str, str]], reasons: list[str]) -> str:
    lesson_block = "\n".join(
        f"- [{lesson.id}] Когда: {_TAGS.defuse(squash(lesson.cue))}\n  Делать: {_TAGS.defuse(squash(lesson.lesson))}\n"
        f"  Почему: {_TAGS.defuse(squash(lesson.why))}\n  Повторов: {lesson.repeats}"
        for lesson in lessons
    )
    reason = _TAGS.defuse(squash("; ".join(r for r in reasons if r)))
    if reason:
        lesson_block = f"Причина переноса: {reason}\n\n{lesson_block}".rstrip()
    rules = "\n\n".join(f"### {Path(path).name}\n{_TAGS.defuse(body)}" for path, body in user_texts)
    return load_template("consolidate_edit.md").substitute(
        nonce=nonce(),
        target=target.key,
        text=_TAGS.defuse(text),
        lessons=lesson_block or "—",
        user_rules=rules or "—",
    )


def _target_line(cfg: Config, target: Target) -> str:
    line = f"{target.key} — {squash(target.description)[:TEXT_MAX]}"
    if target.repo is not None:
        line += f" (репозиторий {cfg.repos[target.repo]})"
    return line


# -- model output ----------------------------------------------------------------

def _line(value) -> str:
    """A model-written one-liner, safe to store, show in the digest and put in a PR."""
    return squash(model_text(value))


def _strings(value) -> list[str]:
    return [model_text(v) for v in value if isinstance(v, str)] if isinstance(value, list) else []


def _verbatim(value) -> str | None:
    """old_string/new_string exactly as given, or None when not a string or not valid UTF-8."""
    if not isinstance(value, str):
        return None
    try:
        value.encode("utf-8")
    except UnicodeEncodeError:  # a lone surrogate cannot be written into any file
        return None
    return value


def _english(text: str, fallback: str) -> str:
    return text if text and not _CYRILLIC.search(text) else fallback


def _redactions(text: str) -> int:
    return scrub(text).count(REDACTED) - text.count(REDACTED)


def _plural(n: int, word: str) -> str:
    return f"{n} {word}" if n == 1 else f"{n} {word}s"


@dataclass
class _Promotion:
    target: Target
    lessons: list[Lesson] = field(default_factory=list)
    files: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class _Edit:
    old: str
    new: str
    summary: str  # English, for the PR and the digest
    why: str  # English, for the PR
    diff: str = ""  # the target before → after, for the digest


class _Run:
    """One consolidate pass; collects what the summary reports."""

    def __init__(self, cfg: Config, state: State, now: datetime, llm, git):
        self.cfg, self.state, self.llm, self.git = cfg, state, llm, git
        self.now = now
        self.now_iso = now.strftime(ISO)
        self.date = now.strftime("%Y-%m-%d")
        self.run_id = now.strftime(RUN_ID_FORMAT)
        since = state.last_consolidate_at or state.installed_at or ""
        self.since = since
        self.new_lessons = [lesson.id for lesson in state.lessons.values() if lesson.created_at >= since]
        self.touched: set[tuple[str, str]] = set()
        self.promoted: list[dict] = []
        self.prs: list[str] = []
        self.removed: list[dict] = []
        self.suggestions: list[dict] = []
        self.eligible_ids: set[str] = set()
        self.user_files: dict[str, list[dict]] = {}
        self.user_texts: dict[str, str] = {}
        self.plan_error = ""
        self._token: str | None = None
        self.removal_cap = 0  # set by plan()
        self.retired = 0
        self.memory_changes = 0  # journaled file changes the archive of this run can undo
        self.upstream: list[dict] = []  # test-writing rules worth carrying into dev-process-toolkit
        self.memory_reads: dict | None = None  # how often the agent opened each memory and notes file
        self.clusters: list[dict] = []  # the same correction across sessions: lessons written or confirmed

    # -- summary ---------------------------------------------------------------

    def regenerate(self) -> None:
        for project_root, memory_dir in sorted(self.touched):
            cursor_index.sync(self.cfg, Path(project_root), Path(memory_dir))

    def aborted(self, reason: str) -> dict:
        """The run stopped before its work; its events stay for the next digest."""
        self.regenerate()
        self.close_archive()
        return {"aborted": scrub(reason)[:ERROR_MAX]}

    def upkeep(self) -> dict:
        """The lessons worth a look and the rules broken since the last run: a key only when it has items."""
        out: dict = {}
        try:
            if items := care.lessons(self.state, self.now):
                out["care"] = items
            if rules := care.broken_rules(journal.load(self.cfg), parse_ts(self.since)):
                out["broken_rules"] = rules
        except Exception as exc:  # advice for the digest must not cost the digest
            self.state.event("care_failed", error=error_text(exc))
        return out

    def summary(self) -> dict:
        self.regenerate()
        self.close_archive()
        upkeep = self.upkeep()
        events = self.state.events
        conflicts = [{"file": e.get("file", ""), "suggestion": e.get("notes", "")}
                     for e in events if e.get("kind") == "user_rule_conflict"]
        return {
            "new_lessons": self.new_lessons,
            # events pile up until a run finishes: every lesson broken while live since the last one
            "violated": list(dict.fromkeys(e.get("lesson_id") for e in events
                                           if e.get("kind") == "lesson_violated" and e.get("lesson_id"))),
            "promoted": self.promoted,
            "prs": self.prs,
            "removed": self.removed,
            "suggestions": self.suggestions + conflicts,
            "upstream": self.upstream,
            "errors": [dict(e) for e in events if e.get("kind") in ERROR_KINDS],
            "advisories": [dict(e) for e in events if e.get("kind") in ADVISORY_KINDS],
            # events pile up until a run finishes, so this counts since the last consolidate
            "skipped_no_project": sum(1 for e in events if e.get("kind") == "no_project"),
            "health": self.health_since(),
            "usage": usage.stats(self.state, self.now),
            **({"memory_reads": self.memory_reads} if self.memory_reads is not None else {}),
            **upkeep,
            **({"clusters": self.clusters} if self.clusters else {}),
        }

    def health_since(self) -> list[dict]:
        """Silent-failure keys seen since the last consolidate, scrubbed; the state keeps them all."""
        entries = health.entries_since(self.state.health, health.parse_ts(self.since), self.cfg.health_ignore)
        return [{k: scrub(v) if isinstance(v, str) else v for k, v in entry.items()} for entry in entries]

    # -- archive and journal -----------------------------------------------------

    def archive_memory(self) -> None:
        """Archive every memory dir this run may change, before it changes any. Raises OSError."""
        for project_root, memory_dir in managed_projects(self.state):
            if Path(memory_dir).is_dir():
                lessons = [lesson for lesson in self.state.lessons.values() if lesson.memory_dir == memory_dir]
                skipped = memory_archive.archive(self.cfg, self.run_id, self.now_iso, project_root, memory_dir,
                                                 lessons)
                for path, error in skipped:  # an unreadable file is not ours to change, so the run goes on
                    self.state.event("archive_file_skipped", file=path, error=error)

    def close_archive(self) -> None:
        """Drop this run's archive when it changed no memory, then keep the newest few. Never raises."""
        if not self.memory_changes:
            memory_archive.discard(self.cfg, self.run_id)
        memory_archive.prune(self.cfg)

    def statuses(self) -> dict[str, str]:
        return {lesson_id: lesson.status for lesson_id, lesson in self.state.lessons.items()}

    def journal(self, kind: str, lesson: Lesson, previous_status: str, **fields) -> None:
        """One append-only journal line per memory mutation. Never raises."""
        if kind in memory_archive.RESTORABLE_KINDS:  # rollback only undoes file changes
            self.memory_changes += 1
        entry = {"ts": self.now_iso, "run_id": self.run_id, "kind": kind, "lesson_id": lesson.id,
                 "project_root": lesson.project_root, "previous_status": previous_status,
                 "status": lesson.status, "file": lesson.file, **fields}
        try:
            memory_archive.append_journal(self.cfg, entry)
        except (OSError, ValueError, UnicodeEncodeError) as exc:
            self.state.event("memory_write_failed", step="journal", lesson_id=lesson.id, error=error_text(exc))

    def journal_changes(self, before: dict[str, str], kinds: dict[tuple[str, str], str]) -> None:
        """Journal the status changes since `before` that `kinds` names."""
        for lesson_id, lesson in self.state.lessons.items():
            kind = kinds.get((before.get(lesson_id, ""), lesson.status))
            if kind is not None:
                self.journal(kind, lesson, before[lesson_id])

    def invalid(self, item: str, reason: str) -> None:
        self.state.event("plan_invalid", item=item[:TEXT_MAX], reason=scrub(reason)[:ERROR_MAX])

    # -- plan ------------------------------------------------------------------

    def plan(self, targets: list[Target]) -> dict | None:
        """The model's plan, {} when there is nothing to plan, None when the model failed."""
        lessons, self.user_files = eligible(self.state, self.now)
        for path in self.user_files:
            try:
                text = Path(path).read_bytes().decode("utf-8")
            except (OSError, UnicodeDecodeError):
                continue  # the user removed or broke the entry: nothing to promote from it
            self.user_texts[path] = scrub(text)[:USER_TEXT_MAX]
        self.eligible_ids = {lesson.id for lesson in lessons}
        auto = [lesson for lesson in self.state.lessons.values() if lesson.status in LIVE]
        self.removal_cap = max(1, -(-len(auto) * REMOVAL_CAP_PERCENT // 100))
        if not auto and not self.user_texts:
            return {}
        prompt = build_plan_prompt(self.cfg, auto, self.eligible_ids, self.user_texts,
                                   self.user_files, targets)
        try:
            plan = self.llm(self.cfg, self.cfg.consolidate_model, prompt, PLAN_SCHEMA,
                            effort=self.cfg.consolidate_effort)
        except LlmError as exc:
            self.plan_error = error_text(exc)
            self.state.event("llm_error", step="plan", error=self.plan_error)
            return None
        if not isinstance(plan, dict):
            self.invalid("plan", "the plan is not an object")
            return {}
        return plan

    def items(self, plan: dict, name: str) -> list[dict]:
        value = plan.get(name, [])
        if not isinstance(value, list):
            self.invalid(name, "not a list")
            return []
        good = [item for item in value if isinstance(item, dict)]
        if len(good) != len(value):
            self.invalid(name, "an item is not an object")
        return good

    def within_cap(self, count: int) -> bool:
        return self.retired + count <= self.removal_cap

    def live_lesson(self, lesson_id: str) -> Lesson | None:
        lesson = self.state.lessons.get(lesson_id)
        return lesson if lesson is not None and lesson.status in LIVE else None

    def suggestions_from(self, plan: dict) -> None:
        for item in self.items(plan, "user_entry_suggestions"):
            path, text = _line(item.get("file")), _line(item.get("suggestion"))
            if not path or not text:
                self.invalid(f"suggestion {path}", "empty file or suggestion")
            elif path not in self.user_files:
                self.invalid(f"suggestion {path}", "not a user entry with a recorded repeat")
            else:
                self.suggestions.append({"file": path, "suggestion": text[:ERROR_MAX * 2]})

    # -- memory cleanup --------------------------------------------------------

    def merge(self, item: dict) -> None:
        keep_id = model_text(item.get("keep"))
        keep = self.live_lesson(keep_id)
        if keep is None:
            self.invalid(f"merge keep {keep_id}", "not a live or staged auto-lesson")
            return
        absorbed: list[Lesson] = []
        for lesson_id in _strings(item.get("absorb")):
            lesson = self.live_lesson(lesson_id)
            if lesson is None or lesson.id == keep.id or lesson in absorbed:
                self.invalid(f"merge absorb {lesson_id}", "not another live or staged auto-lesson")
            elif (lesson.project_root, lesson.memory_dir) != (keep.project_root, keep.memory_dir):
                self.invalid(f"merge absorb {lesson_id}", "lessons of different projects")
            else:
                absorbed.append(lesson)
        if not absorbed:
            return
        memory_dir = Path(keep.memory_dir)
        if not _unchanged(keep):
            self.state.event("memory_write_failed", lesson_id=keep.id, file=keep.file,
                             error="the kept lesson changed on disk; merge skipped")
            return
        if not self.within_cap(len(absorbed)):
            self.invalid(f"merge keep {keep.id}", "removal_cap")
            return
        reason = _line(item.get("reason"))
        before = {lesson.id: lesson.status for lesson in (keep, *absorbed)}
        gone = [lesson for lesson in absorbed if self.delete(lesson, "removed")]
        self.retired += len(gone)
        for lesson in gone:
            self.journal("merge", lesson, before[lesson.id], into=keep.id, reason=reason[:TEXT_MAX])
        if not gone:
            return
        group = (keep, *gone)
        merged = replace(keep, repeats=keep.repeats + sum(lesson.repeats for lesson in gone),
                         evidence=[*keep.evidence, *(e for lesson in gone for e in lesson.evidence)],
                         violations=sum(lesson.violations for lesson in group),
                         last_violated_at=max((x.last_violated_at for x in group if x.last_violated_at), default=None),
                         reads=max(lesson.reads for lesson in group),  # one file read is one read, wherever
                         last_read_at=max((x.last_read_at for x in group if x.last_read_at), default=None),
                         updated_at=self.now_iso)
        try:
            merged.sha256 = write_lesson(memory_dir, merged)
            self.state.lessons[keep.id] = merged
            if merged.status == "staged":
                go_live_if_room(self.cfg, merged, self.now_iso, self.state)
        except _WRITE_ERRORS as exc:
            self.state.event("memory_write_failed", lesson_id=keep.id, file=keep.file, error=error_text(exc))
        if self.state.lessons[keep.id] is merged:
            self.journal("merge", merged, before[keep.id], absorbed=[lesson.id for lesson in gone],
                         reason=reason[:TEXT_MAX])
        for lesson in gone:
            self.removed.append({"file": str(Path(lesson.memory_dir) / lesson.file),
                                 "reason": (i18n.t("merged into ", "склеен с ") + f"{keep.file}: {reason}").rstrip(": ")})

    def remove(self, item: dict) -> None:
        lesson_id = model_text(item.get("lesson_id"))
        lesson = self.live_lesson(lesson_id)
        if lesson is None:
            self.invalid(f"removal {lesson_id}", "not a live or staged auto-lesson")
            return
        if not self.within_cap(1):
            self.invalid(f"removal {lesson_id}", "removal_cap")
            return
        previous = lesson.status
        if self.delete(lesson, "removed"):
            self.retired += 1
            reason = _line(item.get("reason"))
            self.journal("remove", lesson, previous, reason=reason[:TEXT_MAX])
            self.removed.append({"file": str(Path(lesson.memory_dir) / lesson.file), "reason": reason})

    def delete(self, lesson: Lesson, status: str) -> bool:
        try:
            remove_lesson(Path(lesson.memory_dir), lesson)
        except Exception as exc:  # refused or failed: the file stays, and so does the record
            self.state.event("memory_write_failed", lesson_id=lesson.id, file=lesson.file,
                             error=error_text(exc))
            return False
        lesson.status, lesson.updated_at = status, self.now_iso
        self.touched.add((lesson.project_root, lesson.memory_dir))
        return True

    # -- promotions ------------------------------------------------------------

    def promotions(self, plan: dict, targets: list[Target]) -> list[_Promotion]:
        by_key = {t.key: t for t in targets}
        chosen: dict[str, _Promotion] = {}
        used_lessons: set[str] = set()
        used_files: set[str] = set()
        for item in self.items(plan, "promotions"):
            key = model_text(item.get("target"))
            target = by_key.get(key)
            if target is None:
                self.invalid(f"promotion target {key}", "not a known target")
                continue
            if any(not _has_session_evidence(self.state.lessons[lesson_id])
                   for lesson_id in _strings(item.get("lesson_ids")) if lesson_id in self.state.lessons):
                self.invalid(f"promotion {target.key}", "no_evidence")
                continue
            promo = chosen.setdefault(target.key, _Promotion(target))
            for lesson_id in _strings(item.get("lesson_ids")):
                lesson = self.live_lesson(lesson_id)
                if lesson is None or lesson_id not in self.eligible_ids:
                    self.invalid(f"promotion lesson {lesson_id}", "not an eligible auto-lesson")
                elif lesson_id in used_lessons:
                    self.invalid(f"promotion lesson {lesson_id}", "already promoted into another target")
                elif _is_rules(target) and lesson.project_root != str(self.cfg.repos[target.repo]):
                    self.invalid(f"promotion lesson {lesson_id}", f"not a lesson of {target.repo}")
                else:
                    used_lessons.add(lesson_id)
                    promo.lessons.append(lesson)
            for path in _strings(item.get("user_rule_files")):
                if path not in self.user_texts:
                    self.invalid(f"promotion user file {path}", "no recorded repeat")
                elif path in used_files:
                    self.invalid(f"promotion user file {path}", "already promoted into another target")
                elif _is_rules(target) and Path(path).parent != memory_dir_for(
                        self.cfg.projects_dir, self.cfg.repos[target.repo]):
                    self.invalid(f"promotion user file {path}", f"not a memory entry of {target.repo}")
                else:
                    used_files.add(path)
                    promo.files.append(path)
            promo.reasons.append(_line(item.get("reason")))
        return [p for p in chosen.values() if p.lessons or p.files]

    def ask_edit(self, promo: _Promotion, text: str) -> _Edit | None:
        """The model's edit for one target, or None with an event recorded."""
        key = promo.target.key
        prompt = build_edit_prompt(promo.target, text, promo.lessons,
                                   [(f, self.user_texts[f]) for f in promo.files], promo.reasons)
        try:
            edit = self.llm(self.cfg, self.cfg.consolidate_model, prompt, EDIT_SCHEMA,
                            effort=self.cfg.consolidate_effort)
        except LlmError as exc:
            self.state.event("llm_error", step="edit", target=key, error=error_text(exc))
            return None
        if not isinstance(edit, dict):
            self.state.event("edit_failed", target=key, error="the edit is not an object")
            return None
        if edit.get("skip") is True:
            self.state.event("edit_skipped", target=key, reason=_line(edit.get("reason"))[:ERROR_MAX])
            return None
        old, new = _verbatim(edit.get("old_string")), _verbatim(edit.get("new_string"))
        if old is None or new is None:
            self.state.event("edit_failed", target=key, error="old_string/new_string is not valid text")
            return None
        if not old or old == new:
            self.state.event("edit_failed", target=key, error="the edit has no change")
            return None
        too_big = size_error(old, new)
        if too_big is not None:
            self.state.event("edit_failed", target=key, error=too_big)
            return None
        incident = incident_error(old, new, [lesson.id for lesson in promo.lessons])
        if incident is not None:
            self.state.event("edit_failed", target=key, reason=INCIDENT_LOG, error=incident[:ERROR_MAX])
            return None
        if _redactions(new) > _redactions(old):
            self.state.event("edit_failed", target=key, error="the edit adds secret-like text")
            return None
        why = _line(edit.get("why_en"))
        if not why:  # an edit that cannot say what went wrong has nothing to stand on
            self.invalid(f"edit {key}", "no_evidence")
            return None
        return _Edit(old, new,
                     summary=_english(_line(edit.get("summary"))[:TEXT_MAX], FALLBACK_SUMMARY),
                     why=_english(why[:TEXT_MAX], FALLBACK_WHY))

    def retire(self, promo: _Promotion, edit: _Edit, pr_url: str | None, notes: list[str]) -> None:
        for lesson in promo.lessons:
            previous = lesson.status
            if self.delete(lesson, "promoted"):
                self.journal("retire", lesson, previous, target=promo.target.key, pr_url=pr_url)
            else:
                # still in memory and no longer ours to track: the user decides what to do with it
                lesson.status = "user_owned"
        for path in promo.files:
            self.state.user_rule_repeats.pop(path, None)
        self.promoted.append({
            "target": promo.target.key,
            "summary": edit.summary,
            "lesson_ids": [lesson.id for lesson in promo.lessons],
            "user_rule_files": list(promo.files),
            "backup_run_id": None if pr_url else self.run_id,
            "pr_url": pr_url,
            "notes": notes,
            "diff": edit.diff,
        })

    def promote_personal(self, promo: _Promotion) -> None:
        target = promo.target
        try:
            text = target.path.read_bytes().decode("utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            self.state.event("edit_failed", target=target.key, error=error_text(exc))
            return
        edit = self.ask_edit(promo, text)
        if edit is None:
            return
        try:
            notes = apply_personal(self.cfg, self.run_id, target, edit.old, edit.new)
        except (EditError, OSError, UnicodeError) as exc:
            self.state.event("edit_failed", target=target.key, error=error_text(exc))
            return
        after = text.replace(edit.old, edit.new, 1)  # apply_personal checked it occurs exactly once
        self.check_size(target.key, after)
        edit = replace(edit, diff=unified_diff(text, after, f"{target.path.parent.name}/{target.path.name}"))
        self.retire(promo, edit, None, [scrub(note)[:ERROR_MAX] for note in notes])

    def promote_repo(self, name: str, group: list[_Promotion]) -> None:
        """All of one repo's edits: one worktree from origin/main, one commit, one draft PR."""
        repo = self.cfg.repos[name]
        branch = f"docs/agent-reflect-{self.date}"
        dest = self.cfg.worktrees_dir / f"{name}-{self.date}"
        try:
            self.cfg.worktrees_dir.mkdir(parents=True, exist_ok=True)
            self.git.prepare_worktree(repo, branch, dest)
        except (GitError, OSError) as exc:
            self.state.event("git_failed", repo=name, step="prepare_worktree", error=error_text(exc))
            return
        applied: list[tuple[_Promotion, _Edit]] = []
        url: str | None = None
        try:
            for promo in group:
                edit = self.edit_in_worktree(dest, promo)
                if edit is not None:
                    applied.append((promo, edit))
            if not applied:
                return
            title = f"{COMMIT_SUBJECT} {self.date}"
            body = self.pr_body(applied)
            token = self.token()
            self.git.commit_and_push(dest, [promo.target.rel_path for promo, _ in applied],
                                     scrub(f"{title}\n\n{CO_AUTHORED_BY}"), branch)
            url = self.git.create_draft_pr(dest, branch, title, body, token)
            self.record_pr(name, url, branch, applied)  # first, so nothing later can lose it
        except GitError as exc:
            self.state.event("git_failed", repo=name, step="publish", error=error_text(exc))
        finally:
            try:
                self.git.remove_worktree(repo, dest)
            except Exception as exc:
                self.state.event("git_failed", repo=name, step="remove_worktree", error=error_text(exc))
        if url is None:
            return
        for promo, edit in applied:
            self.retire(promo, edit, url, [])

    def record_pr(self, name: str, url: str, branch: str, applied: list[tuple[_Promotion, _Edit]]) -> None:
        """Remember a created PR. Never raises."""
        self.prs.append(url)
        pairs = []
        for promo, _ in applied:
            copy = promo.target.cursor_copy
            if copy is not None:
                current = _plain_file_bytes(copy)
                pairs.append({"rel": promo.target.rel_path, "dst": str(copy),
                              "sha256": None if current is None else _sha(current)})
        self.state.open_prs.append({"repo": name, "url": url, "branch": branch, "status": "open",
                                    "created_at": self.now_iso, "cursor_sync": pairs})

    def edit_in_worktree(self, dest: Path, promo: _Promotion) -> _Edit | None:
        rel = promo.target.rel_path
        path = _inside_worktree(dest, rel)
        if path is None:
            self.state.event("edit_failed", target=promo.target.key,
                             error="the file is missing in the worktree or reached through a symlink")
            return None
        try:
            text = path.read_bytes().decode("utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            self.state.event("edit_failed", target=promo.target.key, error=error_text(exc))
            return None
        edit = self.ask_edit(promo, text)
        if edit is None:
            return None
        try:
            after = apply_edit_text(text, edit.old, edit.new, skill=rel.endswith("SKILL.md"))
            write_atomic(path, after)
        except (EditError, OSError) as exc:
            self.state.event("edit_failed", target=promo.target.key, error=error_text(exc))
            return None
        self.check_size(promo.target.key, after)
        return replace(edit, diff=unified_diff(text, after, rel))

    def check_size(self, key: str, after: str) -> None:
        """An advisory, never a refusal: a skill this long is due for a human's trim."""
        if len(after) > SKILL_LARGE_CHARS:
            self.state.event("skill_large", target=key, chars=len(after), limit=SKILL_LARGE_CHARS)

    def token(self) -> str:
        if self._token is None:
            self._token = self.git.gh_token()
        return self._token

    def sessions_of(self, promo: _Promotion) -> set[str]:
        evidence = [e for lesson in promo.lessons for e in lesson.evidence]
        evidence += [e for path in promo.files for e in self.user_files.get(path, [])]
        return {e["session"] for e in evidence if isinstance(e.get("session"), str)}

    def pr_body(self, applied: list[tuple[_Promotion, _Edit]]) -> str:
        """The PR description, in English only: lesson texts stay in the digest."""
        what, why, sessions = [], [], set()
        for promo, edit in applied:
            key = promo.target.key
            own = self.sessions_of(promo)
            sessions |= own
            counts = [_plural(len(promo.lessons), "lesson")] if promo.lessons else []
            if promo.files:
                counts.append(_plural(len(promo.files), "user memory rule"))
            what.append(f"- `{key}` — {edit.summary}")
            why.append(f"- `{key}` — {edit.why} ({' and '.join(counts)} from {_plural(len(own), 'session')})")
        body = (
            "## What\n\n" + "\n".join(what) + "\n\n"
            "## Why\n\n" + "\n".join(why) + "\n\n"
            "## How verified\n\n"
            "Each edit is a single exact replacement; SKILL.md frontmatter re-validated; "
            f"evidence from {_plural(len(sessions), 'session')}.\n\n"
            f"{PR_FOOTER}\n"
        )
        return scrub(body)


def _is_rules(target: Target) -> bool:
    return target.kind == "repo" and target.rel_path == "CLAUDE.md"


def _has_session_evidence(lesson: Lesson) -> bool:
    return any(isinstance(e, dict) and isinstance(e.get("session"), str) and e["session"]
               for e in lesson.evidence)


def _unchanged(lesson: Lesson) -> bool:
    try:
        raw = (Path(lesson.memory_dir) / lesson.file).read_bytes()
    except OSError:
        return False
    return _sha(raw) == lesson.sha256


def run(cfg: Config, state: State, *, now: datetime, llm=run_structured, git=gitops,
        force: bool = False, dry_run: bool = False, memory_scan=memory_reads.scan) -> dict:
    """One weekly pass. The caller saves `state`.

    {} means it was not due; {"aborted": reason} means it stopped before its work
    (the caller counts that as a failure); anything else is the summary for the digest.
    """
    if not force and not is_due(state, now, cfg.consolidate_interval_days):
        return {}
    if dry_run:
        lessons, files = eligible(state, now)
        return {"dry_run": True, "eligible": [lesson.id for lesson in lessons],
                "user_rule_files": sorted(files), "targets": [t.key for t in registry(cfg)]}

    ctx = _Run(cfg, state, now, llm, git)
    if not autodream_disabled(cfg.claude_settings):
        state.event("autodream_not_disabled", settings=str(cfg.claude_settings))
        return ctx.aborted(i18n.t("autoDreamEnabled is not off in the Claude Code settings, memory left alone",
                             "autoDreamEnabled не выключен в настройках Claude Code, память не тронута"))
    try:
        before = ctx.statuses()
        ctx.touched.update(sync_ownership(state))
        ctx.journal_changes(before, {("live", "rejected"): "reject", ("staged", "rejected"): "reject"})
        ctx.archive_memory()  # nothing of ours has changed memory yet; a failure stops the run here
        before = ctx.statuses()
        for project_root, memory_dir in managed_projects(state):
            if rebalance(state, cfg, project_root, memory_dir, ctx.now_iso):
                ctx.touched.add((project_root, memory_dir))
        ctx.journal_changes(before, {("live", "staged"): "demote", ("staged", "live"): "promote"})
        sync_merged_prs(cfg, state, git=git)
        targets = registry(cfg)
        plan = ctx.plan(targets)
    except Exception as exc:  # nothing promoted yet: stop, and retry on the next tick
        state.event("prepare_failed", error=error_text(exc))
        return ctx.aborted(i18n.t("preparing failed: ", "подготовка упала: ") + error_text(exc))
    if plan is None:
        return ctx.aborted(i18n.t("no plan came back: ", "план не получен: ") + str(ctx.plan_error))

    for item in ctx.items(plan, "merges"):
        ctx.merge(item)
    for item in ctx.items(plan, "removals"):
        ctx.remove(item)
    by_repo: dict[str, list[_Promotion]] = {}
    for promo in ctx.promotions(plan, targets):
        if promo.target.kind != "personal":
            by_repo.setdefault(promo.target.repo, []).append(promo)
            continue
        try:
            ctx.promote_personal(promo)
        except Exception as exc:  # one target's surprise must not stop the others
            state.event("edit_failed", target=promo.target.key, error=error_text(exc))
    for name in sorted(by_repo):
        try:
            ctx.promote_repo(name, by_repo[name])
        except Exception as exc:
            state.event("git_failed", repo=name, step="unexpected", error=error_text(exc))
    ctx.suggestions_from(plan)
    try:
        ctx.upstream = upstream.step(cfg, state, now=now, llm=llm, git=git)
    except Exception as exc:  # a scan surprise must not cost the week's promotions
        state.event("upstream_failed", error=error_text(exc))
    try:
        ctx.memory_reads = memory_scan(cfg, now=now)
    except Exception as exc:  # a count of file reads must not cost the week's digest
        state.event("memory_reads_failed", error=error_text(exc))
    try:
        ctx.clusters = clusters.step(cfg, state, now=now, llm=llm, records=journal.load(cfg))
    except Exception as exc:  # the look across sessions must not cost the week's digest
        state.event("clusters_failed", error=error_text(exc))

    state.last_consolidate_at = ctx.now_iso
    summary = ctx.summary()
    state.events = []
    return summary
