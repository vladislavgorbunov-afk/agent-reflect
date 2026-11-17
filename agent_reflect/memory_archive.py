"""Reversible memory mutations: per-run archives of the auto files, an append-only journal, rollback."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path

from agent_reflect import skills
from agent_reflect.config import RUN_ID_FORMAT, Config, is_run_id
from agent_reflect.ledger import Lesson, State, parse_ts
from agent_reflect.memory import (
    _AUTO_FILE_RE,
    AUTO_PREFIX,
    MemoryIndexError,
    _atomic_write,
    _has_entry,
    add_index_line,
    has_capacity,
    index_has_room,
    is_user_entry,
    read_index,
)
from agent_reflect.projects import sanitize
from agent_reflect.scrub import scrub

KEEP_ARCHIVES = 4
INDEX_COPY = "MEMORY.auto.md"
MANIFEST = "manifest.json"
JOURNAL_KINDS = ("merge", "remove", "retire", "demote", "promote", "reject")
# the changes a rollback can undo: they rewrite or delete an auto file
RESTORABLE_KINDS = ("merge", "remove", "retire")
# a lesson in one of these states is the user's decision: rollback never brings it back
_USERS = ("rejected", "user_owned")


def archives_dir(cfg: Config) -> Path:
    return cfg.backups_dir / "memory"


def journal_file(cfg: Config) -> Path:
    return cfg.state_dir / "journal.jsonl"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_json(path: Path, data) -> None:
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


# -- archive -----------------------------------------------------------------------

def archive(cfg: Config, run_id: str, now_iso: str, project_root: str, memory_dir: str,
            lessons: list[Lesson]) -> list[tuple[str, str]]:
    """Copy memory_dir's auto_*.md files and MEMORY.md's auto lines into the run's archive.

    Plain files under backups/memory/<run_id>/<slug>/ plus a manifest with the lessons'
    records as they were. A file that cannot be read is left out and returned as
    (path, error); OSError means the archive itself cannot be written.
    """
    memory_dir_path = Path(memory_dir)
    dest = archives_dir(cfg) / run_id / memory_dir_path.parent.name
    dest.mkdir(parents=True, exist_ok=True)
    files, skipped = [], []
    for path in sorted(memory_dir_path.glob("auto_*.md")):
        if not _AUTO_FILE_RE.fullmatch(path.name) or path.is_symlink() or not path.is_file():
            continue
        try:
            raw = path.read_bytes()
        except OSError as exc:
            skipped.append((str(path), scrub(str(exc))[:300]))
            continue
        (dest / path.name).write_bytes(raw)
        files.append({"name": path.name, "sha256": _sha(raw)})
    try:
        index = read_index(memory_dir_path)
        lines = [line.rstrip("\r") for line in index.split("\n") if line.startswith(AUTO_PREFIX)]
    except MemoryIndexError:
        lines = None  # unreadable: nothing of the index is archived, so rollback re-adds no line
    (dest / INDEX_COPY).write_text("".join(f"{line}\n" for line in lines or []), encoding="utf-8")
    _write_json(dest / MANIFEST, {
        "run_id": run_id,
        "created_at": now_iso,
        "project_root": project_root,
        "memory_dir": memory_dir,
        "files": files,
        "index_readable": lines is not None,
        "lessons": {lesson.id: asdict(lesson) for lesson in lessons},
    })
    return skipped


def list_archives(cfg: Config) -> list[str]:
    try:
        return sorted(p.name for p in archives_dir(cfg).iterdir() if p.is_dir())
    except OSError:
        return []


def has_archive(cfg: Config, run_id: str) -> bool:
    return is_run_id(run_id) and (archives_dir(cfg) / run_id).is_dir()


def discard(cfg: Config, run_id: str) -> None:
    shutil.rmtree(archives_dir(cfg) / run_id, ignore_errors=True)


def prune(cfg: Config, keep: int = KEEP_ARCHIVES) -> None:
    """Keep the `keep` newest memory archives; run ids sort by time."""
    for run_id in list_archives(cfg)[:-keep]:
        discard(cfg, run_id)


# -- journal -----------------------------------------------------------------------

def _scrubbed(value):
    if isinstance(value, str):
        return scrub(value)
    if isinstance(value, list):
        return [_scrubbed(v) for v in value]
    return value


def append_journal(cfg: Config, entry: dict) -> None:
    """Append one line to the journal; the file is never rewritten. Raises OSError."""
    if entry.get("kind") not in JOURNAL_KINDS:
        raise ValueError(f"unknown journal kind {entry.get('kind')!r}")
    path = journal_file(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps({k: _scrubbed(v) for k, v in entry.items()}, ensure_ascii=False)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def run_changes(cfg: Config, run_id: str) -> dict[str, dict] | None:
    """lesson id -> the journal line of each file change the run made; None when the
    journal holds no line of that run at all (lost or never written)."""
    try:
        lines = journal_file(cfg).read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError):
        return None
    seen, changes = False, {}
    for line in lines:
        try:
            entry = json.loads(line)
        except ValueError:
            continue
        if not isinstance(entry, dict) or entry.get("run_id") != run_id:
            continue
        seen = True
        if entry.get("kind") in RESTORABLE_KINDS and isinstance(entry.get("lesson_id"), str):
            changes[entry["lesson_id"]] = entry
    return changes if seen else None


# -- rollback ----------------------------------------------------------------------

@dataclass
class Restore:
    path: Path  # the auto file to write back
    data: bytes
    project_root: str
    index_line: str | None  # the MEMORY.md line to re-add, when there is room for it
    hook: str
    record: Lesson  # the lesson as archived
    previous_status: str | None  # in state now; None when state has no record of it
    status: str  # the status the lesson gets back
    hold: str | None = None  # set for a promoted lesson whose skill edit stays
    notes: tuple[str, ...] = ()  # what the plan output says about it beyond the file


@dataclass
class RollbackPlan:
    run_id: str
    restores: list[Restore]
    skipped: list[str]  # "path: why": left as it is to keep the user's work
    refused: list[str] = field(default_factory=list)  # "path: why": the archive itself is not trusted


def _archived_hooks(folder: Path) -> dict[str, str]:
    try:
        text = (folder / INDEX_COPY).read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return {}
    hooks = {}
    for line in text.splitlines():
        name = line[len("- ["):].split("]", 1)[0]
        prefix = f"- [{name}]({name}) — "
        if _AUTO_FILE_RE.fullmatch(name) and line.startswith(prefix):
            hooks[name] = line[len(prefix):]
    return hooks


def _state_lesson(state: State, memory_dir: str, name: str) -> Lesson | None:
    return next((lesson for lesson in state.lessons.values()
                 if lesson.file == name and lesson.memory_dir == memory_dir), None)


def _indexed(memory_dir: Path, name: str) -> bool:
    try:
        return _has_entry(read_index(memory_dir), name)
    except MemoryIndexError:
        return False


def hold_note(run_id: str, change: dict | None) -> str:
    pr_url = (change or {}).get("pr_url")
    if pr_url:
        return (f"its edit stays in the draft PR {pr_url}; the lesson stays held "
                "until you remove the lesson by hand")
    return f"skill edit stays — run `agent-reflect rollback {run_id}` to undo it"


def _skill_edit_undone(cfg: Config, run_id: str, change: dict | None) -> bool:
    """The run's skill edit for this lesson was already rolled back (personal targets only)."""
    if not change or change.get("pr_url") or not isinstance(change.get("target"), str):
        return False
    return skills.backup_restored(cfg, run_id, change["target"])


class _Planner:
    """One rollback's choices, with MEMORY.md as it will be after the lines it adds."""

    def __init__(self, cfg: Config, state: State, run_id: str):
        self.cfg, self.state, self.run_id = cfg, state, run_id
        self.changes = run_changes(cfg, run_id)
        self.plan = RollbackPlan(run_id, [], [])
        self.indexes: dict[str, str] = {}

    def skip(self, path, why: str) -> None:
        self.plan.skipped.append(f"{path}: {why}")

    def refuse(self, path, why: str) -> None:
        self.plan.refused.append(f"{path}: {why}")

    def manifest(self, manifest_path: Path) -> None:
        folder = manifest_path.parent
        expected = self.cfg.projects_dir / folder.name / "memory"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            memory_dir, project_root = manifest["memory_dir"], manifest["project_root"]
            names = [entry["name"] for entry in manifest["files"]]
            records = {r["file"]: r for r in manifest.get("lessons", {}).values() if isinstance(r, dict)}
        except (OSError, ValueError, KeyError, TypeError) as exc:
            self.refuse(manifest_path, f"unreadable manifest ({exc})")
            return
        # the archive says where to write; only the place our own layout gives it is accepted
        if memory_dir != str(expected) or not isinstance(project_root, str) \
                or sanitize(project_root) != folder.name:
            self.refuse(manifest_path, f"it names {memory_dir}, not {expected}")
            return
        created_at = manifest.get("created_at")
        hooks = _archived_hooks(folder)
        for name in names:
            if isinstance(name, str) and _AUTO_FILE_RE.fullmatch(name) and Path(name).name == name:
                self.one(folder, expected, project_root, name, records.get(name), hooks.get(name), created_at)
            else:
                self.refuse(manifest_path, f"{name!r} is not an auto-lesson file name")

    @staticmethod
    def looks_changed_by_run(lesson: Lesson | None, created_at) -> bool:
        """What state shows of a weekly run's change: the run's own stamp, or a removal."""
        return lesson is not None and (lesson.status in ("removed", "promoted") or lesson.updated_at == created_at)

    def one(self, folder: Path, memory_dir: Path, project_root: str, name: str, record: dict | None,
            hook: str | None, created_at) -> None:
        saved, path = folder / name, memory_dir / name
        try:
            data = saved.read_bytes()
        except OSError:
            self.refuse(path, "archived copy unreadable")
            return
        if saved.is_symlink() or is_user_entry(saved):
            self.refuse(path, "the archived copy is not agent-reflect's")
            return
        if record is None or record.get("status") not in ("live", "staged") or record.get("sha256") != _sha(data):
            return  # not a lesson of ours the run could have changed
        try:
            archived = Lesson(**record)
        except TypeError as exc:
            self.refuse(path, f"archived record unusable ({exc})")
            return
        lesson = _state_lesson(self.state, str(memory_dir), name)
        if self.changes is None:  # no journal for the run: what state shows is all there is
            if not self.looks_changed_by_run(lesson, created_at):
                return
        elif archived.id not in self.changes:
            if self.looks_changed_by_run(lesson, created_at):
                self.skip(path, f"not in the journal of {self.run_id}, left as it is")
            return
        if lesson is not None and lesson.status in _USERS:
            self.skip(path, f"the lesson is {lesson.status}, the user's decision stands")
            return
        if path.exists() or path.is_symlink():
            try:
                current = path.read_bytes() if path.is_file() and not path.is_symlink() else None
            except OSError:
                current = None
            if current == data:
                return  # unchanged since the run
            if current is None or is_user_entry(path) or lesson is None or _sha(current) != lesson.sha256:
                self.skip(path, "changed after the run by someone else, left as it is")
                return
        # a hold is the state's to keep: one lifted since the archive never comes back from it
        restore = Restore(path=path, data=data, project_root=project_root, index_line=None, hook=hook or "",
                          record=archived, previous_status=None if lesson is None else lesson.status,
                          status="staged", hold=None if lesson is None else lesson.rollback_hold)
        if lesson is not None and lesson.status == "promoted":
            # its text went into a skill: back in memory staged, and held while that edit stands
            change = (self.changes or {}).get(archived.id)
            if _skill_edit_undone(self.cfg, self.run_id, change):
                restore.notes = (f"its skill edit in {change['target']} is already undone, no hold",)
            else:
                restore.hold = self.run_id
                restore.notes = (hold_note(self.run_id, change),)
        elif lesson is not None and lesson.status == "live":
            # state still calls it live: its line is the user's to keep or drop, never re-added
            restore.status = "live" if _indexed(memory_dir, name) else "staged"
        elif hook is not None:
            self.line(restore, memory_dir, name, hook)
        self.plan.restores.append(restore)

    def line(self, restore: Restore, memory_dir: Path, name: str, hook: str) -> None:
        key = str(memory_dir)
        if key not in self.indexes:
            try:
                self.indexes[key] = read_index(memory_dir)
            except MemoryIndexError:
                self.skip(memory_dir / "MEMORY.md", f"unreadable, its line for {name} is not re-added")
                return
        text = self.indexes[key]
        if _has_entry(text, name):
            restore.status = "live"
            return
        line = f"- [{name}]({name}) — {hook}"
        if not index_has_room(text, self.cfg, line):
            restore.notes = (f"no room in {memory_dir / 'MEMORY.md'} for its line, it stays staged",)
            return
        restore.index_line, restore.status = line, "live"
        self.indexes[key] = (text if not text or text.endswith("\n") else text + "\n") + line + "\n"


def plan_rollback(cfg: Config, state: State, run_id: str) -> RollbackPlan | None:
    """What a rollback of `run_id` would write; None when there is no such archive.

    Only lessons the run itself merged, removed or retired come back, and never one
    the user rejected or took over since. Call memory.sync_ownership first, so the
    user's deletions after the run count as rejections.
    """
    if not has_archive(cfg, run_id):
        return None
    planner = _Planner(cfg, state, run_id)
    for manifest_path in sorted((archives_dir(cfg) / run_id).glob(f"*/{MANIFEST}")):
        planner.manifest(manifest_path)
    return planner.plan


def apply_rollback(cfg: Config, state: State, plan: RollbackPlan, now_iso: str | None = None) -> list[str]:
    """Write the plan's files and index lines and reset the lessons' records. Returns problems.

    A lesson whose index line goes back in is live from now (`now_iso`, else the clock).
    """
    now_iso = now_iso or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    problems: list[str] = []
    for restore in plan.restores:
        try:
            _atomic_write(restore.path, restore.data)
        except OSError as exc:
            problems.append(f"{restore.path}: {exc}")
            continue
        memory_dir, name = restore.path.parent, restore.path.name
        line_added = False
        if restore.index_line is not None:
            try:
                if has_capacity(memory_dir, cfg, restore.index_line):
                    add_index_line(memory_dir, name, restore.hook)
                    line_added = True
                else:
                    problems.append(f"{memory_dir / 'MEMORY.md'}: no room for {name}, it stays staged")
            except (OSError, MemoryIndexError, UnicodeEncodeError) as exc:
                problems.append(f"{memory_dir / 'MEMORY.md'}: {exc}")
        # its sha256 is the restored file's; live only with its line, or it reads as the user's rejection
        status = "live" if restore.status == "live" and _indexed(memory_dir, name) else "staged"
        record = replace(restore.record, status=status, rollback_hold=restore.hold)
        if status == "live" and line_added:
            record.live_since = now_iso
        state.lessons[restore.record.id] = record
    return problems


def end_rolled_back_moves(cfg: Config, state: State, run_id: str, restored: set[Path], now_iso: str) -> list[str]:
    """After `rollback run_id` restored the skill backup folders in `restored`, stop watching the
    lessons moved into one of them by that run or a later one: the file is back as it was before
    the run, so their rule is gone from it and no verdict on their move could mean anything.
    The recorded moves into those files are marked rolled back too. Returns the lessons' ids.
    """
    targets = {move.get("target") for move in state.moves
               if isinstance(move, dict) and move.get("run_id") == run_id and not move.get("pr_url")}
    targets |= {change.get("target") for change in (run_changes(cfg, run_id) or {}).values()
                if change.get("kind") == "retire" and not change.get("pr_url")}
    targets = {t for t in targets if isinstance(t, str) and skills.backup_folder(cfg, run_id, t) in restored}
    try:
        since = datetime.strptime(run_id, RUN_ID_FORMAT).replace(tzinfo=timezone.utc)
    except ValueError:
        return []
    ended = []
    for lesson in state.lessons.values():
        at = parse_ts(lesson.promoted_at)
        # a restored target is a personal skill: a move into it never went by PR
        if (lesson.status == "promoted" and not lesson.effect and lesson.promoted_to in targets
                and at is not None and at >= since):
            lesson.effect, lesson.effect_at = "rolled_back", now_iso
            ended.append(lesson.id)
    for move in state.moves:
        at = parse_ts(move.get("at")) if isinstance(move, dict) else None
        if at is not None and at >= since and move.get("target") in targets:
            move["rolled_back"] = True
    return ended


def lift_holds(cfg: Config, state: State, run_id: str, restored: set[Path]) -> tuple[list[str], list[str]]:
    """After `rollback run_id` restored the skill backup folders in `restored`, lift the holds
    whose skill edit that undid: a personal retire (no PR) into one of those targets.

    Returns (lifted ids, ids still held). A hold whose edit stands in a PR, or whose retire
    is not in the journal, stays.
    """
    changes = run_changes(cfg, run_id) or {}
    lifted, kept = [], []
    for lesson in state.lessons.values():
        if lesson.rollback_hold != run_id:
            continue
        change = changes.get(lesson.id)
        undone = (change is not None and change.get("kind") == "retire" and not change.get("pr_url")
                  and isinstance(change.get("target"), str)
                  and skills.backup_folder(cfg, run_id, change["target"]) in restored)
        if undone:
            lesson.rollback_hold = None
            lifted.append(lesson.id)
        else:
            kept.append(lesson.id)
    return lifted, kept
