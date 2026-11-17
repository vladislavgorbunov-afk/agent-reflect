from __future__ import annotations

import fcntl
import json
import os
import re
import tempfile
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterator

# "retired": the user took its line out of the index with `retire`; its file stays, and a repeat brings it back
LESSON_STATUSES = ("live", "staged", "promoted", "rejected", "user_owned", "removed", "retired")


@dataclass
class Lesson:
    id: str
    project_root: str
    memory_dir: str
    file: str
    title: str
    hook: str
    cue: str
    lesson: str
    why: str
    confidence: str
    repeats: int
    evidence: list[dict]  # {"session", "ts", "text"}
    status: str
    sha256: str
    created_at: str
    updated_at: str
    # run id of a memory rollback that brought this lesson back while its skill edit stayed:
    # it is not promoted again until `rollback <run id>` takes the skill edit back too
    rollback_hold: str | None = None
    # when its index line last went in: a session that started before it never saw the lesson
    live_since: str | None = None
    # sessions that did what the lesson forbids while it was live, and the last time one did
    violations: int = 0
    last_violated_at: str | None = None
    # times an agent opened the lesson's file (or named it to Grep/Bash), and the last time one did
    reads: int = 0
    last_read_at: str | None = None
    # "tool": how to run something; "process": how to work, read from the user returning the work
    kind: str = "tool"
    # what the user's correction was about (reflect.FORMS), "" for a lesson that came with none
    form: str = ""
    # a regex over a shell command the lesson warns about: the guard hook shows the lesson
    # when it matches, and a failing command that matches is a repeat without asking the model
    trigger: str = ""
    # times the guard hook showed the lesson to an agent, and the last time it did
    fires: int = 0
    last_fired_at: str | None = None
    # a regex over the path of a file: the guard hook shows the lesson when the agent is about
    # to edit such a file, once a session
    path_trigger: str = ""


@dataclass
class State:
    installed_at: str | None = None
    sessions: dict[str, dict] = field(default_factory=dict)  # str(path) -> {"offset": int}
    lessons: dict[str, Lesson] = field(default_factory=dict)
    rejected_cues: dict[str, list[str]] = field(default_factory=dict)  # project_root -> cues
    user_rule_repeats: dict[str, list[dict]] = field(default_factory=dict)  # abs path -> evidence
    last_consolidate_at: str | None = None
    consecutive_failures: int = 0
    consolidate_failures: int = 0
    open_prs: list[dict] = field(default_factory=list)
    events: list[dict] = field(default_factory=list)
    # subagent transcripts were not read before; their history up to the upgrade is a baseline
    subagent_baseline_done: bool = False
    # silent failures (hook crashes, API errors) per key: alerts for the user, never lessons
    health: dict[str, dict] = field(default_factory=dict)
    # str(path) -> byte offset up to which the transcript's health is merged, kept only while
    # it is ahead of the reading offset (a segment left unread for the cap or an LLM error)
    health_offsets: dict[str, int] = field(default_factory=dict)
    # how far the guard hook's fires log has been counted into the lessons
    guard_offset: int = 0
    # rules the user gave the guard hook by hand (`guard add`): {"name", "on": "command" | "path",
    # "pattern", "text", "project": a root or "" for every project, "fires", "last_fired_at"}
    user_guards: list[dict] = field(default_factory=list)
    # corrections the weekly look across sessions already put into a group: "project|session|turn"
    # -> when; a correction is grouped once, so a lesson the user rejected is not written again
    clustered: dict[str, str] = field(default_factory=dict)
    # the first run that counted lesson reads: the "index does not reach the agent" note waits from here
    reads_since: str | None = None
    # repo name -> the origin/main commit the weekly upstream scan has read up to
    upstream_synced: dict[str, str] = field(default_factory=dict)

    def event(self, kind: str, **fields) -> None:
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        self.events.append({"ts": ts, "kind": kind, **fields})


def load_state(path: Path) -> State:
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return State()
    raw["lessons"] = {k: Lesson(**v) for k, v in raw.get("lessons", {}).items()}
    return State(**raw)


def save_state(state: State, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(asdict(state), ensure_ascii=False, indent=2)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise


@contextmanager
def run_lock(path: Path) -> Iterator[bool]:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            yield False
        else:
            yield True
    finally:
        os.close(fd)


def snapshot_baseline(state: State, transcripts: list[Path], now_iso: str) -> None:
    """Every listed transcript not yet tracked is read from its current end on.

    `transcripts` lists subagent transcripts too, so afterwards their baseline is done.
    """
    if state.installed_at is None:
        state.installed_at = now_iso
    for t in transcripts:
        key = str(t)
        if key in state.sessions:
            continue
        try:
            size = Path(t).stat().st_size
        except FileNotFoundError:
            continue
        state.sessions[key] = {"offset": size}
    state.subagent_baseline_done = True


def parse_ts(value) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        ts = datetime.fromisoformat(value)
    except ValueError:
        return None
    return ts if ts.tzinfo is not None else ts.replace(tzinfo=timezone.utc)


def distinct_sessions(lesson: Lesson, now: datetime, window_days: int = 30) -> int:
    """Distinct non-empty evidence sessions within the last `window_days`.

    Evidence whose ts cannot be parsed counts as inside the window.
    """
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    since = now - timedelta(days=window_days)
    sessions = set()
    for entry in lesson.evidence:
        session = entry.get("session") if isinstance(entry, dict) else None
        if not isinstance(session, str) or not session:
            continue
        ts = parse_ts(entry.get("ts"))
        if ts is None or ts >= since:
            sessions.add(session)
    return len(sessions)


def normalize_cue(cue: str) -> str:
    return re.sub(r"\s+", " ", cue.casefold()).strip(" \t\n.,;:!?…-–—\"'«»()[]")
