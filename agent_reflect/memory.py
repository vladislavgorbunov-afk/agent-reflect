from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from agent_reflect.config import Config
from agent_reflect.ledger import Lesson, State, normalize_cue
from agent_reflect.promptkit import error_text

INDEX_HEADER = "# Memory Index\n\n"
AUTO_PREFIX = "- [auto_"


class MemoryIndexError(Exception):
    """MEMORY.md cannot be read safely; nothing was written."""


class MemoryWriteError(Exception):
    """Refused to write or edit a memory file that is not ours."""


_AUTO_FILE_RE = re.compile(r"auto_[A-Za-z0-9_.\-]+\.md")


def lesson_filename(lesson_id: str) -> str:
    return f"auto_{lesson_id}.md"


def _one_line(text: str) -> str:
    return " ".join(text.split())


def render_lesson(lesson: Lesson) -> str:
    evidence = ", ".join(f"{e['session']}@{e['ts']}" for e in lesson.evidence)
    hook = _one_line(lesson.hook)
    cue = re.sub(r"[.!;\s]+$", "", lesson.cue)
    return (
        "---\n"
        f"name: auto-{lesson.id}\n"
        f"description: {json.dumps(hook, ensure_ascii=False)}\n"
        "metadata:\n"
        "  type: feedback\n"
        "  source: agent-reflect\n"
        f"  lesson_id: {lesson.id}\n"
        f"  confidence: {lesson.confidence}\n"
        f"  repeats: {lesson.repeats}\n"
        f"  evidence_sessions: [{evidence}]\n"
        "---\n\n"
        f"{lesson.lesson}\n\n"
        f"**Why:** {lesson.why}\n\n"
        f"**How to apply:** Когда: {cue}. Делать: {lesson.lesson}\n"
    )


def _atomic_write(path: Path, data: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as fh:
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


def write_lesson(memory_dir: Path, lesson: Lesson) -> str:
    memory_dir = Path(memory_dir)
    if lesson.file != lesson_filename(lesson.id) or not _AUTO_FILE_RE.fullmatch(lesson.file):
        raise MemoryWriteError(f"unexpected lesson file name: {lesson.file!r}")
    target = memory_dir / lesson.file
    if target.exists() and is_user_entry(target):
        raise MemoryWriteError(f"refusing to overwrite a user-owned file: {target}")
    data = render_lesson(lesson).encode("utf-8")
    _atomic_write(target, data)
    return hashlib.sha256(data).hexdigest()


def _read_index(memory_dir: Path) -> tuple[bytes, str]:
    try:
        raw = (Path(memory_dir) / "MEMORY.md").read_bytes()
    except FileNotFoundError:
        return b"", ""
    except OSError as exc:
        raise MemoryIndexError(f"cannot read MEMORY.md: {exc}") from exc
    try:
        return raw, raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise MemoryIndexError(f"MEMORY.md is not UTF-8: {exc}") from exc


def read_index(memory_dir: Path) -> str:
    """MEMORY.md as text, "" when absent. Raises MemoryIndexError when it cannot be read safely."""
    return _read_index(memory_dir)[1]


def index_line(filename: str, hook: str) -> str:
    """The MEMORY.md line for one entry, exactly as add_index_line writes it."""
    return f"- [{filename}]({filename}) — {_one_line(hook)}"


def _has_entry(text: str, filename: str) -> bool:
    marker = f"- [{filename}]("
    return any(line.startswith(marker) for line in text.split("\n"))


def add_index_line(memory_dir: Path, filename: str, hook: str) -> None:
    raw, text = _read_index(memory_dir)
    if _has_entry(text, filename):
        return
    eol = b"\r\n" if b"\r\n" in raw else b"\n"
    if not raw:
        raw = INDEX_HEADER.encode("utf-8")
    elif not raw.endswith(b"\n"):
        raw += eol
    line = index_line(filename, hook).encode("utf-8")
    _atomic_write(Path(memory_dir) / "MEMORY.md", raw + line + eol)


def remove_index_line(memory_dir: Path, filename: str) -> None:
    if not _AUTO_FILE_RE.fullmatch(filename):
        raise MemoryWriteError(f"refusing to remove a non-auto index line: {filename!r}")
    raw, text = _read_index(memory_dir)
    marker = f"- [{filename}]("
    parts = text.split("\n")
    kept = [p for p in parts if not p.startswith(marker)]
    if len(kept) != len(parts):
        _atomic_write(Path(memory_dir) / "MEMORY.md", "\n".join(kept).encode("utf-8"))


def remove_lesson(memory_dir: Path, lesson: Lesson) -> None:
    """Remove an auto-lesson's index line, then its file.

    Only a file we wrote and nobody changed since: the name is the lesson's own,
    the file is not a user entry, and its bytes still hash to lesson.sha256.
    Anything else raises MemoryWriteError and touches nothing.
    """
    memory_dir = Path(memory_dir)
    if lesson.file != lesson_filename(lesson.id) or not _AUTO_FILE_RE.fullmatch(lesson.file):
        raise MemoryWriteError(f"unexpected lesson file name: {lesson.file!r}")
    path = memory_dir / lesson.file
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise MemoryWriteError(f"cannot read lesson file {path}: {exc}") from exc
    if is_user_entry(path):
        raise MemoryWriteError(f"refusing to delete a user-owned file: {path}")
    if hashlib.sha256(raw).hexdigest() != lesson.sha256:
        raise MemoryWriteError(f"refusing to delete a lesson changed since it was written: {path}")
    remove_index_line(memory_dir, lesson.file)
    path.unlink(missing_ok=True)


def has_capacity(memory_dir: Path, cfg: Config, new_line: str) -> bool:
    try:
        _, text = _read_index(memory_dir)
    except MemoryIndexError:
        return False
    return index_has_room(text, cfg, new_line)


def index_has_room(text: str, cfg: Config, new_line: str) -> bool:
    """Whether MEMORY.md with this text can take one more of our lines."""
    lines = text.splitlines()
    auto = sum(1 for ln in lines if ln.startswith(AUTO_PREFIX))
    return (
        auto < cfg.auto_index_cap
        and len(lines) + 1 <= cfg.index_line_limit
        and len(text) + len(new_line) + 1 <= cfg.index_char_limit  # the line lands with its newline
    )


def _lesson_event(state: State, kind: str, lesson: Lesson, **fields) -> None:
    state.event(kind, project=lesson.project_root, lesson_id=lesson.id, title=lesson.title, **fields)


def go_live_if_room(cfg: Config, lesson: Lesson, now_iso: str | None = None,
                    state: State | None = None) -> bool:
    """Add the lesson's index line when MEMORY.md has room. Raises the memory write errors.

    A lesson that goes live records when (`now_iso`, else the clock): the agent of a
    session started before then never saw its line. With a `state` it also records
    a lesson_live event.
    """
    memory_dir = Path(lesson.memory_dir)
    read_index(memory_dir)  # an unreadable MEMORY.md is a failed write, not a full index
    if not has_capacity(memory_dir, cfg, index_line(lesson.file, lesson.hook)):
        return False
    add_index_line(memory_dir, lesson.file, lesson.hook)
    lesson.status = "live"
    lesson.live_since = now_iso or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    if state is not None:
        _lesson_event(state, "lesson_live", lesson)
    return True


_SOURCE_RE = re.compile(r"""\s+source:\s*["']?agent-reflect["']?\s*""")


def is_user_entry(path: Path) -> bool:
    """False only for a missing file or one we wrote; unreadable means user's."""
    try:
        raw = Path(path).read_bytes()
    except FileNotFoundError:
        return False
    except OSError:
        return True
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        return True
    m = re.match(r"---\r?\n(.*?)\r?\n---", text, re.DOTALL)
    if not m:
        return True
    in_meta = False
    for line in m.group(1).splitlines():
        if re.fullmatch(r"metadata:\s*", line):
            in_meta = True
        elif in_meta and line.strip() and not line[0].isspace():
            in_meta = False
        elif in_meta and _SOURCE_RE.fullmatch(line):
            return False
    return True


def _reject(state: State, lesson: Lesson, reason: str) -> None:
    lesson.status = "rejected"
    _lesson_event(state, "lesson_rejected", lesson, reason=reason)
    cues = state.rejected_cues.setdefault(lesson.project_root, [])
    cue = normalize_cue(lesson.cue)
    if cue not in cues:
        cues.append(cue)


def _index_line_deleted(memory_dir: Path, filename: str) -> bool:
    """True only when MEMORY.md exists, reads cleanly and has no line for `filename`."""
    if not (memory_dir / "MEMORY.md").is_file():
        return False
    try:
        return not _has_entry(read_index(memory_dir), filename)
    except MemoryIndexError:
        return False


def _hand_over(state: State, lesson: Lesson) -> None:
    lesson.status = "user_owned"
    _lesson_event(state, "lesson_user_owned", lesson, reason="edited")


def sync_ownership(state: State) -> list[tuple[str, str]]:
    """Read what the user did to our lessons since the last run.

    A deleted file or a deleted index line of a live lesson is a rejection (the
    file goes too); an edited or unreadable file now belongs to the user. Each
    is recorded as an event.
    """
    changed: list[tuple[str, str]] = []
    for lesson in state.lessons.values():
        if lesson.status not in ("live", "staged"):
            continue
        memory_dir = Path(lesson.memory_dir)
        if not memory_dir.is_dir():
            continue
        try:
            raw = (memory_dir / lesson.file).read_bytes()
        except FileNotFoundError:
            _reject(state, lesson, "file_deleted")
            try:
                remove_index_line(memory_dir, lesson.file)
            except (MemoryIndexError, MemoryWriteError, OSError):
                pass
        except OSError:
            _hand_over(state, lesson)
        else:
            if hashlib.sha256(raw).hexdigest() != lesson.sha256:
                _hand_over(state, lesson)
            elif lesson.status == "live" and _index_line_deleted(memory_dir, lesson.file):
                _reject(state, lesson, "index_line_deleted")
                try:
                    remove_lesson(memory_dir, lesson)  # its guards still decide whether the file goes
                except (MemoryIndexError, MemoryWriteError, OSError):
                    pass
            else:
                continue
        pair = (lesson.project_root, lesson.memory_dir)
        if pair not in changed:
            changed.append(pair)
    return changed


def managed_projects(state: State) -> list[tuple[str, str]]:
    """(project_root, memory_dir) of every project with a live or staged auto-lesson."""
    return sorted({(lesson.project_root, lesson.memory_dir) for lesson in state.lessons.values()
                   if lesson.status in ("live", "staged")})


_CONFIDENCE_RANK = {"low": 0, "medium": 1, "high": 2}


def _fits(text: str, cfg: Config) -> bool:
    return len(text.splitlines()) <= cfg.index_line_limit and len(text) <= cfg.index_char_limit


def rebalance(state: State, cfg: Config, project_root, memory_dir, now_iso: str | None = None) -> bool:
    """Keep MEMORY.md under the load limit: our lines give way to the user's. True if changed.

    Over the line or char limit, live lessons lose their index line (low confidence
    first, then medium, then the oldest) until it fits; then, while there is room,
    eligible staged lessons go live, most repeated first. Lines that are not our
    live lessons' are never touched; an unreadable MEMORY.md changes nothing.
    """
    memory_dir = Path(memory_dir)
    if not memory_dir.is_dir():
        return False
    ours = [lesson for lesson in state.lessons.values()
            if lesson.project_root == str(project_root) and lesson.memory_dir == str(memory_dir)]
    changed = False
    try:
        text = read_index(memory_dir)
        live = sorted((lesson for lesson in ours if lesson.status == "live" and _has_entry(text, lesson.file)),
                      key=lambda lesson: (_CONFIDENCE_RANK.get(lesson.confidence, 0), lesson.created_at))
        for lesson in live:
            if _fits(text, cfg):
                break
            remove_index_line(memory_dir, lesson.file)
            lesson.status = "staged"
            _lesson_event(state, "lesson_unlisted", lesson, reason="no_room")
            changed = True
            text = read_index(memory_dir)
        if not _fits(text, cfg):
            return changed
        staged = sorted((lesson for lesson in ours if lesson.status == "staged"
                         and (lesson.confidence == "high" or lesson.repeats >= 2)),
                        key=lambda lesson: (-lesson.repeats, -_CONFIDENCE_RANK.get(lesson.confidence, 0),
                                            lesson.created_at))
        for lesson in staged:
            changed = go_live_if_room(cfg, lesson, now_iso, state) or changed
    except MemoryIndexError:
        return changed
    except (MemoryWriteError, OSError, UnicodeEncodeError) as exc:
        state.event("memory_write_failed", project=str(project_root), step="rebalance", error=error_text(exc))
    return changed


def autodream_disabled(settings_path: Path) -> bool:
    try:
        data = json.loads(Path(settings_path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return isinstance(data, dict) and data.get("autoDreamEnabled") is False
