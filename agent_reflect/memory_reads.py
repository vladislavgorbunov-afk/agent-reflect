"""Which memory and notes files the agent opens: counted afresh from the last month of transcripts.

The user's own records live in each project's memory dir (`<projects>/<project>/memory/*.md`)
and in the global notes dir (`~/.claude/notes/*.md`). Every `Read` of such a file and every
`Bash`/`Grep` call that names one counts as one read of it, each file at most once per call.
The indexes (MEMORY.md, CLAUDE.md) load on their own and the auto-lessons are usage.py's, so
neither is listed. Nothing is stored: the weekly run scans again, and transcripts older than
about a month are gone anyway.
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

from agent_reflect.config import Config
from agent_reflect.ledger import parse_ts
from agent_reflect.projects import sanitize
from agent_reflect.transcript import list_transcripts

WINDOW_DAYS = 30
NOTES = "notes"
_INDEXES = frozenset({"MEMORY.md", "CLAUDE.md"})
_LESSON_PREFIX = "auto_"
_MENTION_TOOLS = ("Bash", "Grep")
# A file name in a shell command: no quote, separator, glob or variable in it, and nothing
# that continues the name after `.md` (reference.md.bak is no read of reference.md).
_NAME = r"(?P<name>[^/\s\"'`\\|;&<>()*?\[\]{}$]+\.md)(?![\w.-])"


def _notes_dir(cfg: Config) -> Path:
    return cfg.home / ".claude" / NOTES


def _listed(name: str) -> bool:
    return name.endswith(".md") and name not in _INDEXES and not name.startswith(_LESSON_PREFIX)


class Matcher:
    """The memory and notes files one tool call reads, as (directory, file name) pairs."""

    def __init__(self, cfg: Config):
        home = rf"(?:{re.escape(str(cfg.home))}|~|\$HOME|\$\{{HOME\}})"
        projects = re.escape(cfg.projects_dir.relative_to(cfg.home).as_posix())
        notes = re.escape(_notes_dir(cfg).relative_to(cfg.home).as_posix())
        self.projects_dir = cfg.projects_dir
        self.notes_dir = _notes_dir(cfg)
        self.memory_re = re.compile(rf"{home}/{projects}/(?P<project>[^/\s\"'`]+)/memory/{_NAME}")
        self.notes_re = re.compile(rf"{home}/{notes}/{_NAME}")

    def _target(self, m: re.Match) -> tuple[Path, str] | None:
        name = m.group("name")
        if not _listed(name):
            return None
        if m.re is self.memory_re:
            return self.projects_dir / m.group("project") / "memory", name
        return self.notes_dir, name

    def files(self, tool: str, tool_input) -> list[tuple[Path, str]]:
        if tool == "Read":
            path = tool_input.get("file_path") if isinstance(tool_input, dict) else None
            if not isinstance(path, str):
                return []
            m = self.memory_re.fullmatch(path) or self.notes_re.fullmatch(path)
            target = self._target(m) if m else None
            return [target] if target else []
        if tool not in _MENTION_TOOLS:
            return []
        if isinstance(tool_input, dict):
            text = "\n".join(v for v in tool_input.values() if isinstance(v, str))
        else:
            text = tool_input if isinstance(tool_input, str) else ""
        found = [self._target(m) for regex in (self.memory_re, self.notes_re) for m in regex.finditer(text)]
        return list(dict.fromkeys(t for t in found if t))


def _tool_calls(path: Path):
    """(tool, input, timestamp) of each tool call in a transcript that could name a file; bad lines are skipped."""
    try:
        with path.open("rb") as f:
            for raw in f:
                if b'"tool_use"' not in raw or b".md" not in raw:
                    continue
                try:
                    entry = json.loads(raw.decode("utf-8", errors="replace"))
                except ValueError:
                    continue
                if not isinstance(entry, dict) or entry.get("type") != "assistant":
                    continue
                msg = entry.get("message")
                content = msg.get("content") if isinstance(msg, dict) else None
                if not isinstance(content, list):
                    continue
                ts = entry.get("timestamp") if isinstance(entry.get("timestamp"), str) else None
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "tool_use":
                        yield block.get("name"), block.get("input"), ts
    except OSError:
        return


def location_name(cfg: Config, folder: str) -> str:
    """A project folder's name for the digest: the repo's name, else the slug past the repos' parent or home."""
    for name, root in cfg.repos.items():
        if sanitize(str(root)) == folder:
            return name
    parents = {sanitize(str(root.parent)) for root in cfg.repos.values()} | {sanitize(str(cfg.home))}
    for prefix in sorted(parents, key=len, reverse=True):
        if folder.startswith(prefix + "-") and len(folder) > len(prefix) + 1:
            return folder[len(prefix) + 1:]
    return folder


def _on_disk(directory: Path) -> list[str]:
    try:
        return sorted(p.name for p in directory.iterdir() if _listed(p.name) and p.is_file())
    except OSError:
        return []


def _locations(cfg: Config) -> list[tuple[str, Path]]:
    """The notes dir, then every project memory dir by name; a missing dir is simply left out."""
    found = []
    try:
        folders = sorted(d for d in cfg.projects_dir.iterdir() if (d / "memory").is_dir())
    except OSError:
        folders = []
    for folder in folders:
        found.append((location_name(cfg, folder.name), folder / "memory"))
    return [(NOTES, _notes_dir(cfg))] + sorted(found)


def scan(cfg: Config, *, now: datetime, days: int = WINDOW_DAYS) -> dict:
    """Reads of each memory and notes file on disk within the last `days`. Read-only; never raises on bad data."""
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    cutoff = now - timedelta(days=days)
    matcher = Matcher(cfg)
    reads: dict[tuple[Path, str], list] = {}  # (dir, name) -> [count, latest ts]
    for path in list_transcripts(cfg.projects_dir, ()):
        try:
            if datetime.fromtimestamp(path.stat().st_mtime, timezone.utc) < cutoff:
                continue
        except OSError:
            continue
        for tool, tool_input, ts in _tool_calls(path):
            when = parse_ts(ts)
            if when is not None and when < cutoff:
                continue
            for key in matcher.files(tool, tool_input):
                seen = reads.setdefault(key, [0, None])
                seen[0] += 1
                if ts and (seen[1] is None or ts > seen[1]):
                    seen[1] = ts
    locations = []
    for name, directory in _locations(cfg):
        files = []
        for f in _on_disk(directory):
            count, last = reads.get((directory, f), (0, None))
            files.append({"name": f, "reads": count, "last_read": last})
        if files:
            locations.append({"name": name, "files": files})
    return {"days": days, "locations": locations}
