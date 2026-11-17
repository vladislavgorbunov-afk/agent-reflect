"""Lessons at the point of use: the rules the guard hook matches against, and its fires.

A lesson with a trigger (a regex over a shell command) is shown to the agent by
`guard_hook.py` when it runs such a command; one with a path trigger (a regex over a file's
path) when it is about to edit such a file. The user's own rules (`guard add`) are shown the
same way. This module keeps the hook's rules file in step with the ledger, counts the hook's
fires, and wires the hook into the user's Claude Code settings.
"""
from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path

from agent_reflect.config import Config
from agent_reflect.ledger import State
from agent_reflect.projects import sanitize
from agent_reflect.promptkit import squash

RULES_NAME = "guard-rules.json"
FIRES_NAME = "guard-fires.jsonl"
# a lesson the agent has in its index, or one already moved into a skill: both are trusted
ACTIVE = ("live", "promoted")
TEXT_MAX = 500
TRIGGER_MAX = 200
HOOK_MARK = "agent_reflect/guard_hook.py"  # what tells our entry in settings.json from the user's
HOOK_MATCHER = "^(Bash|Edit|Write|NotebookEdit)$"
USER_PREFIX = "user:"  # the id of a rule the user added by hand, in the rules file and the fires log
NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,39}")
# paths no trigger may match: one that does says nothing about which file it is
_ANY_FILE = ("/Users/x/project/README.md", "/tmp/notes.txt", "/Users/x/project/src/main.py")


def guard_dir(cfg: Config) -> Path:
    return cfg.home / ".claude" / "agent-reflect"


def hook_command(cfg: Config) -> str:
    script = cfg.repo_dir / "agent_reflect" / "guard_hook.py"
    return f'/usr/bin/python3 "{script}" 2>/dev/null || true'


def _compiles(pattern: str) -> bool:
    try:
        re.compile(pattern)
    except re.error:
        return False
    return bool(pattern)


def rules(state: State) -> list[dict]:
    """A rule per trigger that compiles: of every trusted lesson, then of the user's own rules.

    `on` says what the pattern is matched against: a shell command or the path of a file being edited.
    """
    out = []
    for lesson in sorted(state.lessons.values(), key=lambda item: item.id):
        if lesson.status not in ACTIVE:
            continue
        for on, pattern in (("command", lesson.trigger), ("path", lesson.path_trigger)):
            if _compiles(pattern):
                out.append({"id": lesson.id, "file": lesson.file, "project": lesson.project_root,
                            "slug": sanitize(lesson.project_root), "on": on, "pattern": pattern,
                            "text": squash(lesson.lesson)[:TEXT_MAX]})
    for rule in state.user_guards:
        if not isinstance(rule, dict) or rule.get("on") not in ("command", "path"):
            continue
        pattern, project = str(rule.get("pattern") or ""), str(rule.get("project") or "")
        if _compiles(pattern):
            out.append({"id": USER_PREFIX + str(rule.get("name", "")), "file": "", "project": project,
                        "slug": sanitize(project) if project else "", "all": not project, "on": rule["on"],
                        "pattern": pattern, "text": squash(str(rule.get("text") or ""))[:TEXT_MAX]})
    return out


def _atomic_write(path: Path, text: str) -> None:
    """Replace the file `path` names. A link stays a link (its target is replaced), and a file
    that was there keeps its mode: settings.json is the user's, often linked from their dotfiles."""
    path = Path(os.path.realpath(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        mode = os.stat(path).st_mode & 0o7777
    except OSError:
        mode = None
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
        if mode is not None:
            os.chmod(tmp, mode)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def sync(cfg: Config, state: State) -> bool:
    """Write the hook's rules file when the rules changed; True when it was written."""
    path = guard_dir(cfg) / RULES_NAME
    text = json.dumps({"rules": rules(state)}, ensure_ascii=False, indent=1) + "\n"
    try:
        if path.read_text(encoding="utf-8") == text:
            return False
    except (OSError, UnicodeDecodeError):
        pass
    _atomic_write(path, text)
    return True


def merge_fires(cfg: Config, state: State) -> int:
    """Count the hook's fires since the last run into the lessons; returns how many were new."""
    path = guard_dir(cfg) / FIRES_NAME
    try:
        with path.open("rb") as fh:
            fh.seek(0, 2)
            if state.guard_offset > fh.tell():  # the log was cleared
                state.guard_offset = 0
            fh.seek(state.guard_offset)
            data = fh.read()
    except OSError:
        return 0
    complete = data[: data.rfind(b"\n") + 1]
    added = 0
    for raw in complete.split(b"\n"):
        try:
            entry = json.loads(raw.decode("utf-8", errors="replace"))
        except ValueError:
            continue
        if not isinstance(entry, dict):
            continue
        fired, ts = str(entry.get("lesson_id") or ""), entry.get("ts")
        later = lambda last: isinstance(ts, str) and (not last or ts > last)  # noqa: E731
        if fired.startswith(USER_PREFIX):
            rule = next((r for r in state.user_guards
                         if isinstance(r, dict) and r.get("name") == fired[len(USER_PREFIX):]), None)
            if rule is None:
                continue
            rule["fires"] = int(rule.get("fires") or 0) + 1
            if later(rule.get("last_fired_at")):
                rule["last_fired_at"] = ts
            added += 1
            continue
        lesson = state.lessons.get(fired)
        if lesson is None:
            continue
        lesson.fires += 1
        if later(lesson.last_fired_at):
            lesson.last_fired_at = ts
        added += 1
    state.guard_offset += len(complete)
    return added


def check_trigger(trigger: str, on: str = "command") -> str:
    """Why a trigger cannot be used for commands (or for file paths, `on="path"`), or "" when it can."""
    if not trigger:
        return "empty"
    if len(trigger) > TRIGGER_MAX:
        return "too long"
    try:
        pattern = re.compile(trigger)
    except re.error as exc:
        return f"not a regex: {exc}"
    if on == "path":
        # an extension alone (no folder in the pattern) is every such file of every project
        broad = "/" not in trigger or pattern.search("") or any(pattern.search(p) for p in _ANY_FILE)
        return "matches any file" if broad else ""
    if pattern.search("") or pattern.search("ls"):
        return "matches any command"
    return ""


def _load_settings(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} is not a JSON object")
    return data


def _is_ours(entry) -> bool:
    return isinstance(entry, dict) and any(
        isinstance(h, dict) and HOOK_MARK in str(h.get("command", "")) for h in entry.get("hooks") or [])


def install_hook(cfg: Config) -> bool:
    """Add the guard hook to the user's settings.json, keeping everything else; True when it changed."""
    path = cfg.claude_settings
    settings = _load_settings(path)
    hooks = settings.setdefault("hooks", {})
    entries = hooks.setdefault("PreToolUse", [])
    ours = {"matcher": HOOK_MATCHER, "hooks": [{"type": "command", "command": hook_command(cfg), "timeout": 5}]}
    kept = [e for e in entries if not _is_ours(e)]
    if len(kept) == len(entries) - 1 and ours in entries:
        return False
    hooks["PreToolUse"] = [*kept, ours]
    _atomic_write(path, json.dumps(settings, ensure_ascii=False, indent=2) + "\n")
    return True


def uninstall_hook(cfg: Config) -> bool:
    """Take our guard hook out of settings.json; True when it was there."""
    path = cfg.claude_settings
    try:
        settings = _load_settings(path)
    except ValueError:
        return False
    entries = (settings.get("hooks") or {}).get("PreToolUse")
    if not isinstance(entries, list) or not any(_is_ours(e) for e in entries):
        return False
    kept = [e for e in entries if not _is_ours(e)]
    if kept:
        settings["hooks"]["PreToolUse"] = kept
    else:
        del settings["hooks"]["PreToolUse"]
        if not settings["hooks"]:
            del settings["hooks"]
    _atomic_write(path, json.dumps(settings, ensure_ascii=False, indent=2) + "\n")
    return True
