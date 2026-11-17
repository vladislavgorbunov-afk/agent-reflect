"""Lessons at the point of use: the rules the guard hook matches against, and its fires.

A lesson with a trigger (a regex over a shell command) is shown to the agent by
`guard_hook.py` when it runs such a command; one with a path trigger (a regex over a file's
path) when it is about to edit such a file; one with an error trigger (a regex over what a
failed shell command printed) right after such a failure. The user's own rules (`guard add`)
are shown the same way. This module keeps the hook's rules file in step with the ledger, counts the hook's
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
# there while some rule is over an error: the after-Bash hook starts Python only then
ERROR_RULES_MARK = "guard-error-rules"
FIRES_NAME = "guard-fires.jsonl"
# a lesson the agent has in its index, or one already moved into a skill: both are trusted
ACTIVE = ("live", "promoted")
TEXT_MAX = 500
FIRED_SESSIONS_MAX = 50
FIRE_TIMES_MAX = 20
TRIGGER_MAX = 200
HOOK_MARK = "agent_reflect/guard_hook.py"  # what tells our entry in settings.json from the user's
HOOK_MATCHER = "^(Bash|Edit|Write|NotebookEdit)$"
# after a shell command: what it printed is matched against the error triggers
AFTER_EVENTS = ("PostToolUse", "PostToolUseFailure")
AFTER_MATCHER = "^Bash$"
ON = ("command", "path", "error")
USER_PREFIX = "user:"  # the id of a rule the user added by hand, in the rules file and the fires log
NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,39}")
# paths no trigger may match: one that does says nothing about which file it is
_ANY_FILE = ("/Users/x/project/README.md", "/tmp/notes.txt", "/Users/x/project/src/main.py")
# what any failure prints: an error trigger that matches one of these says nothing about which mistake it is
_ANY_FAILURE = ("", "error", "Error: failed", "failed", "Exit code 1", "Traceback (most recent call last):",
                "command failed with exit code 2", "Permission denied", "No such file or directory",
                "command not found", "fatal: not a git repository")


def matches_any_failure(pattern: re.Pattern) -> bool:
    return any(pattern.search(text) for text in _ANY_FAILURE)


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

    `on` says what the pattern is matched against: a shell command, the path of a file being
    edited, or what a failed shell command printed.
    """
    out = []
    for lesson in sorted(state.lessons.values(), key=lambda item: item.id):
        if lesson.status not in ACTIVE:
            continue
        for on, pattern in lesson.patterns():
            if _compiles(pattern):
                rule = {"id": lesson.id, "file": lesson.file, "project": lesson.project_root,
                        "slug": sanitize(lesson.project_root), "on": on, "pattern": pattern,
                        "text": squash(lesson.lesson)[:TEXT_MAX]}
                if on == "command" and lesson.block:
                    rule["block"] = True
                out.append(rule)
    for rule in state.user_guards:
        if not isinstance(rule, dict) or rule.get("on") not in ON:
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
    """Write the hook's rules file when the rules changed; True when it was written.

    The error-rules mark follows: the after-Bash hook runs only while it is there.
    """
    path = guard_dir(cfg) / RULES_NAME
    current = rules(state)
    mark = guard_dir(cfg) / ERROR_RULES_MARK
    if any(rule["on"] == "error" for rule in current):
        mark.parent.mkdir(parents=True, exist_ok=True)
        mark.touch()
    else:
        mark.unlink(missing_ok=True)
    text = json.dumps({"rules": current}, ensure_ascii=False, indent=1) + "\n"
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
        session = str(entry.get("session") or "")
        if entry.get("on", "command") == "command" and session and isinstance(ts, str):
            times = lesson.fire_times.pop(session, [])  # popped and put back: the dict keeps the latest last
            lesson.fire_times[session] = [*times, ts][-FIRE_TIMES_MAX:]
            while len(lesson.fire_times) > FIRED_SESSIONS_MAX:
                lesson.fire_times.pop(next(iter(lesson.fire_times)))
        if entry.get("blocked") is True:
            state.event("lesson_blocked", project=lesson.project_root, session=session, lesson_id=lesson.id,
                        title=lesson.title)
        lesson.fires += 1
        if later(lesson.last_fired_at):
            lesson.last_fired_at = ts
        added += 1
    state.guard_offset += len(complete)
    return added


def check_trigger(trigger: str, on: str = "command") -> str:
    """Why a trigger cannot be used for commands (or file paths, `on="path"`, or the output of a
    failed command, `on="error"`), or "" when it can."""
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
    if on == "error":
        return "matches any failure" if matches_any_failure(pattern) else ""
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


def _our_entries(cfg: Config) -> dict[str, dict]:
    """Our entry for each event: before a tool runs (shell commands and edits), after a shell command.

    After a shell command the hook starts only while some rule is over an error: a shell test is
    cheap, Python's start after every command is not.
    """
    hook = {"type": "command", "command": hook_command(cfg), "timeout": 5}
    after = {**hook, "command": f'test -e "{guard_dir(cfg) / ERROR_RULES_MARK}" && ' + hook_command(cfg)}
    return {"PreToolUse": {"matcher": HOOK_MATCHER, "hooks": [hook]},
            **{event: {"matcher": AFTER_MATCHER, "hooks": [dict(after)]} for event in AFTER_EVENTS}}


def install_hook(cfg: Config) -> bool:
    """Add the guard hook to the user's settings.json, keeping everything else; True when it changed."""
    path = cfg.claude_settings
    settings = _load_settings(path)
    hooks = settings.setdefault("hooks", {})
    changed = False
    for event, ours in _our_entries(cfg).items():
        entries = hooks.get(event) or []
        kept = [e for e in entries if not _is_ours(e)]
        if len(kept) == len(entries) - 1 and ours in entries:
            continue
        hooks[event] = [*kept, ours]
        changed = True
    if changed:
        _atomic_write(path, json.dumps(settings, ensure_ascii=False, indent=2) + "\n")
    return changed


def uninstall_hook(cfg: Config) -> bool:
    """Take our guard hook out of settings.json; True when it was there."""
    path = cfg.claude_settings
    try:
        settings = _load_settings(path)
    except ValueError:
        return False
    hooks = settings.get("hooks")
    if not isinstance(hooks, dict):
        return False
    changed = False
    for event in _our_entries(cfg):
        entries = hooks.get(event)
        if not isinstance(entries, list) or not any(_is_ours(e) for e in entries):
            continue
        kept = [e for e in entries if not _is_ours(e)]
        if kept:
            hooks[event] = kept
        else:
            del hooks[event]
        changed = True
    if not changed:
        return False
    if not hooks:
        del settings["hooks"]
    _atomic_write(path, json.dumps(settings, ensure_ascii=False, indent=2) + "\n")
    return True
