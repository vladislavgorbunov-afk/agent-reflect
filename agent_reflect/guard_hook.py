#!/usr/bin/env python3
"""PreToolUse hook: shows an agent-reflect lesson, or a rule of the user's, at the moment it applies.

A shell command is matched against the rules for commands; a file about to be edited (Edit,
Write, NotebookEdit) against the rules for paths. A rule for a path is shown once a session:
the agent edits the same file many times, and the reminder is about the work, not the call.

Stands alone on purpose (stdlib only, no import of the package): Claude Code starts it before
every shell command and every edit. It never blocks: the text is added as context beside the
tool's result, and each showing is one line in the fires log, which the nightly run counts.
Any failure here is silent, so a broken rules file can never stop a tool.
"""
from __future__ import annotations

import json
import os
import re
import sys
import time

GUARD_DIR = os.environ.get("AGENT_REFLECT_GUARD_DIR") or os.path.expanduser("~/.claude/agent-reflect")
RULES_NAME = "guard-rules.json"
FIRES_NAME = "guard-fires.jsonl"
SHOWN_MAX = 3  # rules one call can bring up
EDIT_TOOLS = {"Edit": "file_path", "Write": "file_path", "NotebookEdit": "notebook_path"}
USER_PREFIX = "user:"
_FIRES_TAIL = 262_144  # of the fires log read to see what this session was already shown
# a heredoc: what follows `<<WORD` up to the line that is WORD alone is a file's text, not a command
_HEREDOC_RE = re.compile(r"(<<-?[ \t]*(['\"]?)([A-Za-z_]\w*)\2[^\n]*)\n.*?\n[ \t]*\3[ \t]*(?=\n|$)", re.S)
_HEADS = {
    "command": "agent-reflect: эта команда похожа на ту, что уже подводила в этом проекте.",
    "path": "agent-reflect: за работу с этим файлом пользователь уже возвращал результат в этом проекте.",
    "user": "agent-reflect: к этому действию относится правило пользователя.",
}


def shell_text(command: str) -> str:
    """The command without the bodies of its heredocs."""
    return _HEREDOC_RE.sub(lambda m: m.group(1), command)


def matching(rules: list, on: str, text: str, transcript: str, cwd: str) -> list[dict]:
    """The rules for `on` of this session's project (or of every project) whose pattern the text matches."""
    hits = []
    for rule in rules:
        if not isinstance(rule, dict) or rule.get("on", "command") != on:
            continue
        slug, project = str(rule.get("slug", "")), str(rule.get("project", ""))
        in_project = rule.get("all") is True or (bool(slug) and f"/projects/{slug}/" in transcript) or (
            bool(project) and (cwd == project or cwd.startswith(project + "/")))
        if not in_project:
            continue
        try:
            if re.search(str(rule.get("pattern", "")), text):
                hits.append(rule)
        except re.error:
            continue
    return hits[:SHOWN_MAX]


def shown_in(session: str) -> set[str]:
    """The rules for paths this session was already shown, read from the end of the fires log."""
    seen: set[str] = set()
    try:
        with open(os.path.join(GUARD_DIR, FIRES_NAME), "rb") as fh:
            fh.seek(0, 2)
            fh.seek(max(0, fh.tell() - _FIRES_TAIL))
            lines = fh.read().decode("utf-8", errors="replace").split("\n")
    except OSError:
        return seen
    for line in lines:
        try:
            entry = json.loads(line)
        except ValueError:
            continue
        if isinstance(entry, dict) and entry.get("session") == session and entry.get("on") == "path":
            seen.add(str(entry.get("lesson_id")))
    return seen


def render(hits: list[dict], on: str) -> str:
    """The lessons under the head for `on`, then the user's own rules under theirs."""
    lessons = [r for r in hits if not str(r.get("id", "")).startswith(USER_PREFIX)]
    own = [r for r in hits if str(r.get("id", "")).startswith(USER_PREFIX)]
    out = []
    if lessons:
        out += [_HEADS[on], *(f"- {r.get('text', '')} (урок {r.get('file', '')})" for r in lessons)]
    if own:
        out += [_HEADS["user"],
                *(f"- {r.get('text', '')} (правило «{str(r.get('id', ''))[len(USER_PREFIX):]}»)" for r in own)]
    return "\n".join(out)


def main() -> int:
    try:
        data = json.load(sys.stdin)
        tool, tool_input = data.get("tool_name"), data.get("tool_input") or {}
        if tool == "Bash":
            on, text = "command", tool_input.get("command")
            text = shell_text(text) if isinstance(text, str) else text
        elif tool in EDIT_TOOLS:
            on, text = "path", tool_input.get(EDIT_TOOLS[tool])
        else:
            return 0
        if not isinstance(text, str) or not text:
            return 0
        with open(os.path.join(GUARD_DIR, RULES_NAME), encoding="utf-8") as fh:
            rules = json.load(fh).get("rules", [])
        hits = matching(rules, on, text, str(data.get("transcript_path") or ""), str(data.get("cwd") or ""))
        session = data.get("session_id")
        if hits and on == "path":
            seen = shown_in(session)
            hits = [rule for rule in hits if str(rule.get("id")) not in seen]
        if not hits:
            return 0
        try:
            with open(os.path.join(GUARD_DIR, FIRES_NAME), "a", encoding="utf-8") as fh:
                for rule in hits:
                    fh.write(json.dumps({"ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                         "lesson_id": rule.get("id"), "session": session, "on": on}) + "\n")
        except OSError:
            pass  # the lesson is still shown
        print(json.dumps({"hookSpecificOutput": {
            "hookEventName": "PreToolUse", "additionalContext": render(hits, on),
        }}, ensure_ascii=False))
    except Exception:
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
