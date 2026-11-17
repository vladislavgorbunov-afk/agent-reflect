#!/usr/bin/env python3
"""Tool hook: shows an agent-reflect lesson, or a rule of the user's, at the moment it applies.

Before a tool runs (PreToolUse), a shell command is matched against the rules for commands and
a file about to be edited (Edit, Write, NotebookEdit) against the rules for paths. After a
shell command (PostToolUse, PostToolUseFailure), what it printed is matched against the rules
for errors: for a mistake the command does not show, only its error does. A rule for a path or
an error is shown once a session: the agent edits the same file many times and meets the same
error again while it works on it, and the reminder is about the work, not the call.

Stands alone on purpose (stdlib only, no import of the package): Claude Code starts it before
every shell command and every edit, and after every shell command. The text is added as
context beside the tool's result, and each showing is one line in the fires log, which the
nightly run counts. It refuses a command only when it matches a lesson the user made blocking
(`agent-reflect guard block`). Any failure here is silent, so a broken rules file can never stop a tool.
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
_OUTPUT_EDGE = 10_000  # of a long output, its start and its end are matched: errors print at either
# a command that prints files or history: what it shows is no error of its own (as transcript._reads_files)
_READERS = frozenset({"cat", "head", "tail", "less", "grep", "rg"})
_GIT_READERS = frozenset({"diff", "show", "log"})
_LEADING_CD_RE = re.compile(r"""\s*cd\s+(?:"[^"]*"|'[^']*'|\S+)\s*&&\s*""")
# what shows that a shell command failed although the tool did not say so (as transcript._SILENT_RE)
_SILENT_SCAN_MAX = 6000
_SILENT_RE = re.compile(
    r"^Traceback \(most recent call last\):|^Exit code [1-9]|: command not found"
    r"|Permission denied|^zsh: permission denied:|Operation not permitted|exited with (?:code|status) [1-9]",
    re.MULTILINE,
)
AFTER_EVENTS = ("PostToolUse", "PostToolUseFailure")
ONCE_A_SESSION = ("path", "error")
# a heredoc: what follows `<<WORD` up to the line that is WORD alone is a file's text, not a command
_HEREDOC_RE = re.compile(r"(<<-?[ \t]*(['\"]?)([A-Za-z_]\w*)\2[^\n]*)\n.*?\n[ \t]*\3[ \t]*(?=\n|$)", re.S)
_HEADS = {
    "command": "agent-reflect: эта команда похожа на ту, что уже подводила в этом проекте.",
    "path": "agent-reflect: за работу с этим файлом пользователь уже возвращал результат в этом проекте.",
    "error": "agent-reflect: эта ошибка уже встречалась в этом проекте, и вот что тогда помогло.",
    "user": "agent-reflect: к этому действию относится правило пользователя.",
}
_BLOCK_HEAD = ("agent-reflect: команда отклонена. Она совпала с уроком, который пользователь сделал "
               "блокирующим: его нарушали, хотя этот хук его показывал.")
_BLOCK_TAIL = ("Перепиши команду по уроку. Если урок к этой команде не относится, скажи об этом пользователю: "
               "снять блок может только он, в своём терминале: {unblock}.")


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
    return hits


def failed_quietly(text: str) -> bool:
    """True when the output of a command the tool did not mark as failed shows a failure anyway."""
    return _SILENT_RE.search(text[:_SILENT_SCAN_MAX]) is not None


def reads_files(command: str) -> bool:
    """True when the command's first word (after a leading `cd … &&`) prints files or history."""
    m = _LEADING_CD_RE.match(command)
    words = command[m.end() if m else 0:].split()
    if not words:
        return False
    first, rest = words[0].rsplit("/", 1)[-1], words[1:]
    if first in _READERS:
        return True
    if first == "sed":
        return rest[:1] == ["-n"]
    if first == "git":
        if rest[:1] == ["-C"]:
            rest = rest[2:]
        return bool(rest) and rest[0] in _GIT_READERS
    return False


def output_text(data: dict) -> str:
    """What a shell command printed, as the hook gets it after the command: an error, or its output."""
    if data.get("is_interrupt") is True:
        return ""  # the user stopped it: no error of the command's own
    error = data.get("error")
    if isinstance(error, str) and error:
        text = error
    else:
        response = data.get("tool_response")
        if isinstance(response, dict):
            text = "\n".join(str(response.get(k) or "") for k in ("stdout", "stderr")).strip()
        else:
            text = response if isinstance(response, str) else ""
    if len(text) > 2 * _OUTPUT_EDGE:
        text = text[:_OUTPUT_EDGE] + "\n" + text[-_OUTPUT_EDGE:]
    return text


def shown_in(session: str) -> set[tuple[str, str]]:
    """(rule id, on) of the rules for paths and errors this session was already shown, from the end of the fires log."""
    seen: set[tuple[str, str]] = set()
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
        if isinstance(entry, dict) and entry.get("session") == session and entry.get("on") in ONCE_A_SESSION:
            seen.add((str(entry.get("lesson_id")), str(entry.get("on"))))
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


def _rules() -> list:
    try:
        with open(os.path.join(GUARD_DIR, RULES_NAME), encoding="utf-8") as fh:
            rules = json.load(fh).get("rules", [])
    except (OSError, ValueError, AttributeError):
        return []
    return rules if isinstance(rules, list) else []


def render_block(blocking: list[dict]) -> str:
    """Why the command was refused: the blocking lessons, and who can lift the block."""
    unblock = ", ".join(f"`agent-reflect guard unblock {r.get('id', '')}`" for r in blocking)
    return "\n".join([_BLOCK_HEAD, *(f"- {r.get('text', '')} (урок {r.get('file', '')})" for r in blocking),
                      _BLOCK_TAIL.format(unblock=unblock)])


def main() -> int:
    try:
        data = json.load(sys.stdin)
        tool, tool_input = data.get("tool_name"), data.get("tool_input") or {}
        event = data.get("hook_event_name") or "PreToolUse"
        command = tool_input.get("command") if tool == "Bash" and isinstance(tool_input, dict) else None
        if event in AFTER_EVENTS:
            if tool != "Bash":
                return 0
            on, text = "error", output_text(data)
            if event == "PostToolUse" and ((isinstance(command, str) and reads_files(command))
                                           or not failed_quietly(text)):
                return 0  # it worked, or it printed what it read: as the night, no failure of its own
        elif tool == "Bash":
            on, text = "command", tool_input.get("command")
            text = shell_text(text) if isinstance(text, str) else text
        elif tool in EDIT_TOOLS:
            on, text = "path", tool_input.get(EDIT_TOOLS[tool])
        else:
            return 0
        if not isinstance(text, str) or not text:
            return 0
        rules = _rules()
        hits = matching(rules, on, text, str(data.get("transcript_path") or ""), str(data.get("cwd") or ""))
        session = data.get("session_id")
        if hits and on in ONCE_A_SESSION:
            seen = shown_in(session)
            hits = [rule for rule in hits if (str(rule.get("id")), on) not in seen]
        if not hits:
            return 0
        blocking = [r for r in hits if r.get("block") is True and on == "command" and event == "PreToolUse"]
        hits = (blocking or hits)[:SHOWN_MAX]  # cut only now: a block or an unseen rule is never crowded out
        blocking = hits if blocking else []
        try:
            with open(os.path.join(GUARD_DIR, FIRES_NAME), "a", encoding="utf-8") as fh:
                for rule in hits:
                    fire = {"ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                            "lesson_id": rule.get("id"), "session": session, "on": on}
                    if blocking:
                        fire["blocked"] = True
                    fh.write(json.dumps(fire) + "\n")
        except OSError:
            pass  # the lesson is still shown
        if blocking:
            out = {"hookEventName": event, "permissionDecision": "deny",
                   "permissionDecisionReason": render_block(blocking)}
        else:
            out = {"hookEventName": event, "additionalContext": render(hits, on)}
        print(json.dumps({"hookSpecificOutput": out}, ensure_ascii=False))
    except Exception:
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
