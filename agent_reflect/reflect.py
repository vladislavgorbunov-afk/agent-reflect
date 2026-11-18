from __future__ import annotations

import hashlib
import logging
import re
from collections import Counter
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable

from agent_reflect import cursor_index, guard, health, i18n, session_roots, usage
from agent_reflect.config import Config
from agent_reflect.guard_hook import shell_text
from agent_reflect.ledger import Lesson, State, normalize_cue, parse_ts, snapshot_baseline
from agent_reflect.llm import LlmError, run_structured, take_spent
from agent_reflect.projects import memory_dir_for, resolve_project_root
from agent_reflect.memory import (
    MemoryIndexError,
    MemoryWriteError,
    autodream_disabled,
    go_live_if_room,
    is_user_entry,
    lesson_filename,
    managed_projects,
    read_index,
    rebalance,
    sync_ownership,
    write_lesson,
)
from agent_reflect.promptkit import DataTags, error_text, load_template, model_text, nonce, squash
from agent_reflect.scrub import scrub
from agent_reflect.transcript import (
    SIGNAL_WEIGHTS,
    Segment,
    _render_signal,
    _render_turn,
    is_subagent,
    list_transcripts,
    parent_session_id,
    read_segment,
    session_cwd,
    session_title,
)

log = logging.getLogger(__name__)

VERDICTS = ("NO_ENTRY", "NEW", "REPEAT", "USER_RULE_REPEAT", "LESSON_VIOLATED")
CONFIDENCES = ("high", "medium", "low")
TITLE_MAX = 60
HOOK_MAX = 120
CONFLICT_NOTES_MAX = 300
NOTES_LOG_MAX = 1000
_STRING_FIELDS = (
    "lesson_id", "memory_file", "conflict_file", "title", "hook", "cue", "lesson", "why", "evidence",
)
_NEW_FIELDS = ("title", "hook", "cue", "lesson", "why", "evidence", "confidence")
MAX_LLM_ERRORS = 3
# What a correction of the user was about. self_report: the agent said the work was done or
# checked and it was not; constraint: it broke a rule or a limit it had been given; intent: it
# did something other than what was asked; operational: a tool or a command used wrongly.
FORMS = ("self_report", "constraint", "intent", "operational", "other")
# A lesson about a tool or a command, or one about how to work (read from the user's corrections).
KINDS = ("tool", "process", "fact")
RECHECK_MAX = 300
MAX_ENTRIES = 3  # lessons and repeats one model answer may carry
MAX_EPISODES = 40  # of one answer: a count, so a long part with many corrections must not be cut short
# Which reading of the user's corrections counted them. Raise it when the prompt or the dialogue
# view changes what counts as a correction: counts of different detectors are not compared.
DETECTOR = 2
GIST_MAX = 200  # of the model's sentence about a correction
SAID_MAX = 160  # of the user's own words kept with it
RULE_MIN = 12  # characters of a quoted rule that tell which rule it is
TRIGGER_MAX = 200
QUOTE_MIN = 4  # characters of a quote that prove anything
REPLY_WEIGHT = 2  # what an answer of the user to the agent adds to a segment's weight
_FAILING = ("tool_error", "silent_failure")
# subagent segments worth a model call: a fix that worked, or a hook that stopped the agent
_SUBAGENT_SIGNALS = ("recovery", "hook_block")
NOTIFY_AFTER_FAILURES = 3
AGING_PER_DAY = 0.5  # weight a waiting segment gains per day of age
WEIGHT_REPEATS_MAX = 3  # occurrences of one signal kind that add to a segment's weight
_WRITE_ERRORS = (MemoryWriteError, MemoryIndexError, OSError, UnicodeEncodeError)

_REF_SCHEMA: dict = {
    "type": "object",
    "properties": {"ref": {"type": "string"}, "quote": {"type": "string"}},
    "required": ["ref", "quote"],
    "additionalProperties": False,
}
ENTRY_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": list(VERDICTS)},
        **{name: {"type": "string"} for name in _STRING_FIELDS},
        "confidence": {"type": "string", "enum": list(CONFIDENCES)},
        "kind": {"type": "string", "enum": list(KINDS)},
        "form": {"type": "string", "enum": list(FORMS)},
        "trigger": {"type": "string"},
        "path_trigger": {"type": "string"},
        "error_trigger": {"type": "string"},
        "recheck": {"type": "string"},
        "refs": {"type": "array", "items": _REF_SCHEMA},
    },
    "required": ["verdict"],
    "additionalProperties": False,
}
# `notes` comes first so the model reasons before it commits; then every correction of the
# user it found (`episodes`), then what follows from the segment (`entries`, none = NO_ENTRY).
REFLECT_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "notes": {"type": "string"},
        "episodes": {"type": "array", "items": {
            "type": "object",
            "properties": {"ref": {"type": "string"}, "form": {"type": "string", "enum": list(FORMS)},
                           "gist": {"type": "string"}, "rule": {"type": "string"}},
            "required": ["ref", "form"],
            "additionalProperties": False,
        }},
        "entries": {"type": "array", "items": ENTRY_SCHEMA},
    },
    "required": ["notes", "entries"],
    "additionalProperties": False,
}
# The second look at a new lesson: do the quoted passages carry it?
VERIFY_SCHEMA: dict = {
    "type": "object",
    "properties": {"notes": {"type": "string"}, "supported": {"type": "boolean"}},
    "required": ["notes", "supported"],
    "additionalProperties": False,
}

# asks the second look about a lesson and the passages it quotes: (does it hold, why)
Verify = Callable[[dict, list[str]], tuple[bool | None, str]]

_TAGS = DataTags(("user_rules", "project_rules", "memory_index", "auto_lessons", "rejected_cues",
                  "session_segment"))
USER_RULES_MAX = 16_000
_VERIFY_TAGS = DataTags(("lesson", "passages", "session_segment"))
_REF_RE = re.compile(r"\[?(u?)(\d+)\]?")
_SHOWN_TURN_RE = re.compile(r"^\[u(\d+)\] ", re.M)
_QUOTE_EDGES = "«»\"'“”„`…. "
_ELLIPSIS_RE = re.compile(r"\s*(?:…|\.{3})\s*")
QUOTE_SHOWN_MAX = 160

# What makes a lesson checkable in a later session: something it names exactly.
_BACKTICK_RE = re.compile(r"`[^`\n]+`")
_FILE_RE = re.compile(
    r"(?:^|[\w-])\.(?:md|mdc|py|sh|bash|zsh|json|jsonl|yaml|yml|toml|ini|cfg|env|txt|log|lock|plist|xml|"
    r"html|css|js|mjs|cjs|ts|tsx|jsx|dart|arb|swift|kt|kts|java|gradle|rb|go|rs|sql)$",
    re.IGNORECASE,
)
# a dotted ASCII name: a hostname or Name.ext; one-letter abbreviations (e.g, i.e) are not names
_DOTTED_RE = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_-]*(?:\.[A-Za-z0-9_-]+)*\.[A-Za-z][A-Za-z0-9]{0,9}")
_SLASHED_WORDS_RE = re.compile(r"[а-яёА-ЯЁ]+(?:/[а-яёА-ЯЁ]+)+")  # и/или, до/после
_FLAG_RE = re.compile(r"--[A-Za-z][\w-]*")
_IDENT_RE = re.compile(r"[A-Za-z0-9_]{4,}")
_CAMEL_RE = re.compile(r"[a-z][A-Z]")
_TOKEN_EDGES = ".,;:!?()[]{}<>«»\"'…—–"
NOT_CONCRETE_MAX = 200


def _is_dotted_name(token: str) -> bool:
    if not _DOTTED_RE.fullmatch(token):
        return False
    parts = token.split(".")
    return len(parts) > 2 or len(parts[0]) > 1


def is_concrete(lesson: str) -> bool:
    """True when the lesson names something exactly.

    That is a backtick span, a path-like token (with a `/` or a file extension), a
    dotted name (a hostname or Name.ext), a --flag, or an identifier of 4+ characters
    with `_` or an inner capital. A `/` between Russian words (и/или) is no path.
    """
    if _BACKTICK_RE.search(lesson):
        return True
    for raw in lesson.split():
        token = raw.lstrip(_TOKEN_EDGES.replace(".", "")).rstrip(_TOKEN_EDGES)  # a dotfile keeps its dot
        slashed = "/" in token and any(c.isalnum() for c in token) and not _SLASHED_WORDS_RE.fullmatch(token)
        if slashed or _FILE_RE.search(token) or _FLAG_RE.fullmatch(token) or _is_dotted_name(token):
            return True
        for ident in _IDENT_RE.findall(token):
            if any(c.isalpha() for c in ident) and ("_" in ident or _CAMEL_RE.search(ident)):
                return True
    return False


def read_user_rules(cfg: Config) -> str:
    """The user's own rules for every project (~/.claude/CLAUDE.md), "" when there are none to read.

    Every session loads them, so a lesson that says the same again only makes the memory longer.
    """
    try:
        text = (cfg.home / ".claude" / "CLAUDE.md").read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return ""
    return scrub(text.strip())[:USER_RULES_MAX]


def read_project_rules(project_root: Path) -> str:
    """The project's own rules (its CLAUDE.md), "" when there are none to read: loaded by every session there."""
    try:
        text = (Path(project_root) / "CLAUDE.md").read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return ""
    return scrub(text.strip())[:USER_RULES_MAX]


def build_prompt(segment_text: str, project_root: Path, index_text: str,
                 auto_lessons: list[Lesson], rejected: list[str], user_rules: str = "",
                 project_rules: str = "") -> str:
    return load_template("reflect.md").substitute(
        nonce=nonce(),
        project=str(project_root),
        user_rules=_TAGS.defuse(user_rules.strip()) or "—",
        project_rules=_TAGS.defuse(project_rules.strip()) or "—",
        index=_TAGS.defuse(index_text.strip()) or "—",
        auto_lessons=_TAGS.one_per_line(_auto_line(lesson) for lesson in auto_lessons),
        rejected=_TAGS.one_per_line(rejected),
        segment=_TAGS.defuse(segment_text.strip()),
    )


def build_verify_prompt(fields: dict, cited: list[str], segment_text: str = "") -> str:
    """The second look gets what the first one read: the lesson, the passages it quotes and the same part."""
    return load_template("verify.md").substitute(
        nonce=nonce(),
        lesson=_VERIFY_TAGS.defuse("\n".join(
            [f"{name}: {squash(fields.get(name, ''))}" for name in ("cue", "lesson", "why")]
            + ([f"kind: fact", f"recheck: {squash(fields.get('recheck', ''))}"] if fields.get("kind") == "fact" else []))),
        passages=_VERIFY_TAGS.defuse("\n".join(cited).strip()) or "—",
        segment=_VERIFY_TAGS.defuse(segment_text.strip()) or "—",
    )


def entries_of(answer) -> tuple[str, list[dict], list[dict]]:
    """The notes, the entries and the raw episodes of one model answer.

    An answer in the old shape, one verdict at its top level, is one entry; an answer with
    no entries is one NO_ENTRY.
    """
    if not isinstance(answer, dict):
        return "", [{"verdict": None}], []
    notes = model_text(answer.get("notes"))
    episodes = answer.get("episodes") if isinstance(answer.get("episodes"), list) else []
    if "entries" not in answer:
        return notes, [answer], episodes
    listed = answer.get("entries")
    entries = [e for e in listed if isinstance(e, dict)][:MAX_ENTRIES] if isinstance(listed, list) else []
    return notes, entries or [{"verdict": "NO_ENTRY"}], episodes


def _norm(text: str) -> str:
    return squash(text).casefold()


def shown_turns(text: str) -> set[int]:
    """The numbers of the user's messages a rendered part shows."""
    return {int(n) for n in _SHOWN_TURN_RE.findall(text)}


def episodes_of(segment: Segment, raw: list, rules_text: str = "") -> list[dict]:
    """The corrections the model named that the segment really holds, each as the journal keeps it.

    The turn, what it was about, when, the user's own words and whether the message answered the
    agent (`reply`: only those are counted against the user's answers); the model's sentence about it
    (`gist`) when it gave one; and for a broken rule the rule it quoted, when that quote stands
    in `rules_text` (the user's rules, the project's and the memory index).
    """
    turns = {t.n: t for t in segment.turns}
    rules = _norm(rules_text)
    found: dict[int, dict] = {}
    for item in raw:
        if not isinstance(item, dict):
            continue
        m = _REF_RE.fullmatch(model_text(item.get("ref")))
        form = item.get("form")
        if not (m and m.group(1) and int(m.group(2)) in turns and form in FORMS) or int(m.group(2)) in found:
            continue
        turn = turns[int(m.group(2))]
        episode = {"turn": turn.n, "form": form, "ts": turn.ts, "said": scrub(squash(turn.text))[:SAID_MAX],
                   "reply": turn.is_reply}
        if gist := scrub(squash(model_text(item.get("gist"))))[:GIST_MAX]:
            episode["gist"] = gist
        rule = squash(model_text(item.get("rule"))).strip(_QUOTE_EDGES)
        if form == "constraint" and len(rule) >= RULE_MIN and _stands(_norm(rule), rules):
            episode["rule"] = scrub(rule)[:GIST_MAX]
        found[turn.n] = episode
    return [found[n] for n in sorted(found)][:MAX_EPISODES]


@dataclass
class _Refs:
    """What a lesson's quotes came to when checked against the segment."""
    reason: str = ""  # why they do not hold, "" when every one does
    quote: str = ""  # the quote that was not found, for the event
    user_turns: tuple[int, ...] = ()  # the user's messages quoted in the user's own words
    cited: tuple[str, ...] = ()  # the quoted signals and messages, as the model was shown them


def _stands(quote: str, hay: str) -> bool:
    """The quote is in the text; one the model cut with an ellipsis is there piece by piece, in order."""
    if quote in hay:
        return True
    pieces = [piece for piece in _ELLIPSIS_RE.split(quote) if piece]
    if len(pieces) < 2 or any(len(piece) < QUOTE_MIN for piece in pieces):
        return False
    at = 0
    for piece in pieces:
        at = hay.find(piece, at)
        if at < 0:
            return False
        at += len(piece)
    return True


def check_refs(segment: Segment, refs) -> _Refs:
    """Every quote must stand, word for word, in the signal or the message it names.

    What the agent answered to a message is kept as the `before` of the next one, so a quote
    of that answer may name either message.
    """
    if not isinstance(refs, list) or not refs:
        return _Refs(reason="no quotes")
    turns = {t.n: t for t in segment.turns}
    by_seq = {t.seq: t for t in segment.turns}
    answered = {t.n: reply for t, reply in zip(segment.turns, segment.turns[1:])}
    user_turns: list[int] = []
    cited: list[str] = []
    for item in refs:
        if not isinstance(item, dict):
            return _Refs(reason="a reference is not an object")
        ref, quote = model_text(item.get("ref")), _norm(model_text(item.get("quote"))).strip(_QUOTE_EDGES)
        m = _REF_RE.fullmatch(ref)
        if m is None:
            return _Refs(reason=f"unknown reference {ref!r}")
        n = int(m.group(2))
        if m.group(1):
            turn = turns.get(n)
            if turn is None:
                return _Refs(reason=f"no message [u{n}] in the segment")
            hay, shown = (turn.text, turn.before, turn.after), [_render_turn(turn)]
            if len(quote) >= QUOTE_MIN and _stands(quote, _norm(turn.text)) and n not in user_turns:
                user_turns.append(n)
            reply = answered.get(n)
            if reply is not None and not any(_stands(quote, _norm(h)) for h in hay):
                hay, shown = (reply.before,), [*shown, _render_turn(reply)]
        else:
            if not 1 <= n <= len(segment.signals):
                return _Refs(reason=f"no signal [{n}] in the segment")
            s = segment.signals[n - 1]
            hay, shown = (s.text, s.tool_input, s.after, s.before, s.command), [_render_signal(n, s)]
            # a correction is shown twice, as a signal and as the message right before it: the
            # user's words quoted through the signal are the user's words all the same
            said = by_seq.get(s.seq - 1) if s.kind == "user_correction" else None
            if (said is not None and len(quote) >= QUOTE_MIN and _stands(quote, _norm(said.text))
                    and said.n not in user_turns):
                user_turns.append(said.n)
        if len(quote) < QUOTE_MIN or not any(_stands(quote, _norm(h)) for h in hay):
            return _Refs(reason=f"the quote is not in {ref}", quote=quote[:QUOTE_SHOWN_MAX])
        cited.extend(text for text in shown if text not in cited)
    return _Refs(user_turns=tuple(user_turns), cited=tuple(cited))


def valid_trigger(segment: Segment, trigger: str) -> tuple[str, str]:
    """The trigger if it can be trusted, else "" and why not.

    A trigger is a regex over a shell command. It must match a command that failed in this
    segment and must match neither the command that fixed it nor an empty command: only then
    does it tell the failing shape from the working one. Commands are matched as the hook
    matches them, without the bodies of their heredocs.
    """
    if not trigger:
        return "", ""
    if len(trigger) > TRIGGER_MAX:
        return "", "too long"
    try:
        pattern = re.compile(trigger)
    except re.error as exc:
        return "", f"not a regex: {exc}"
    if pattern.search("") or pattern.search("ls"):
        return "", "matches any command"
    failing = [shell_text(s.command) for s in segment.signals if s.kind in _FAILING and s.command]
    if not any(pattern.search(c) for c in failing):
        return "", "matches no failing command of the segment"
    if any(pattern.search(shell_text(s.command)) for s in segment.signals if s.kind == "recovery" and s.command):
        return "", "matches the command that worked"
    return trigger, ""


def valid_error_trigger(segment: Segment, trigger: str) -> tuple[str, str]:
    """The error trigger if it can be trusted, else "" and why not.

    An error trigger is a regex over what a failed shell command printed, for a mistake its
    command does not show. It must match the output of a failure in this segment, no output
    every failure prints, and not what the command that fixed it printed: only then does it
    tell this failure from a run that worked.
    """
    if not trigger:
        return "", ""
    if why_not := guard.check_trigger(trigger, "error"):
        return "", why_not
    pattern = re.compile(trigger)
    if not any(pattern.search(s.text) for s in segment.signals if s.kind in _FAILING and s.command and s.text):
        return "", "matches no failing output of the segment"
    if any(pattern.search(s.text) for s in segment.signals if s.kind == "recovery" and s.text):
        return "", "matches the output that worked"
    return trigger, ""


def valid_path_trigger(segment: Segment, trigger: str) -> tuple[str, str]:
    """The path trigger if it can be trusted, else "" and why not.

    It is a regex over the path of a file. It must match a file the agent wrote to in this
    segment and no file at large: only then does it name the kind of work the lesson is about.
    """
    if not trigger:
        return "", ""
    why_not = guard.check_trigger(trigger, "path")
    if why_not:
        return "", why_not
    if not any(re.search(trigger, path) for path in segment.edited):
        return "", "matches no file the agent edited in the segment"
    return trigger, ""


def _auto_line(lesson: Lesson) -> str:
    """One auto-lesson for the prompt; a live one says since when its line is in the index, a moved
    one since when it is in a skill."""
    if lesson.status == "promoted":
        since = f" [перенесён в скилл {lesson.promoted_at}]"
    else:
        since = f" [в индексе с {_live_since(lesson)}]" if lesson.status == "live" else ""
    return f"{lesson.id}{since}: {lesson.cue} → {lesson.lesson}"


def prompt_lessons(state: State, project_root: Path) -> list[Lesson]:
    """The auto-lessons of a project the model may name: in memory or waiting, and those moved into a
    skill since moves were recorded, whose mistake coming back tells whether the move helped."""
    return [lesson for lesson in state.lessons.values()
            if lesson.project_root == str(project_root)
            and (lesson.status in ("live", "staged", "retired") or _watched(lesson))]


def _watched(lesson: Lesson) -> bool:
    """A lesson moved out of memory into a skill, with no verdict yet: its mistake is counted from the move,
    while its PR waits too; the verdict counts only what came after the rule was there (effect.check)."""
    return lesson.status == "promoted" and bool(lesson.promoted_at) and not lesson.effect


def _live_since(lesson: Lesson) -> str:
    """When the lesson went live; one that went live before this was recorded dates from its creation."""
    return lesson.live_since or lesson.created_at


def apply_verdict(state: State, cfg: Config, segment: Segment, project_root: Path,
                  memory_dir: Path, verdict: dict, now_iso: str,
                  verify: Verify | None = None) -> str:
    """Turn one reflector verdict (one entry of an answer) into memory writes.

    Returns "no_entry", "new_live", "new_staged", "repeat", "violated", "after_promotion", "user_rule_repeat",
    "rejected_cue", "rejected_not_concrete", "rejected_bad_quote", "rejected_unsupported",
    "bad_verdict" or "write_failed". A bad verdict or a refused lesson writes nothing.
    `verify` is the second look at a new lesson that came with quotes: True when the quoted
    passages carry it, False when they do not, None when it could not be asked, and in a
    sentence why.
    """
    kind = verdict.get("verdict") if isinstance(verdict, dict) else None
    if kind not in VERDICTS:
        kind = scrub(str(kind))
        return _bad(state, segment, project_root, kind, f"unknown verdict {kind!r}")
    project_root, memory_dir = Path(project_root), Path(memory_dir)
    _record_conflict(state, segment, project_root, memory_dir, verdict)
    if kind == "NO_ENTRY":
        return "no_entry"
    if kind == "NEW":
        return _apply_new(state, cfg, segment, project_root, memory_dir, verdict, now_iso, verify)
    if kind in ("REPEAT", "LESSON_VIOLATED"):
        moved = state.lessons.get(_text(verdict, "lesson_id"))
        if moved is not None and _watched(moved):
            return _after_promotion(state, segment, project_root, moved, _text(verdict, "evidence"), "model", kind)
    if kind == "REPEAT":
        return _repeat(state, cfg, segment, project_root,
                       _text(verdict, "lesson_id"), _text(verdict, "evidence"), now_iso)
    if kind == "LESSON_VIOLATED":
        return _violated(state, cfg, segment, project_root,
                         _text(verdict, "lesson_id"), _text(verdict, "evidence"), now_iso)
    return _apply_user_rule_repeat(state, segment, project_root, memory_dir, verdict)


def _apply_new(state: State, cfg: Config, segment: Segment, project_root: Path,
               memory_dir: Path, verdict: dict, now_iso: str,
               verify: Verify | None = None) -> str:
    fields = {name: _text(verdict, name) for name in _NEW_FIELDS}
    fields["title"] = squash(fields["title"])[:TITLE_MAX].rstrip()
    fields["hook"] = squash(fields["hook"])[:HOOK_MAX].rstrip()
    missing = [name for name, value in fields.items() if not value]
    if missing:
        return _bad(state, segment, project_root, "NEW", f"missing {', '.join(missing)}")
    if fields["confidence"] not in CONFIDENCES:
        return _bad(state, segment, project_root, "NEW", f"unknown confidence {fields['confidence']!r}")
    kind = verdict.get("kind") if verdict.get("kind") in KINDS else "tool"
    project = str(project_root)
    if segment.agent_id and fields["confidence"] == "high":
        # a subagent has no user: nothing in its segment confirms the better path, so it waits
        fields["confidence"] = "medium"
    # how to work and what holds in a project are both read from the record: neither stands without quotes
    refs = check_refs(segment, verdict.get("refs")) if "refs" in verdict or kind in ("process", "fact") else None
    if refs is not None and refs.reason:
        # the evidence is not in the segment as quoted: nothing else about the lesson can be trusted
        state.event("lesson_bad_quote", project=project, session=segment.session_id,
                    reason=refs.reason, quote=scrub(refs.quote),
                    lesson=scrub(squash(fields["lesson"]))[:NOT_CONCRETE_MAX])
        return "rejected_bad_quote"
    if kind == "process":
        # how to work is read from the user returning the work: without their words there is none
        if segment.agent_id or not refs.user_turns:
            state.event("lesson_bad_quote", project=project, session=segment.session_id,
                        reason="a process lesson quotes no message of the user",
                        lesson=scrub(squash(fields["lesson"]))[:NOT_CONCRETE_MAX])
            return "rejected_bad_quote"
        if fields["confidence"] == "high" and len(refs.user_turns) < 2:
            # one correction is one opinion; the same correction twice is a pattern
            fields["confidence"] = "medium"
    elif kind == "fact" and not _text(verdict, "recheck"):
        # a fact nobody can check again goes stale unseen
        state.event("lesson_not_concrete", project=project, session=segment.session_id,
                    reason="a fact says no way to check it again",
                    lesson=scrub(squash(fields["lesson"]))[:NOT_CONCRETE_MAX])
        return "rejected_not_concrete"
    if kind != "process" and not is_concrete(fields["lesson"]):
        # a well-formed verdict the quality bar turned down: an outcome, not an error
        state.event("lesson_not_concrete", project=project, session=segment.session_id,
                    lesson=scrub(squash(fields["lesson"]))[:NOT_CONCRETE_MAX])
        return "rejected_not_concrete"
    if normalize_cue(fields["cue"]) in state.rejected_cues.get(project, []):
        state.event("rejected_cue", project=project, session=segment.session_id, cue=fields["cue"])
        return "rejected_cue"
    recheck = squash(_text(verdict, "recheck"))[:RECHECK_MAX] if kind == "fact" else ""
    if verify is not None and refs is not None:
        supported, reason = verify({**fields, "kind": kind, "recheck": recheck}, list(refs.cited))
        if supported is False:
            state.event("lesson_unsupported", project=project, session=segment.session_id,
                        reason=scrub(squash(reason))[:NOTES_LOG_MAX], cue=scrub(squash(fields["cue"])),
                        lesson=scrub(squash(fields["lesson"]))[:NOT_CONCRETE_MAX])
            return "rejected_unsupported"
        if supported is None and fields["confidence"] == "high":
            fields["confidence"] = "medium"  # nobody looked twice: it waits for a repeat
    trigger = error_trigger = ""
    if kind == "tool":
        trigger, why_not = valid_trigger(segment, _text(verdict, "trigger"))
        if why_not:
            state.event("trigger_dropped", project=project, session=segment.session_id, reason=why_not)
        error_trigger, why_not = valid_error_trigger(segment, _text(verdict, "error_trigger"))
        if why_not:
            state.event("trigger_dropped", project=project, session=segment.session_id, on="error", reason=why_not)
    path_trigger, why_not = valid_path_trigger(segment, _text(verdict, "path_trigger"))
    if why_not:
        state.event("trigger_dropped", project=project, session=segment.session_id, on="path", reason=why_not)
    lesson_id = hashlib.sha1((project + fields["cue"] + now_iso).encode("utf-8")).hexdigest()[:8]
    existing = state.lessons.get(lesson_id)
    if existing is not None:
        if existing.project_root == project and normalize_cue(existing.cue) == normalize_cue(fields["cue"]):
            return _repeat(state, cfg, segment, project_root, lesson_id, fields["evidence"], now_iso)
        return _bad(state, segment, project_root, "NEW", f"lesson id {lesson_id} already exists")

    lesson = Lesson(
        id=lesson_id,
        project_root=project,
        memory_dir=str(memory_dir),
        file=lesson_filename(lesson_id),
        title=fields["title"],
        hook=fields["hook"],
        cue=fields["cue"],
        lesson=fields["lesson"],
        why=fields["why"],
        confidence=fields["confidence"],
        repeats=1,
        evidence=[_evidence(segment, fields["evidence"])],
        status="staged",
        sha256="",
        created_at=now_iso,
        updated_at=now_iso,
        kind=kind,
        form=verdict.get("form") if verdict.get("form") in FORMS else "",
        trigger=trigger,
        path_trigger=path_trigger,
        error_trigger=error_trigger,
        recheck=recheck,
    )
    try:
        lesson.sha256 = write_lesson(memory_dir, lesson)
    except _WRITE_ERRORS as exc:
        return _write_failed(state, segment, lesson, exc)
    state.lessons[lesson_id] = lesson
    outcome = "new_staged"
    if lesson.confidence == "high":
        try:
            if go_live_if_room(cfg, lesson, now_iso, state):
                outcome = "new_live"
        except _WRITE_ERRORS as exc:
            outcome = _write_failed(state, segment, lesson, exc)
    state.event("lesson_new", project=project, session=segment.session_id,
                lesson_id=lesson_id, status=lesson.status, title=lesson.title)
    cursor_index.sync(cfg, project_root, memory_dir)
    return outcome


def _checked_lesson(state: State, segment: Segment, project_root: Path, kind: str, lesson_id: str,
                    evidence: str, statuses: tuple[str, ...]) -> Lesson | str:
    """The lesson a REPEAT or LESSON_VIOLATED names, or "bad_verdict" with the reason recorded."""
    lesson = state.lessons.get(lesson_id)
    if lesson is None or lesson.project_root != str(project_root):
        return _bad(state, segment, project_root, kind, f"no lesson {lesson_id!r} in this project")
    if lesson.status not in statuses:
        return _bad(state, segment, project_root, kind, f"lesson {lesson_id} is {lesson.status}")
    if not evidence:
        return _bad(state, segment, project_root, kind, "missing evidence")
    try:
        on_disk = (Path(lesson.memory_dir) / lesson.file).read_bytes()
    except OSError as exc:
        return _bad(state, segment, project_root, kind, f"lesson file unreadable: {exc}")
    if hashlib.sha256(on_disk).hexdigest() != lesson.sha256:
        lesson.status = "user_owned"
        state.event("lesson_user_owned", project=lesson.project_root, session=segment.session_id,
                    lesson_id=lesson_id, reason=f"the user edited the lesson; {kind} not applied")
        return "bad_verdict"
    return lesson


def _after_promotion(state: State, segment: Segment, project_root: Path, lesson: Lesson, evidence: str,
                     how: str, kind: str) -> str:
    """The mistake of a lesson moved into a skill, made again: kept for the move's verdict, memory untouched.

    It counts once a session, and only for a session that started after the move: an earlier one
    had the lesson in its memory, not in the skill.
    """
    if lesson.project_root != str(project_root):
        return _bad(state, segment, project_root, kind, f"no lesson {lesson.id!r} in this project")
    if not evidence:
        return _bad(state, segment, project_root, kind, "missing evidence")
    started, moved = parse_ts(segment.session_start or segment.first_ts), parse_ts(lesson.promoted_at)
    if started is None or moved is None or started < moved:
        # it had the lesson in memory, not in the skill: it says nothing about the move, and is no error
        if not any(e.get("kind") == "after_promotion_skipped" and e.get("session") == segment.session_id
                   and e.get("lesson_id") == lesson.id for e in state.events):
            state.event("after_promotion_skipped", project=lesson.project_root, session=segment.session_id,
                        lesson_id=lesson.id, reason="the session started before the move")
        return "no_entry"
    counted = segment.session_id not in {a.get("session") for a in lesson.after}
    if counted:
        # when it started: a session begun before a PR's merge never had the skill (effect counts from the merge)
        lesson.after.append({**_evidence(segment, evidence), "how": how,
                             "started": segment.session_start or segment.first_ts})
    state.event("lesson_after_promotion", project=lesson.project_root, session=segment.session_id,
                lesson_id=lesson.id, target=lesson.promoted_to, how=how, counted=counted)
    return "after_promotion"


def _new_session(lesson: Lesson, segment: Segment) -> bool:
    """A repeat or a violation counts once per session: the same session saying it twice proves nothing new."""
    return segment.session_id not in {e.get("session") for e in lesson.evidence}


def _not_a_violation(lesson: Lesson, segment: Segment) -> str:
    """Why the segment cannot have broken the live lesson, or "" when it provably had it in its index."""
    if segment.agent_id:
        return "a subagent may not have loaded the index"
    started = parse_ts(segment.session_start or segment.first_ts)
    if started is None:
        return "the session start is unknown"
    live = parse_ts(_live_since(lesson))
    if live is None or started < live:
        return "the session started before the lesson went live"
    return ""


def _violated(state: State, cfg: Config, segment: Segment, project_root: Path,
              lesson_id: str, evidence: str, now_iso: str, after_showing: bool = False) -> str:
    """The agent did what a live lesson forbids, in a session that had the lesson in its index."""
    lesson = state.lessons.get(lesson_id)
    if lesson is not None and lesson.status == "live":
        reason = _not_a_violation(lesson, segment)
        if reason:  # the agent never had the lesson before it: it would have helped, which is a REPEAT
            state.event("violation_downgraded", project=str(project_root), session=segment.session_id,
                        lesson_id=lesson_id, reason=reason)
            return _repeat(state, cfg, segment, project_root, lesson_id, evidence, now_iso)
    checked = _checked_lesson(state, segment, project_root, "LESSON_VIOLATED", lesson_id, evidence, ("live",))
    if not isinstance(checked, Lesson):
        return checked
    lesson = checked
    new_session = _new_session(lesson, segment)
    updated = replace(lesson, violations=lesson.violations + int(new_session), last_violated_at=now_iso,
                      evidence=[*lesson.evidence, {**_evidence(segment, evidence), "violated": True,
                                                   **({"after_showing": True} if after_showing else {})}],
                      updated_at=now_iso)
    try:
        updated.sha256 = write_lesson(Path(lesson.memory_dir), updated)
    except _WRITE_ERRORS as exc:
        return _write_failed(state, segment, lesson, exc)
    state.lessons[lesson_id] = updated
    state.event("lesson_violated", project=updated.project_root, session=segment.session_id,
                lesson_id=lesson_id, violations=updated.violations, counted=new_session)
    cursor_index.sync(cfg, project_root, Path(lesson.memory_dir))
    return "violated"


def _repeat(state: State, cfg: Config, segment: Segment, project_root: Path,
            lesson_id: str, evidence: str, now_iso: str) -> str:
    checked = _checked_lesson(state, segment, project_root, "REPEAT", lesson_id, evidence,
                              ("live", "staged", "retired"))
    if not isinstance(checked, Lesson):
        return checked
    lesson = checked
    memory_dir = Path(lesson.memory_dir)
    new_session = _new_session(lesson, segment)
    # a lesson the user retired for lack of use is needed again once its mistake comes back
    status = "staged" if lesson.status == "retired" and new_session else lesson.status
    updated = replace(lesson, repeats=lesson.repeats + int(new_session), status=status,
                      evidence=[*lesson.evidence, _evidence(segment, evidence)], updated_at=now_iso)
    try:
        updated.sha256 = write_lesson(memory_dir, updated)
    except _WRITE_ERRORS as exc:
        return _write_failed(state, segment, lesson, exc)
    state.lessons[lesson_id] = updated
    outcome = "repeat"
    if new_session and updated.status == "staged":
        try:
            go_live_if_room(cfg, updated, now_iso, state)
        except _WRITE_ERRORS as exc:
            outcome = _write_failed(state, segment, updated, exc)
    state.event("lesson_repeat", project=updated.project_root, session=segment.session_id,
                lesson_id=lesson_id, repeats=updated.repeats, counted=new_session, status=updated.status)
    cursor_index.sync(cfg, project_root, memory_dir)
    return outcome


def _apply_user_rule_repeat(state: State, segment: Segment, project_root: Path,
                            memory_dir: Path, verdict: dict) -> str:
    name = _text(verdict, "memory_file")
    path = _user_entry(state, memory_dir, name)
    if path is None:
        return _bad(state, segment, project_root, "USER_RULE_REPEAT", f"{name!r} is not a user entry here")
    evidence = _text(verdict, "evidence")
    if not evidence:
        return _bad(state, segment, project_root, "USER_RULE_REPEAT", "missing evidence")
    state.user_rule_repeats.setdefault(str(path), []).append(_evidence(segment, evidence))
    state.event("user_rule_repeat", project=str(project_root), session=segment.session_id, file=str(path))
    return "user_rule_repeat"


def _record_conflict(state: State, segment: Segment, project_root: Path,
                     memory_dir: Path, verdict: dict) -> None:
    """Note a user entry the session contradicted; the entry itself is never touched."""
    path = _user_entry(state, memory_dir, _text(verdict, "conflict_file"))
    if path is not None:
        state.event("user_rule_conflict", project=str(project_root), session=segment.session_id,
                    file=str(path), notes=squash(_text(verdict, "notes"))[:CONFLICT_NOTES_MAX])


def _user_entry(state: State, memory_dir: Path, name: str) -> Path | None:
    """The path of the user entry `name` in memory_dir, or None if it is not one."""
    if not name.endswith(".md") or Path(name).name != name or name == "MEMORY.md":
        return None
    path = memory_dir / name
    if path.is_file() and (is_user_entry(path) or _taken_over_by_user(state, memory_dir, name)):
        return path
    return None


def _taken_over_by_user(state: State, memory_dir: Path, name: str) -> bool:
    """An auto-lesson the user edited is theirs now and counts as a user entry."""
    return any(
        lesson.status == "user_owned" and lesson.file == name and lesson.memory_dir == str(memory_dir)
        for lesson in state.lessons.values()
    )


def _log_verdict(segment: Segment, verdict, outcome: str) -> None:
    """One line per model answer, so a quiet NO_ENTRY can be read back later."""
    fields = verdict if isinstance(verdict, dict) else {}
    agent = f" {segment.agent_id}" if segment.agent_id else ""
    # scrubbed again once the lines are joined: a secret split by a newline only matches then
    log.info("verdict %s → %s · session %s%s · %s", scrub(str(fields.get("verdict"))), outcome,
             segment.session_id, agent, scrub(squash(model_text(fields.get("notes"))))[:NOTES_LOG_MAX])


def _verdict_record(segment: Segment, path: Path, project_root: Path, title: str | None, verdict,
                    outcome: str, now_iso: str) -> dict:
    """What the journal keeps of one model answer: who it was about, what it said and what came of it."""
    fields = verdict if isinstance(verdict, dict) else {}
    return {
        "ts": now_iso, "session": segment.session_id, "agent": segment.agent_id,
        "transcript": str(path), "project": str(project_root), "title": title,
        "verdict": model_text(fields.get("verdict")) or None, "outcome": outcome,
        "lesson_id": model_text(fields.get("lesson_id")) or None,
        "file": model_text(fields.get("memory_file")) or model_text(fields.get("conflict_file")) or None,
        # scrubbed again once the lines are joined: a secret split by a newline only matches then
        "notes": scrub(squash(model_text(fields.get("notes"))))[:NOTES_LOG_MAX],
    }


def _text(verdict: dict, name: str) -> str:
    return model_text(verdict.get(name))


def _evidence(segment: Segment, text: str) -> dict:
    return {"session": segment.session_id, "ts": segment.last_ts, "text": text}


def _bad(state: State, segment: Segment, project_root: Path, kind: str, reason: str) -> str:
    state.event("bad_verdict", project=str(project_root), session=segment.session_id,
                verdict=kind, reason=reason)
    return "bad_verdict"


def _write_failed(state: State, segment: Segment, lesson: Lesson, exc: Exception) -> str:
    state.event("memory_write_failed", project=lesson.project_root, session=segment.session_id,
                lesson_id=lesson.id, file=lesson.file, error=str(exc))
    return "write_failed"


def project_for(cfg: Config, cwd: str | None, noted: str | None = None) -> Path | None:
    """The project a session belongs to, or None for scratch dirs and Claude's own tree.

    Git answers for the folder the session started in. When that folder is gone (a worktree
    removed before the night), `noted` answers: the root the session-start hook wrote down.
    """
    root = resolve_project_root(cwd)
    if root is None and noted:
        root = Path(noted)
    if root is None or any(Path(root).is_relative_to(d) for d in cfg.no_project_roots):
        return None
    return root


@dataclass
class _Candidate:
    """One unread segment of one transcript, held between reading it and the model call."""
    path: Path
    info: dict
    size: int  # of the file when it was read
    segment: Segment
    cwd: str | None
    project_root: Path | None
    weight: float
    noted: str | None = None  # the root the session-start hook wrote down, for a folder gone by now


def _weight(segment: Segment, age_seconds: float) -> float:
    """What the segment's signals weigh, plus half a point per day of age so nothing starves.

    Each kind counts at most WEIGHT_REPEATS_MAX times: one command failing in a loop is one problem.
    """
    kinds = Counter(s.kind for s in segment.signals)
    signals = sum(SIGNAL_WEIGHTS.get(kind, 1) * min(n, WEIGHT_REPEATS_MAX) for kind, n in kinds.items())
    replies = REPLY_WEIGHT * min(len(segment.replies), WEIGHT_REPEATS_MAX)
    return signals + replies + AGING_PER_DAY * max(0.0, age_seconds) / 86400


def worth_asking(segment: Segment) -> bool:
    """Whether the model can learn anything from the segment.

    A session of the user: any signal, or an answer of the user to the agent, since a
    correction needs no failed tool. A subagent has no user and fails tools all day while it
    looks around: only a fix that worked or a hook that stopped it is worth a call.
    """
    if segment.agent_id:
        return any(s.kind in _SUBAGENT_SIGNALS for s in segment.signals)
    return bool(segment.signals) or bool(segment.replies)


def trigger_hits(state: State, cfg: Config, segment: Segment, project_root: Path, now_iso: str) -> list[dict]:
    """Repeats and violations a lesson's trigger proves without the model.

    A failing shell command that matches the trigger of a lesson of this project, or whose
    output matches its error trigger, is the lesson's situation again: a violation when the
    session had the lesson in its index, else a repeat. Each session counts once, so a segment
    of a session already counted adds nothing, except that a violation by a command run after
    the hook had shown the lesson marks the session as one that read it and went on.
    """
    hits: list[dict] = []
    for lesson in list(state.lessons.values()):
        moved = _watched(lesson)
        if (not (lesson.trigger or lesson.error_trigger) or lesson.project_root != str(project_root)
                or not (moved or lesson.status in ("live", "staged"))):
            continue
        matches = _trigger_matches(lesson, segment)
        if not matches:
            continue
        if moved:
            if segment.session_id in {a.get("session") for a in lesson.after}:
                continue
            evidence, what, _ = matches[0]
            outcome = _after_promotion(state, segment, project_root, lesson, evidence, "trigger", "REPEAT")
            if outcome != "after_promotion":
                continue
        else:
            shown = [m for m in matches if _after_showing(lesson, segment, m[2])]
            evidence, what, _ = (shown or matches)[0]
            if not _new_session(lesson, segment):
                if shown and lesson.status == "live":
                    _mark_after_showing(lesson, segment)
                continue
            if lesson.status == "live":
                outcome = _violated(state, cfg, segment, project_root, lesson.id, evidence, now_iso, bool(shown))
            else:
                outcome = _repeat(state, cfg, segment, project_root, lesson.id, evidence, now_iso)
        hits.append({"verdict": "LESSON_VIOLATED" if outcome == "violated" else "REPEAT",
                     "lesson_id": lesson.id, "outcome": outcome, "notes": what + evidence})
    return hits


def _after_showing(lesson: Lesson, segment: Segment, sig: Signal) -> bool:
    """True when the failing command matched the lesson's command trigger and the hook had shown the
    lesson in this session before the agent made the call: the hook's text reaches the agent with the
    result of the command it fired on, so a showing at least a second before the call came back to the
    agent first. Commands called together fire together and read nothing of each other's showing.

    A signal without its call time (a transcript read before it was kept) needs two showings before
    its result instead.
    """
    if not lesson.trigger or not sig.command:
        return False
    try:
        if not re.search(lesson.trigger, shell_text(sig.command)):
            return False
    except re.error:
        return False
    times = [ts for ts in map(parse_ts, lesson.fire_times.get(segment.session_id, [])) if ts is not None]
    call = parse_ts(sig.called)
    if call is not None:
        return any(ts <= call - _SHOWN_BEFORE for ts in times)
    at = parse_ts(sig.ts)
    return at is not None and sum(1 for ts in times if ts < at) >= 2


# the hook logs its showings to the second: one this far before a call surely came before it
_SHOWN_BEFORE = timedelta(seconds=1)


def _mark_after_showing(lesson: Lesson, segment: Segment) -> None:
    """A session already counted turns out to have broken the lesson after the hook showed it too."""
    for entry in lesson.evidence:
        if isinstance(entry, dict) and entry.get("session") == segment.session_id and entry.get("violated") is True:
            entry["after_showing"] = True


def _trigger_matches(lesson: Lesson, segment: Segment) -> list[tuple[str, str, Signal]]:
    """Every failed shell command a lesson's trigger or error trigger matches: (evidence, note, signal).

    The trigger is read against the command, the error trigger against what it printed.
    """
    checks = []
    for trigger, on_output, note in (
            (lesson.trigger, False, i18n.t("the lesson's command pattern matched, the model was not called: ",
                                           "совпал шаблон команды урока, модель не вызывалась: ")),
            (lesson.error_trigger, True, i18n.t("the lesson's error pattern matched, the model was not called: ",
                                                "совпал шаблон ошибки урока, модель не вызывалась: "))):
        if not trigger:
            continue
        try:
            checks.append((re.compile(trigger), on_output, note))
        except re.error:
            continue
    found = []
    for n, sig in enumerate(segment.signals, 1):
        if sig.kind not in _FAILING or not sig.command:
            continue
        for pattern, on_output, note in checks:
            text = sig.text if on_output else shell_text(sig.command)
            m = pattern.search(text) if text else None
            if m is None:
                continue
            if not on_output:
                found.append((f"[{n}] {squash(text)[:160]}", note, sig))
            else:
                # the command, and the output where the pattern matched: a long log's error is often at its end
                shown = squash(text[max(0, m.start() - 40):m.end() + 40])[:120]
                found.append((f"[{n}] {squash(shell_text(sig.command))[:80]} → …{shown}…", note, sig))
            break
    return found


def _set_offset(state: State, path: Path, record: dict) -> None:
    """Move a transcript's reading offset; its health bookkeeping goes once reading caught up."""
    key = str(path)
    state.sessions[key] = record
    if state.health_offsets.get(key, 0) <= record["offset"]:
        state.health_offsets.pop(key, None)


def _session_file(path: Path) -> Path:
    """The transcript of the session a transcript belongs to: a subagent's parent, else itself."""
    if is_subagent(path):
        # <project dir>/<session-id>/subagents/agent-*.jsonl sits beside <project dir>/<session-id>.jsonl
        return path.parents[2] / f"{parent_session_id(path)}.jsonl"
    return path


def noted_root(roots: dict[str, str], path: Path) -> str | None:
    """What the session-start hook noted for the session of this transcript (subagents: their parent's)."""
    return roots.get(str(_session_file(path)))


def _start_cwd(state: State, path: Path) -> str | None:
    """Where a transcript's session started. A subagent works for its parent's project even
    when it runs in the session's scratchpad, so its cwd is the parent's; its own is the fallback."""
    if is_subagent(path):
        parent = _session_file(path)
        cwd = state.sessions.get(str(parent), {}).get("cwd")
        if not cwd and parent.is_file():
            cwd = session_cwd(parent)
        if cwd:
            return cwd
    return session_cwd(path)


def _advance(state: State, cand: _Candidate, dry_run: bool) -> None:
    if not dry_run:
        _set_offset(state, cand.path, {**cand.info, "offset": cand.segment.end, "cwd": cand.cwd})


def _finish(state: State, cfg: Config, cand: _Candidate, result: dict, now_iso: str, dry_run: bool) -> None:
    """The segment is done with: count what its failing commands prove by trigger, then move past it."""
    if not dry_run and any(lesson.trigger or lesson.error_trigger for lesson in state.lessons.values()):
        try:
            root = cand.project_root or project_for(cfg, cand.cwd, cand.noted)
            if root is not None:
                for hit in trigger_hits(state, cfg, cand.segment, root, now_iso):
                    result["outcomes"][hit["outcome"]] += 1
                    result["verdicts"].append({
                        "ts": now_iso, "session": cand.segment.session_id, "agent": cand.segment.agent_id,
                        "transcript": str(cand.path), "project": str(root), "title": None, "file": None,
                        **hit})
        except Exception as exc:  # a bug here must not keep the segment for another night
            state.event("trigger_crashed", transcript=str(cand.path), error=error_text(exc))
            result["errors"] += 1
    _advance(state, cand, dry_run)


def _guard(state: State, result: dict, kind: str, work: Callable[[], object]) -> None:
    """The guard hook's files are a side channel: a failure there never stops the run, but it is
    an error of the night. A hook nobody feeds shows stale rules, or none, and says nothing."""
    try:
        work()
    except Exception as exc:
        state.event(kind, error=error_text(exc))
        result["errors"] += 1


def _verifier(cfg: Config, llm, result: dict, segment_text: str) -> Verify:
    """The second look at a new lesson: one more model call over the part the lesson was read from."""
    def verify(fields: dict, cited: list[str]) -> tuple[bool | None, str]:
        result["verify_calls"] += 1
        try:
            answer = llm(cfg, cfg.verify_model, build_verify_prompt(fields, cited, segment_text), VERIFY_SCHEMA,
                         effort=cfg.reflect_effort)
        except LlmError as exc:
            log.warning("the lesson was not verified: %s", error_text(exc))
            return None, error_text(exc)
        if not isinstance(answer, dict) or not isinstance(answer.get("supported"), bool):
            return None, "no answer"
        reason = model_text(answer.get("notes"))
        log.info("verify → %s · %s", answer["supported"], scrub(squash(reason))[:NOTES_LOG_MAX])
        return answer["supported"], reason
    return verify


def _transcript_failed(state: State, result: dict, path: Path, info: dict, size: int,
                       exc: Exception, dry_run: bool) -> None:
    """A transcript that trips a bug is skipped to its end, so it never stalls later nights."""
    result["errors"] += 1
    if not dry_run:
        state.event("transcript_failed", transcript=str(path), error=error_text(exc))
        _set_offset(state, path, {**info, "offset": size})


def _merge_health(state: State, cfg: Config, path: Path, segment: Segment, added: Counter) -> None:
    """Count a segment's health events and lesson reads exactly once, whether or not its offset advances.

    Every segment read in the run is merged, before any model call and whatever the cap.
    A segment left unread (cap, LLM error) is read again next night, so the byte offset
    its health was merged up to is kept in `state.health_offsets` and events at or before
    it are not counted again.
    """
    key = str(path)
    done = state.health_offsets.get(key, 0)
    if done > segment.end:  # the file shrank and was read again from its start
        done = 0
    added.update(health.merge(state.health, [e for e in segment.health if e.end > done], cfg.health_ignore))
    usage.merge(state.lessons, [r for r in segment.reads if r.end > done])
    state.health_offsets[key] = max(done, segment.end)


def _notify_health(cfg: Config, state: State, added: Counter, now: datetime, now_iso: str,
                   notify: Callable[[str], None] | None) -> None:
    """One notice per run for keys that fired often tonight; a failed notice never breaks the run."""
    keys = health.due(state.health, added, cfg.health_notify_threshold, now)
    if not keys or notify is None:
        return
    message = health.notify_message(state.health, keys)
    try:
        notify(message)
    except Exception as exc:
        log.warning("health notice failed: %s", error_text(exc))
        return
    log.info("health notice: %s", message)
    for key in keys:
        state.health[key]["notified_at"] = now_iso


def run(cfg: Config, state: State, *, now: datetime, llm=run_structured,
        notify: Callable[[str], None] | None = None,
        max_segments: int | None = None, dry_run: bool = False) -> dict:
    """One nightly pass over new transcript segments. The caller saves `state`."""
    cap = cfg.max_segments_per_run if max_segments is None else max_segments
    result = {"segments_seen": 0, "llm_calls": 0, "outcomes": Counter(), "errors": 0,
              "no_signals": 0,  # segments read that held nothing to ask the model about
              "deferred": 0,  # candidates the cap or three model errors kept from the model
              "verify_calls": 0,  # second looks at new lessons, on top of llm_calls
              "episodes": Counter(),  # the user's corrections the model found, by what they were about
              "replies": 0,  # the user's answers to the agent in the segments the model read
              "detector": DETECTOR,
              "verdicts": []}  # one record per entry of a model answer, for the journal
    outcomes: Counter = result["outcomes"]
    if dry_run:
        result["would_call"] = []  # rendered size of each segment that would go to the model
    now_iso = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    take_spent()  # whatever was metered before this run is not its cost

    if not autodream_disabled(cfg.claude_settings):
        state.event("autodream_not_disabled", settings=str(cfg.claude_settings))
        return result
    transcripts = list_transcripts(cfg.projects_dir, cfg.own_project_prefixes)
    if state.installed_at is None:
        if not dry_run:
            snapshot_baseline(state, transcripts, now_iso)
        return result
    if not dry_run and state.reads_since is None:
        state.reads_since = now_iso  # lesson reads are counted from this run on
    if not dry_run:
        _guard(state, result, "guard_fires_unread", lambda: guard.merge_fires(cfg, state))
        session_roots.rotate(cfg)
    roots = session_roots.load(cfg)
    if not state.subagent_baseline_done:
        # the first run that reads subagents: their history up to now is no backlog
        untracked = [t for t in transcripts if is_subagent(t) and str(t) not in state.sessions]
        if dry_run:
            untracked_set = {str(t) for t in untracked}
            transcripts = [t for t in transcripts if str(t) not in untracked_set]
        else:
            snapshot_baseline(state, untracked, now_iso)
    if not dry_run:
        changed = set(sync_ownership(state))
        for project_root, memory_dir in managed_projects(state):
            if rebalance(state, cfg, project_root, memory_dir, now_iso):
                changed.add((project_root, memory_dir))
        for project_root, memory_dir in sorted(changed):
            cursor_index.sync(cfg, Path(project_root), Path(memory_dir))

    # Reading every unread segment is cheap and deterministic, so it all happens first;
    # only then do the heaviest segments go to the model, up to the cap.
    candidates: list[_Candidate] = []
    health_added: Counter = Counter()
    for path in transcripts:
        try:
            stat = path.stat()
        except OSError:
            continue
        info = state.sessions.get(str(path), {})
        offset = info.get("offset", 0)
        if now.timestamp() - stat.st_mtime < cfg.quiet_seconds or stat.st_size <= offset:
            continue
        # Everything up to the model call is guarded per transcript: one file that trips
        # a bug is skipped to its end instead of stalling every later night.
        try:
            segment = read_segment(path, offset)
            if not dry_run:
                _merge_health(state, cfg, path, segment, health_added)
            # a session cds around: its project is where it started, for every segment
            cwd = info.get("cwd") or _start_cwd(state, path)
            result["segments_seen"] += 1
            cand = _Candidate(path, info, stat.st_size, segment, cwd, None,
                              _weight(segment, now.timestamp() - stat.st_mtime), noted_root(roots, path))
            if not worth_asking(segment):
                result["no_signals"] += 1
                _finish(state, cfg, cand, result, now_iso, dry_run)
                continue
            cand.project_root = project_for(cfg, cwd, cand.noted)
            if cand.project_root is None:
                outcomes["no_project"] += 1
                if not dry_run:
                    state.event("no_project", transcript=str(path), cwd=cwd)
                _advance(state, cand, dry_run)
                continue
        except OSError as exc:  # the transcript cannot be read now: retry next night
            state.event("segment_unreadable", transcript=str(path), error=error_text(exc))
            continue
        except Exception as exc:
            _transcript_failed(state, result, path, info, stat.st_size, exc, dry_run)
            continue
        candidates.append(cand)

    if not dry_run:
        _notify_health(cfg, state, health_added, now, now_iso, notify)

    candidates.sort(key=lambda c: -c.weight)  # stable: equal weights keep the oldest first
    llm_errors = 0
    user_rules = read_user_rules(cfg)
    titles: dict[Path, str | None] = {}  # read once per transcript, and only for what reaches the model
    for i, cand in enumerate(candidates):
        calls = len(result["would_call"]) if dry_run else result["llm_calls"]
        if llm_errors >= MAX_LLM_ERRORS or calls >= cap:
            result["deferred"] = len(candidates) - i
            break  # the rest keep their offsets for the next night
        segment, project_root, path = cand.segment, cand.project_root, cand.path
        try:
            texts = segment.render_parts(cfg.segment_max_chars, cfg.segment_max_parts)
            if dry_run:
                result["would_call"].extend(len(text) for text in texts)
                continue
            memory_dir = memory_dir_for(cfg.projects_dir, project_root)
            try:
                index_text = read_index(memory_dir)
            except MemoryIndexError as exc:
                index_text = ""
                state.event("index_unreadable", project=str(project_root), error=error_text(exc))
            project_rules = read_project_rules(project_root)
        except OSError as exc:
            state.event("segment_unreadable", transcript=str(path), error=error_text(exc))
            continue
        except Exception as exc:
            _transcript_failed(state, result, path, cand.info, cand.size, exc, dry_run)
            continue
        unread = False  # the first part got no answer: the whole segment waits for the next night
        for part, text in enumerate(texts):
            try:
                # read again for every part: an earlier part may have added a lesson
                auto = prompt_lessons(state, project_root)
                prompt = build_prompt(text, project_root, index_text, auto,
                                      state.rejected_cues.get(str(project_root), []), user_rules, project_rules)
            except Exception as exc:
                _transcript_failed(state, result, path, cand.info, cand.size, exc, dry_run)
                unread = True
                break
            result["llm_calls"] += 1
            if path not in titles:
                try:
                    titles[path] = session_title(path)
                except Exception as exc:  # the title is for the report: a bug in reading it must not cost the call
                    log.warning("session title not read for %s: %s", path.name, error_text(exc))
                    titles[path] = None
            try:
                answer = llm(cfg, cfg.reflect_model, prompt, REFLECT_SCHEMA, effort=cfg.reflect_effort)
            except LlmError as exc:
                state.event("llm_error", transcript=str(path), error=error_text(exc))
                result["errors"] += 1
                outcomes["llm_error"] += 1
                record = _verdict_record(segment, path, project_root, titles[path], None, "llm_error", now_iso)
                result["verdicts"].append({**record, "notes": error_text(exc)})
                llm_errors += 1
                unread = part == 0  # a later part lost: what the earlier ones gave stays, the rest is skipped
                break
            notes, entries, raw_episodes = entries_of(answer)
            verify = _verifier(cfg, llm, result, text)
            episodes = episodes_of(segment, raw_episodes, "\n".join((user_rules, project_rules, index_text)))
            result["episodes"].update(e["form"] for e in episodes)
            # the answers this part shows: one a capped segment left out was never the model's to judge
            shown = shown_turns(text)
            replies = sum(1 for turn in segment.replies if turn.n in shown)
            result["replies"] += replies
            for n, entry in enumerate(entries):
                try:
                    outcome = apply_verdict(state, cfg, segment, project_root, memory_dir, entry, now_iso, verify)
                except Exception as exc:  # a deterministic crash must not block later nights
                    state.event("apply_crashed", transcript=str(path), error=error_text(exc))
                    result["errors"] += 1
                    outcome = "apply_crashed"
                outcomes[outcome] += 1
                # the answer's notes go with its first entry; a later one is told by its own `why` or evidence
                told = {**entry, "notes": notes if n == 0 else (entry.get("why") or entry.get("evidence") or notes)}
                _log_verdict(segment, told, outcome)
                record = _verdict_record(segment, path, project_root, titles[path], told, outcome, now_iso)
                if n == 0 and episodes:
                    record["episodes"] = episodes
                if n == 0:
                    record["replies"] = replies  # what the corrections are counted against
                result["verdicts"].append(record)
        if not unread:
            _finish(state, cfg, cand, result, now_iso, dry_run)

    if spent := take_spent():
        result["spent"] = spent
    if not dry_run:
        # the hook's rules follow tonight's lessons: a new trigger works from the next command on
        _guard(state, result, "guard_rules_unwritten", lambda: guard.sync(cfg, state))
        state.consecutive_failures = state.consecutive_failures + 1 if result["errors"] else 0
        if state.consecutive_failures >= NOTIFY_AFTER_FAILURES and notify is not None:
            n = state.consecutive_failures
            notify(i18n.t(f"agent-reflect: the nightly review has failed {n} times in a row, see logs/reflect.log",
                          f"agent-reflect: ночной разбор падает {n} раз подряд, см. logs/reflect.log"))
    return result

