from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path

from agent_reflect import health as _health
from agent_reflect.health import HealthEvent
from agent_reflect.i18n import t
from agent_reflect.promptkit import squash
from agent_reflect.scrub import scrub
from agent_reflect.usage import LessonRead, lesson_ids

SIGNAL_KINDS = ("tool_error", "hook_block", "user_rejection", "user_interrupt", "user_correction",
                "silent_failure", "recovery")
# How much a signal weighs when the nightly cap picks which segments go to the model first.
SIGNAL_WEIGHTS = {"hook_block": 3, "user_correction": 3, "user_rejection": 2, "user_interrupt": 1,
                  "tool_error": 1, "silent_failure": 1, "recovery": 2}
# Failing results a streak needs before a later fix of the same tool counts as a recovery.
_FAILING = ("tool_error", "silent_failure")
RECOVERY_MIN_FAILURES = 2
# A shell result the CLI did not flag as an error, but whose output shows one. A closed,
# line-anchored set: the bare words error/failed/timeout are everywhere in normal output.
_SHELL_TOOLS = ("Bash", "PowerShell")
_SILENT_SCAN_MAX = 6000
_SILENT_RE = re.compile(
    r"^Traceback \(most recent call last\):|^Exit code [1-9]|: command not found"
    r"|Permission denied|^zsh: permission denied:|Operation not permitted|exited with (?:code|status) [1-9]",
    re.MULTILINE,
)
# Commands that print files or history: what they show is content, not their own failure.
_READERS = frozenset({"cat", "head", "tail", "less", "grep", "rg"})
_GIT_READERS = frozenset({"diff", "show", "log"})
# Commands that only look around: between a failing streak and its fix they neither end
# the streak nor count as the fix, and their output (a cat of .env, say) is never shown.
# Not listed on purpose: env, echo, find, git branch, git remote — each can also change things.
_LOOKERS = _READERS | {"ls", "pwd", "which", "type", "wc", "file", "stat", "tree",
                       "printenv", "du", "df", "ps", "whoami", "date"}
_GIT_LOOKERS = _GIT_READERS | {"status", "rev-parse", "blame", "ls-files"}
_LEADING_CD_RE = re.compile(r"""\s*cd\s+(?:"[^"]*"|'[^']*'|\S+)\s*&&\s*""")
# Whole words only: "стой" must not fire inside "простой" or "достойные".
# A bare "нет" / "no" counts only as the message's first word: "ответ нет" is no correction.
CORRECTION_RE = re.compile(
    r"(?i)^нет\b|^no\b|\b(не так|не то|стой(те)?|стоп|я( же)? (говорил|просил|сказал)а?|неправильно"
    r"|зачем ты|а зачем|а почему|почему ты|не нужно|не надо|(опять|снова) ты|ты (опять|снова)|ты решаешь"
    r"|wrong|that's not|i said|stop|why did you)\b"
)
# Entries the CLI writes into the user's turn that the user never typed.
_NOT_TYPED = ("isMeta", "isCompactSummary", "isVisibleInTranscriptOnly")

_INPUT_MAX = 300
_TEXT_MAX = 800
_AFTER_MAX = 300
_BEFORE_MAX = 600
_USER_MSG_MAX = 500
# a turn of the dialogue: the agent's words before the user's message keep their head (what it
# claimed) and their tail (what it offered or asked)
_TURN_BEFORE_HEAD = 380
_TURN_BEFORE_TAIL = 180
_WORK_TOOLS_MAX = 6
# tools that write a file, and where their input names it
EDIT_TOOLS = {"Edit": "file_path", "Write": "file_path", "NotebookEdit": "notebook_path"}
_TURN_FILES_MAX = 4  # files a turn names; the rest are counted
_FILE_MAX = 160
_EDITED_MAX = 500
# how far back from a segment's start the agent's last words to the user are looked for
CARRY_BYTES = 262_144
# A reply this short with no correction word ("го", "да", "1") tells nothing about the agent's work.
REPLY_MIN_CHARS = 12
# signals whose loss costs least when a segment outgrows what the model is shown
_DROPPABLE = ("tool_error", "silent_failure")
MAX_PARTS = 3
_TASK_MAX = 800
_RECOVERY_TEXT_MAX = 400
_COMMAND_MAX = 4000
_INTERRUPT = "[Request interrupted by user"
_SUBAGENTS = "subagents"
_AGENT_PREFIX = "agent-"
TASK_HEADING = "## Task given to the subagent"
# A session with no ai-title is named by its first words; a subagent by its parent, marked.
TITLE_FALLBACK_MAX = 60


@dataclass(frozen=True)
class Signal:
    kind: str
    ts: str | None
    tool: str | None
    tool_input: str
    text: str
    after: str
    before: str = ""  # what the agent last said to the user, for the user's own signals
    # a recovery: how many failing results it ended, and their 1-based numbers among the signals
    fixed_after: int = 0
    resolves: tuple[int, ...] = ()
    # the whole shell command of the call, for matching a lesson's trigger; never shown to the model
    command: str = ""
    seq: int = 0  # its place among the segment's signals and turns, in the order they happened
    called: str | None = None  # when the agent made the call, as against `ts`, when its result came


@dataclass(frozen=True)
class Turn:
    """One typed message of the user with what surrounds it: `[uN]` in the dialogue."""
    n: int  # 1-based among the user's messages of the segment
    ts: str | None
    text: str
    before: str = ""  # the agent's latest words to the user, when it said any since the last message
    after: str = ""  # what the agent then did until the next message: tools by count, failures
    seq: int = 0

    @property
    def is_reply(self) -> bool:
        """An answer to something the agent said that is long enough to judge the agent's work by."""
        return bool(self.before) and (len(self.text) >= REPLY_MIN_CHARS or CORRECTION_RE.search(self.text) is not None)


@dataclass(frozen=True)
class Segment:
    transcript: Path
    session_id: str
    start: int
    end: int
    cwd: str | None
    first_ts: str | None
    last_ts: str | None
    signals: tuple[Signal, ...]
    user_messages: tuple[str, ...]
    task_context: str = ""  # a subagent's dispatch prompt from its parent agent
    agent_id: str = ""  # a subagent's file stem; session_id is then its parent's
    # hook failures and API errors: alerts for the user, never signals for the model
    health: tuple[HealthEvent, ...] = ()
    # the first timestamp of the whole transcript: when the session loaded its memory
    session_start: str | None = None
    # the agent opening auto-lesson files: counted, never shown to the model
    reads: tuple[LessonRead, ...] = ()
    # the user's messages again, each with the agent's words before it and its work after it
    turns: tuple[Turn, ...] = ()
    # the files the agent wrote to, once each, in the order it first did
    edited: tuple[str, ...] = ()

    @property
    def replies(self) -> tuple[Turn, ...]:
        """The user's answers to the agent: what a correction can only be among."""
        return tuple(t for t in self.turns if t.is_reply)

    def render(self, max_chars: int) -> str:
        """The segment as one text of at most `max_chars`; what does not fit is counted, not shown."""
        return self.render_parts(max_chars, 1)[0]

    def render_parts(self, max_chars: int, max_parts: int = MAX_PARTS) -> list[str]:
        """The segment as up to `max_parts` texts of at most `max_chars` each, in the order it happened.

        Signals keep their numbers `[n]` and messages theirs `[uN]` across the parts. When even
        `max_parts` texts cannot hold it all, the plain failures no recovery points at go first,
        the oldest first: the dialogue and the fixes are what a lesson is read from.
        """
        started = (f" · started {self.session_start}"
                   if self.session_start and self.session_start != self.first_ts else "")
        head = scrub(
            f"session {self.session_id} · {self.first_ts} → {self.last_ts}{started} · cwd {self.cwd}\n"
        )
        if self.task_context:
            # one line, so the parent's text can never open a section of its own
            head += f"\n{TASK_HEADING}\n" + scrub(" ".join(self.task_context.split())) + "\n"
        resolved = {n for s in self.signals for n in s.resolves}
        # (order, 0 for a signal / 1 for a turn, text, may be dropped first)
        items = [(s.seq, 0, scrub(_render_signal(n, s)), s.kind in _DROPPABLE and n not in resolved)
                 for n, s in enumerate(self.signals, 1)]
        items += [(t.seq, 1, scrub(_render_turn(t)), False) for t in self.turns]
        items.sort(key=lambda it: (it[0], it[1]))
        sig_head, talk_head = "\n## Signals\n", "\n## Dialogue\n"
        room = max_chars - len(head) - len(sig_head) - len(talk_head) - 120  # part and omitted notes
        dropped = 0
        excess = sum(len(it[2]) + 1 for it in items) - room * max_parts
        if excess > 0:
            kept = []
            for it in items:
                if excess > 0 and it[3]:
                    excess -= len(it[2]) + 1
                    dropped += 1
                else:
                    kept.append(it)
            items = kept
        parts: list[list] = [[]]
        used = 0
        left_out = [0, 0]  # signals, messages that no part had room for
        for it in items:
            size = len(it[2]) + 1
            if used + size > room and parts[-1]:
                if len(parts) == max_parts:
                    left_out[it[1]] += 1
                    continue
                parts.append([])
                used = 0
            parts[-1].append(it)
            used += size
        out: list[str] = []
        for i, part in enumerate(parts, 1):
            text = head
            if len(parts) > 1:
                text += f"part {i} of {len(parts)}\n"
            text += sig_head + "\n".join(it[2] for it in part if it[1] == 0)
            if i == len(parts) and (dropped or left_out[0]):
                text += f"\n… {dropped + left_out[0]} more signals omitted\n"
            talk = [it[2] for it in part if it[1] == 1]
            if talk:
                text += talk_head + "\n".join(talk)
            if i == len(parts) and left_out[1]:
                text += f"\n… {left_out[1]} more messages omitted\n"
            out.append(text[:max_chars])
        return out


def _one_line(text: str) -> str:
    """Tool output on one line, so it can never open a section of the segment."""
    return " ".join(text.split())


def _render_signal(n: int, s: Signal) -> str:
    if s.kind == "recovery":
        refs = ", ".join(f"[{i}]" for i in s.resolves)
        return (f"[{n}] {s.kind} {s.ts} tool={s.tool}\n"
                f"fixed_after: {s.fixed_after} failures of {s.tool}" + (f" ({refs})" if refs else "") + "\n"
                f"input: {s.tool_input}\nresult: {_one_line(s.text)}\nnext: {s.after}\n")
    return (f"[{n}] {s.kind} {s.ts} tool={s.tool}\n"
            + (f"before: {s.before}\n" if s.before else "")
            + f"input: {s.tool_input}\nerror: {_one_line(s.text)}\nnext: {s.after}\n")


def _render_turn(t: Turn) -> str:
    """A message of the user as the dialogue shows it; the lines around it are the agent's."""
    return ((f"before: {_one_line(t.before)}\n" if t.before else "")
            + f"[u{t.n}] {_one_line(t.text)}\n"
            + (f"then: {t.after}\n" if t.after else ""))


def _clip_ends(text: str, head: int, tail: int) -> str:
    """Scrubbed text with its middle cut out when it is longer than `head + tail`."""
    text = scrub(text)
    if len(text) <= head + tail:
        return text
    return text[:head] + " … " + text[-tail:]


def _work_text(work: Counter, failed: int, files: list[str] = ()) -> str:
    """What the agent did between two messages: its tools by count, most used first, and the files it wrote to."""
    if not work:
        return ""
    top = work.most_common(_WORK_TOOLS_MAX)
    text = ", ".join(f"{name.rsplit('__', 1)[-1]}×{n}" for name, n in top)
    rest = sum(work.values()) - sum(n for _, n in top)
    if rest:
        text += f", other×{rest}"
    if failed:
        text += f" · failed: {failed}"
    if files:
        more = len(files) - _TURN_FILES_MAX
        text += " · edited: " + ", ".join(files[:_TURN_FILES_MAX]) + (f" (+{more})" if more > 0 else "")
    return text


def _edited_file(name: str, inp) -> str:
    """The file a tool call writes to, one line and scrubbed; "" for any other call."""
    key = EDIT_TOOLS.get(name)
    path = inp.get(key) if key and isinstance(inp, dict) else None
    return _clip(_one_line(path), _FILE_MAX) if isinstance(path, str) and path.strip() else ""


def _same_input(name: str, inp) -> str:
    """What makes two calls the same call: a shell command with its spacing evened out, else the whole input.

    A shell call that differs only in its description or timeout is the same command again.
    """
    if name in _SHELL_TOOLS and isinstance(inp, dict) and isinstance(inp.get("command"), str):
        return " ".join(inp["command"].split())
    try:
        return json.dumps(inp, ensure_ascii=False, sort_keys=True)
    except (TypeError, ValueError):
        return repr(inp)


def _clip(text: str, limit: int) -> str:
    """Scrub first, then clip: a secret cut at the limit would evade the regexes."""
    text = scrub(text)
    return text if len(text) <= limit else text[:limit]


def _blocks(content) -> list[dict]:
    if isinstance(content, list):
        return [b for b in content if isinstance(b, dict)]
    return []


def _result_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            b.get("text", "") for b in content if isinstance(b, dict) and isinstance(b.get("text"), str)
        )
    return ""


def _user_text(content) -> str:
    if isinstance(content, str):
        return content
    return "\n".join(
        b["text"] for b in _blocks(content) if b.get("type") == "text" and isinstance(b.get("text"), str)
    )


def _error_kind(text: str) -> str:
    if "hook error" in text or "Refusing:" in text:
        return "hook_block"
    if "doesn't want to proceed" in text or "doesn't want to take this action" in text:
        return "user_rejection"
    return "tool_error"


def _silent_failure_line(text: str) -> str | None:
    """The line of a shell result that shows a failure, if its first 6000 chars hold one."""
    text = text[:_SILENT_SCAN_MAX]
    m = _SILENT_RE.search(text)
    if m is None:
        return None
    start = text.rfind("\n", 0, m.start()) + 1
    end = text.find("\n", m.end())
    return text[start:] if end < 0 else text[start:end]


def _words(command: str) -> list[str]:
    """The command's words after a leading `cd … &&`, the first one without its directory."""
    m = _LEADING_CD_RE.match(command)
    words = command[m.end() if m else 0:].split()
    if words:
        words[0] = words[0].rsplit("/", 1)[-1]
    return words


def _first_word_in(command: str, plain: frozenset, git: frozenset) -> bool:
    words = _words(command)
    if not words:
        return False
    first, rest = words[0], words[1:]
    if first in plain:
        return True
    if first == "sed":
        return rest[:1] == ["-n"]
    if first == "git":
        if rest[:1] == ["-C"]:
            rest = rest[2:]
        return bool(rest) and rest[0] in git
    return False


def _reads_files(command: str) -> bool:
    """True when the command's first word (after a leading `cd … &&`) prints files or history."""
    return _first_word_in(command, _READERS, _GIT_READERS)


def _looks_around(command: str) -> bool:
    """True for a command that only reads or lists: no fix, whatever it prints."""
    return _first_word_in(command, _LOOKERS, _GIT_LOOKERS)


def _program(command: str) -> str:
    words = _words(command)
    return words[0] if words else ""


def _tool_input(inp) -> str:
    return _clip(inp if isinstance(inp, str) else json.dumps(inp, ensure_ascii=False), _INPUT_MAX)


def _typed_by_user(entry: dict) -> bool:
    """False for meta entries, compaction summaries and anything without a human origin."""
    if any(entry.get(flag) is True for flag in _NOT_TYPED):
        return False
    origin = entry.get("origin")
    return origin is None or (isinstance(origin, dict) and origin.get("kind") == "human")


def is_subagent(path: Path) -> bool:
    """A subagent's transcript: <project dir>/<session-id>/subagents/agent-*.jsonl."""
    path = Path(path)
    return path.parent.name == _SUBAGENTS and path.name.startswith(_AGENT_PREFIX)


def parent_session_id(path: Path) -> str:
    """The session a transcript belongs to: the <session-id> folder of a subagent, else the file stem."""
    path = Path(path)
    return path.parent.parent.name if is_subagent(path) else path.stem


def _head_entries(path: Path):
    """The complete entries of a file from its start, parsed one at a time."""
    with Path(path).open("rb") as f:
        for raw_line in f:
            if not raw_line.endswith(b"\n"):
                break
            try:
                e = json.loads(raw_line.decode("utf-8", errors="replace"))
            except ValueError:
                continue
            if isinstance(e, dict):
                yield e


def session_cwd(path: Path) -> str | None:
    """Where the session started: the cwd of the first complete entry that has one."""
    for e in _head_entries(path):
        if isinstance(e.get("cwd"), str) and e["cwd"]:
            return e["cwd"]
    return None


def session_start(path: Path) -> str | None:
    """The first timestamp of the transcript, read from its head."""
    for e in _head_entries(path):
        if isinstance(e.get("timestamp"), str) and e["timestamp"]:
            return e["timestamp"]
    return None


def _own_title(path: Path) -> str | None:
    """The last ai-title of a transcript, else its first typed words; None when it has neither."""
    title = first = None
    for e in _head_entries(path):
        if e.get("type") == "ai-title":
            named = squash(e["aiTitle"]) if isinstance(e.get("aiTitle"), str) else ""
            title = named or title
        elif first is None and e.get("type") == "user" and _typed_by_user(e):
            msg = e.get("message")
            text = squash(_user_text(msg.get("content") if isinstance(msg, dict) else None))
            if text and not text.startswith(("<", _INTERRUPT)):
                first = text
    if title is not None:
        return scrub(title)
    return _clip(first, TITLE_FALLBACK_MAX) if first else None


def session_title(path: Path) -> str | None:
    """What a session is called: its last ai-title, else its first words; a subagent's is its parent's, marked.

    None when the file cannot be read or holds nothing to name it by.
    """
    path = Path(path)
    try:
        if not is_subagent(path):
            return _own_title(path)
        parent = _own_title(path.parents[2] / f"{parent_session_id(path)}.jsonl")
    except OSError:
        return None
    return parent + t(" (subagent)", " (сабагент)") if parent else None


def task_context(path: Path) -> str:
    """A subagent's task: the first user text of its transcript, scrubbed and clipped."""
    for e in _head_entries(path):
        if e.get("type") != "user":
            continue
        msg = e.get("message")
        text = _user_text(msg.get("content") if isinstance(msg, dict) else None).strip()
        if text:
            return _clip(text, _TASK_MAX)
    return ""


def _recovery(streak: list, name: str | None, command: str, call: str, ts: str | None, inp: str,
              content) -> dict | None:
    """The recovery a successful result makes of the failing streak, if any; updates the streak.

    Only the same tool as the last failure can fix it, and for a shell only the same program;
    any other success (a diagnostic Read, an `npm install`) leaves the streak as it is, and so
    does a command that only reads or lists (`cat`, `ls`, `git status`), whose output is never
    shown. The first matching success ends the streak: a recovery when that tool (that program)
    failed at least twice in it and the call changed, otherwise nothing: the same call working
    on a retry is luck, not a fix.
    """
    if not streak or name != streak[-1][1]:
        return None
    shell = name in _SHELL_TOOLS
    if shell and (_looks_around(command) or _program(command) != streak[-1][3]):
        return None
    program = streak[-1][3]
    failures = [f for f in streak if f[1] == name and (not shell or f[3] == program)]
    streak.clear()
    if len(failures) < RECOVERY_MIN_FAILURES or call == failures[-1][2]:
        return None
    return {"kind": "recovery", "ts": ts, "tool": name, "tool_input": inp,
            "text": _clip(_result_text(content), _RECOVERY_TEXT_MAX), "after": "",
            "fixed_after": len(failures), "resolves": tuple(f[0] + 1 for f in failures)}


def _carry(path: Path, start: int) -> tuple[str, bool]:
    """What the agent last said to the user before `start`, and whether that was after the user's last message.

    A segment starts where the last run stopped reading, which is usually right after the agent's
    closing words. The user's answer the next morning opens the next segment, and without this
    it would have nothing before it: no answer, so nothing a correction can be read from.
    Only the last `CARRY_BYTES` are read; what is not found there is as if it was not said.
    """
    if start <= 0:
        return "", False
    begin = max(0, start - CARRY_BYTES)
    try:
        with path.open("rb") as f:
            f.seek(begin)
            data = f.read(start - begin)
    except OSError:
        return "", False
    lines = data.split(b"\n")
    if begin > 0:
        lines = lines[1:]  # the window opens in the middle of a line
    last_said, said_since = "", False
    for raw_line in lines:
        try:
            e = json.loads(raw_line.decode("utf-8", errors="replace"))
        except ValueError:
            continue
        if not isinstance(e, dict):
            continue
        msg = e.get("message")
        content = msg.get("content") if isinstance(msg, dict) else None
        if e.get("type") == "assistant":
            texts = [b["text"] for b in _blocks(content) if b.get("type") == "text" and isinstance(b.get("text"), str)]
            if texts:
                last_said, said_since = "\n".join(texts), True
        elif e.get("type") == "user" and _typed_by_user(e):
            text = _user_text(content).strip()
            if text and not text.startswith((_INTERRUPT, "<")):
                said_since = False  # a message of the user: what was said before it was answered already
    return last_said, said_since


def read_segment(path: Path, start: int) -> Segment:
    with path.open("rb") as f:
        f.seek(0, 2)
        if start > f.tell():
            start = 0
        f.seek(start)
        data = f.read()
    last_nl = data.rfind(b"\n")
    complete = data[: last_nl + 1] if last_nl >= 0 else b""
    end = start + len(complete)
    # a subagent's user turns are its parent agent's dispatch, never a human's words
    subagent = is_subagent(path)

    session_id = ""
    cwd: str | None = None
    first_ts = last_ts = None
    uses: dict[str, tuple[str, str]] = {}
    called: dict[str, str | None] = {}  # tool_use id -> when the assistant message that made the call was written
    commands: dict[str, str] = {}  # tool_use id -> a shell command, as it was given
    full_inputs: dict[str, str] = {}  # tool_use id -> its whole input, for the recovery contrast
    # the failing results since the last fix or the user's word: (index in raw, tool, call, program)
    streak: list[tuple[int, str | None, str, str]] = []
    raw: list[dict] = []  # mutable signal records
    awaiting: list[dict] = []  # signals still waiting for the next assistant entry
    user_messages: list[str] = []
    turns: list[dict] = []
    work: Counter = Counter()  # the agent's tool calls since the user's last message
    failed = 0  # and how many of their results failed
    files: list[str] = []  # and the files they wrote to
    edited: list[str] = []  # every file written to in the segment
    seq = 0  # one count for signals and turns, in the order they happened
    # the agent's latest text to the user, and whether it came since the user's last message:
    # both start from what was said just before the segment
    last_said, said_since = ("", False) if subagent else _carry(path, start)
    health: list[HealthEvent] = []
    reads: list[LessonRead] = []
    pos = start  # byte offset just past the current line

    for raw_line in complete.split(b"\n"):
        pos += len(raw_line) + 1
        if not raw_line:
            continue
        try:
            e = json.loads(raw_line.decode("utf-8", errors="replace"))
        except ValueError:
            continue
        if not isinstance(e, dict):
            continue
        ts = e.get("timestamp") if isinstance(e.get("timestamp"), str) else None
        if not session_id and isinstance(e.get("sessionId"), str):
            session_id = e["sessionId"]
        if cwd is None and isinstance(e.get("cwd"), str):
            cwd = e["cwd"]
        if ts:
            first_ts = first_ts or ts
            last_ts = ts
        msg = e.get("message")
        content = msg.get("content") if isinstance(msg, dict) else None
        typ = e.get("type")
        if typ in ("attachment", "system"):
            event = _health.from_entry(e, end=pos)
            if event is not None:
                health.append(event)
            continue

        if typ == "assistant":
            nxt = ""
            for b in _blocks(content):
                if b.get("type") == "tool_use":
                    work[b.get("name", "") or "?"] += 1
                    uses[b.get("id", "")] = (b.get("name", ""), _tool_input(b.get("input", {})))
                    called[b.get("id", "")] = ts
                    full_inputs[b.get("id", "")] = _same_input(b.get("name", ""), b.get("input", {}))
                    reads.extend(LessonRead(lid, ts, pos) for lid in lesson_ids(b.get("name", ""), b.get("input")))
                    wrote = _edited_file(b.get("name", ""), b.get("input"))
                    if wrote and wrote not in files:
                        files.append(wrote)
                    if wrote and wrote not in edited and len(edited) < _EDITED_MAX:
                        edited.append(wrote)
                    inp = b.get("input")
                    if isinstance(inp, dict) and isinstance(inp.get("command"), str):
                        commands[b.get("id", "")] = inp["command"]
            texts = [b["text"] for b in _blocks(content)
                     if b.get("type") == "text" and isinstance(b.get("text"), str)]
            if texts:
                nxt = last_said = "\n".join(texts)
                said_since = True
            else:
                for b in _blocks(content):
                    if b.get("type") == "tool_use":
                        nxt = f"{b.get('name', '')}({_tool_input(b.get('input', {}))})"
                        break
            if nxt and awaiting:
                for rec in awaiting:
                    rec["after"] = _clip(nxt, _AFTER_MAX)
                awaiting = []
        elif typ == "user":
            for b in _blocks(content):
                if b.get("type") != "tool_result":
                    continue
                use_id = b.get("tool_use_id", "")
                name, inp = uses.get(use_id, (None, ""))
                rec = None
                if b.get("is_error") is True:
                    text = _result_text(b.get("content"))
                    rec = {"kind": _error_kind(text), "ts": ts, "tool": name or None,
                           "tool_input": inp, "text": _clip(text, _TEXT_MAX), "after": ""}
                elif name in _SHELL_TOOLS and not _reads_files(commands.get(use_id, "")):
                    line = _silent_failure_line(_result_text(b.get("content")))
                    if line is not None:
                        rec = {"kind": "silent_failure", "ts": ts, "tool": name,
                               "tool_input": inp, "text": _clip(line, _TEXT_MAX), "after": ""}
                command = commands.get(use_id, "")
                if rec is None:  # a success
                    rec = _recovery(streak, name, command, full_inputs.get(use_id, ""), ts, inp, b.get("content"))
                    if rec is None:
                        continue
                elif rec["kind"] in _FAILING:
                    failed += 1
                    streak.append((len(raw), name or None, full_inputs.get(use_id, ""), _program(command)))
                seq += 1
                rec["seq"] = seq
                rec["command"] = _clip(command, _COMMAND_MAX)
                rec["called"] = called.get(use_id)
                raw.append(rec)
                awaiting.append(rec)
            if subagent or not _typed_by_user(e):
                continue
            text = _user_text(content).strip()
            if not text:
                continue
            if text.startswith(_INTERRUPT):
                streak.clear()  # what follows is the user's steer, not the agent's own fix
                seq += 1
                rec = {"kind": "user_interrupt", "ts": ts, "tool": None, "tool_input": "",
                       "text": _clip(text, _TEXT_MAX), "after": "", "before": _clip(last_said, _BEFORE_MAX),
                       "seq": seq}
                raw.append(rec)
                awaiting.append(rec)
            elif not text.startswith("<"):
                user_messages.append(_clip(text, _USER_MSG_MAX))
                if turns:
                    turns[-1]["after"] = _work_text(work, failed, files)
                work, failed, files = Counter(), 0, []
                seq += 1
                turns.append({"n": len(user_messages), "ts": ts, "text": _clip(text, _USER_MSG_MAX),
                              "before": (_clip_ends(last_said, _TURN_BEFORE_HEAD, _TURN_BEFORE_TAIL)
                                         if said_since else ""),
                              "after": "", "seq": seq})
                said_since = False
                if CORRECTION_RE.search(text):
                    streak.clear()
                    seq += 1
                    rec = {"kind": "user_correction", "ts": ts, "tool": None, "tool_input": "",
                           "text": _clip(text, _TEXT_MAX), "after": "",
                           "before": _clip(last_said, _BEFORE_MAX), "seq": seq}
                    raw.append(rec)
                    awaiting.append(rec)

    if turns:
        turns[-1]["after"] = _work_text(work, failed, files)
    if subagent:
        # the parent session: N subagents of one task are one session of evidence
        session_id = parent_session_id(path)
    owner = session_id or parent_session_id(path)
    return Segment(
        transcript=path, session_id=session_id, start=start, end=end, cwd=cwd,
        first_ts=first_ts, last_ts=last_ts,
        signals=tuple(Signal(**r) for r in raw), user_messages=tuple(user_messages),
        task_context=task_context(path) if subagent else "",
        agent_id=path.stem if subagent else "",
        health=tuple(replace(event, session_id=owner) for event in health),
        session_start=first_ts if start == 0 else session_start(path),
        reads=tuple(reads),
        turns=tuple(Turn(**t) for t in turns),
        edited=tuple(edited),
    )


def list_transcripts(projects_dir: Path, exclude_prefixes: tuple[str, ...]) -> list[Path]:
    """Session transcripts and their subagents' ones, oldest first.

    Project folders named with an excluded prefix are skipped; a file that vanishes
    or cannot be read is left out, and nothing here raises.
    """
    found: list[tuple[float, str, Path]] = []
    try:
        dirs = [d for d in projects_dir.iterdir() if d.is_dir() and not d.name.startswith(exclude_prefixes)]
    except OSError:
        return []
    for d in dirs:
        for pattern in ("*.jsonl", f"*/{_SUBAGENTS}/{_AGENT_PREFIX}*.jsonl"):
            try:
                for p in d.glob(pattern):
                    try:
                        if p.is_file():
                            found.append((p.stat().st_mtime, str(p), p))
                    except OSError:
                        continue
            except OSError:
                continue
    return [p for _, _, p in sorted(found)]
