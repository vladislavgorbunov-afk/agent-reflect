"""How big a skill edit may be to land unreviewed, and the diff the digest shows for it."""
from __future__ import annotations

import difflib
import re
from collections import Counter

from agent_reflect.i18n import t
from agent_reflect.scrub import scrub

REPLACE_MAX = 300  # chars of existing text an edit may replace; an insertion may be any size
DIFF_MAX_LINES = 60
INCIDENT_LOG = "incident_log"
SKILL_LARGE_CHARS = 24_000  # past this a skill gets an advisory, never a refusal
# lines an edit may add to a file every session loads whole (a project's CLAUDE.md): a rule that
# goes in takes the place of one it covers, so the file the agent always reads does not grow
ALWAYS_ON_NET_LINES = 1
# Standards whose names look like ticket keys.
_NOT_TICKETS = frozenset({"UTF", "SHA", "ISO", "RFC", "HTTP", "TLS", "SSL", "AES", "RSA", "MD", "CVE", "IPV"})
# What an incident log leaves in a skill: a date, a PR or issue number, a ticket key, a session id.
_INCIDENT_PATTERNS: tuple[tuple[str, re.Pattern], ...] = (
    ("a date", re.compile(r"\b\d{4}-\d{2}-\d{2}\b")),
    # only after PR/pull/issue or as a list item's first word: "step #1" is no reference
    ("a PR or issue number", re.compile(r"(?im)(?:\b(?:PR|pull|issue)s?\b[^\S\n]*|^[^\S\n]*[-*][^\S\n]*)#\d+\b")),
    ("a ticket key", re.compile(r"\b([A-Z]{2,})-\d+\b")),
    ("a session-like id", re.compile(r"\b(?=[0-9a-f]*[a-f])(?=[0-9a-f]*\d)[0-9a-f]{8,}\b")),
    # a hex or UUID-shaped id after the word: "session cookie2fa" is a word, "сессия a1b2c3" an id
    ("a session id", re.compile(r"(?i)\b(?:сесси[а-я]*|session)\s*(?:id)?\s*[:#=]?\s*[`'\"]?"
                                r"((?=[0-9a-f-]*\d)[0-9a-f]{6,}(?:-[0-9a-f]{4,})*)\b")),
)


def size_error(old: str, new: str) -> str | None:
    """Why an edit is too big to apply, or None: insertions and small replacements may land."""
    if old in new or len(old) <= REPLACE_MAX:
        return None
    return (f"the edit replaces {len(old)} chars of existing text; only an insertion "
            f"or a replacement of at most {REPLACE_MAX} chars is applied")


def growth_error(old: str, new: str, limit: int = ALWAYS_ON_NET_LINES) -> str | None:
    """Why an edit adds too many lines to a file every session loads, or None; blank lines do not count."""
    def lines(text: str) -> int:
        return sum(1 for line in text.splitlines() if line.strip())
    added = lines(new) - lines(old)
    if added <= limit:
        return None
    return (f"the edit adds {added} lines to a file every session loads; at most {limit}: "
            f"a new rule takes the place of one it covers")


def unified_diff(before: str, after: str, name: str) -> str:
    """A scrubbed unified diff of one file, at most DIFF_MAX_LINES lines."""
    lines = list(difflib.unified_diff(before.splitlines(), after.splitlines(),
                                      fromfile=f"a/{name}", tofile=f"b/{name}", lineterm=""))
    if len(lines) > DIFF_MAX_LINES:
        rest = len(lines) - (DIFF_MAX_LINES - 1)
        lines = lines[:DIFF_MAX_LINES - 1] + [t(f"… diff cut, {rest} more lines", f"… diff обрезан, ещё строк: {rest}")]
    return scrub("\n".join(lines))


def _found(pattern: re.Pattern, text: str) -> Counter:
    found = Counter()
    for m in pattern.finditer(text):
        if m.re.groups and m.group(1) in _NOT_TICKETS:
            continue
        found[m.group(0)] += 1
    return found


def incident_error(old: str, new: str, lesson_ids=()) -> str | None:
    """Why an edit reads as an incident log, or None.

    The edit may add a rule, not a story: what `new_string` adds beyond `old_string`
    must hold no date, PR/issue number, ticket key, session id or id of a lesson it
    promotes. Occurrences that were already in `old_string` are not counted again.
    """
    for lesson_id in lesson_ids:
        if new.count(lesson_id) > old.count(lesson_id):
            return f"{INCIDENT_LOG}: the edit names lesson id {lesson_id}; state the rule, not its source"
    for what, pattern in _INCIDENT_PATTERNS:
        added = _found(pattern, new) - _found(pattern, old)
        if added:
            return f"{INCIDENT_LOG}: the edit adds {what} ({next(iter(added))}); state the rule, not the incident"
    return None
