"""The health channel: alerts for the user, never lessons for the agent.

A hook that fails on every call or an API that keeps erroring is no mistake the agent
could learn from. These events are counted per key in `State.health`, shown in
`agent-reflect recent` and the weekly digest, and raise a notification when one key
fires often in a night. They never reach the model and never become memory.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from agent_reflect.i18n import t, times as _times
from agent_reflect.scrub import scrub

HOOK_FAILURE = "hook_failure"
API_ERROR = "api_error"
SESSIONS_MAX = 20
DETAIL_MAX = 200
CODE_MAX = 40
NOTIFY_KEYS_MAX = 3
RENOTIFY_AFTER = timedelta(days=7)
_HOME_RE = re.compile(r"/Users/[^/\s\"']+/")


@dataclass(frozen=True)
class HealthEvent:
    kind: str  # hook_failure | api_error
    key: str
    ts: str | None
    session_id: str
    detail: str
    hook_name: str = ""  # matched by Config.health_ignore along with the command
    end: int = 0  # byte offset just past the entry in its transcript, so a re-read never counts it twice


def _first_line(text) -> str:
    if not isinstance(text, str):
        return ""
    for line in text.splitlines():
        if line.strip():
            return scrub(line.strip())[:DETAIL_MAX]
    return ""


def _hook_failure(entry: dict, session_id: str, end: int) -> HealthEvent | None:
    att = entry.get("attachment")
    if not isinstance(att, dict) or att.get("type") != "hook_non_blocking_error":
        return None
    code = att.get("exitCode")
    if not _int(code) or code == 0:
        return None
    command = _HOME_RE.sub("~/", str(att.get("command") or ""))
    key = f"{att.get('hookEvent') or ''}|{scrub(command)}|{code}"
    detail = _first_line(att.get("stderr")) or _first_line(att.get("stdout"))
    hook_name = att.get("hookName") if isinstance(att.get("hookName"), str) else ""
    return HealthEvent(HOOK_FAILURE, key, _ts(entry), session_id, detail, hook_name, end)


def _int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _api_error(entry: dict, session_id: str, end: int) -> HealthEvent | None:
    """An API error the CLI gave up on: an attempt it will still retry is no failure yet."""
    if entry.get("subtype") != "api_error":
        return None
    attempt, most = entry.get("retryAttempt"), entry.get("maxRetries")
    if _int(attempt) and _int(most) and attempt < most:
        return None
    error = entry.get("error") if isinstance(entry.get("error"), dict) else {}
    status = error.get("status")
    code = str(status).strip() if _int(status) or isinstance(status, str) else ""
    code = scrub(code).replace("|", "/")[:CODE_MAX]
    key = f"{API_ERROR}|{code}" if code else API_ERROR
    detail = _first_line(error.get("message")) or _first_line(entry.get("content"))
    return HealthEvent(API_ERROR, key, _ts(entry), session_id, detail, "", end)


def _ts(entry: dict) -> str | None:
    return entry.get("timestamp") if isinstance(entry.get("timestamp"), str) else None


def from_entry(entry: dict, session_id: str = "", end: int = 0) -> HealthEvent | None:
    """The health event an entry of a transcript carries, if any."""
    typ = entry.get("type")
    if typ == "attachment":
        return _hook_failure(entry, session_id, end)
    if typ == "system":
        return _api_error(entry, session_id, end)
    return None


def _key_parts(key: str) -> tuple[str, str, str]:
    """A hook failure key's (hookEvent, command, exit code); the command may hold `|` itself."""
    event, _, rest = key.partition("|")
    command, _, code = rest.rpartition("|")
    return event, command, code


def is_ignored(kind: str, key: str, hook_name: str, ignore: tuple[str, ...]) -> bool:
    """A hook failure whose command or hook name holds an ignored substring; never an API error.

    The command is matched with its quotes stripped, after `/Users/<name>/` became `~/`.
    """
    if kind != HOOK_FAILURE:
        return False
    command = _key_parts(key)[1].replace('"', "").replace("'", "")
    return any(word and (word in command or word in hook_name) for word in ignore)


def _entry(health: dict, key: str, kind: str, hook_name: str) -> dict:
    entry = health.setdefault(key, {})
    for name, default in (("kind", kind), ("count", 0), ("first_ts", None), ("last_ts", None),
                          ("sessions", []), ("detail", ""), ("notified_at", None)):
        entry.setdefault(name, default)
    if hook_name and not entry.get("hook_name"):
        entry["hook_name"] = hook_name
    return entry


def merge(health: dict, events, ignore: tuple[str, ...] = ()) -> dict[str, int]:
    """Add `events` into `health`; returns how many each key gained. Ignored events are dropped."""
    added: dict[str, int] = {}
    for ev in events:
        if is_ignored(ev.kind, ev.key, ev.hook_name, ignore):
            continue
        entry = _entry(health, ev.key, ev.kind, ev.hook_name)
        entry["count"] += 1
        if ev.ts:
            if not entry["first_ts"] or ev.ts < entry["first_ts"]:
                entry["first_ts"] = ev.ts
            if not entry["last_ts"] or ev.ts >= entry["last_ts"]:
                entry["last_ts"] = ev.ts
                if ev.detail:
                    entry["detail"] = ev.detail
        if ev.detail and not entry["detail"]:
            entry["detail"] = ev.detail
        sessions = entry["sessions"]
        if ev.session_id and ev.session_id not in sessions and len(sessions) < SESSIONS_MAX:
            sessions.append(ev.session_id)
        added[ev.key] = added.get(ev.key, 0) + 1
    return added


def parse_ts(value) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        ts = datetime.fromisoformat(value)
    except ValueError:
        return None
    return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)


def due(health: dict, added: dict[str, int], threshold: int, now: datetime) -> list[str]:
    """Keys that fired at least `threshold` times this run and were not announced in the last week."""
    keys = []
    for key, n in added.items():
        entry = health.get(key)
        if entry is None or n < threshold:
            continue
        last = parse_ts(entry.get("notified_at"))
        if last is None or now - last >= RENOTIFY_AFTER:
            keys.append(key)
    return sorted(keys, key=lambda k: -health[k].get("count", 0))


def _command_name(command: str) -> str:
    """The basename of the script a hook command runs: the first token with a `/`, else the first."""
    tokens = [t.strip("\"'") for t in command.split()]
    path = next((t for t in tokens if "/" in t), tokens[0] if tokens else "")
    return path.rstrip("/").rsplit("/", 1)[-1] or command


def _phrase(key: str, entry: dict) -> str:
    count = entry.get("count", 0)
    if entry.get("kind") == HOOK_FAILURE:
        _, command, code = _key_parts(key)
        name, times = _command_name(command), _times(count)
        return t(f"hook {name} fails (code {code}), {times}", f"хук {name} падает (код {code}), {times}")
    code = key.partition("|")[2]
    return t("API answers with an error", "API отвечает ошибкой") + (f" {code}" if code else "") + f", {_times(count)}"



def notify_message(health: dict, keys: list[str]) -> str:
    parts = [_phrase(key, health[key]) for key in keys[:NOTIFY_KEYS_MAX]]
    if len(keys) > NOTIFY_KEYS_MAX:
        parts.append(f"+{len(keys) - NOTIFY_KEYS_MAX}")
    return scrub(t("agent-reflect: silent failures — ", "agent-reflect: тихие сбои — ") + "; ".join(parts)
                 + t("; see agent-reflect recent", "; см. agent-reflect recent"))


def entries_since(health: dict, since: datetime | None, ignore: tuple[str, ...] = ()) -> list[dict]:
    """Entries last seen at or after `since` (all when None), newest first, each with its `key`."""
    out = []
    for key, entry in health.items():
        if not isinstance(entry, dict):
            continue
        if is_ignored(entry.get("kind", ""), key, entry.get("hook_name") or "", ignore):
            continue
        last = parse_ts(entry.get("last_ts"))
        if since is not None and (last is None or last < since):
            continue
        out.append({"key": key, **entry})
    return sorted(out, key=lambda e: e.get("last_ts") or "", reverse=True)


def describe(entry: dict) -> str:
    """One line for `recent` and the digest: kind, what, how often, exit code, last seen, detail."""
    key = entry.get("key", "")
    if entry.get("kind") == HOOK_FAILURE:
        event, what, code = _key_parts(key)
        what = f"{event} {what}".strip()
    else:
        what, code = API_ERROR, key.partition("|")[2]
    times, last = _times(int(entry.get("count", 0) or 0)), entry.get("last_ts") or "?"
    return (f"- {entry.get('kind', '?')} · {what} · {times} · " + t("code ", "код ") + f"{code or '—'} · "
            + t("last ", "последний ") + f"{last} · {entry.get('detail') or '—'}")
