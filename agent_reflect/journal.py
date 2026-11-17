"""The journal of runs, state/journal.jsonl: one JSON record per line, kept for RETAIN_DAYS.

It lives apart from state.events, which consolidate clears, and holds what the reports read:
a `run` record per run, a `verdict` per decision of the model and an `event` per state event.
"""
from __future__ import annotations

import glob
import json
import os
import re
import tempfile
from collections import Counter
from datetime import datetime, timedelta, timezone, tzinfo
from pathlib import Path

from .config import Config
from .ledger import State, parse_ts
from .transcript import session_title

ISO = "%Y-%m-%dT%H:%M:%SZ"
RETAIN_DAYS = 60

# What a log written by Python logging ("2026-10-23 03:31:43,154 INFO ...", local time) holds of the runs.
LOGS = ("reflect.log", "backfill.log")
_LOG_TIME = "%Y-%m-%d %H:%M:%S"
_VERDICT_LINE = re.compile(
    r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d),\d+ INFO verdict (\S+) → (\S+) · session (\S+)(?: (agent-\S+))? · (.*)$")
_DONE_LINE = re.compile(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d),\d+ INFO (reflect|backfill) done: (\{.*\})$")
_CRASH_LINE = re.compile(r"^\d{4}-\d\d-\d\d \d\d:\d\d:\d\d,\d+ ERROR (?:reflect|backfill) crashed\b")
_Titles = dict[tuple[str, str], str | None]  # (session, agent) -> title, so a session is looked up once
SCHEDULE_HOUR = 3  # the scheduled reflect starts at 03:30 local; a run begun in that hour was the schedule's


def _iso(moment: datetime) -> str:
    if moment.tzinfo is not None:
        moment = moment.astimezone(timezone.utc)
    return moment.strftime(ISO)


def events_added(state: State, before: list[dict], start: int) -> list[dict]:
    """The events a run added to `state.events`, which it began with as `before`, `start` items long.

    consolidate clears state.events when it finishes, so the run's events are those past `start`
    in the list it started with plus whatever replaced it.
    """
    return before[start:] + (state.events if state.events is not before else [])


def live_counts(state: State) -> dict[str, int]:
    """How many lessons are live in each project: what the agents are shown of them."""
    return dict(Counter(lesson.project_root for lesson in state.lessons.values() if lesson.status == "live"))


def run_id(started: datetime) -> str:
    """The `run` of the records of a run that started at `started`."""
    return _iso(started)


def run_records(*, kind: str, trigger: str, started: datetime, ended: datetime, result: dict,
                events: list[dict], failed: str | None = None, digest: str | None = None,
                live: dict[str, int] | None = None) -> list[dict]:
    """The run record, then a record per verdict in `result` and one per event; `run` is the start time.

    `live` is the run's `live_counts` at its end; a run read from the logs has none, and no `live` field.
    """
    run = run_id(started)
    head = {
        "type": "run", "run": run, "kind": kind, "trigger": trigger,
        "started_at": run, "ended_at": _iso(ended),
        "segments_seen": result.get("segments_seen", 0), "no_signals": result.get("no_signals", 0),
        "deferred": result.get("deferred", 0), "llm_calls": result.get("llm_calls", 0),
        "outcomes": dict(result.get("outcomes") or {}), "errors": result.get("errors", 0),
        "failed": failed, "digest": digest,
    }
    if result.get("episodes"):
        # the user's corrections the model found, by what they were about
        head["episodes"] = dict(result["episodes"])
    if result.get("replies"):
        # the user's answers to the agent in what the model read: what the corrections are counted against
        head["replies"] = result["replies"]
    if result.get("detector"):
        head["detector"] = result["detector"]  # which reading of corrections counted them (reflect.DETECTOR)
    if spent := result.get("spent"):
        head["spent"] = spent_summary(spent)
    if live is not None:
        head["live"] = dict(live)
    return [head,
            *({"type": "verdict", "run": run, **v} for v in result.get("verdicts", [])),
            *({"type": "event", "run": run, **e} for e in events)]


def spent_summary(spent: dict) -> dict:
    """What a run's model calls cost, as the journal keeps it: calls, tokens in and out, dollars by API prices."""
    count = lambda name: int(spent.get(name) or 0)  # noqa: E731
    return {"calls": count("calls"),
            "in": count("input_tokens") + count("cache_read_input_tokens") + count("cache_creation_input_tokens"),
            "out": count("output_tokens"), "cost_usd": round(float(spent.get("cost_usd") or 0), 4)}


def _recorded_at(record: dict) -> datetime | None:
    """When the record was made: a run ends, the others carry `ts`, and the run's start is the fallback."""
    return parse_ts(record.get("ended_at" if record.get("type") == "run" else "ts")) or parse_ts(record.get("run"))


def _local(stamp: str, tz: tzinfo) -> datetime | None:
    try:
        return datetime.strptime(stamp, _LOG_TIME).replace(tzinfo=tz)
    except ValueError:
        return None


def _is_count(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _done_result(payload: str) -> dict | None:
    """The counters of a `done` line as `run_records` takes them; None for a dry run or JSON that is no result.

    What the line does not say is worked out: calls the model never answered are `llm_error`, and what was
    read but neither asked about nor left without a project held no signals. Counters a newer log carries win.
    """
    try:
        done = json.loads(payload)
    except ValueError:
        return None
    if not isinstance(done, dict) or "would_call" in done:
        return None
    outcomes = done.get("outcomes") or {}
    counts = {key: done.get(key, 0) for key in ("segments_seen", "llm_calls", "errors")}
    given = {key: done[key] for key in ("no_signals", "deferred") if key in done}
    numbers = (*counts.values(), *given.values(), *(outcomes.values() if isinstance(outcomes, dict) else (None,)))
    if not all(_is_count(n) for n in numbers):
        return None
    outcomes = dict(outcomes)
    answered = sum(n for key, n in outcomes.items() if key != "no_project")
    if counts["llm_calls"] > answered:
        outcomes["llm_error"] = outcomes.get("llm_error", 0) + counts["llm_calls"] - answered
    no_signals = counts["segments_seen"] - counts["llm_calls"] - outcomes.get("no_project", 0)
    return {**counts, "outcomes": outcomes, "no_signals": max(0, no_signals), "deferred": 0, **given}


def _title(cfg: Config, session: str, agent: str, titles: _Titles) -> str | None:
    """The title of a session's transcript (a subagent's, with `agent`), None if it is gone; `titles` caches."""
    if (session, agent) not in titles:
        name = glob.escape(session) + (f"/subagents/{glob.escape(agent)}.jsonl" if agent else ".jsonl")
        try:
            path = next(cfg.projects_dir.glob(f"*/{name}"), None)
            titles[session, agent] = session_title(path) if path else None
        except (OSError, ValueError):
            titles[session, agent] = None
    return titles[session, agent]


def _log_runs(cfg: Config, path: Path, tz: tzinfo, titles: _Titles) -> list[list[dict]]:
    """The records of each run a log tells of, run first: its verdict lines are those above its `done` line.

    A run that crashed has no `done` line: its verdict lines, above its `crashed` line, belong to no run.
    """
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").split("\n")
    except OSError:
        return []
    runs: list[list[dict]] = []
    verdicts: list[tuple[datetime, dict]] = []  # since the last `done` or `crashed` line
    for line in lines:
        if _CRASH_LINE.match(line):
            verdicts = []
        elif m := _VERDICT_LINE.match(line):
            if (at := _local(m[1], tz)) is not None:
                verdicts.append((at, {
                    "ts": _iso(at), "session": m[4], "agent": m[5] or "", "transcript": None, "project": None,
                    "title": _title(cfg, m[4], m[5] or "", titles), "verdict": m[2], "outcome": m[3],
                    "lesson_id": None, "file": None, "notes": m[6]}))
        elif m := _DONE_LINE.match(line):
            ran, verdicts = verdicts, []
            ended, result = _local(m[1], tz), _done_result(m[3])
            if ended is None or result is None:
                continue
            started = ran[0][0] if ran else ended
            result["verdicts"] = [verdict for _, verdict in ran]
            trigger = "schedule" if m[2] == "reflect" and started.hour == SCHEDULE_HOUR else "manual"
            runs.append(run_records(kind=m[2], trigger=trigger, started=started, ended=ended, result=result, events=[]))
    return runs


def import_logs(cfg: Config, tz: tzinfo | None = None) -> list[dict]:
    """The runs and verdicts the reflect and backfill logs tell of, as the journal would hold them.

    Seeds a journal that does not exist yet, so the first report has a history. Log times are local
    (`tz`). A log that cannot be read, a dry run and a line that does not parse add nothing; never raises.
    """
    tz = tz or datetime.now().astimezone().tzinfo
    titles: _Titles = {}
    runs = [run for name in LOGS for run in _log_runs(cfg, cfg.logs_dir / name, tz, titles)]
    runs.sort(key=lambda records: records[0]["started_at"])  # the two logs interleave; ISO times sort as times
    return [record for records in runs for record in records]


def load(cfg: Config, tz: tzinfo | None = None) -> list[dict]:
    """Every record in the file; a line that is not a JSON object (one cut short by a crash) is skipped.

    Without a file the records are those of the logs (see `import_logs`); nothing is written.
    """
    try:
        raw = cfg.journal_file.read_bytes()
    except FileNotFoundError:
        return import_logs(cfg, tz)
    records = []
    # not splitlines(): it also breaks on U+2028, U+2029 and U+0085, which json.dumps writes raw
    for line in raw.decode("utf-8", "replace").split("\n"):
        try:
            record = json.loads(line)
        except ValueError:
            continue
        if isinstance(record, dict):
            records.append(record)
    return records


def append(cfg: Config, records: list[dict], now: datetime, tz: tzinfo | None = None) -> None:
    """Add `records` and drop what is older than RETAIN_DAYS, rewriting the file atomically.

    The first call, with no file yet, writes the records of the logs before `records`.
    A record whose time cannot be read is kept. Raises OSError.
    """
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    cutoff = now - timedelta(days=RETAIN_DAYS)
    kept = [r for r in [*load(cfg, tz), *records] if (at := _recorded_at(r)) is None or at >= cutoff]
    path = cfg.journal_file
    path.parent.mkdir(parents=True, exist_ok=True)
    data = "".join(json.dumps(r, ensure_ascii=False, default=str) + "\n" for r in kept)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
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
