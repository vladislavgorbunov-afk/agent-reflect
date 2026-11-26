"""`agent-reflect baseline`: how often the user sent work back in past days, read by today's detector.

The nightly run counts the user's corrections from the day it learnt to; the weeks before were
counted by a word list, or not at all. This reads those weeks' sessions again with the same
prompt and the same checks the nightly run uses, keeps only the count (corrections per answer
of the user, by day), and writes it to `state/baseline.json`. No lesson is written, no offset
moves, memory is not touched: the model's entries are thrown away.

What was asked is kept in `state/baseline-cache.jsonl`, so a run that was cut short, or capped,
goes on from where it stopped.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import date, datetime, tzinfo
from pathlib import Path

from . import reflect, session_roots
from .config import Config
from .journal import spent_summary
from .ledger import State, parse_ts
from .llm import LlmError, run_structured, take_spent
from .memory import MemoryIndexError, read_index
from .projects import memory_dir_for
from .promptkit import error_text
from .transcript import Segment, is_subagent, list_transcripts, read_segment

log = logging.getLogger(__name__)

MAX_PARTS = 60  # of one session: the nightly run reads a day's slice, this reads the whole session
JOBS = 3  # model calls at a time
CACHE_NAME = "baseline-cache.jsonl"
ISO = "%Y-%m-%dT%H:%M:%SZ"


@dataclass
class _Part:
    """One text of a session that holds answers of the user from the days asked about."""
    path: Path
    project_root: Path
    segment: Segment
    text: str
    turns: dict[int, str]  # the answers in it from those days: number -> its local day
    key: str = ""


def _day(ts: str | None, tz: tzinfo) -> date | None:
    moment = parse_ts(ts)
    return moment.astimezone(tz).date() if moment else None


def plan(cfg: Config, state: State, since: date, until: date, tz: tzinfo) -> list[_Part]:
    """The parts to ask about: of every session of the user in a project, those with an answer in the days."""
    floor = datetime.combine(since, datetime.min.time(), tz).timestamp()
    parts: list[_Part] = []
    roots = session_roots.load(cfg)
    for path in list_transcripts(cfg.projects_dir, cfg.own_project_prefixes):
        try:
            if is_subagent(path) or path.stat().st_mtime < floor:
                continue
            segment = read_segment(path, 0)
            days = {t.n: day for t in segment.replies
                    if (day := _day(t.ts, tz)) is not None and since <= day <= until}
            if not days:
                continue
            project_root = reflect.project_for(cfg, reflect._start_cwd(state, path), reflect.noted_root(roots, path))
            if project_root is None:
                continue
            for text in segment.render_parts(cfg.segment_max_chars, MAX_PARTS):
                shown = reflect.shown_turns(text)
                if turns := {n: day.isoformat() for n, day in days.items() if n in shown}:
                    digest = hashlib.sha1(text.encode("utf-8", "replace")).hexdigest()[:16]
                    # no days in the key: what the model said of a text holds for any days asked about
                    parts.append(_Part(path, project_root, segment, text, turns,
                                       f"{reflect.DETECTOR}:{path}:{digest}"))
        except OSError as exc:
            log.warning("baseline: %s not read: %s", path.name, error_text(exc))
    return parts


def _load_cache(cfg: Config) -> dict[str, list]:
    cache: dict[str, list] = {}
    try:
        lines = (cfg.state_dir / CACHE_NAME).read_text(encoding="utf-8").split("\n")
    except OSError:
        return cache
    for line in lines:
        try:
            entry = json.loads(line)
        except ValueError:
            continue
        if isinstance(entry, dict) and isinstance(entry.get("key"), str) and isinstance(entry.get("episodes"), list):
            cache[entry["key"]] = entry["episodes"]
    return cache


def _remember(cfg: Config, key: str, episodes: list) -> None:
    try:
        cfg.state_dir.mkdir(parents=True, exist_ok=True)
        with (cfg.state_dir / CACHE_NAME).open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"key": key, "episodes": episodes}, ensure_ascii=False) + "\n")
    except OSError as exc:  # the answer still counts in this run
        log.warning("baseline: cache not written: %s", error_text(exc))


def _write(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(json.dumps(data, ensure_ascii=False, indent=1) + "\n")
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def run(cfg: Config, state: State, *, since: date, until: date, now: datetime, llm=run_structured,
        tz: tzinfo | None = None, dry_run: bool = False, max_calls: int | None = None, jobs: int = JOBS) -> dict:
    """Count the corrections of the days from `since` to `until` and write the baseline; returns it.

    `state` is only read (where each session started). A dry run asks nothing and writes nothing:
    it says how many sessions and model calls the count would take.
    """
    tz = tz or datetime.now().astimezone().tzinfo
    parts = plan(cfg, state, since, until, tz)
    if dry_run:
        return {"sessions": len({p.path for p in parts}), "calls": len(parts),
                "chars": sum(len(p.text) for p in parts), "replies": sum(len(p.turns) for p in parts)}
    cache = _load_cache(cfg)
    pending = [p for p in parts if p.key not in cache]
    left = max(0, len(pending) - max_calls) if max_calls is not None else 0
    pending = pending[:max_calls] if max_calls is not None else pending
    take_spent()
    user_rules = reflect.read_user_rules(cfg)
    context: dict[Path, tuple[str, str]] = {}  # project -> its memory index and its own rules

    def context_of(project_root: Path) -> tuple[str, str]:
        if project_root not in context:
            try:
                index = read_index(memory_dir_for(cfg.projects_dir, project_root))
            except (MemoryIndexError, OSError):
                index = ""
            context[project_root] = (index, reflect.read_project_rules(project_root))
        return context[project_root]

    for part in parts:
        context_of(part.project_root)  # read before the threads start

    def ask(part: _Part):
        index, project_rules = context[part.project_root]
        prompt = reflect.build_prompt(part.text, part.project_root, index, [], [], user_rules, project_rules)
        try:
            answer = llm(cfg, cfg.reflect_model, prompt, reflect.REFLECT_SCHEMA, effort=cfg.reflect_effort)
        except LlmError as exc:
            return exc
        _, _, raw = reflect.entries_of(answer)
        found = reflect.episodes_of(part.segment, raw, "\n".join((user_rules, project_rules, index)))
        # every correction of the part is kept: which of them fall on the days asked about is decided when counting
        return [{"turn": e["turn"], "form": e["form"]} for e in found]

    errors = 0
    with ThreadPoolExecutor(max_workers=max(1, jobs)) as pool:
        for i, (part, outcome) in enumerate(zip(pending, pool.map(ask, pending)), 1):
            if isinstance(outcome, Exception):
                errors += 1
                log.warning("baseline: %d/%d %s not answered: %s", i, len(pending), part.path.name, error_text(outcome))
                continue
            cache[part.key] = outcome
            _remember(cfg, part.key, outcome)
            log.info("baseline: %d/%d %s · answers %d, sent back %d", i, len(pending), part.path.name,
                     len(part.turns), sum(1 for e in outcome if e["turn"] in part.turns))

    days: dict[str, Counter] = {}
    forms: Counter = Counter()
    sessions: set[Path] = set()
    for part in parts:
        if part.key not in cache:
            continue
        sessions.add(part.path)
        for day in part.turns.values():
            days.setdefault(day, Counter())["replies"] += 1
        for episode in cache[part.key]:
            day = part.turns.get(episode.get("turn"))
            if day is not None and episode.get("form") in reflect.FORMS:
                days[day]["episodes"] += 1
                forms[episode["form"]] += 1
    result = {
        "made_at": now.strftime(ISO), "since": since.isoformat(), "until": until.isoformat(),
        "detector": reflect.DETECTOR, "sessions": len(sessions), "calls": len(pending), "errors": errors,
        "partial": bool(left or errors), "left": left,
        "replies": sum(day["replies"] for day in days.values()),
        "episodes": sum(day["episodes"] for day in days.values()),
        "forms": dict(forms.most_common()),
        "days": {day: {"replies": days[day]["replies"], "episodes": days[day]["episodes"]} for day in sorted(days)},
    }
    if spent := take_spent():
        result["spent"] = spent_summary(spent)
    _write(cfg.baseline_file, result)
    return result
