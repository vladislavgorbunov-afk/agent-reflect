"""The weekly look across sessions: the same correction in several sessions is a lesson no single night could see.

Each night the reflector reads one session and journals every correction of the user in one
sentence (`gist`). Once a week this puts the corrections of the last weeks of a project side
by side and asks the model which of them are the same mistake. A group of at least three
corrections from at least two sessions either counts as a repeat of a lesson the project
already has (so a waiting lesson goes live), or becomes a new process lesson with the
sessions as its evidence. At most `MAX_NEW` new lessons a project a week.
"""
from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

from agent_reflect import cursor_index, guard
from agent_reflect.config import Config
from agent_reflect.ledger import Lesson, State, normalize_cue, parse_ts
from agent_reflect.llm import LlmError, run_structured
from agent_reflect.memory import (
    MemoryIndexError, MemoryWriteError, go_live_if_room, lesson_filename, write_lesson,
)
from agent_reflect.projects import memory_dir_for
from agent_reflect.promptkit import DataTags, error_text, load_template, model_text, nonce, squash
from agent_reflect.reflect import FORMS, HOOK_MAX, TITLE_MAX
from agent_reflect.scrub import scrub

WINDOW = timedelta(days=30)  # how far back the corrections are taken from
MIN_EPISODES = 3  # corrections a group needs
MIN_SESSIONS = 2  # and the sessions they must come from
MAX_NEW = 2  # new lessons a project gets a week
MAX_SHOWN = 150  # corrections of a project the model is shown, the latest ones
SAID_SHOWN = 6  # of the user's words a digest item keeps
ISO = "%Y-%m-%dT%H:%M:%SZ"
_NEW_FIELDS = ("title", "hook", "cue", "lesson", "why")
_WRITE_ERRORS = (MemoryWriteError, MemoryIndexError, OSError, UnicodeEncodeError)
_KNOWN = ("live", "staged", "retired")
# lessons that left our hands but still say the thing: the user's own now, or moved into a skill.
# A group that names one is covered already; nothing is written for it.
_COVERED = ("user_owned", "promoted")

_STR = {"type": "string"}
SCHEMA: dict = {
    "type": "object",
    "properties": {
        "notes": _STR,
        "clusters": {"type": "array", "items": {
            "type": "object",
            "properties": {"members": {"type": "array", "items": _STR}, "lesson_id": _STR,
                           **{name: _STR for name in _NEW_FIELDS}},
            "required": ["members"],
            "additionalProperties": False,
        }},
    },
    "required": ["notes", "clusters"],
    "additionalProperties": False,
}
_TAGS = DataTags(("episodes", "auto_lessons", "rejected_cues"))


def _key(project: str, session: str, turn) -> str:
    return f"{project}|{session}|{turn}"


def gather(records: list[dict], since: datetime, used: frozenset | set = frozenset()) -> dict[str, list[dict]]:
    """The corrections journaled since `since` that say what went wrong, by project, numbered e1, e2, ….

    A broken rule that was quoted is left out: the rule is written already, and the digest counts it.
    So is a correction whose key is in `used`: an earlier week put it into a group.
    """
    found: dict[str, list[dict]] = {}
    for record in records:
        at = parse_ts(record.get("ts"))
        project = record.get("project")
        if (record.get("type") != "verdict" or record.get("agent") or not isinstance(project, str) or not project
                or at is None or at < since):
            continue
        for episode in record.get("episodes") or []:
            if not isinstance(episode, dict) or not episode.get("gist") or episode.get("rule"):
                continue
            session = str(record.get("session") or "")
            key = _key(project, session, episode.get("turn"))
            if key in used:
                continue
            items = found.setdefault(project, [])
            items.append({"id": f"e{len(items) + 1}", "session": session, "key": key,
                          "ts": episode.get("ts") or record.get("ts"), "form": episode.get("form"),
                          "gist": str(episode["gist"]), "said": str(episode.get("said") or "")})
    return found


def build_prompt(project: str, episodes: list[dict], lessons: list[Lesson], rejected: list[str]) -> str:
    return load_template("clusters.md").substitute(
        nonce=nonce(),
        project=project,
        episodes=_TAGS.one_per_line(
            f"{e['id']} | {str(e['ts'])[:10]} | {e['session'][:8]} | {e['form']} | {e['gist']} | «{e['said']}»"
            for e in episodes),
        auto_lessons=_TAGS.one_per_line(f"{lesson.id} [{lesson.status}]: {lesson.cue} → {lesson.lesson}"
                                        for lesson in lessons),
        rejected=_TAGS.one_per_line(rejected),
    )


def _sessions(members: list[dict]) -> list[str]:
    return list(dict.fromkeys(e["session"] for e in members))


def _evidence(members: list[dict], sessions: list[str]) -> list[dict]:
    """One line of evidence per session: its first correction, in the user's own words."""
    first = {session: next(e for e in members if e["session"] == session) for session in sessions}
    return [{"session": session, "ts": e["ts"], "text": f"«{e['said']}» — {e['gist']}"} for session, e in first.items()]


def _item(outcome: str, lesson: Lesson, members: list[dict]) -> dict:
    return {"outcome": outcome, "lesson_id": lesson.id, "title": lesson.title, "sessions": len(_sessions(members)),
            "count": len(members), "said": [e["said"] for e in members if e["said"]][:SAID_SHOWN]}


def _repeat(cfg: Config, state: State, lesson: Lesson, members: list[dict], now_iso: str) -> dict | None:
    """Count the group's sessions the lesson has not seen yet; a waiting lesson then goes live. None when none is new."""
    seen = {e.get("session") for e in lesson.evidence if isinstance(e, dict)}
    fresh = [session for session in _sessions(members) if session not in seen]
    memory_dir = Path(lesson.memory_dir)
    try:
        unchanged = hashlib.sha256((memory_dir / lesson.file).read_bytes()).hexdigest() == lesson.sha256
    except OSError:
        unchanged = False
    if not fresh or not unchanged:  # an edited file is the user's: the nightly run hands it over
        return None
    status = "staged" if lesson.status == "retired" else lesson.status
    updated = replace(lesson, repeats=lesson.repeats + len(fresh), status=status, updated_at=now_iso,
                      evidence=[*lesson.evidence, *_evidence(members, fresh)])
    updated.sha256 = write_lesson(memory_dir, updated)
    state.lessons[lesson.id] = updated
    if updated.status == "staged":
        go_live_if_room(cfg, updated, now_iso, state)
    state.event("lesson_repeat", project=updated.project_root, session=fresh[0], lesson_id=updated.id,
                repeats=updated.repeats, counted=True, status=updated.status, source="clusters")
    return _item("repeat", updated, members)


def _new(cfg: Config, state: State, project: str, memory_dir: Path, cluster: dict, members: list[dict],
         now: datetime) -> dict | None:
    fields = {name: model_text(cluster.get(name)) for name in _NEW_FIELDS}
    fields["title"] = squash(fields["title"])[:TITLE_MAX].rstrip()
    fields["hook"] = squash(fields["hook"])[:HOOK_MAX].rstrip()
    if not all(fields.values()):
        state.event("cluster_dropped", project=project, reason="a new lesson without " + ", ".join(
            name for name, value in fields.items() if not value))
        return None
    cue = normalize_cue(fields["cue"])
    now_iso = now.strftime(ISO)
    same = next((lesson for lesson in state.lessons.values() if lesson.project_root == project
                 and lesson.status in _KNOWN and normalize_cue(lesson.cue) == cue), None)
    if same is not None:
        return _repeat(cfg, state, same, members, now_iso)
    sessions = _sessions(members)
    forms = Counter(e["form"] for e in members if e["form"] in FORMS)
    lesson_id = hashlib.sha1((project + fields["cue"] + now_iso).encode("utf-8")).hexdigest()[:8]
    if lesson_id in state.lessons:
        return None
    # dated just before the run: the digest of this run shows it among the groups, and the next
    # one, which lists the lessons created since this run, must not call it new again
    created = (now - timedelta(seconds=1)).strftime(ISO)
    lesson = Lesson(
        id=lesson_id, project_root=project, memory_dir=str(memory_dir), file=lesson_filename(lesson_id),
        title=fields["title"], hook=fields["hook"], cue=fields["cue"], lesson=fields["lesson"], why=fields["why"],
        confidence="medium", repeats=len(sessions), evidence=_evidence(members, sessions), status="staged",
        sha256="", created_at=created, updated_at=now_iso, kind="process",
        form=forms.most_common(1)[0][0] if forms else "")
    lesson.sha256 = write_lesson(memory_dir, lesson)
    state.lessons[lesson_id] = lesson
    go_live_if_room(cfg, lesson, now_iso, state)
    state.event("lesson_new", project=project, session=sessions[0], lesson_id=lesson_id, status=lesson.status,
                title=lesson.title, source="clusters")
    return _item("new", lesson, members)


def step(cfg: Config, state: State, *, now: datetime, llm=run_structured, records: list[dict]) -> list[dict]:
    """Look across the sessions of each project; returns what the digest says of the groups that were written.

    A project the model did not answer about, or whose memory could not be written, is an event;
    the others go on.
    """
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    now_iso = now.strftime(ISO)
    out: list[dict] = []
    since = now - WINDOW
    # a mark older than the window marks a correction that is not offered any more anyway
    state.clustered = {key: ts for key, ts in state.clustered.items()
                       if (at := parse_ts(ts)) is not None and at >= since}
    for project, episodes in sorted(gather(records, since, set(state.clustered)).items()):
        episodes = episodes[-MAX_SHOWN:]
        if len(episodes) < MIN_EPISODES or len(_sessions(episodes)) < MIN_SESSIONS:
            continue
        lessons = [lesson for lesson in state.lessons.values()
                   if lesson.project_root == project and lesson.status in (*_KNOWN, *_COVERED)]
        prompt = build_prompt(project, episodes, lessons, state.rejected_cues.get(project, []))
        try:
            answer = llm(cfg, cfg.consolidate_model, prompt, SCHEMA, effort=cfg.consolidate_effort)
        except LlmError as exc:
            state.event("clusters_failed", project=project, error=error_text(exc))
            continue
        by_id = {e["id"]: e for e in episodes}
        memory_dir = memory_dir_for(cfg.projects_dir, Path(project))
        used: set[str] = set()
        made = 0
        changed = False
        listed = answer.get("clusters") if isinstance(answer, dict) else None
        for cluster in listed if isinstance(listed, list) else []:
            if not isinstance(cluster, dict) or not isinstance(cluster.get("members"), list):
                continue
            ids = list(dict.fromkeys(m.strip("[] ") for m in cluster["members"] if isinstance(m, str)))
            members = [by_id[i] for i in ids if i in by_id and i not in used]
            if len(members) < MIN_EPISODES or len(_sessions(members)) < MIN_SESSIONS:
                continue
            known = state.lessons.get(model_text(cluster.get("lesson_id")))
            if known is not None and known.project_root != project:
                known = None
            cue = model_text(cluster.get("cue"))
            item, settled = None, True  # settled: these corrections have had their group, written or not
            try:
                if known is not None and known.status in _KNOWN:
                    item = _repeat(cfg, state, known, members, now_iso)
                elif known is not None and known.status in _COVERED:
                    pass  # the user's own lesson, or a skill, says it already
                elif known is None and cue and normalize_cue(cue) in state.rejected_cues.get(project, []):
                    state.event("rejected_cue", project=project, session="", cue=cue)
                elif made < MAX_NEW:
                    item = _new(cfg, state, project, memory_dir, cluster, members, now)
                    made += int(item is not None and item["outcome"] == "new")
                    settled = item is not None
                else:
                    settled = False  # over this week's cap: next week may write it
            except _WRITE_ERRORS as exc:
                state.event("memory_write_failed", project=project, step="clusters", error=error_text(exc))
                continue
            if settled:
                used.update(e["id"] for e in members)
                state.clustered.update({e["key"]: now_iso for e in members})
            if item is not None:
                out.append({**item, "said": [scrub(text) for text in item["said"]]})
                changed = True
        if changed:
            guard.sync(cfg, state)
            cursor_index.sync(cfg, Path(project), memory_dir)
    return out
