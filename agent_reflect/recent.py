"""`agent-reflect recent`: the night as the journal tells it, a chart and cards for what needs reading.

A pure function of the state, the journal records and `now`; the one thing read is MEMORY.md, for its gauge.
"""
from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone, tzinfo
from pathlib import Path
from typing import Callable, NamedTuple

from . import chart, health, i18n, usage
from .config import Config
from .i18n import quoted, t
from .ledger import Lesson, State, parse_ts
from .memory import lesson_filename
from .promptkit import squash
from .scrub import scrub

DAY = timedelta(hours=24)
CHART_DAYS = 7
SECTION_COLUMNS = 61  # a section's ── line runs up to this column
CARD_COLUMNS = 76  # no line of cards, decisions or quiet failures runs past this column
REASON_MAX = 60  # the short reason of a decision
REASON_MIN = 12  # a decision's title too long for its line gives way to this much of the reason
_FRAME = "│  "
_TEXT_COLUMNS = CARD_COLUMNS - len(_FRAME)
_ERROR, _NOTHING = "🟥", "⬜"
_SENTENCE_END = re.compile(r"(?<=[.!?])\s")
_QUOTE = re.compile(r"«[^«»]+»")
_columns = chart.columns  # how many terminal columns a text takes

# a lesson's events, weightiest first: its card is headed by the weightiest it had in the window
_LESSON_EVENTS = ("lesson_rejected", "lesson_unlisted", "lesson_user_owned", "lesson_violated",
                  "lesson_after_promotion", "lesson_blocked", "lesson_live", "lesson_new", "lesson_repeat")
# texts are (English, Russian) pairs, said with i18n.t
_LESSON_HEADS = {
    "lesson_unlisted": ("➖ lesson taken out of MEMORY.md: no room left", "➖ урок убран из MEMORY.md: не хватило места"),
    "lesson_user_owned": ("✋ the lesson is yours now: you edited its file", "✋ урок перешёл к тебе: ты правил файл"),
    "lesson_violated": ("⚠️ lesson broken after it went live", "⚠️ урок нарушен после публикации"),
    "lesson_after_promotion": ("↩️ a moved lesson's mistake came back", "↩️ ошибка перенесённого урока повторилась"),
    "lesson_blocked": ("⛔ the hook refused a command by this lesson", "⛔ хук отклонил команду по этому уроку"),
    "lesson_live": ("➕ lesson now visible to agents", "➕ урок стал виден агентам"),
    "lesson_new": ("🆕 new lesson", "🆕 новый урок"),
    "lesson_repeat": ("🔁 lesson repeated", "🔁 урок повторился"),
}
_REJECTED = {"file_deleted": ("🗑 lesson rejected: you deleted its file", "🗑 урок отклонён: ты удалил файл"),
             "index_line_deleted": ("🗑 lesson rejected: you removed its line from MEMORY.md",
                                    "🗑 урок отклонён: ты убрал строку из MEMORY.md")}
# what went wrong, by the 🟥 outcome of a decision or by the kind of an error event
_ERROR_OUTCOMES = {"llm_error": ("the model did not answer", "модель не ответила"),
                   "bad_verdict": ("the model's answer could not be read", "ответ модели не разобран"),
                   "write_failed": ("the lesson was not written", "урок не записался"),
                   "apply_crashed": ("the model's decision was not applied", "решение модели не применилось")}
_ERROR_EVENTS = {"transcript_failed": ("transcript not read", "транскрипт не разобран"),
                 "memory_write_failed": ("memory not written", "память не записалась"),
                 "index_unreadable": ("MEMORY.md unreadable", "MEMORY.md не читается"),
                 "segment_unreadable": ("segment unreadable", "отрезок не читается"),
                 "autodream_not_disabled": ("autoDream is not off", "autoDream не выключен"),
                 "guard_rules_unwritten": ("hook rules not written", "правила хука не записались"),
                 "guard_fires_unread": ("hook showings not read", "показы хука не прочитаны")}
# events that say why a decision came out 🟥
_DECISION_FAILURES = ("llm_error", "bad_verdict", "memory_write_failed", "apply_crashed")


class _Card(NamedTuple):
    head: str
    body: list[str]  # lines without the frame, already wrapped
    foot: list[str]  # the first goes after └─, the rest under it


class _Tally(NamedTuple):
    """What the report and the notification say of a window, computed once for both."""
    nightly: dict | None
    runs: list[dict]
    verdicts: list[dict]
    events: list[dict]
    changes: int
    memory: list[str]
    attention: list[_Card]
    errors: int
    seen: int
    asked: int
    alerts: list[dict]


def _text(value) -> str:
    return value if isinstance(value, str) else ""


def _count(value) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def _started(run: dict) -> datetime | None:
    return parse_ts(run.get("started_at"))


def _at(record: dict) -> datetime | None:
    return parse_ts(record.get("ts")) or parse_ts(record.get("run"))


def _local_day(record: dict, tz: tzinfo):
    """The local day a run counts on in the chart: the one it ended on."""
    return (parse_ts(record.get("ended_at")) or _started(record)).astimezone(tz).date()


def _segments(n: int) -> str:
    return i18n.count(n, chart.SEGMENT_EN, chart.SEGMENT_RU)


def _project(path) -> str:
    return Path(_text(path)).name or _text(path)


def window(records: list[dict], now: datetime, hours: int | None,
           tz: tzinfo | None = None) -> tuple[datetime, dict | None, list[dict]]:
    """Where the report starts, the nightly run it is about, and the runs from that start on, oldest first.

    By default the start is that of the nightly run: of the scheduled reflects of the past 24 hours,
    the first on the local (`tz`) day of the last one. launchd also starts the job at login, so a run
    after a reboot is scheduled too, and it must not cut the night off. Without a scheduled reflect the
    start is 24 hours back and there is no nightly run. `hours` makes it the last `hours` hours, with no
    nightly run either.
    """
    tz = tz or datetime.now().astimezone().tzinfo
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    runs = sorted((r for r in records if r.get("type") == "run" and _started(r) is not None), key=_started)
    start = now - (DAY if hours is None else timedelta(hours=hours))
    nightly = None
    if hours is None:
        scheduled = [run for run in runs if run.get("kind") == "reflect" and run.get("trigger") == "schedule"
                     and _started(run) >= start]
        if scheduled:
            day = _started(scheduled[-1]).astimezone(tz).date()
            nightly = next(run for run in scheduled if _started(run).astimezone(tz).date() == day)
            start = _started(nightly)
    return start, nightly, [run for run in runs if _started(run) >= start]


def _words(text: str) -> list[str]:
    """The words of `text`; a one-letter word stays with the next, so no line ends on «с» or «в»."""
    words: list[str] = []
    glue = False
    for word in text.split():
        if glue:
            words[-1] += " " + word
        else:
            words.append(word)
        glue = len(word) == 1 and word.isalpha()
    return words


def _wrap(text: str, width: int, lead: str = "", hang: str = "") -> list[str]:
    """`text` in lines of at most `width` columns, the first after `lead` and the others after `hang`.

    A word longer than a line is cut. The text is scrubbed before it is cut: a secret split
    over two lines would no longer match.
    """
    lines: list[str] = []
    line, fresh = lead, True  # fresh: nothing on the line but its indent
    for word in _words(scrub(squash(text))):
        while word:
            joined = line + word if fresh else f"{line} {word}"
            if _columns(joined) <= width:
                line, fresh, word = joined, False, ""
            elif not fresh:
                lines.append(line)
                line, fresh = hang, True
            else:
                cut = len(word)
                while cut > 1 and _columns(line + word[:cut]) > width:
                    cut -= 1
                lines.append(line + word[:cut])
                line, word = hang, word[cut:]
    if not fresh:
        lines.append(line)
    return lines


def _para(text: str, lead: str = "", hang: str = "") -> list[str]:
    """Card text: what fits beside the frame."""
    return _wrap(text, _TEXT_COLUMNS, lead, hang)


def _draw(cards: list[_Card]) -> list[str]:
    out: list[str] = []
    for i, card in enumerate(cards, 1):
        if i > 1:
            out.append("")
        out.append(f"┌─ {i}/{len(cards)} {card.head}")
        out += [_FRAME + line for line in card.body]
        out += [("└─ " if j == 0 else "   ") + line for j, line in enumerate(card.foot)]
    return out


def _rule(title: str) -> str:
    head = f"── {title} "
    return head + "─" * max(2, SECTION_COLUMNS - _columns(head))


def _clip(text: str, columns: int) -> str:
    """`text` within `columns` columns: cut with "…", at the end of a word unless that loses over a third.

    The text is scrubbed before it is cut: the first few characters of a secret would no longer match.
    """
    text = scrub(text)
    if _columns(text) <= columns:
        return text
    cut = len(text)
    while cut and _columns(text[:cut] + "…") > columns:
        cut -= 1
    if text[cut:cut + 1] != " " and (space := text.rfind(" ", 0, cut)) >= cut - cut // 3:
        cut = space
    return text[:cut].rstrip() + "…" if cut else ""


def _sentence(notes) -> str:
    """The first sentence of the model's notes, without its full stop."""
    sentence = _SENTENCE_END.split(squash(_text(notes)), maxsplit=1)[0]
    return sentence[:-1] if sentence.endswith(".") and not sentence.endswith("..") else sentence


def _short(notes) -> str:
    return _clip(_sentence(notes), REASON_MAX)


def _name(verdict: dict | None, session) -> str:
    """The session's title as the decision recorded it, else its short id."""
    title = squash(_text(verdict.get("title"))) if verdict else ""
    return title or t("session ", "сессия ") + _text(session)[:8]


def _source(journal: list[dict], event: dict, prefer: Callable[[dict], bool]) -> dict | None:
    """The decision an event of a session came from: one of its run, the `prefer`red, the main session's."""
    same = [v for v in journal if v.get("session") == event.get("session")]
    return min(same, key=lambda v: (v.get("run") != event.get("run"), not prefer(v), bool(v.get("agent"))),
               default=None)


# -- the night's lessons --------------------------------------------------------


def _lesson_head(lesson: Lesson | None, events: list[dict]) -> str:
    kinds = {event["kind"] for event in events}
    if lesson is not None and lesson.status == "live" and "lesson_live" in kinds:
        kinds.discard("lesson_unlisted")  # taken out and put back: rebalance can do both in one call
    kind = min(kinds, key=_LESSON_EVENTS.index)
    if kind == "lesson_rejected":
        reasons = [_text(e.get("reason")) for e in events if e["kind"] == kind]
        return t(*_REJECTED.get(reasons[-1], ("🗑 lesson rejected", "🗑 урок отклонён")))
    return t(*_LESSON_HEADS[kind])


def _since(moment: datetime, now: datetime, tz: tzinfo) -> str:
    """03:30 on the day of the report, 21.10 before it."""
    local = moment.astimezone(tz)
    return f"{local:%H:%M}" if local.date() == now.astimezone(tz).date() else chart.short_date(local.date())


def _lesson_numbers(lesson: Lesson, events: list[dict], now: datetime, tz: tzinfo) -> str:
    repeats = sum(1 for e in events if e["kind"] == "lesson_repeat" and e.get("counted"))
    sessions = len({e.get("session") for e in lesson.evidence if isinstance(e, dict) and e.get("session")})
    in_sessions = i18n.count(sessions, ("session", "sessions"), ("сессия", "сессии", "сессий"))
    total = t(f"{in_sessions} in all", f"всего {in_sessions}")
    if repeats:
        times = i18n.count(repeats, ("repeat", "repeats"), ("повтор", "повтора", "повторов"))
        total = t(f"🔁 +{times} overnight, {total}", f"🔁 +{times} за ночь, {total}")
    parts = [total]
    since = parse_ts(lesson.live_since or lesson.created_at)
    if lesson.status == "live" and since is not None:
        parts.append(t("👀 since ", "👀 с ") + _since(since, now, tz))
    parts.append(t("📖 reads: ", "📖 прочтений: ") + str(lesson.reads))
    return " · ".join(parts)


def _lesson_card(lesson_id: str, lesson: Lesson | None, events: list[dict], now: datetime, tz: tzinfo) -> _Card:
    """All a lesson's events of the window on one card; one gone from the state has its events' title only."""
    project = lesson.project_root if lesson else next((e["project"] for e in events if _text(e.get("project"))), "")
    head = _lesson_head(lesson, events) + (f" · {_project(project)}" if project else "")
    title = lesson.title if lesson else next((e["title"] for e in events if _text(e.get("title"))), "")
    body = _para(quoted(title or t("lesson ", "урок ") + lesson_id))
    if lesson is None:
        return _Card(head, body, [lesson_filename(lesson_id)])
    body += _para(lesson.cue, t("when:    ", "когда:   "), " " * 9)
    body += _para(lesson.lesson, t("do:      ", "делать:  "), " " * 9)
    body += _para(_lesson_numbers(lesson, events, now, tz))
    if lesson.status == "staged":
        waits = lesson.confidence != "high" and lesson.repeats < 2
        body.append(t("⏳ waits for a second case", "⏳ ждёт второго случая") if waits
                    else t("⏳ no room in MEMORY.md", "⏳ в MEMORY.md нет места"))
        if waits:
            body += _para(t("accept now: ", "принять сразу: ") + f"agent-reflect approve {lesson.id}")
    if lesson.violations:
        body.append(t("⚠️ broken after it went live: ", "⚠️ нарушен после публикации: ") + i18n.times(lesson.violations))
    if lesson.status == "promoted":
        body += _para(_moved_line(lesson, now, tz))
    if hooks := hook_patterns(lesson):
        body += _para(t("🪝 hook: ", "🪝 хук: ") + hooks)
    if lesson.kind == "fact" and lesson.recheck:
        body += _para(lesson.recheck, t("🔎 how to recheck: ", "🔎 как перепроверить: "), " " * 9)
    # a moved lesson's file is gone: the skill holds its rule now
    return _Card(head, body, [lesson.promoted_to or t("a skill", "скилл")] if lesson.status == "promoted"
                 else [lesson.file])


def _moved_line(lesson: Lesson, now: datetime, tz: tzinfo) -> str:
    """Where a moved lesson went and since when, and how many sessions made its mistake since."""
    after = len({a.get("session") for a in lesson.after if isinstance(a, dict) and a.get("session")})
    moved, since = parse_ts(lesson.promoted_at), parse_ts(lesson.moved_since)
    if lesson.effect == "rolled_back":
        when = t("the move was undone by a rollback", "перенос отменён откатом")
    elif since is not None:
        when = t("in the skill since ", "в скилле с ") + _since(since, now, tz)
    elif lesson.promoted_pr:
        when = t("waits for its PR's merge", "ждёт мержа PR")
    else:
        when = t("moved into a skill ", "перенесён в скилл ") + (_since(moved, now, tz) if moved is not None else "?")
    return f"🛠 {when}; " + t(f"sessions with the mistake since it left memory: {after}",
                              f"сессий с ошибкой после переноса: {after}")


def hook_patterns(lesson: Lesson) -> str:
    """The lesson's patterns for the hook, each with what it is matched against, and whether it blocks."""
    names = {"command": t("command", "команда"), "error": t("error", "ошибка"), "path": t("file", "файл")}
    parts = [f"{names[on]} /{pattern}/" for on, pattern in lesson.patterns()]
    if parts and lesson.block and lesson.trigger:
        parts.append(t("blocks", "блокирует"))
    return " · ".join(parts)


def _put_back(lesson: Lesson | None, events: list[dict]) -> bool:
    """Live before the window and after it, its line only taken out and put back: rebalance can do both in one call."""
    return (lesson is not None and lesson.status == "live" and events[0]["kind"] == "lesson_unlisted"
            and {event["kind"] for event in events} == {"lesson_unlisted", "lesson_live"})


def _memory(state: State, runs: list[dict], events: list[dict], now: datetime, tz: tzinfo) -> tuple[int, list[str]]:
    """The number of lesson cards and the section's lines: the cards, then a line per weekly clean-up.

    A lesson whose line was only taken out and put back changed nothing, and has no card.
    """
    by_lesson: dict[str, list[dict]] = {}
    for event in events:
        if event.get("kind") in _LESSON_EVENTS and _text(event.get("lesson_id")):
            by_lesson.setdefault(event["lesson_id"], []).append(event)
    cards = [_lesson_card(lesson_id, state.lessons.get(lesson_id), lesson_events, now, tz)
             for lesson_id, lesson_events in by_lesson.items()
             if not _put_back(state.lessons.get(lesson_id), lesson_events)]
    lines = _draw(cards)
    digests = [t("🧹 the weekly clean-up ran: ", "🧹 прошла недельная чистка: ") + _text(run.get("digest")) for run in runs
               if run.get("kind") == "consolidate" and _text(run.get("digest"))]
    if lines and digests:
        lines.append("")
    return len(cards), lines + digests


# -- what needs the user --------------------------------------------------------


def _conflict_card(event: dict, journal: list[dict]) -> _Card:
    file = _project(event.get("file"))
    source = _source(journal, event, lambda v: v.get("file") == file)
    notes = squash(_text(event.get("notes")))
    said = quoted(_name(source, event.get("session"))) + (f": {notes}" if notes else "")
    return _Card(t("🤨 a note looks out of date", "🤨 запись, похоже, устарела"), [file, *_para(said)] if file else _para(said),
                 [t("👉 check the way again and fix the note", "👉 перепроверить способ и поправить запись")])


def _rule_card(event: dict, journal: list[dict], state: State, tz: tzinfo, first: bool) -> _Card:
    """A broken rule of the user's; the meme is on the `first` such card of the report only."""
    file = _project(event.get("file"))
    source = _source(journal, event, lambda v: v.get("outcome") == "user_rule_repeat")
    notes = _text(source.get("notes")) if source else ""
    quote = m[0] if (m := _QUOTE.search(notes)) else _short(notes)
    said = quoted(_name(source, event.get("session"))) + (f": {quote}" if quote else "")
    installed = parse_ts(state.installed_at)
    since = t(" since ", " с ") + chart.short_date(installed.astimezone(tz).date()) if installed else ""
    count = len(state.user_rule_repeats.get(_text(event.get("file")), []))
    head = (t("🔁 here we go again: your rule was broken", "🔁 опять двадцать пять: нарушено твоё правило") if first
            else t("🔁 your rule was broken", "🔁 нарушено твоё правило"))
    return _Card(head, [file, *_para(said)] if file else _para(said), [t("broken", "нарушений") + f"{since}: {count}"])


def _fixed(verdict: dict, journal: list[dict]) -> bool:
    """A later decision on the same session and subagent was no error: a later run read it fine."""
    at = _at(verdict)
    key = (verdict.get("session"), verdict.get("agent") or "")
    return at is not None and any(
        (v.get("session"), v.get("agent") or "") == key and chart.square(v.get("outcome")) != _ERROR
        and (later := _at(v)) is not None and later > at for v in journal)


def _covered(event: dict, verdicts: list[dict]) -> bool:
    """An error of a session's decision is told by that decision's 🟥."""
    return bool(event.get("session")) and any(
        v.get("run") == event.get("run") and v.get("session") == event.get("session")
        and chart.square(v.get("outcome")) == _ERROR for v in verdicts)


def _failure(verdict: dict, events: list[dict]) -> str:
    """Why a decision is 🟥: the error its run recorded for it, else the notes."""
    for event in events:
        if (event.get("run") == verdict.get("run") and event.get("kind") in _DECISION_FAILURES
                and (event.get("session") == verdict.get("session") and verdict.get("session")
                     or event.get("transcript") == verdict.get("transcript") and verdict.get("transcript"))):
            if reason := _text(event.get("error")) or _text(event.get("reason")):
                return reason
    return _text(verdict.get("notes"))


def _where(event: dict, journal: list[dict]) -> str:
    """The session, transcript or project an error event is about."""
    if event.get("session"):
        return quoted(_name(_source(journal, event, lambda v: True), event["session"]))
    if transcript := _text(event.get("transcript")):
        title = next((v["title"] for v in journal if v.get("transcript") == transcript and _text(v.get("title"))),
                     None)
        return quoted(squash(title)) if title else Path(transcript).name
    return _project(event.get("project")) or _text(event.get("settings"))


def _error_card(head: str, where: str, text: str, log: str) -> _Card:
    return _Card(f"🔥 {head}", _para(where) + _para(text), [t("👉 see ", "👉 смотри ") + f"logs/{log}.log"])


def _error_cards(runs: list[dict], verdicts: list[dict], events: list[dict], journal: list[dict]) -> list[_Card]:
    """A card per error of the window that no later run took back: decisions, error events, crashed runs."""
    logs = {run.get("run"): _text(run.get("kind")) or "reflect" for run in runs}
    cards = [_error_card(t(*_ERROR_OUTCOMES.get(v.get("outcome"), ("error", "ошибка"))), quoted(_name(v, v.get("session"))),
                         _failure(v, events), logs[v.get("run")])
             for v in verdicts if chart.square(v.get("outcome")) == _ERROR and not _fixed(v, journal)]
    cards += [_error_card(t(*_ERROR_EVENTS[e["kind"]]), _where(e, journal),
                          _text(e.get("error")) or _text(e.get("reason")), logs[e.get("run")])
              for e in events if e.get("kind") in _ERROR_EVENTS and not _covered(e, verdicts)]
    cards += [_error_card(t("the run crashed", "прогон упал"), "", _text(run["failed"]), logs[run.get("run")])
              for run in runs if run.get("failed") and run.get("kind") != "consolidate"]
    return cards


def _attention(state: State, runs: list[dict], verdicts: list[dict], events: list[dict], journal: list[dict],
               now: datetime, tz: tzinfo) -> tuple[list[_Card], int]:
    """The cards for the user, and how many of them are errors."""
    cards = [_conflict_card(e, journal) for e in events if e.get("kind") == "user_rule_conflict"]
    rules = [e for e in events if e.get("kind") == "user_rule_repeat"]
    cards += [_rule_card(e, journal, state, tz, first=i == 0) for i, e in enumerate(rules)]
    errors = _error_cards(runs, verdicts, events, journal)
    errors += [_Card(t("🔥 the weekly clean-up failed", "🔥 недельная чистка не прошла"), _para(_text(run["failed"])),
                     [t("👉 see ", "👉 смотри ") + "logs/consolidate.log"])
               for run in runs if run.get("kind") == "consolidate" and run.get("failed")]
    cards += errors
    if usage.stats(state, now)["stalled"]:
        cards.append(_Card(t("📭 agents do not read the lessons", "📭 агенты не читают уроки"), [],
                           _para(usage.stalled_note(), "👉 ", "   ")))
    return cards, len(errors)


# -- the decisions --------------------------------------------------------------


def _decision(square: str, verdict: dict) -> str:
    """A decision in one line within CARD_COLUMNS: its title, then as much of the reason as fits."""
    head = f"  {square} "
    room = CARD_COLUMNS - _columns(head)
    title, reason = _name(verdict, verdict.get("session")), _sentence(verdict.get("notes"))
    if not reason:
        return head + _clip(title, room)
    title = _clip(title, room - len(" — ") - min(REASON_MIN, _columns(reason)))
    return f"{head}{title} — {_clip(reason, min(REASON_MAX, room - _columns(title) - len(' — ')))}"


def _reviewed(runs: list[dict], verdicts: list[dict], events: list[dict], verbose: bool) -> list[str]:
    """A line a decision that found something, the rest folded (or each with its notes), then the counters."""
    lines, nothing = [], []
    for v in verdicts:
        square = chart.square(v.get("outcome"))
        if square == _NOTHING:
            nothing.append(v)
            continue
        lines.append(_decision(square, v))
    if verbose:
        for v in nothing:
            notes = _text(v.get("notes"))
            lines += _wrap(_name(v, v.get("session")) + (f" — {notes}" if notes.strip() else ""),
                           CARD_COLUMNS, f"  {_NOTHING} ", "     ")
    elif nothing:
        n = len(nothing)
        lines.append(f"  {_NOTHING} … " + (t(f"and {n} more “nothing”", f"и ещё {i18n.times(n)} «ничего»")
                                            + t(" (with reasons: recent -v)", " (с причинами: recent -v)")))
    returned: dict[str, int] = {}
    for run in runs:
        for form, n in (run.get("episodes") or {}).items() if isinstance(run.get("episodes"), dict) else ():
            returned[form] = returned.get(form, 0) + _count(n)
    if total := sum(returned.values()):
        # the user's corrections the model found in what it read: the number the lessons should bring down
        by_form = sorted(returned.items(), key=lambda item: -item[1])
        lines += _wrap(t("you sent work back: ", "ты возвращал работу: ") + f"{total} — "
                       + ", ".join(f"{chart.form_name(form)}: {n}" for form, n in by_form if n),
                       CARD_COLUMNS, "  🔁 ", "     ")
    if cost := chart.spent_line(runs):
        lines += _wrap(cost, CARD_COLUMNS, "  ", "     ")
    quiet = sum(_count(run.get("no_signals")) for run in runs)
    # a run journaled live has an event per segment with no project; one read from the logs only counts them
    outside = sum(max(_count((run.get("outcomes") or {}).get("no_project")),
                      sum(1 for e in events if e.get("run") == run.get("run") and e.get("kind") == "no_project"))
                  for run in runs)
    deferred = sum(_count(run.get("deferred")) for run in runs)
    if quiet:
        lines.append(f"  😴 {_segments(quiet)} " + t("with no errors and no replies of yours: the model did not look",
                                                        "без ошибок и без твоих ответов модель не смотрела"))
    if outside:
        lines.append(t("  🫥 outside projects: ", "  🫥 вне проектов: ") + _segments(outside))
    if deferred:
        lines.append(t("  ⏭ put off to the next night: ", "  ⏭ отложено до следующей ночи: ") + _segments(deferred))
    return lines


# -- the report -----------------------------------------------------------------


def _chart(cfg: Config, state: State, records: list[dict], runs: list[dict], now: datetime, tz: tzinfo) -> list[str]:
    """The last seven local days, the days of the window's runs marked; then live lessons and MEMORY.md."""
    last = now.astimezone(tz).date()
    first = last - timedelta(days=CHART_DAYS - 1)
    installed = parse_ts(state.installed_at)
    installed_day = installed.astimezone(tz).date() if installed else None
    counts = chart.day_counts(records, tz)
    days_of_runs = [_local_day(run, tz) for run in runs]
    lines = chart.day_rows(counts, first, last, installed_day, marked=frozenset(days_of_runs),
                           current=days_of_runs[-1] if days_of_runs else None)
    if legend := chart.legend(counts, first, last):
        lines += legend.split("\n")
    days = [first + timedelta(days=i) for i in range(CHART_DAYS)]
    days = [day for day in days if installed_day is None or day >= installed_day]
    margin = "   "
    extra = [chart.live_line(state.lessons, days, tz, records),
             *chart.memory_gauges(cfg, state.lessons, detailed=False, width=CARD_COLUMNS - len(margin))]
    if extra := [margin + line for line in extra if line]:
        lines += ["", *extra]
    return lines


def _header(nightly: dict | None, hours: int | None, runs: list[dict], errors: int, tz: tzinfo) -> str:
    if hours is not None:
        parts = [t(f"last {hours} h", f"последние {hours} ч")]
    elif nightly is None:
        parts = [t("last 24 h", "последние 24 ч"), t("🦗 the nightly run did not start", "🦗 ночной прогон не запускался")]
    else:
        started = _started(nightly).astimezone(tz)
        parts = [_night_of(started.date()), f"{started:%H:%M}"]
    if errors:
        parts.append(t("🔥 errors: ", "🔥 ошибок: ") + f"{errors} · this is fine 🐶")
    elif nightly is not None:
        parts.append(t("✅ no errors", "✅ без ошибок"))
    if len(runs) > 1:
        parts.append(t("runs: ", "прогонов: ") + str(len(runs)))
    return "🌙 agent-reflect · " + " · ".join(parts)


def _night_of(day) -> str:
    return t("night of ", "ночь на ") + chart.date_phrase(day)


def _section(title: str, lines: list[str]) -> list[str]:
    return [_rule(title), "", *lines, ""] if lines else [_rule(title), ""]


def _clock(now: datetime, tz: tzinfo | None) -> tuple[datetime, tzinfo]:
    """`now` with a zone (UTC when it has none), and `tz` or the machine's."""
    return (now if now.tzinfo else now.replace(tzinfo=timezone.utc)), (tz or datetime.now().astimezone().tzinfo)


def _tally(cfg: Config, state: State, records: list[dict], now: datetime, hours: int | None, tz: tzinfo) -> _Tally:
    start, nightly, runs = window(records, now, hours, tz)
    ids = {run.get("run") for run in runs}
    journal = [r for r in records if r.get("type") == "verdict"]
    verdicts = [v for v in journal if v.get("run") in ids]
    events = [r for r in records if r.get("type") == "event" and r.get("run") in ids]
    changes, memory = _memory(state, runs, events, now, tz)
    attention, errors = _attention(state, runs, verdicts, events, journal, now, tz)
    return _Tally(nightly, runs, verdicts, events, changes, memory, attention, errors,
                  seen=sum(_count(run.get("segments_seen")) for run in runs),
                  asked=sum(_count(run.get("llm_calls")) for run in runs),
                  alerts=health.entries_since(state.health, start, cfg.health_ignore))


def render(cfg: Config, state: State, records: list[dict], now: datetime, *, hours: int | None = None,
           verbose: bool = False, tz: tzinfo | None = None) -> str:
    """The report for the window (see `window`): header, chart, memory, what needs the user, decisions, health.

    `verbose` lists every decision that found nothing, with the model's notes. Times are local (`tz`).
    """
    now, tz = _clock(now, tz)
    tl = _tally(cfg, state, records, now, hours, tz)
    changed = i18n.count(tl.changes, ("change", "changes"), ("изменение", "изменения", "изменений"))
    reviewed = t(f"{tl.asked} of {tl.seen}", f"{tl.asked} из {tl.seen}")
    need = len(tl.attention)

    out = [_header(tl.nightly, hours, tl.runs, tl.errors, tz),
           t(f"   ✍️ memory: {changed} · 🚨 need you: {need} · 🗂 reviewed: {reviewed}",
             f"   ✍️ в памяти: {changed} · 🚨 требуют тебя: {need} · 🗂 разобрано: {reviewed}"), ""]
    out += _section(t(f"📊 LAST {CHART_DAYS} DAYS", f"📊 ПОСЛЕДНИЕ {CHART_DAYS} ДНЕЙ"),
                    _chart(cfg, state, records, tl.runs, now, tz))
    out += _section(t("✍️ MEMORY · ", "✍️ ПАМЯТЬ · ") + changed, tl.memory)
    out += _section(t("🚨 NEEDS YOU · ", "🚨 ТРЕБУЕТ ТЕБЯ · ") + str(need), _draw(tl.attention))
    out += _section(t("🗂 REVIEWED · ", "🗂 РАЗОБРАНО · ") + reviewed, _reviewed(tl.runs, tl.verdicts, tl.events, verbose))
    out += _section(t("🩺 SILENT FAILURES · ", "🩺 ТИХИЕ СБОИ · ") + str(len(tl.alerts)) + ("" if tl.alerts else " 👌"),
                    [line for entry in tl.alerts for line in _wrap(health.describe(entry), CARD_COLUMNS, "  ", "    ")])
    return scrub("\n".join(out).rstrip("\n"))


def notice(cfg: Config, state: State, records: list[dict], now: datetime, *,
           tz: tzinfo | None = None) -> tuple[str, str, str]:
    """The night's report as a notification: the night, its status, and the totals of the report's sections."""
    now, tz = _clock(now, tz)
    tl = _tally(cfg, state, records, now, None, tz)
    night = (_night_of(_started(tl.nightly).astimezone(tz).date()) if tl.nightly
             else t("last 24 h", "последние 24 ч"))
    if tl.errors:
        status = t("🔥 errors: ", "🔥 ошибок: ") + str(tl.errors)
    elif tl.changes or tl.attention or tl.alerts:
        status = t("✅ no errors", "✅ без ошибок")
    else:
        status = t("😴 quiet night · ✅ no errors", "😴 тихая ночь · ✅ без ошибок")
    need, failures = len(tl.attention), len(tl.alerts)
    totals = t(f"✍️ Memory {tl.changes} · 🚨 Needs you {need} · 🗂 Reviewed {tl.asked} of {tl.seen} · 🩺 Failures {failures}",
               f"✍️ Память {tl.changes} · 🚨 Требует тебя {need} · 🗂 Разобрано {tl.asked} из {tl.seen} · 🩺 Сбои {failures}")
    return "🌙 agent-reflect · " + night, status, totals
