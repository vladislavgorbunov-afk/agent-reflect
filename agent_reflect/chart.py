"""Emoji and ASCII drawings for the reports: runs per day, live lessons per day, MEMORY.md fill.

Pure functions of journal run records and lessons; the one thing read is MEMORY.md, in memory_gauges.
"""
from __future__ import annotations

from collections import Counter
from datetime import date, datetime, timedelta, tzinfo
from math import ceil
from pathlib import Path
from typing import Callable, NamedTuple
from unicodedata import east_asian_width

from . import i18n, memory
from .config import Config
from .i18n import t
from .ledger import Lesson, parse_ts

ROW_MAX = 25  # squares in one row of the chart
# no line of the legend runs past this column: the widest a row gets (" ▶ пт 23  ", ROW_MAX squares of two
# columns, "  101  ← эта ночь"), so the mockup's legend, 77 columns, stays on one line
LEGEND_COLUMNS = 77
CLI_BYTE_LIMIT = 25_000  # Claude Code loads no more of MEMORY.md than this
BAR_WIDTH = 20
WARN_RATIO = 0.8  # a limit this full gets a warning

_ERROR = "🟥"
SEGMENT_EN, SEGMENT_RU = ("segment", "segments"), ("отрезок", "отрезка", "отрезков")
# (square, its name in the legend, the outcomes it stands for) in the order a row draws them;
# an outcome not listed is ⬜. no_project is no decision of the model and is never drawn.
_COLOURS = (
    ("⬜", ("nothing", "ничего"), ("no_entry", "rejected_cue", "rejected_not_concrete", "rejected_bad_quote",
                                   "rejected_unsupported")),
    ("🟦", ("repeat", "повтор"), ("repeat", "after_promotion")),
    ("🟩", ("new lesson", "новый урок"), ("new_live", "new_staged")),
    ("🟪", ("lesson broken", "нарушен урок"), ("violated",)),
    ("🟧", ("your rule", "твоё правило"), ("user_rule_repeat",)),
    (_ERROR, ("error", "ошибка"), ("llm_error", "bad_verdict", "write_failed", "apply_crashed")),
)
# what a correction of the user was about, as the reports name it (reflect.FORMS): (English, Russian)
_FORM_NAMES = {"self_report": ("said “done” when it was not", "сказал «готово» — не готово"),
               "constraint": ("broke a rule", "нарушил правило"), "intent": ("misunderstood", "понял не так"),
               "operational": ("got a tool wrong", "ошибся с инструментом"), "other": ("other", "прочее")}


def form_name(form: str) -> str:
    names = _FORM_NAMES.get(form)
    return t(*names) if names else form


_SQUARE_OF = {outcome: square for square, _, outcomes in _COLOURS for outcome in outcomes}
_NOT_A_DECISION = "no_project"
_INDENT = " " * 10  # under the squares of a row
_WEEKDAYS = {"en": ("Mo", "Tu", "We", "Th", "Fr", "Sa", "Su"), "ru": ("пн", "вт", "ср", "чт", "пт", "сб", "вс")}
_MONTHS = {"en": ("January", "February", "March", "April", "May", "June",
                  "July", "August", "September", "October", "November", "December"),
           "ru": ("января", "февраля", "марта", "апреля", "мая", "июня",
                  "июля", "августа", "сентября", "октября", "ноября", "декабря")}
_LABEL_WIDTH = len("Mo 23")  # a weekday is two letters in every language
_EMOJI_STYLE = "\ufe0f"  # after a character, draws it as a two-column emoji: ✍️ ⚠️


def columns(text: str) -> int:
    """How many terminal columns `text` takes: an emoji takes two."""
    width = 0
    for i, ch in enumerate(text):
        if ch == _EMOJI_STYLE:
            continue
        wide = east_asian_width(ch) in "WF" or ord(ch) >= 0x1F000 or text[i + 1:i + 2] == _EMOJI_STYLE
        width += 2 if wide else 1
    return width


def square(outcome: str) -> str:
    return _SQUARE_OF.get(outcome, "⬜")


def day_label(d: date) -> str:
    return f"{_WEEKDAYS[i18n.current()][d.weekday()]} {d.day}"


def _month(d: date) -> str:
    return _MONTHS[i18n.current()][d.month - 1]


def short_date(d: date) -> str:
    """A day of the month without the year: 21.10 / 21 Oct."""
    return f"{d:%d.%m}" if i18n.is_ru() else f"{d.day} {_month(d)[:3]}"


def date_phrase(d: date) -> str:
    return f"{d.day} {_month(d)}"


def period_phrase(first: date, last: date) -> str:
    if first == last:
        return date_phrase(first)
    if (first.year, first.month) == (last.year, last.month):
        return f"{first.day}–{last.day} {_month(last)}"
    return f"{date_phrase(first)} – {date_phrase(last)}"


def _ended(run: dict) -> datetime | None:
    """When a run ended, its start for one with no end; None for a record that is no run."""
    if run.get("type") != "run":
        return None
    return parse_ts(run.get("ended_at")) or parse_ts(run.get("started_at"))


def day_counts(runs: list[dict], tz: tzinfo | None = None) -> dict[date, Counter]:
    """The outcomes of the model's decisions by the local day the run ended; a day a run ended on is a key."""
    tz = tz or datetime.now().astimezone().tzinfo
    counts: dict[date, Counter] = {}
    for run in runs:
        if (ended := _ended(run)) is None:
            continue
        day = counts.setdefault(ended.astimezone(tz).date(), Counter())
        for outcome, n in (run.get("outcomes") or {}).items():
            if outcome != _NOT_A_DECISION and isinstance(n, int) and n > 0:
                day[outcome] += n
    return counts


def _days(first: date, last: date) -> list[date]:
    return [first + timedelta(days=i) for i in range((last - first).days + 1)]


def _by_square(day: Counter) -> Counter:
    squares: Counter = Counter()
    for outcome, n in day.items():
        squares[square(outcome)] += n
    return squares


def _scale(counts: dict[date, Counter], first: date, last: date) -> int:
    """How many decisions one square stands for: the least that keeps every row within ROW_MAX squares."""
    days = [_by_square(counts[d]) for d in _days(first, last) if d in counts]
    k = max(1, ceil(max((sum(day.values()) for day in days), default=0) / ROW_MAX))
    while any(sum(ceil(n / k) for n in day.values()) > ROW_MAX for day in days):
        k += 1  # a colour's few decisions still take a square each, so the plain division can fall short
    return k


def day_rows(counts: dict[date, Counter], first: date, last: date, installed: date | None, *,
             marked: frozenset[date] = frozenset(), current: date | None = None) -> list[str]:
    """A row per day from `first` to `last`: weekday and date, a square per decision, the number of them.

    Days before `installed` are left out; a later day no run ended on sleeps. `marked` days start
    with ▶ and `current` is the night of the report.
    """
    k = _scale(counts, first, last)
    rows = []
    for d in _days(first, last):
        if installed is not None and d < installed:
            continue
        head = (" ▶ " if d in marked else "   ") + day_label(d).ljust(_LABEL_WIDTH)  # squares stay in a column
        if d not in counts:
            rows.append(f"{head}  💤 " + t("no run", "прогона не было"))
            continue
        day = _by_square(counts[d])
        squares = "".join(sq * ceil(day[sq] / k) for sq, _, _ in _COLOURS)
        row = f"{head}  {squares}  {sum(day.values())}"
        rows.append(row + "  ← " + t("this night", "эта ночь") if d == current else row)
    return rows


def legend(counts: dict[date, Counter], first: date, last: date) -> str:
    """The colours drawn from `first` to `last` and, when a square stands for several, how many; "" if none.

    A legend wider than LEGEND_COLUMNS goes on under itself: its lines are joined by newlines.
    """
    drawn = sum((_by_square(counts[d]) for d in _days(first, last) if d in counts), Counter())
    names = [f"{sq} {t(*name)}" for sq, name, _ in _COLOURS if drawn[sq] > 0]
    if not names:
        return ""
    if (k := _scale(counts, first, last)) > 1:
        names.append(t("1 square = ", "1 квадрат = ") + i18n.count(k, SEGMENT_EN, SEGMENT_RU))
    lines: list[str] = []
    for name in names:
        if lines and columns(f"{lines[-1]} · {name}") <= LEGEND_COLUMNS:
            lines[-1] += f" · {name}"
        else:
            lines.append(_INDENT + name)
    return "\n".join(lines)


def error_notes(counts: dict[date, Counter], first: date, last: date) -> list[str]:
    """A line for each day that has 🟥, with how many."""
    notes = []
    for d in _days(first, last):
        if errors := _by_square(counts.get(d, Counter()))[_ERROR]:
            notes.append(f"{_INDENT}{_ERROR} {day_label(d)}: " + t("errors", "ошибок") + f": {errors}. This is fine 🔥🐶")
    return notes


def _tokens(n: int) -> str:
    return f"{round(n / 1000)}" + t("k", " тыс.") if n >= 1000 else str(n)


def spent_line(runs: list[dict]) -> str:
    """What the model calls of `runs` (journal run records) cost; "" when none of them says."""
    spent = [run["spent"] for run in runs if isinstance(run.get("spent"), dict)]
    count = lambda name: sum(n for s in spent if isinstance(n := s.get(name), (int, float)))  # noqa: E731
    calls = int(count("calls"))
    if not calls:
        return ""
    tokens_in, tokens_out = _tokens(int(count("in"))), _tokens(int(count("out")))
    line = (t("💸 model: ", "💸 модель: ") + i18n.count(calls, ("call", "calls"), ("вызов", "вызова", "вызовов")) + " · "
            + t(f"{tokens_in} tokens in, {tokens_out} out", f"{tokens_in} токенов на вход, {tokens_out} на выход"))
    cost = count("cost_usd")
    return line + (f" · ${i18n.decimal(cost, 2)} " + t("at API prices", "по ценам API") if cost >= 0.005 else "")


def live_line(lessons: dict[str, Lesson], days: list[date], tz: tzinfo | None = None, runs: list[dict] = ()) -> str:
    """How many lessons were live at the end of each of `days`.

    A day takes the count of the last of its `runs` (journal records) that has one, summed over the
    projects; a day without, the number of lessons live now that had gone live by its end.
    """
    if not days:
        return ""
    tz = tz or datetime.now().astimezone().tzinfo
    live_from = []
    for lesson in lessons.values():
        since = parse_ts(lesson.live_since or lesson.created_at)
        if lesson.status == "live" and since is not None:
            live_from.append(since.astimezone(tz).date())
    counted: dict[date, tuple[datetime, int]] = {}  # day -> the end of its last run with a count, the count
    for run in runs:
        if (ended := _ended(run)) is None or not isinstance(run.get("live"), dict):
            continue
        day = ended.astimezone(tz).date()
        if day not in counted or ended >= counted[day][0]:
            counted[day] = (ended, sum(n for n in run["live"].values() if isinstance(n, int)))
    live = [counted[day][1] if day in counted else sum(1 for d in live_from if d <= day) for day in days]
    trend = "📈" if live[-1] > live[0] else "📉" if live[-1] < live[0] else "➖"
    return t("🌱 live lessons: ", "🌱 живых уроков: ") + f"{' → '.join(map(str, live))} {trend}"


class _Limit(NamedTuple):
    name: str
    used: int
    limit: int
    rest: str  # what is left of it or, past it, how far over, in words
    as_count: bool = False  # shown as "used of limit" rather than a percentage

    @property
    def ratio(self) -> float:
        return self.used / self.limit


def _thousands(en: str, ru: str) -> Callable[[int, bool], str]:
    """Thousands of a unit, said the same left and over: ~4.6k bytes, ~4,6 тыс. байт."""
    return lambda n, over: f"~{i18n.decimal(n / 1000)}" + t(f"k {en}", f" тыс. {ru}")


def _rows(n: int, over: bool) -> str:
    return i18n.count(n, ("line", "lines"), ("строку" if over else "строка", "строки", "строк"))


def _places(n: int, over: bool) -> str:
    return i18n.count(n, ("lesson slot", "lesson slots"), ("место", "места", "мест")) + t("", " для уроков")


def _limit(name: str, used: int, limit: int, amount: Callable[[int, bool], str], as_count: bool = False) -> _Limit:
    """`amount` says how many of the limit's units a number is: as what is left, or with `over`, past it."""
    if used > limit:
        over = amount(used - limit, True)
        return _Limit(name, used, limit, t(f"over the limit by {over}", f"сверх лимита на {over}"), as_count)
    left = amount(limit - used, False)
    return _Limit(name, used, limit, t(f"{left} left", f"осталось {left}"), as_count)


def _limits(cfg: Config, text: str) -> list[_Limit]:
    """The four limits MEMORY.md runs into: what agent-reflect keeps it under, and what Claude Code reads of it."""
    lines = text.splitlines()
    auto = sum(1 for line in lines if line.startswith(memory.AUTO_PREFIX))
    return [
        _limit(t("chars", "символы"), len(text), cfg.index_char_limit, _thousands("chars", "символов")),
        _limit(t("bytes", "байты"), len(text.encode("utf-8")), CLI_BYTE_LIMIT, _thousands("bytes", "байт")),
        _limit(t("lines", "строки"), len(lines), cfg.index_line_limit, _rows),
        _limit(t("lessons", "уроки"), auto, cfg.auto_index_cap, _places, as_count=True),
    ]


def _bar(ratio: float) -> str:
    filled = min(BAR_WIDTH, round(ratio * BAR_WIDTH))
    return "█" * filled + "░" * (BAR_WIDTH - filled)


def _warning(limit: _Limit, sep: str) -> str:
    return f"{sep}⚠️ {limit.rest}" if limit.ratio >= WARN_RATIO else ""


def memory_gauges(cfg: Config, lessons: dict[str, Lesson], *, detailed: bool, width: int | None = None) -> list[str]:
    """How full MEMORY.md is, for each project that has live or staged lessons.

    A line a project with the fullest of its limits, or with `detailed` a header and a line a limit.
    A project's line that its warning would take past `width` columns has the warning on the next
    line, under the bar.
    """
    projects: dict[str, str] = {}  # memory dir -> name of the project
    for lesson in lessons.values():
        if lesson.status in ("live", "staged"):
            projects.setdefault(lesson.memory_dir, Path(lesson.project_root).name or lesson.project_root)
    out = []
    for memory_dir, project in sorted(projects.items(), key=lambda item: (item[1], item[0])):
        try:
            limits = _limits(cfg, memory.read_index(Path(memory_dir)))
        except memory.MemoryIndexError:
            out.append(f"🗃 MEMORY.md {project}  " + t("unreadable", "не читается"))
            continue
        if not detailed:
            fullest = max(limits, key=lambda limit: limit.ratio)
            head = f"🗃 MEMORY.md {project}  "
            line = f"{head}{_bar(fullest.ratio)} {round(fullest.ratio * 100)}%"
            warning = _warning(fullest, " ")
            if warning and width is not None and columns(line + warning) > width:
                out += [line, _warning(fullest, " " * columns(head))]
            else:
                out.append(line + warning)
            continue
        out.append(f"🗃 MEMORY.md · {project}")
        for limit in limits:
            shown = (t(f"{limit.used} of {limit.limit}", f"{limit.used} из {limit.limit}") if limit.as_count
                     else f"{round(limit.ratio * 100)}%")
            out.append(f"   {limit.name.ljust(9)}{_bar(limit.ratio)}  {shown}" + _warning(limit, "  "))
    return out
