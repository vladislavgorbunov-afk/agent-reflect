"""The weekly digest file, the night's report file and the macOS notification."""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
from collections import Counter
from datetime import date, datetime, timedelta, tzinfo
from pathlib import Path

from . import chart, health, i18n, memory_reads, usage
from .config import Config
from .i18n import quoted, t
from .ledger import Lesson, parse_ts
from .promptkit import squash
from .reflect import DETECTOR
from .scrub import scrub

_DASH = "—"
_NOTIFY_TIMEOUT = 15
# a new notification of a group takes the place of the one before, so unread nights do not pile up
NIGHT_GROUP = "agent-reflect.night"
WEEK_GROUP = "agent-reflect.week"
# the counted sections in their order: the summary key, the emoji, the heading, the name in the notification;
# the heading and the name are (English, Russian) pairs
SECTIONS = (
    ("new_lessons", "🆕", ("New lessons", "Новые уроки"), ("Lessons", "Уроки")),
    ("violated", "⚠️", ("Broken after going live", "Нарушены после публикации"), ("Broken", "Нарушены")),
    ("promoted", "🛠", ("Moved into skills", "Перенесено в скиллы"), ("Into skills", "В скиллы")),
    ("prs", "🔀", ("Draft PR", "Draft PR"), ("PR", "PR")),
    ("removed", "🗑", ("Removed from memory", "Убрано из памяти"), ("Removed", "Убрано")),
    ("suggestions", "💡", ("Suggestions on your notes", "Предложения по твоим записям"), ("Suggestions", "Предложения")),
    ("upstream", "🧰", ("Candidates for dev-process-toolkit", "Кандидаты в dev-process-toolkit"),
     ("Candidates", "Кандидаты")),
    ("errors", "🔥", ("Run errors", "Ошибки прогонов"), ("Errors", "Ошибки")),
    ("advisories", "📣", ("Warnings", "Предупреждения"), ("Warnings", "Предупреждения")),
    ("health", "🩺", ("Silent failures", "Тихие сбои"), ("Failures", "Сбои")),
)
# sections the digest has only when they hold something
EXTRA_SECTIONS = (
    ("clusters", "🧩", ("Repeated corrections", "Повторяющиеся поправки"), ("Corrections", "Поправки")),
    ("broken_rules", "📜", ("Broken rules", "Нарушенные правила"), ("Rules", "Правила")),
    ("care", "🧹", ("Lesson care", "Уход за уроками"), ("Care", "Уход")),
)


def guard_hint() -> str:
    return t("- A rule that can be told by a command or by a file can be shown to the agent as it acts: "
             "`agent-reflect guard add <name> <regex> --text \"<what to show>\"` (for a file, with `--path`).",
             "- Правило, которое узнаётся по команде или по файлу, можно показывать агенту в момент действия: "
             "`agent-reflect guard add <имя> <regex> --text \"<что показать>\"` (для файла — с `--path`).")


def week_tag(now: datetime) -> str:
    year, week, _ = now.isocalendar()
    return f"{year}-W{week:02d}"


def _counts(summary: dict) -> tuple[int, int, int]:
    n = len(summary.get("new_lessons", []))
    m = len({p.get("target") for p in summary.get("promoted", [])})
    k = len(summary.get("prs", []))
    return n, m, k


def notice(summary: dict, first: date, last: date) -> tuple[str, str, str]:
    """The digest as a notification: its days, lessons into skills, and what each section with items counts."""
    n, m, _ = _counts(summary)
    title = f"📅 agent-reflect · {chart.period_phrase(first, last)}"
    subtitle = (f"🧠 {i18n.count(n, ('lesson', 'lessons'), ('урок', 'урока', 'уроков'))} → "
                f"🛠 {i18n.count(m, ('skill', 'skills'), ('скилл', 'скилла', 'скиллов'))}")
    totals = [f"{emoji} {t(*name)} {len(summary[key])}" for key, emoji, _, name in (*SECTIONS, *EXTRA_SECTIONS)
              if summary.get(key)]
    return title, subtitle, " · ".join(totals) or t("😴 no changes", "😴 без изменений")


def _lesson_line(lesson_id: str, lessons: dict[str, Lesson]) -> str:
    lesson = lessons.get(lesson_id)
    if lesson is None:
        return f"- {lesson_id}"
    project = Path(lesson.project_root).name or lesson.project_root
    sessions = ", ".join(dict.fromkeys(e.get("session", "") for e in lesson.evidence))
    violated = f", {violations_phrase(lesson.violations)}" if lesson.violations else ""
    return (f"- **{lesson.title}** ({project}, {lesson.confidence}, " + t("repeats: ", "повторов: ")
            + f"{lesson.repeats}{violated}) — " + t("When: ", "Когда: ") + f"{lesson.cue} → {lesson.lesson}. "
            + t("Sessions: ", "Сессии: ") + sessions)


def violations_phrase(count: int) -> str:
    return t("broken after going live: ", "нарушен после публикации: ") + i18n.times(count)


def _promoted_lines(item: dict, lessons: dict[str, Lesson]) -> list[str]:
    head = f"- `{item.get('target', '')}` — {item.get('summary', '')}"
    if item.get("pr_url"):
        head += f" ({item['pr_url']})"
    elif item.get("backup_run_id"):
        head += t(" (backup: ", " (бэкап: ") + f"{item['backup_run_id']})"
    out = [head]
    ids = list(item.get("lesson_ids") or [])
    files = list(item.get("user_rule_files") or [])
    if ids or files:
        parts = [str(i) for i in ids] + [str(f) for f in files]
        out.append(t("  - Lessons and notes: ", "  - Уроки и записи: ") + ", ".join(parts))
    out.extend(f"  - {note}" for note in item.get("notes") or [])
    if item.get("diff"):
        out.extend(_fenced_diff(str(item["diff"])))
    return out


def _fenced_diff(diff: str) -> list[str]:
    """The diff as a ```diff block inside the list item, fenced longer than any backticks in it."""
    longest = max((len(run) for run in re.findall(r"`+", diff)), default=0)
    fence = "`" * max(3, longest + 1)
    return [f"  {fence}diff", *(f"  {line}" for line in diff.splitlines()), f"  {fence}"]


def _advisory_line(event: dict) -> str:
    if event.get("kind") == "skill_large":
        target, chars, limit = event.get("target", ""), event.get("chars", "?"), event.get("limit", "?")
        return t(f"- `{target}` grew to {chars} characters (limit {limit}): worth trimming by hand",
                 f"- `{target}` разросся до {chars} символов (порог {limit}): стоит сократить вручную")
    if event.get("kind") == "upstream_rebaselined":
        old, new = str(event.get("old", ""))[:7], str(event.get("new", ""))[:7]
        repo = event.get("repo", "")
        return t(f"- {repo}: mark {old} is gone from the history of origin/main, the new base is {new}; "
                 "the commits between them did not make it into the dev-process-toolkit candidates",
                 f"- {repo}: метка {old} пропала из истории origin/main, новая база — {new}; "
                 "коммиты между ними в кандидаты dev-process-toolkit не попали")
    return _error_line(event)


def _usage_lines(stats: dict, lessons: dict[str, Lesson]) -> list[str]:
    """The reads total, one line per live lesson, and the fixed note when the index does not reach the agent."""
    out = [usage.line(stats), ""]
    live = sorted((lesson for lesson in lessons.values() if lesson.status == "live"),
                  key=lambda lesson: (-lesson.reads, lesson.title))
    for lesson in live:
        project = Path(lesson.project_root).name or lesson.project_root
        violated = f", {violations_phrase(lesson.violations)}" if lesson.violations else ""
        # a lesson with a trigger reaches the agent through the Bash hook, whether or not its file is opened
        shown = t(", shown by the hook: ", ", показан хуком: ") + i18n.times(lesson.fires) if lesson.trigger else ""
        out.append(f"- **{lesson.title}** ({project}) — " + t("read: ", "прочитан: ")
                   + f"{i18n.times(lesson.reads)}{shown}{violated}")
    if stats.get("stalled"):
        out += ["", usage.stalled_note()]
    return out


MEMORY_TOP = 5


def memory_note() -> str:
    return t("A rule that sits whole in an index line or in CLAUDE.md works without its file being opened, "
             "so “never opened” is a reason to look at the note again, not a verdict.",
             "Правило, которое целиком стоит в строке индекса или в CLAUDE.md, работает и без открытия файла, "
             "так что «не открывали» — повод пересмотреть запись, а не приговор.")


def _files_word(count: int) -> str:
    if not i18n.is_ru():
        return "file" if count == 1 else "files"
    return "файла" if count % 10 == 1 and count % 100 != 11 else "файлов"


def _memory_lines(data: dict) -> list[str]:
    """Per location: how many of its files the agent never opened, their names, and the most opened ones."""
    out = [memory_note()]
    for location in data.get("locations") or []:
        files = location.get("files") or []
        unread = sorted(f.get("name", "") for f in files if not f.get("reads"))
        files_of = f"{len(unread)} " + t("of", "из") + f" {len(files)} {_files_word(len(files))}"
        line = files_of + t(" never opened", " не открывали")
        out += ["", f"### {location.get('name', '')}", "", line + (": " + ", ".join(unread) if unread else "")]
        top = sorted((f for f in files if f.get("reads")), key=lambda f: (-f["reads"], f.get("name", "")))
        if top:
            out.append(t("Opened most: ", "Чаще всего открывали: ")
                       + ", ".join(f"{f.get('name', '')} ({f['reads']})" for f in top[:MEMORY_TOP]))
    return out


def _care_line(item: dict, lessons: dict[str, Lesson]) -> str:
    lesson = lessons.get(item.get("lesson_id"))
    lesson_id = item.get("lesson_id", "")
    name = f"**{lesson.title}** ({Path(lesson.project_root).name or lesson.project_root})" if lesson else lesson_id
    if item.get("reason") == "stale":
        days = int(item.get("days") or 0)
        days_word = i18n.count(days, ("day", "days"), ("день", "дня", "дней"))
        return t(f"- {name} — {days_word} in the index with no reads, hook showings or repeats. "
                 f"Take its line out of the index, keeping the file: `agent-reflect retire {lesson_id}`",
                 f"- {name} — {days_word} в индексе без чтений, показов хуком "
                 f"и повторов. Убрать строку из индекса, файл останется: `agent-reflect retire {lesson_id}`")
    phrase = violations_phrase(int(item.get("violations") or 0))
    return t(f"- {name} — {phrase}, and it has no hook pattern. "
             f"Set one: `agent-reflect guard set {lesson_id} <regex>` (for a file, with `--path`)",
             f"- {name} — {phrase}, а шаблона для хука нет. "
             f"Поставить: `agent-reflect guard set {lesson_id} <regex>` (для файла — с `--path`)")


def _rule_line(item: dict) -> str:
    count, sessions = int(item.get("count") or 0), int(item.get("sessions") or 0)
    said = _plain(item.get("said"))
    return (f"- {quoted(_plain(item.get('rule')))} — {i18n.times(count)} "
            + _in_sessions(sessions) + (t("; for example: ", "; например: ") + quoted(said) if said else ""))


def _in_sessions(n: int) -> str:
    return t("in ", "в ") + i18n.count(n, ("session", "sessions"), ("сессии", "сессиях", "сессиях"))


def _cluster_lines(item: dict, lessons: dict[str, Lesson]) -> list[str]:
    lesson = lessons.get(item.get("lesson_id"))
    what = t(*{"new": ("new lesson", "новый урок"), "repeat": ("lesson confirmed", "подтверждён урок")}.get(
        item.get("outcome"), ("no lesson written", "урок не записан")))
    title = lesson.title if lesson else _plain(item.get("title"))
    sessions = int(item.get("sessions") or 0)
    count = int(item.get("count") or len(item.get("said") or []))
    corrections = i18n.count(count, ("correction", "corrections"), ("поправка", "поправки", "поправок"))
    out = [f"- **{title}** — {what}; {corrections} " + _in_sessions(sessions)
           + (f". {_plain(item.get('note'))}" if item.get("note") else "")]
    out += [f"  - {quoted(_plain(said))}" for said in (item.get("said") or [])[:CLUSTER_SAID_MAX]]
    return out


CLUSTER_SAID_MAX = 4


def _error_line(event: dict) -> str:
    rest = " ".join(f"{k}={v}" for k, v in event.items() if k not in ("ts", "kind"))
    return f"- {event.get('ts', '')} {event.get('kind', '')}" + (f" {rest}" if rest else "")


def _plain(value) -> str:
    """Free text on one list line: whitespace squashed, and backslashes and backticks escaped,
    so nothing in it opens a code span."""
    return re.sub(r"([\\`])", r"\\\1", squash(str(value or "")))


def _code(text: str) -> str:
    """`text` as one inline code span, fenced longer than any backticks in it."""
    longest = max((len(run) for run in re.findall(r"`+", text)), default=0)
    fence = "`" * (longest + 1)
    pad = " " if text.startswith("`") or text.endswith("`") else ""
    return f"{fence}{pad}{text}{pad}{fence}"


def _upstream_lines(candidates: list[dict], week: str, cfg: Config | None, path: Path | None) -> list[str]:
    lines = []
    for c in candidates:
        place = squash(str(c.get("place") or ""))
        where = f" → {_code(place)}." if place else ""
        lines.append(f"- {_plain(c.get('repo'))} {_plain(c.get('sha'))} {_plain(c.get('date'))} "
                     f"{quoted(_plain(c.get('subject')))} — {_plain(c.get('rule'))}{where} {_plain(c.get('why'))}".rstrip())
    if lines:
        cfg = cfg or Config.default()  # a session in any project reads them: absolute paths only
        source, every = path or path_for(cfg, week), cfg.upstream_candidates_file
        lines.append(t(f"- To move them: ask an agent in a session to “move the candidates from {source} "
                       "into dev-process-toolkit” — each rule is generalized by hand, never copied word for word. "
                       f"Every candidate, past weeks' too: {every}",
                       f"- Перенести: попроси агента в сессии «перенеси кандидатов из {source} "
                       "в dev-process-toolkit» — правило обобщается вручную, дословно не копируется. "
                       f"Все кандидаты, и прошлых недель тоже: {every}"))
    return lines


def episodes_line(records: list[dict], first: date, last: date, tz: tzinfo | None = None) -> str:
    """How often the user sent work back in the days from `first` to `last`; "" when no session was judged.

    This is the number the lessons are there to bring down: the user's corrections the model
    found, the sessions they were in, and the sessions it looked at.
    """
    judged: set[str] = set()
    with_episodes: set[str] = set()
    forms: Counter = Counter()
    # per reading of a session (a run's records of it): its corrections on answers of the user, and
    # those answers when counted. A correction on any other message is in the total, not in the rate.
    readings: dict[tuple, list] = {}
    for record in records:
        if record.get("type") != "verdict" or record.get("agent") or record.get("outcome") == "llm_error":
            continue
        at = parse_ts(record.get("ts"))
        if at is None or not first <= at.astimezone(tz).date() <= last:
            continue
        session = str(record.get("session") or "")
        judged.add(session)
        episodes = [e for e in record.get("episodes") or [] if isinstance(e, dict)]
        reading = readings.setdefault((record.get("run"), session), [0, None])
        reading[0] += sum(1 for e in episodes if e.get("reply") is True)
        if isinstance(record.get("replies"), int):
            reading[1] = (reading[1] or 0) + record["replies"]
        if episodes:
            with_episodes.add(session)
            forms.update(str(e.get("form")) for e in episodes)
    if not judged:
        return ""
    total = sum(forms.values())
    line = t("🔁 work sent back: ", "🔁 возвраты работы: ") + str(total)
    answers = sum(replies for _, replies in readings.values() if replies)
    if answers:
        # only the readings that counted the answers: one without would add corrections and nothing to hold them to
        on_answers = sum(n for n, replies in readings.values() if replies)
        rate = _rate(on_answers, answers)
        line += t(f" · {rate} per 100 of your replies ({on_answers} of {answers})",
                  f" · {rate} на 100 твоих ответов ({on_answers} из {answers})")
    line += t(f" · in {len(with_episodes)} of {len(judged)} sessions", f" · в {len(with_episodes)} из {len(judged)} сессий")
    if total:
        line += " · " + ", ".join(f"{chart.form_name(form)}: {n}" for form, n in forms.most_common())
    return line


def _rate(episodes: int, answers: int) -> str:
    return i18n.decimal(episodes / answers * 100)


def _day_month(iso_day) -> str:
    try:
        return chart.short_date(date.fromisoformat(str(iso_day)))
    except ValueError:
        return str(iso_day)


def load_baseline(cfg: Config) -> dict | None:
    """The baseline `agent-reflect baseline` wrote, None when there is none to read."""
    try:
        data = json.loads(cfg.baseline_file.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def baseline_line(baseline: dict | None, detector: int) -> str:
    """The rate of returned work in the weeks the baseline covers; "" without one.

    A baseline counted by another reading of corrections than today's (`detector`) says so:
    the two numbers are not to be compared.
    """
    if not baseline or not isinstance(baseline.get("replies"), int) or baseline["replies"] <= 0:
        return ""
    episodes = int(baseline.get("episodes") or 0)
    days = f"{_day_month(baseline.get('since'))}–{_day_month(baseline.get('until'))}"
    rate, replies, sessions = _rate(episodes, baseline["replies"]), baseline["replies"], baseline.get("sessions", 0)
    line = t(f"📏 baseline {days}: {rate} per 100 of your replies ({episodes} of {replies}, sessions: {sessions})",
             f"📏 базовая линия {days}: {rate} на 100 твоих ответов ({episodes} из {replies}, сессий: {sessions})")
    other = t(" — counted by another detector, not to be compared", " — считал другой детектор, сравнивать нельзя")
    return line + (other if baseline.get("detector") != detector else "")


def week_block(cfg: Config, lessons: dict[str, Lesson], records: list[dict], first: date, last: date,
               installed: date | None, tz: tzinfo | None = None) -> list[str]:
    """The runs chart of the days from `first` to `last`, with live lessons and MEMORY.md, as a ```text block.

    `first` is the day of the previous consolidate: the digest covers the days since. `records` are the journal's.
    """
    counts = chart.day_counts(records, tz)
    rows = chart.day_rows(counts, first, last, installed)
    if legend := chart.legend(counts, first, last):
        rows += legend.split("\n")
    rows += chart.error_notes(counts, first, last)
    days = [first + timedelta(days=i) for i in range((last - first).days + 1)]
    live = chart.live_line(lessons, [day for day in days if installed is None or day >= installed], tz, records)
    gauges: list[str] = []
    for line in chart.memory_gauges(cfg, lessons, detailed=True):
        gauges += ["", line] if gauges and line.startswith("🗃") else [line]  # a blank line between projects
    out = [t("## 📊 Week · ", "## 📊 Неделя · ") + chart.period_phrase(first, last), "", "```text", *rows]
    if live:
        out += ["", live]
    if returned := episodes_line(records, first, last, tz):
        out += ["", returned]
        if base := baseline_line(load_baseline(cfg), DETECTOR):
            out.append(base)
    in_week = [r for r in records if r.get("type") == "run" and (at := parse_ts(r.get("ended_at"))) is not None
               and first <= at.astimezone(tz).date() <= last]
    if cost := chart.spent_line(in_week):
        out += ["", cost]
    if gauges:
        out += ["", *gauges]
    return out + ["```"]


def render(summary: dict, lessons: dict[str, Lesson], week: str, cfg: Config | None = None, *,
           chart: list[str] = (), path: Path | None = None) -> str:
    """The digest of `week`, with the `chart` block below its total.

    `cfg` names the files it points to (the installed tool's when None); `path` is where the digest is
    written (the week's file when None), the file the upstream candidates line sends the agent to.
    """
    n, m, k = _counts(summary)
    items = {key: summary.get(key) or [] for key, *_ in SECTIONS}
    # each section's lines; its heading counts its items
    lines: dict[str, list[str]] = {
        "new_lessons": [_lesson_line(i, lessons) for i in items["new_lessons"]],
        "violated": [_lesson_line(i, lessons) for i in items["violated"]],
        "promoted": [line for p in items["promoted"] for line in _promoted_lines(p, lessons)],
        "prs": [f"- {url}" for url in items["prs"]],
        "removed": [f"- {r.get('file', '')} — {r.get('reason', '')}" for r in items["removed"]],
        "suggestions": [f"- {s.get('file', '')} — {s.get('suggestion', '')}" for s in items["suggestions"]],
        "upstream": _upstream_lines(items["upstream"], week, cfg, path),
        "errors": [_error_line(e) for e in items["errors"]],
        "advisories": [_advisory_line(e) for e in items["advisories"]],
        "health": [health.describe(e) for e in items["health"]],
    }
    lessons_n = i18n.count(n, ("lesson", "lessons"), ("урок", "урока", "уроков"))
    skills_n = i18n.count(m, ("skill", "skills"), ("скилл", "скилла", "скиллов"))
    total = t("Total: ", "Итого: ") + f"{lessons_n}, {skills_n}, {k} PR"
    out = [t("# agent-reflect — week ", "# agent-reflect — неделя ") + week, "", total, ""]
    if summary.get("skipped_no_project"):
        out += [t("Segments with no project skipped: ", "Отрезков без проекта пропущено: ")
                + str(summary["skipped_no_project"]), ""]
    if chart:
        out += [*chart, ""]
    for key, emoji, title, _ in SECTIONS:
        out += [f"## {emoji} {t(*title)} · {len(items[key])}", ""] + (lines[key] or [_DASH]) + [""]
    extra = {
        "clusters": [line for c in summary.get("clusters") or [] for line in _cluster_lines(c, lessons)],
        "broken_rules": [_rule_line(r) for r in summary.get("broken_rules") or []],
        "care": [_care_line(c, lessons) for c in summary.get("care") or []],
    }
    if extra["broken_rules"]:
        extra["broken_rules"].append(guard_hint())
    for key, emoji, title, _ in EXTRA_SECTIONS:
        if summary.get(key):
            out += [f"## {emoji} {t(*title)} · {len(summary[key])}", ""] + extra[key] + [""]
    if "usage" in summary:
        out += [t("## Lesson reads", "## Чтения уроков"), ""] + _usage_lines(summary["usage"], lessons) + [""]
    if "memory_reads" in summary:
        days = summary["memory_reads"].get("days", memory_reads.WINDOW_DAYS)
        out += [t(f"## Memory: not opened in {days} days", f"## Память: не открывали за {days} дней"), ""]
        out += _memory_lines(summary["memory_reads"]) + [""]
    return scrub("\n".join(out))


def path_for(cfg: Config, week: str) -> Path:
    return cfg.digests_dir / f"{week}.md"


def free_path(cfg: Config, week: str) -> Path:
    """The week's file; when a digest of the week is there already, <week>-2.md, <week>-3.md and so on."""
    path, n = path_for(cfg, week), 1
    while path.exists():
        n += 1
        path = cfg.digests_dir / f"{week}-{n}.md"
    return path


def write(cfg: Config, text: str, week: str, path: Path | None = None) -> Path:
    cfg.digests_dir.mkdir(parents=True, exist_ok=True)
    return _write(path or path_for(cfg, week), text)


def write_night(cfg: Config, report: str, night: date) -> Path:
    """The night's report (`recent`) into reports/<night>.md, fenced so that a Markdown view keeps its columns."""
    cfg.reports_dir.mkdir(parents=True, exist_ok=True)
    return _write(cfg.reports_dir / f"{night.isoformat()}.md", f"```text\n{report}\n```\n")


def _write(path: Path, text: str) -> Path:
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(scrub(text).encode("utf-8"))
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return path


def _quoted(text: str) -> str:
    """`text` as an AppleScript string."""
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'


def notify(message: str, open_path: Path | None = None, *, title: str = "agent-reflect", subtitle: str | None = None,
           group: str | None = None, run=subprocess.run, which=shutil.which) -> None:
    """Best effort: a failed notification never fails the run.

    A click opens `open_path` in its default app; `group` replaces the group's previous notification.
    """
    try:
        if which("terminal-notifier"):
            cmd = ["terminal-notifier", "-title", title]
            if subtitle:
                cmd += ["-subtitle", subtitle]
            cmd += ["-message", message]
            if group:
                cmd += ["-group", group]
            if open_path:
                cmd += ["-open", open_path.as_uri()]
        else:
            script = f"display notification {_quoted(message)} with title {_quoted(title)}"
            if subtitle:
                script += f" subtitle {_quoted(subtitle)}"
            cmd = ["osascript", "-e", script]
        run(cmd, check=False, capture_output=True, timeout=_NOTIFY_TIMEOUT)
    except Exception:
        pass
