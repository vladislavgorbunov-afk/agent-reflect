from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from agent_reflect import recent, usage
from agent_reflect.config import Config
from agent_reflect.ledger import Lesson, State

LOCAL = timezone(timedelta(hours=4))
NOW = datetime(2026, 10, 23, 5, 0, tzinfo=timezone.utc)  # 09:00 local, the morning after
NIGHT = "2026-10-22T23:30:00Z"  # the scheduled run, 03:30 local on 23 October
LATER = "2026-10-23T03:00:00Z"  # a manual run after it, 07:00 local
WEBAPP = "/Users/me/git/webapp"
LESSON, OLDER = "4c326d27", "9a1f03bb"
TITLE = "На macOS нет timeout — используй gtimeout"
PR = "5c39f173-1e0a-4f7e-9d42-8b1f2a3c4d5e"
AUTOTESTS = "a7d2e9c4-77b1-4c2e-a0f3-5d6e7f8a9b0c"
STORES = "b3e1f0aa-2c4d-4e6f-8a0b-1c2d3e4f5a6b"
FIRST = "e1d5a0b2-3f4e-4a5b-9c6d-7e8f9a0b1c2d"  # the session the lesson was made from
CONFLICT_NOTES = ("три new_page с isolatedContext упали с «browser is already running», "
                  "а в записи сказано, что способ проверен")


def _run(started: str, *, ended: str | None = None, kind: str = "reflect", trigger: str = "schedule",
         outcomes: dict | None = None, **extra) -> dict:
    return {"type": "run", "run": started, "kind": kind, "trigger": trigger, "started_at": started,
            "ended_at": ended or started, "segments_seen": 0, "no_signals": 0, "deferred": 0, "llm_calls": 0,
            "outcomes": outcomes or {}, "errors": 0, "failed": None, "digest": None, **extra}


def _verdict(session: str, outcome: str, *, run: str = NIGHT, agent: str = "", title: str | None = None,
             notes: str = "", file: str | None = None, lesson_id: str | None = None) -> dict:
    name = f"{session}/subagents/{agent}" if agent else session
    return {"type": "verdict", "run": run, "ts": run, "session": session, "agent": agent,
            "transcript": f"/home/.claude/projects/-Users-me-git-webapp/{name}.jsonl", "project": WEBAPP,
            "title": title, "verdict": outcome.upper(), "outcome": outcome, "lesson_id": lesson_id, "file": file,
            "notes": notes}


def _event(kind: str, *, run: str = NIGHT, ts: str | None = None, **fields) -> dict:
    return {"type": "event", "run": run, "ts": ts or run, "kind": kind, **fields}


def _lesson(lesson_id: str, memory_dir: Path, *, status: str = "live", title: str = "Урок", cue: str = "когда-то",
            lesson: str = "делай так", confidence: str = "medium", repeats: int = 1, sessions: tuple = (FIRST,),
            created_at: str = "2026-10-21T23:31:00Z", live_since: str | None = None, **extra) -> Lesson:
    return Lesson(id=lesson_id, project_root=WEBAPP, memory_dir=str(memory_dir), file=f"auto_{lesson_id}.md",
                  title=title, hook="h", cue=cue, lesson=lesson, why="w", confidence=confidence, repeats=repeats,
                  evidence=[{"session": s, "ts": created_at, "text": "t"} for s in sessions], status=status,
                  sha256="", created_at=created_at, updated_at=created_at, live_since=live_since, **extra)


def _index(memory_dir: Path) -> None:
    """The mockup's MEMORY.md: 113 lines, two of them our lessons', 20 414 bytes, 82% of what the CLI loads."""
    auto = [f"- [auto_{i}.md](auto_{i}.md) - h" for i in (LESSON, OLDER)]
    text = "\n".join(auto + ["я" * 4_205] + ["x"] * 110) + "\n"
    text = text.replace("x", "x" * (16_209 - len(text) + 1), 1)
    memory_dir.mkdir(parents=True, exist_ok=True)
    (memory_dir / "MEMORY.md").write_text(text, encoding="utf-8")
    assert (memory_dir / "MEMORY.md").stat().st_size == 20_414


@dataclass
class Night:
    cfg: Config
    state: State
    records: list[dict]
    run: dict  # the scheduled run of the night, in `records`
    memory_dir: Path

    def render(self, **kw) -> str:
        return recent.render(self.cfg, self.state, self.records, NOW, tz=LOCAL, **kw)

    def lines(self, **kw) -> list[str]:
        return self.render(**kw).split("\n")


@pytest.fixture
def night(tmp_path) -> Night:
    """The night on 23 October from the spec's mockup, and the three days before it for the chart."""
    cfg = Config.under(tmp_path)
    memory_dir = cfg.projects_dir / "-Users-me-git-webapp" / "memory"
    _index(memory_dir)
    rule = str(memory_dir / "feedback_file_links_clickable.md")
    state = State(
        installed_at="2026-10-20T19:37:35Z",
        lessons={
            LESSON: _lesson(LESSON, memory_dir, title=TITLE, cue="нужно ограничить время команды в оболочке на macOS",
                            lesson="gtimeout из coreutils или параметр timeout у Bash", repeats=3,
                            sessions=(FIRST, PR, AUTOTESTS), live_since="2026-10-22T23:30:05Z"),
            OLDER: _lesson(OLDER, memory_dir, created_at="2026-10-20T23:31:00Z", live_since="2026-10-20T23:31:00Z"),
        },
        user_rule_repeats={rule: [{"session": AUTOTESTS, "ts": "2026-10-22T20:00:00Z", "text": "а ссылки где"}]},
    )
    run = _run(NIGHT, ended="2026-10-22T23:31:43Z", segments_seen=28, no_signals=9, llm_calls=19,
               outcomes={"no_entry": 16, "repeat": 2, "user_rule_repeat": 1})
    records = [
        _run("2026-10-20T19:45:00Z", ended="2026-10-20T19:50:00Z", kind="backfill", trigger="manual",
             outcomes={"no_entry": 1}),
        _run("2026-10-20T23:30:00Z", ended="2026-10-20T23:33:00Z",
             outcomes={"no_entry": 19, "repeat": 1, "new_live": 1, "user_rule_repeat": 1, "llm_error": 3}),
        _run("2026-10-21T23:30:00Z", ended="2026-10-21T23:31:00Z", outcomes={"no_entry": 7, "new_staged": 1}),
        run,
        _verdict(PR, "repeat", title="Webapp PR 256", lesson_id=LESSON,
                 notes="timeout 580 fvm … → command not found. Агент снова обернул команду в timeout."),
        _verdict(AUTOTESTS, "repeat", agent="agent-a1b2c3d4", lesson_id=LESSON,
                 title="Автотесты dev-process-toolkit и магазин-e2e (сабагент)",
                 notes="то же с bun test. Сабагент запустил тесты через timeout и получил command not found."),
        _verdict(AUTOTESTS, "user_rule_repeat", title="Автотесты dev-process-toolkit и магазин-e2e",
                 file="feedback_file_links_clickable.md",
                 notes="статус без ссылок. Пользователь переспросил «а ссылки где», хотя правило про это есть."),
        _verdict(STORES, "no_entry", title="Кнопки сторов в web2app", file="reference_mcp_chrome_profile.md",
                 notes="Нового урока нет, но запись о профиле Chrome противоречит сессии."),
        *(_verdict(f"{i:08x}-0000-4000-8000-000000000000", "no_entry", title=f"Сессия {i}",
                   notes="Ничего, что стоило бы запомнить.") for i in range(1, 16)),
        _event("lesson_live", ts="2026-10-22T23:30:05Z", project=WEBAPP, lesson_id=LESSON, title=TITLE),
        _event("lesson_repeat", ts="2026-10-22T23:30:05Z", project=WEBAPP, session=PR, lesson_id=LESSON,
               repeats=2, counted=True, status="live"),
        _event("lesson_repeat", ts="2026-10-22T23:31:02Z", project=WEBAPP, session=AUTOTESTS, lesson_id=LESSON,
               repeats=3, counted=True, status="live"),
        _event("user_rule_conflict", ts="2026-10-22T23:31:20Z", project=WEBAPP, session=STORES,
               file=str(memory_dir / "reference_mcp_chrome_profile.md"), notes=CONFLICT_NOTES),
        _event("user_rule_repeat", ts="2026-10-22T23:31:30Z", project=WEBAPP, session=AUTOTESTS, file=rule),
    ]
    return Night(cfg, state, records, run, memory_dir)


def _section(lines: list[str], name: str) -> list[str]:
    """The lines under the section header that names `name`, without the blank lines around them."""
    start = next(i for i, line in enumerate(lines) if line.startswith("── ") and name in line)
    end = next((i for i in range(start + 1, len(lines)) if lines[i].startswith("── ")), len(lines))
    body = lines[start + 1:end]
    while body and not body[0]:
        body.pop(0)
    while body and not body[-1]:
        body.pop()
    return body


def _card(lines: list[str], head: str) -> list[str]:
    """The card whose first line starts with `head`, through its └─ line."""
    start = next(i for i, line in enumerate(lines) if line.startswith(head))
    end = next(i for i in range(start, len(lines)) if lines[i].startswith("└─"))
    return lines[start:end + 1]


def _render(tmp_path, state: State, records: list[dict], **kw) -> list[str]:
    return recent.render(Config.under(tmp_path), state, records, NOW, tz=LOCAL, **kw).split("\n")


def test_header_names_nightly_run_and_counts(night):
    assert night.lines()[:2] == [
        "🌙 agent-reflect · ночь на 23 октября · 03:30 · ✅ без ошибок",
        "   ✍️ в памяти: 1 изменение · 🚨 требуют тебя: 2 · 🗂 разобрано: 19 из 28",
    ]


def test_sections_come_in_mockup_order_and_end_in_one_column(night):
    heads = [line for line in night.lines() if line.startswith("── ")]
    assert heads == [
        "── 📊 ПОСЛЕДНИЕ 7 ДНЕЙ ──────────────────────────────────────",
        "── ✍️ ПАМЯТЬ · 1 изменение ──────────────────────────────────",
        "── 🚨 ТРЕБУЕТ ТЕБЯ · 2 ──────────────────────────────────────",
        "── 🗂 РАЗОБРАНО · 19 из 28 ──────────────────────────────────",
        "── 🩺 ТИХИЕ СБОИ · 0 👌 ─────────────────────────────────────",
    ]


def test_chart_section_matches_mockup(night):
    assert _section(night.lines(), "📊 ПОСЛЕДНИЕ 7 ДНЕЙ") == [
        "   вт 20  ⬜  1",
        "   ср 21  ⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜🟦🟩🟧🟥🟥🟥  25",
        "   чт 22  ⬜⬜⬜⬜⬜⬜⬜🟩  8",
        " ▶ пт 23  ⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜🟦🟦🟧  19  ← эта ночь",
        "          ⬜ ничего · 🟦 повтор · 🟩 новый урок · 🟧 твоё правило · 🟥 ошибка",
        "",
        "   🌱 живых уроков: 0 → 1 → 1 → 2 📈",
        "   🗃 MEMORY.md webapp  ████████████████░░░░ 82% ⚠️ осталось ~4,6 тыс. байт",
    ]


def test_live_lessons_of_a_night_are_those_its_run_counted(night):
    # three lessons went live and out again since: today's lessons would say 2
    night.run["live"] = {WEBAPP: 4, "/Users/me/git/other": 1}
    assert "   🌱 живых уроков: 0 → 1 → 1 → 5 📈" in _section(night.lines(), "📊 ПОСЛЕДНИЕ 7 ДНЕЙ")


def test_a_gauge_too_wide_for_the_terminal_puts_its_warning_under_the_bar(night):
    for lesson in night.state.lessons.values():
        lesson.project_root = "/Users/me/git/webapp-with-a-much-longer-name"
    chart = _section(night.lines(), "📊 ПОСЛЕДНИЕ 7 ДНЕЙ")
    assert chart[-2:] == ["   🗃 MEMORY.md webapp-with-a-much-longer-name  ████████████████░░░░ 82%",
                          " " * 48 + "⚠️ осталось ~4,6 тыс. байт"]
    assert all(recent._columns(line) <= 76 for line in chart[-2:])


def test_memory_card_matches_mockup(night):
    assert _section(night.lines(), "ПАМЯТЬ") == [
        "┌─ 1/1 ➕ урок стал виден агентам · webapp",
        "│  «На macOS нет timeout — используй gtimeout»",
        "│  когда:   нужно ограничить время команды в оболочке на macOS",
        "│  делать:  gtimeout из coreutils или параметр timeout у Bash",
        "│  🔁 +2 повтора за ночь, всего 3 сессии · 👀 с 03:30 · 📖 прочтений: 0",
        "└─ auto_4c326d27.md",
    ]


def test_attention_cards_match_mockup(night):
    assert _section(night.lines(), "ТРЕБУЕТ ТЕБЯ") == [
        "┌─ 1/2 🤨 запись, похоже, устарела",
        "│  reference_mcp_chrome_profile.md",
        "│  «Кнопки сторов в web2app»: три new_page с isolatedContext упали",
        "│  с «browser is already running», а в записи сказано, что способ проверен",
        "└─ 👉 перепроверить способ и поправить запись",
        "",
        "┌─ 2/2 🔁 опять двадцать пять: нарушено твоё правило",
        "│  feedback_file_links_clickable.md",
        "│  «Автотесты dev-process-toolkit и магазин-e2e»: «а ссылки где»",
        "└─ нарушений с 20.10: 1",
    ]


def test_only_the_first_rule_card_has_the_meme(night):
    other = str(night.memory_dir / "feedback_no_force_push.md")
    night.records.append(_event("user_rule_repeat", ts="2026-10-22T23:31:40Z", project=WEBAPP, session=PR,
                                file=other))
    heads = [line for line in _section(night.lines(), "ТРЕБУЕТ ТЕБЯ") if line.startswith("┌─ ")]
    assert heads == ["┌─ 1/3 🤨 запись, похоже, устарела", "┌─ 2/3 🔁 опять двадцать пять: нарушено твоё правило",
                     "┌─ 3/3 🔁 нарушено твоё правило"]


def test_reviewed_section_lists_what_was_not_nothing(night):
    assert _section(night.lines(), "РАЗОБРАНО") == [
        "  🟦 Webapp PR 256 — timeout 580 fvm … → command not found",
        "  🟦 Автотесты dev-process-toolkit и магазин-e2e (сабагент) — то же с bun…",  # 78 columns in full
        "  🟧 Автотесты dev-process-toolkit и магазин-e2e — статус без ссылок",
        "  ⬜ … и ещё 16 раз «ничего» (с причинами: recent -v)",
        "  😴 9 отрезков без ошибок и без твоих ответов модель не смотрела",
    ]
    assert night.lines()[-1] == "── 🩺 ТИХИЕ СБОИ · 0 👌 ─────────────────────────────────────"


@pytest.mark.parametrize("title, notes", [
    # a title that fits, the reason cut to what is left of the line
    ("Автотесты dev-process-toolkit и магазин-e2e (сабагент)",
     "Нового урока нет, но запись о профиле Chrome противоречит тому, что было в сессии, и это стоит проверить."),
    # a title too long for a line: it gives way, a few words of the reason still show
    ("Очень длинное название сессии, которое модель придумала по первой реплике пользователя целиком",
     "Повтор урока про timeout: агент снова обернул команду в timeout, которого на macOS нет."),
    ("Очень длинное название сессии, которое модель придумала по первой реплике пользователя целиком", ""),
    ("Короткое 🚀 название", "Повтор урока про timeout: агент снова обернул команду в timeout на macOS."),
])
def test_reviewed_lines_fit_the_terminal(night, title, notes):
    night.records.append(_verdict("c0ffee04-1111", "repeat", title=title, notes=notes))
    for verbose in (False, True):
        reviewed = _section(night.lines(verbose=verbose), "РАЗОБРАНО")
        assert all(recent._columns(line) <= 76 for line in reviewed)
    line = [line for line in reviewed if line.startswith("  🟦 ")][-1]  # the decisions keep journal order
    shown, _, reason = line.removeprefix("  🟦 ").partition(" — ")
    assert shown == title or (shown.endswith("…") and title.startswith(shown[:-1]))
    if notes:
        assert len(reason) >= 12 and notes.startswith(reason.rstrip("…").rstrip())


def test_reviewed_section_counts_the_work_the_user_sent_back():
    runs = [{"episodes": {"self_report": 27, "intent": 20}}, {"episodes": {"intent": 3, "constraint": 14, "other": 0}},
            {"episodes": "нет"}, {}]
    lines = recent._reviewed(runs, [], [], False)
    assert lines == ["  🔁 ты возвращал работу: 64 — сказал «готово» — не готово: 27, понял не",
                     "     так: 23, нарушил правило: 14"]
    assert all(recent._columns(line) <= 76 for line in lines)
    assert recent._reviewed([{"episodes": {}}, {}], [], [], False) == []


def test_reviewed_section_collapses_nothing(night):
    lines = night.lines()
    assert "  ⬜ … и ещё 16 раз «ничего» (с причинами: recent -v)" in lines
    assert "  😴 9 отрезков без ошибок и без твоих ответов модель не смотрела" in lines

    verbose = night.lines(verbose=True)
    assert "  ⬜ … и ещё 16 раз «ничего» (с причинами: recent -v)" not in verbose
    assert len([line for line in verbose if line.startswith("  ⬜ ")]) == 16
    stores = verbose.index("  ⬜ Кнопки сторов в web2app — Нового урока нет, но запись о профиле Chrome")
    assert verbose[stores + 1] == "     противоречит сессии."
    assert "  😴 9 отрезков без ошибок и без твоих ответов модель не смотрела" in verbose


def test_verbose_reasons_wrap_at_76(night):
    night.records.append(_verdict("c0ffee00-1111", "no_entry", title="Длинная", notes="причина " * 30))
    lines = _section(night.lines(verbose=True), "РАЗОБРАНО")
    start = lines.index("  ⬜ Длинная — причина причина причина причина причина причина причина")
    tail = lines[start + 1:start + 4]
    assert all(line.startswith("     причина") and len(line) <= 76 for line in tail)
    assert " ".join(line.split(" — ", 1)[-1].strip() for line in lines[start:start + 4]) == " ".join(["причина"] * 30)


def test_short_reason_is_the_first_sentence_within_60(night):
    night.records += [
        _verdict("c0ffee01-1111", "repeat", title="Кратко", notes="Одно предложение. Второе."),
        _verdict("c0ffee02-1111", "repeat", title="Длинно", notes="очень " * 20 + "длинное первое предложение."),
    ]
    lines = night.lines()
    assert "  🟦 Кратко — Одно предложение" in lines
    long = next(line for line in lines if line.startswith("  🟦 Длинно — "))
    reason = long.removeprefix("  🟦 Длинно — ")
    assert len(reason) == 60 and reason.endswith("…") and reason.startswith("очень очень")


def test_cards_are_numbered_and_separated(night):
    night.run["outcomes"]["llm_error"] = 1
    night.records.append(_verdict("c0ffee03-1111", "llm_error", title="Упавшая", notes="claude timed out"))
    night.records.append(_event("lesson_new", project=WEBAPP, session=PR, lesson_id="deadbeef", status="staged",
                                title="Урок, которого уже нет"))
    lines = night.lines()
    for name in ("ПАМЯТЬ", "ТРЕБУЕТ ТЕБЯ"):
        cards = "\n".join(_section(lines, name)).split("\n\n")
        assert len(cards) >= 2
        for i, card in enumerate(cards, 1):
            card_lines = card.split("\n")
            assert card_lines[0].startswith(f"┌─ {i}/{len(cards)} ")
            assert card_lines[-1].startswith("└─ ")
            assert all(line.startswith("│  ") for line in card_lines[1:-1])


def test_window_starts_at_last_scheduled_run(night):
    night.records += [
        _run("2026-10-22T08:15:00Z", kind="backfill", trigger="manual", segments_seen=5, llm_calls=5,
             outcomes={"no_entry": 5}),  # 12:15 local the day before
        _run(LATER, ended="2026-10-23T03:01:00Z", trigger="manual", segments_seen=2, no_signals=1, llm_calls=1,
             outcomes={"no_entry": 1}),
    ]
    start, nightly, runs = recent.window(night.records, NOW, None)
    assert start == datetime(2026, 10, 22, 23, 30, tzinfo=timezone.utc)
    assert nightly is night.run
    assert [run["run"] for run in runs] == [NIGHT, LATER]

    lines = night.lines()
    assert lines[0] == "🌙 agent-reflect · ночь на 23 октября · 03:30 · ✅ без ошибок · прогонов: 2"
    assert lines[1].endswith("🗂 разобрано: 20 из 30")


def test_a_run_at_login_does_not_cut_the_night_off(tmp_path):
    # launchd also starts the job at login (RunAtLoad), so the run after a reboot is a scheduled one too
    lesson = _lesson(LESSON, tmp_path, title=TITLE, live_since="2026-10-22T23:30:05Z")
    nightly = _run(NIGHT, ended="2026-10-22T23:31:00Z", segments_seen=1, llm_calls=1, outcomes={"repeat": 1})
    login = _run("2026-10-23T10:05:00Z", ended="2026-10-23T10:06:00Z")  # 14:05 local
    records = [
        _run("2026-10-22T12:00:00Z"),  # 16:00 local the day before, a login too: within 24 h, not this night
        nightly,
        _event("lesson_live", ts="2026-10-22T23:30:05Z", project=WEBAPP, lesson_id=LESSON, title=TITLE),
        login,
    ]
    now = datetime(2026, 10, 23, 11, 0, tzinfo=timezone.utc)  # 15:00 local

    assert recent.window(records, now, None, LOCAL) == (datetime(2026, 10, 22, 23, 30, tzinfo=timezone.utc),
                                                         nightly, [nightly, login])
    lines = recent.render(Config.under(tmp_path), State(lessons={LESSON: lesson}), records, now,
                          tz=LOCAL).split("\n")
    assert lines[0] == "🌙 agent-reflect · ночь на 23 октября · 03:30 · ✅ без ошибок · прогонов: 2"
    assert _section(lines, "ПАМЯТЬ")[0] == "┌─ 1/1 ➕ урок стал виден агентам · webapp"


def test_no_nightly_run_falls_back_to_24h_and_says_so(tmp_path):
    records = [
        _run("2026-10-21T23:30:00Z", segments_seen=9, llm_calls=9),  # the night before: older than 24 h
        _run(LATER, trigger="manual", segments_seen=2, llm_calls=1, outcomes={"no_entry": 1}),
    ]
    start, nightly, runs = recent.window(records, NOW, None)
    assert (start, nightly, [run["run"] for run in runs]) == (NOW - timedelta(hours=24), None, [LATER])

    lines = _render(tmp_path, State(installed_at="2026-10-20T19:37:35Z"), records)
    assert lines[0] == "🌙 agent-reflect · последние 24 ч · 🦗 ночной прогон не запускался"
    assert lines[1].endswith("🗂 разобрано: 1 из 2")


def test_hours_gives_rolling_window(night):
    start, nightly, runs = recent.window(night.records, NOW, 2)
    assert (start, nightly, runs) == (NOW - timedelta(hours=2), None, [])

    lines = night.lines(hours=2)
    assert lines[0] == "🌙 agent-reflect · последние 2 ч"
    assert lines[1] == "   ✍️ в памяти: 0 изменений · 🚨 требуют тебя: 0 · 🗂 разобрано: 0 из 0"

    lines = night.lines(hours=6)
    assert lines[0] == "🌙 agent-reflect · последние 6 ч"
    assert lines[1] == "   ✍️ в памяти: 1 изменение · 🚨 требуют тебя: 2 · 🗂 разобрано: 19 из 28"


def test_error_resolved_later_is_not_a_card(night):
    night.run["outcomes"]["llm_error"] = 1
    night.records += [
        _verdict("s1", "llm_error", title="Упавшая", notes="claude timed out\nafter 600s"),
        _run(LATER, trigger="manual", segments_seen=1, llm_calls=1, outcomes={"no_entry": 1}),
        _verdict("s1", "no_entry", run=LATER, title="Упавшая", notes="Ничего."),
    ]
    lines = night.lines()
    assert lines[0] == "🌙 agent-reflect · ночь на 23 октября · 03:30 · ✅ без ошибок · прогонов: 2"
    assert not any("🔥" in line for line in lines)
    assert "🟥" in next(line for line in lines if line.startswith(" ▶ пт 23"))
    assert "  🟥 Упавшая — claude timed out after 600s" in lines  # still told among the decisions


def test_error_of_a_subagent_is_not_resolved_by_its_session(night):
    night.records += [
        _verdict("s1", "llm_error", agent="agent-x", title="Упавшая (сабагент)", notes="claude timed out"),
        _run(LATER, trigger="manual"),
        _verdict("s1", "no_entry", run=LATER, notes="Ничего."),
    ]
    assert "🔥 ошибок: 1" in night.lines()[0]


def test_unresolved_error_is_a_card_and_in_header(night):
    night.run["outcomes"]["llm_error"] = 1
    night.records.append(_verdict("s1", "llm_error", title="Упавшая", notes="claude timed out\nafter 600s"))
    lines = night.lines()
    assert lines[0] == "🌙 agent-reflect · ночь на 23 октября · 03:30 · 🔥 ошибок: 1 · this is fine 🐶"
    assert lines[1].startswith("   ✍️ в памяти: 1 изменение · 🚨 требуют тебя: 3 ·")
    assert _card(lines, "┌─ 3/3") == [
        "┌─ 3/3 🔥 модель не ответила",
        "│  «Упавшая»",
        "│  claude timed out after 600s",
        "└─ 👉 смотри logs/reflect.log",
    ]


def test_error_events_and_a_crashed_run_are_cards(night):
    later = _run(LATER, kind="backfill", trigger="manual", failed="RuntimeError: boom")
    night.records += [
        later,
        _event("transcript_failed", transcript=f"/home/.claude/projects/-Users-me-git-webapp/{PR}.jsonl",
               error="bad\njson"),
        _event("index_unreadable", project=WEBAPP, error="MEMORY.md is not UTF-8"),
        # a failed write of a decision is told by that decision's 🟥, not twice
        _verdict("s2", "write_failed", title="Незаписанная", notes="Новый урок."),
        _event("memory_write_failed", project=WEBAPP, session="s2", lesson_id="x", file="auto_x.md",
               error="disk full"),
    ]
    lines = night.lines()
    assert "🔥 ошибок: 4 · this is fine 🐶 · прогонов: 2" in lines[0]
    attention = _section(lines, "ТРЕБУЕТ ТЕБЯ")
    assert _card(attention, "┌─ 3/6") == [
        "┌─ 3/6 🔥 урок не записался", "│  «Незаписанная»", "│  disk full", "└─ 👉 смотри logs/reflect.log"]
    assert _card(attention, "┌─ 4/6") == [
        "┌─ 4/6 🔥 транскрипт не разобран", "│  «Webapp PR 256»", "│  bad json", "└─ 👉 смотри logs/reflect.log"]
    assert _card(attention, "┌─ 5/6") == [
        "┌─ 5/6 🔥 MEMORY.md не читается", "│  webapp", "│  MEMORY.md is not UTF-8", "└─ 👉 смотри logs/reflect.log"]
    assert _card(attention, "┌─ 6/6") == [
        "┌─ 6/6 🔥 прогон упал", "│  RuntimeError: boom", "└─ 👉 смотри logs/backfill.log"]


def test_failed_consolidate_is_a_card(night):
    night.records.append(_run("2026-10-23T00:00:00Z", kind="consolidate",
                              failed="план не получен: claude timed out after 600s"))
    lines = night.lines()
    assert lines[0] == ("🌙 agent-reflect · ночь на 23 октября · 03:30 · 🔥 ошибок: 1 · this is fine 🐶"
                        " · прогонов: 2")
    assert _card(lines, "┌─ 3/3") == [
        "┌─ 3/3 🔥 недельная чистка не прошла",
        "│  план не получен: claude timed out after 600s",
        "└─ 👉 смотри logs/consolidate.log",
    ]


def test_consolidate_in_window_links_digest(night):
    night.records.append(_run("2026-10-23T00:00:00Z", kind="consolidate", digest="/x/digests/2026-W44.md"))
    lines = night.lines()
    memory = _section(lines, "ПАМЯТЬ")
    assert memory[-3:] == ["└─ auto_4c326d27.md", "", "🧹 прошла недельная чистка: /x/digests/2026-W44.md"]
    assert lines[1].startswith("   ✍️ в памяти: 1 изменение · 🚨 требуют тебя: 2 ·")


def test_staged_lesson_card_says_why_it_waits(night):
    waits = _lesson("11111111", night.memory_dir, status="staged", title="Ждёт", confidence="medium")
    full = _lesson("22222222", night.memory_dir, status="staged", title="Некуда", confidence="high")
    night.state.lessons |= {waits.id: waits, full.id: full}
    night.records += [_event("lesson_new", project=WEBAPP, session=FIRST, lesson_id=lesson.id, status="staged",
                             title=lesson.title) for lesson in (waits, full)]
    lines = night.lines()
    assert _card(lines, "┌─ 2/3") == [
        "┌─ 2/3 🆕 новый урок · webapp",
        "│  «Ждёт»",
        "│  когда:   когда-то",
        "│  делать:  делай так",
        "│  всего 1 сессия · 📖 прочтений: 0",
        "│  ⏳ ждёт второго случая",
        "│  принять сразу: agent-reflect approve 11111111",
        "└─ auto_11111111.md",
    ]
    assert "│  ⏳ в MEMORY.md нет места" in _card(lines, "┌─ 3/3")
    assert lines[1].startswith("   ✍️ в памяти: 3 изменения ·")


def test_unlisted_and_rejected_lessons_get_cards(night):
    unlisted = _lesson("33333333", night.memory_dir, status="staged", confidence="high", repeats=2)
    deleted = _lesson("44444444", night.memory_dir, status="rejected")
    unlined = _lesson("55555555", night.memory_dir, status="rejected")
    night.state.lessons |= {lesson.id: lesson for lesson in (unlisted, deleted, unlined)}
    night.records += [
        _event("lesson_unlisted", project=WEBAPP, lesson_id=unlisted.id, title="Урок", reason="no_room"),
        _event("lesson_rejected", project=WEBAPP, lesson_id=deleted.id, title="Урок", reason="file_deleted"),
        _event("lesson_rejected", project=WEBAPP, lesson_id=unlined.id, title="Урок", reason="index_line_deleted"),
    ]
    lines = night.lines()
    assert [line for line in lines if line.startswith("┌─ ")][1:4] == [
        "┌─ 2/4 ➖ урок убран из MEMORY.md: не хватило места · webapp",
        "┌─ 3/4 🗑 урок отклонён: ты удалил файл · webapp",
        "┌─ 4/4 🗑 урок отклонён: ты убрал строку из MEMORY.md · webapp",
    ]
    assert "│  ⏳ в MEMORY.md нет места" in _card(lines, "┌─ 2/4")
    assert lines[1].startswith("   ✍️ в памяти: 4 изменения ·")


@pytest.mark.parametrize("kinds, status, head", [
    (("lesson_new", "lesson_live"), "live", "➕ урок стал виден агентам"),
    (("lesson_new",), "staged", "🆕 новый урок"),
    (("lesson_repeat",), "staged", "🔁 урок повторился"),
    (("lesson_repeat", "lesson_violated"), "live", "⚠️ урок нарушен после публикации"),
    (("lesson_violated", "lesson_user_owned"), "user_owned", "✋ урок перешёл к тебе: ты правил файл"),
    (("lesson_live", "lesson_unlisted"), "staged", "➖ урок убран из MEMORY.md: не хватило места"),
    # it went live in the window, and whatever took its line out put it back: the lesson is live
    (("lesson_live", "lesson_unlisted"), "live", "➕ урок стал виден агентам"),
    (("lesson_unlisted", "lesson_live", "lesson_repeat"), "live", "➕ урок стал виден агентам"),
    (("lesson_live", "lesson_rejected"), "rejected", "🗑 урок отклонён: ты удалил файл"),
])
def test_card_is_headed_by_the_weightiest_event(tmp_path, kinds, status, head):
    lesson = _lesson("77777777", tmp_path, status=status)
    records = [_run(NIGHT)] + [_event(kind, project=WEBAPP, lesson_id=lesson.id, reason="file_deleted")
                               for kind in kinds]
    lines = _render(tmp_path, State(lessons={lesson.id: lesson}), records)
    assert f"┌─ 1/1 {head} · webapp" in lines


def test_a_lesson_taken_out_and_put_back_is_no_change(tmp_path):
    # live before the window and after it: rebalance took its line out and put it back in one call
    lesson = _lesson("77777777", tmp_path, live_since=NIGHT)
    records = [_run(NIGHT), _event("lesson_unlisted", project=WEBAPP, lesson_id=lesson.id, reason="no_room"),
               _event("lesson_live", project=WEBAPP, lesson_id=lesson.id)]
    lines = _render(tmp_path, State(lessons={lesson.id: lesson}), records)
    assert lines[1].startswith("   ✍️ в памяти: 0 изменений ·")
    assert _section(lines, "ПАМЯТЬ") == []


def test_violated_lesson_says_how_often(night):
    night.state.lessons[OLDER].violations = 2
    night.records.append(_event("lesson_violated", project=WEBAPP, session=PR, lesson_id=OLDER, violations=2,
                                counted=True))
    card = _card(night.lines(), "┌─ 2/2")
    assert card[0] == "┌─ 2/2 ⚠️ урок нарушен после публикации · webapp"
    assert card[-3:] == ["│  всего 1 сессия · 👀 с 21.10 · 📖 прочтений: 0",
                         "│  ⚠️ нарушен после публикации: 2 раза", "└─ auto_9a1f03bb.md"]


def test_user_owned_event_without_title_takes_the_lessons(night):
    night.state.lessons[OLDER].status = "user_owned"
    night.state.lessons[OLDER].title = "Свой урок"
    night.records.append(_event("lesson_user_owned", project=WEBAPP, session=PR, lesson_id=OLDER,
                                reason="the user edited the lesson; REPEAT not applied"))
    card = _card(night.lines(), "┌─ 2/2")
    assert card[:2] == ["┌─ 2/2 ✋ урок перешёл к тебе: ты правил файл · webapp", "│  «Свой урок»"]


def test_card_for_lesson_missing_from_state(night):
    night.records.append(_event("lesson_new", project=WEBAPP, session=PR, lesson_id="deadbeef", status="staged",
                                title="Урок, которого уже нет"))
    assert _card(night.lines(), "┌─ 2/2") == [
        "┌─ 2/2 🆕 новый урок · webapp",
        "│  «Урок, которого уже нет»",
        "└─ auto_deadbeef.md",
    ]


def test_no_project_is_a_count_not_attention(night):
    night.records.append(_event("no_project", transcript="/home/.claude/projects/-tmp/x.jsonl",
                                cwd="/private/tmp/scratch"))
    lines = night.lines()
    assert "  🫥 вне проектов: 1 отрезок" in lines
    assert lines[1].startswith("   ✍️ в памяти: 1 изменение · 🚨 требуют тебя: 2 ·")
    assert "/private/tmp/scratch" not in "\n".join(lines)


def test_no_project_of_a_run_without_events_is_counted_from_its_outcomes(night):
    night.run["outcomes"]["no_project"] = 3
    night.run["deferred"] = 2
    lines = night.lines()
    assert "  🫥 вне проектов: 3 отрезка" in lines
    assert "  ⏭ отложено до следующей ночи: 2 отрезка" in lines


def test_lessons_nobody_reads_are_a_card(night):
    night.state.reads_since = "2026-10-01T00:00:00Z"
    for i in range(3):
        lesson = _lesson(f"8888888{i}", night.memory_dir, created_at="2026-10-02T00:00:00Z")
        night.state.lessons[lesson.id] = lesson
    lines = night.lines()
    card = lines[lines.index("┌─ 3/3 📭 агенты не читают уроки"):]
    foot = card[1:card.index("") if "" in card else None]
    assert foot[0].startswith("└─ 👉 ")
    assert all(line.startswith("      ") for line in foot[1:])
    assert " ".join(line.strip() for line in foot) == "└─ 👉 " + usage.stalled_note()
    assert len(foot[0]) <= 75 and all(len(line) <= 76 for line in foot[1:])  # 👉 takes two columns


def test_quiet_failures_since_the_window_start(night):
    def entry(last_ts, detail):
        return {"kind": "hook_failure", "count": 3, "first_ts": last_ts, "last_ts": last_ts, "sessions": ["s1"],
                "detail": detail, "notified_at": None}
    night.state.health = {
        "PreToolUse|~/hooks/play-sound.sh|127": entry("2026-10-23T01:00:00.000Z", "jq: command not found"),
        "Stop|~/hooks/old.sh|1": entry("2026-10-22T20:00:00Z", "старое"),  # before the night began
        "PreToolUse|~/x/dev-process-toolkit/gate.sh|127": entry("2026-10-23T01:00:00Z", "ignored"),
    }
    lines = night.lines()
    assert "── 🩺 ТИХИЕ СБОИ · 1 ────────────────────────────────────────" in lines
    alert = _section(lines, "ТИХИЕ СБОИ")
    assert alert[0].startswith("  - hook_failure · PreToolUse ~/hooks/play-sound.sh · 3 раза · код 127")
    assert all(line.startswith("    ") and recent._columns(line) <= 76 for line in alert[1:])
    assert " ".join(line.strip() for line in alert).endswith("· jq: command not found")
    assert "старое" not in "\n".join(alert) and "gate.sh" not in "\n".join(lines)


def test_output_is_scrubbed(night):
    secret = "ghp_" + "a" * 36
    night.records.append(_verdict("s9", "repeat", title="Секрет", notes=f"в логе был ключ {secret} и всё"))
    quiet = next(r for r in night.records if r["type"] == "verdict" and r["session"] == STORES)
    quiet["notes"] = f"токен {secret}"  # shown with -v
    for out in (night.render(), night.render(verbose=True)):
        assert secret not in out and "[REDACTED]" in out


def test_a_secret_cut_short_is_still_scrubbed(night):
    # cut to its first few characters a token no longer looks like one: it is scrubbed before the cut
    night.records.append(_verdict("c0ffee05-1111", "repeat", title="x" * 60 + "ghp_" + "a" * 36))
    line = next(line for line in night.lines() if line.startswith("  🟦 xxxx"))
    assert "ghp_" not in line and "[REDACTED]" in line
    assert recent._clip("x" * 60 + "ghp_" + "a" * 36, 66) == "x" * 60 + "[REDA…"


def test_long_text_wraps_at_76(night):
    notes = " ".join(["объяснение"] * 36) + " /Users/me/" + "очень-длинный-путь/" * 8
    conflict = next(r for r in night.records if r.get("kind") == "user_rule_conflict")
    conflict["notes"] = notes
    card = _card(night.lines(), "┌─ 1/2")
    body = [line for line in card if line.startswith("│  ")]
    assert len(body) > 4
    assert all(len(line) <= 76 for line in body)
    words = " ".join(line[3:] for line in body[1:]).split()
    assert "".join(words) == "".join(f"«Кнопки сторов в web2app»: {notes}".split())


def test_empty_journal_renders(tmp_path):
    lines = _render(tmp_path, State(), [])
    assert lines[:2] == ["🌙 agent-reflect · последние 24 ч · 🦗 ночной прогон не запускался",
                         "   ✍️ в памяти: 0 изменений · 🚨 требуют тебя: 0 · 🗂 разобрано: 0 из 0"]
    heads = [line.rstrip("─ ") for line in lines if line.startswith("── ")]
    assert heads == ["── 📊 ПОСЛЕДНИЕ 7 ДНЕЙ", "── ✍️ ПАМЯТЬ · 0 изменений", "── 🚨 ТРЕБУЕТ ТЕБЯ · 0",
                     "── 🗂 РАЗОБРАНО · 0 из 0", "── 🩺 ТИХИЕ СБОИ · 0 👌"]
    assert _section(lines, "ПАМЯТЬ") == _section(lines, "ТРЕБУЕТ ТЕБЯ") == _section(lines, "РАЗОБРАНО") == []


def test_missing_title_uses_short_session_id(night):
    night.records.append(_verdict("5c39f173-9999-4000-8000-000000000000", "repeat", notes="Снова timeout."))
    assert "  🟦 сессия 5c39f173 — Снова timeout" in night.lines()


def test_journal_records_of_odd_shape_do_not_break_the_report(night):
    night.records += [
        {"type": "run", "run": "вчера", "started_at": "вчера"},
        {"type": "verdict", "run": NIGHT, "outcome": "repeat", "notes": None, "title": 7, "session": None},
        {"type": "event", "run": NIGHT, "kind": "lesson_new"},
        {"type": "event", "run": NIGHT, "kind": "user_rule_conflict", "notes": None},
        {"something": "else"},
    ]
    assert night.lines()[0].startswith("🌙 agent-reflect · ночь на 23 октября · 03:30")


def _notice(night: Night) -> tuple[str, str, str]:
    return recent.notice(night.cfg, night.state, night.records, NOW, tz=LOCAL)


def test_notice_sums_the_night_up_by_the_report_sections(night):
    assert _notice(night) == (
        "🌙 agent-reflect · ночь на 23 октября",
        "✅ без ошибок",
        "✍️ Память 1 · 🚨 Требует тебя 2 · 🗂 Разобрано 19 из 28 · 🩺 Сбои 0",
    )


def test_notice_puts_errors_in_the_status(night):
    night.run["outcomes"]["llm_error"] = 1
    night.records.append(_verdict("s1", "llm_error", title="Упавшая", notes="claude timed out"))
    _, status, totals = _notice(night)
    assert status == "🔥 ошибок: 1"
    assert "🚨 Требует тебя 3" in totals


def test_notice_of_a_quiet_night_keeps_every_total(tmp_path):
    state = State(installed_at="2026-10-20T19:37:35Z")
    records = [_run(NIGHT, ended="2026-10-22T23:31:00Z", segments_seen=4, no_signals=4)]
    assert recent.notice(Config.under(tmp_path), state, records, NOW, tz=LOCAL) == (
        "🌙 agent-reflect · ночь на 23 октября",
        "😴 тихая ночь · ✅ без ошибок",
        "✍️ Память 0 · 🚨 Требует тебя 0 · 🗂 Разобрано 0 из 4 · 🩺 Сбои 0",
    )


def test_a_quiet_failure_is_not_a_quiet_night(tmp_path):
    state = State(installed_at="2026-10-20T19:37:35Z", health={"PreToolUse|~/hooks/play-sound.sh|127": {
        "kind": "hook_failure", "count": 3, "first_ts": "2026-10-23T01:00:00Z", "last_ts": "2026-10-23T01:00:00Z",
        "sessions": ["s1"], "detail": "jq: command not found", "notified_at": None}})
    _, status, totals = recent.notice(Config.under(tmp_path), state, [_run(NIGHT)], NOW, tz=LOCAL)
    assert status == "✅ без ошибок"
    assert totals.endswith("🩺 Сбои 1")
