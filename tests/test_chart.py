from __future__ import annotations

from collections import Counter
from datetime import date, timedelta, timezone

import pytest

from agent_reflect import chart, i18n
from agent_reflect.config import Config
from agent_reflect.ledger import Lesson

LOCAL = timezone(timedelta(hours=4))
D20, D21, D22, D23, D24 = (date(2026, 10, day) for day in range(20, 25))

# the nights of the spec's mockup, by local day
MOCKUP = {
    D20: Counter(no_entry=1),
    D21: Counter(no_entry=19, repeat=1, new_staged=1, user_rule_repeat=1, llm_error=3),
    D22: Counter(no_entry=7, new_staged=1),
    D23: Counter(no_entry=16, repeat=2, user_rule_repeat=1),
}


@pytest.fixture
def cfg(tmp_path):
    return Config.under(tmp_path)


def _run(ended_at: str, outcomes: dict | None = None, **extra) -> dict:
    return {"type": "run", "run": ended_at, "kind": "reflect", "trigger": "schedule", "started_at": ended_at,
            "ended_at": ended_at, "outcomes": outcomes or {}, **extra}


def _lesson(lid="ab12cd34", *, status="live", memory_dir="/m", project_root="/Users/me/git/webapp",
            created_at="2026-10-20T10:00:00Z", live_since=None) -> Lesson:
    return Lesson(id=lid, project_root=project_root, memory_dir=memory_dir, file=f"auto_{lid}.md", title="t",
                  hook="h", cue="c", lesson="l", why="w", confidence="high", repeats=1, evidence=[], status=status,
                  sha256="", created_at=created_at, updated_at=created_at, live_since=live_since)


def _index(memory_dir, *, lines: int, auto: int, chars: int, cyrillic: int = 0) -> str:
    """Write a MEMORY.md of exactly `lines` lines and `chars` characters, `cyrillic` of them two-byte letters."""
    head = [f"- [auto_{i}.md](auto_{i}.md) - h" for i in range(auto)]
    text = "\n".join(head + ["я" * cyrillic] + ["x"] * (lines - auto - 1)) + "\n"
    text = text.replace("x", "x" * (chars - len(text) + 1), 1)  # the first filler line takes the slack
    assert (len(text.splitlines()), len(text)) == (lines, chars)
    memory_dir.mkdir(parents=True, exist_ok=True)
    (memory_dir / "MEMORY.md").write_text(text, encoding="utf-8")
    return text


def _gauge_lesson(tmp_path, **index) -> Lesson:
    memory_dir = tmp_path / "memory"
    _index(memory_dir, **index)
    return _lesson(memory_dir=str(memory_dir))


def test_square_follows_the_colour_table():
    squares = {o: chart.square(o) for o in (
        "no_entry", "rejected_cue", "rejected_not_concrete", "repeat", "new_live", "new_staged", "violated",
        "user_rule_repeat", "llm_error", "bad_verdict", "write_failed", "apply_crashed", "something_new")}

    assert squares == {
        "no_entry": "⬜", "rejected_cue": "⬜", "rejected_not_concrete": "⬜", "something_new": "⬜",
        "repeat": "🟦", "new_live": "🟩", "new_staged": "🟩", "violated": "🟪", "user_rule_repeat": "🟧",
        "llm_error": "🟥", "bad_verdict": "🟥", "write_failed": "🟥", "apply_crashed": "🟥"}


def test_rows_match_mockup():
    rows = chart.day_rows(MOCKUP, D20, D23, D20, marked=frozenset({D23}), current=D23)

    assert rows == [
        "   вт 20  ⬜  1",
        "   ср 21  " + "⬜" * 19 + "🟦🟩🟧🟥🟥🟥  25",
        "   чт 22  " + "⬜" * 7 + "🟩  8",
        " ▶ пт 23  " + "⬜" * 16 + "🟦🟦🟧  19  ← эта ночь",
    ]


def test_marked_day_that_is_not_current_has_no_arrow_text():
    [_, _, _, row] = chart.day_rows(MOCKUP, D20, D23, D20, marked=frozenset({D23}))

    assert row == " ▶ пт 23  " + "⬜" * 16 + "🟦🟦🟧  19"


def test_rows_use_local_day():
    runs = [_run("2026-10-22T23:31:43Z", {"no_entry": 2}), _run("2026-10-22T19:59:59Z", {"repeat": 1})]

    counts = chart.day_counts(runs, LOCAL)

    assert counts == {D23: Counter(no_entry=2), D22: Counter(repeat=1)}
    assert chart.day_rows(counts, D22, D23, None) == ["   чт 22  🟦  1", "   пт 23  ⬜⬜  2"]


def test_day_counts_skips_no_project_and_what_is_not_a_run():
    runs = [
        _run("2026-10-23T03:31:43Z", {"no_entry": 1, "no_project": 8, "repeat": 0}),
        {"type": "verdict", "run": "2026-10-23T03:30:00Z", "outcome": "repeat"},
        {"type": "event", "run": "2026-10-23T03:30:00Z", "ts": "2026-10-23T03:30:05Z"},
        _run("2026-10-23T05:00:00Z", {"no_entry": 2}),
    ]

    assert chart.day_counts(runs, LOCAL) == {D23: Counter(no_entry=3)}


def test_day_counts_keeps_a_day_whose_runs_decided_nothing():
    runs = [_run("2026-10-23T03:31:43Z", {"no_project": 3}, kind="consolidate")]

    assert chart.day_counts(runs, LOCAL) == {D23: Counter()}
    assert chart.day_rows({D23: Counter()}, D23, D23, None) == ["   пт 23    0"]


def test_day_counts_falls_back_to_the_start_of_a_run_without_an_end():
    run = _run("2026-10-22T23:31:43Z", {"no_entry": 1})
    run["ended_at"] = None

    assert chart.day_counts([run], LOCAL) == {D23: Counter(no_entry=1)}


def test_day_without_run_after_install_sleeps():
    rows = chart.day_rows(MOCKUP, D22, D24, D20)

    assert rows[-1] == "   сб 24  💤 прогона не было"


def test_days_before_install_are_hidden():
    rows = chart.day_rows(MOCKUP, D20, D23, D22, current=D23)

    assert [row[3:8] for row in rows] == ["чт 22", "пт 23"]


def test_single_digit_day_keeps_the_squares_aligned():
    counts = {date(2026, 11, 1): Counter(repeat=1), date(2026, 11, 2): Counter()}

    rows = chart.day_rows(counts, date(2026, 10, 31), date(2026, 11, 2), None)

    assert rows == ["   сб 31  💤 прогона не было", "   вс 1   🟦  1", "   пн 2     0"]


def test_long_row_is_scaled():
    counts = {D23: Counter(no_entry=100, repeat=1), D22: Counter(repeat=2)}

    [short, long_] = chart.day_rows(counts, D22, D23, None)

    assert short == "   чт 22  🟦  2"
    assert long_.count("⬜") == 20 and long_.count("🟦") == 1 and long_.endswith("  101")
    assert chart.legend(counts, D22, D23).endswith("1 квадрат = 5 отрезков")


@pytest.mark.parametrize("n, squares, scale", [(25, 25, ""), (26, 13, "1 квадрат = 2 отрезка")])
def test_a_row_takes_25_squares_at_most(n, squares, scale):
    counts = {D23: Counter(no_entry=n)}

    [row] = chart.day_rows(counts, D23, D23, None)

    assert row == "   пт 23  " + "⬜" * squares + f"  {n}"
    assert chart.legend(counts, D23, D23) == "          ⬜ ничего" + (f" · {scale}" if scale else "")


def test_scale_keeps_every_row_within_the_limit():
    # 115 and one of each of the five other colours: ceil(120 / 25) is 5, but ceil(115 / 5) + 5 squares is 28,
    # so one square stands for 6
    counts = {D23: Counter(no_entry=115, repeat=1, new_live=1, violated=1, user_rule_repeat=1, llm_error=1)}

    [row] = chart.day_rows(counts, D23, D23, None)

    assert row.count("⬜") == 20 and row.count("🟥") == 1 and row.endswith("  120")
    assert sum(row.count(sq) for sq in "⬜🟦🟩🟪🟧🟥") == 25
    assert chart.legend(counts, D23, D23).endswith("1 квадрат = 6 отрезков")


def test_unscaled_period_has_no_scale_in_the_legend():
    assert "квадрат =" not in chart.legend(MOCKUP, D20, D23)


def test_legend_lists_the_colours_of_the_period_in_table_order():
    assert chart.legend(MOCKUP, D20, D23) == "          ⬜ ничего · 🟦 повтор · 🟩 новый урок · 🟧 твоё правило · 🟥 ошибка"
    assert chart.legend({D22: Counter(violated=1, repeat=1)}, D20, D23) == "          🟦 повтор · 🟪 нарушен урок"
    assert chart.legend(MOCKUP, D22, D22) == "          ⬜ ничего · 🟩 новый урок"
    assert chart.legend({D22: Counter(violated=1, repeat=1)}, D20, D21) == ""


def test_legend_names_every_colour_and_goes_on_under_itself():
    counts = {D23: Counter(no_entry=1, repeat=1, new_live=1, violated=1, user_rule_repeat=1, bad_verdict=1)}

    assert chart.legend(counts, D23, D23).split("\n") == [
        "          ⬜ ничего · 🟦 повтор · 🟩 новый урок · 🟪 нарушен урок",
        "          🟧 твоё правило · 🟥 ошибка",
    ]


def test_a_legend_with_a_scale_stays_within_the_chart_width():
    counts = {D23: Counter(no_entry=100, repeat=1, new_live=1, violated=1, user_rule_repeat=1, bad_verdict=1)}

    lines = chart.legend(counts, D23, D23).split("\n")

    assert lines == ["          ⬜ ничего · 🟦 повтор · 🟩 новый урок · 🟪 нарушен урок",
                     "          🟧 твоё правило · 🟥 ошибка · 1 квадрат = 5 отрезков"]
    assert all(chart.columns(line) <= chart.LEGEND_COLUMNS for line in lines)


def test_the_mockup_legend_stays_on_one_line():
    legend = chart.legend(MOCKUP, D20, D23)

    assert "\n" not in legend and chart.columns(legend) == 77


def test_error_notes_name_the_day():
    assert chart.error_notes(MOCKUP, D20, D23) == ["          🟥 ср 21: ошибок: 3. This is fine 🔥🐶"]


def test_error_notes_count_every_kind_of_error_and_every_day():
    counts = {D22: Counter(bad_verdict=1, write_failed=1), D23: Counter(apply_crashed=1), D24: Counter(repeat=1)}

    assert chart.error_notes(counts, D20, D24) == [
        "          🟥 чт 22: ошибок: 2. This is fine 🔥🐶", "          🟥 пт 23: ошибок: 1. This is fine 🔥🐶"]
    assert chart.error_notes(counts, D24, D24) == []


def test_live_line_counts_lessons_live_by_day_end():
    lessons = {
        "a": _lesson("a", live_since="2026-10-21T10:00:00Z"),
        # 20:30 UTC on the 22nd is 00:30 on the 23rd locally: it counts from the 23rd
        "b": _lesson("b", live_since="2026-10-22T20:30:00Z"),
        # no live_since: created_at stands in
        "c": _lesson("c", created_at="2026-10-21T01:00:00Z"),
        "gone": _lesson("gone", status="removed", live_since="2026-10-20T01:00:00Z"),
        "waits": _lesson("waits", status="staged", live_since="2026-10-20T01:00:00Z"),
    }

    assert chart.live_line(lessons, [D20, D21, D22, D23], LOCAL) == "🌱 живых уроков: 0 → 2 → 2 → 3 📈"


def test_live_line_matches_mockup():
    lessons = {"a": _lesson("a", live_since="2026-10-21T10:00:00Z"),
               "b": _lesson("b", live_since="2026-10-23T03:30:05Z")}

    assert chart.live_line(lessons, [D20, D21, D22, D23], LOCAL) == "🌱 живых уроков: 0 → 1 → 1 → 2 📈"


def test_live_line_takes_a_day_from_its_last_run_that_counted_live_lessons():
    lessons = {"a": _lesson("a", live_since="2026-10-21T10:00:00Z"),
               "b": _lesson("b", live_since="2026-10-23T03:30:05Z")}
    runs = [
        _run("2026-10-22T10:00:00Z", live={"/p/one": 2}),  # 14:00 on the 22nd locally: the last that counted
        _run("2026-10-21T20:00:00Z", live={"/p/one": 3, "/p/two": 1}),  # 00:00 on the 22nd
        _run("2026-10-22T12:00:00Z"),  # counted nothing: a run read from the logs
        {"type": "verdict", "run": "2026-10-22T13:00:00Z", "ended_at": "2026-10-22T13:00:00Z", "live": {"/p": 9}},
    ]

    # the 20th, 21st and 23rd have no count: the lessons that are live now tell them
    assert chart.live_line(lessons, [D20, D21, D22, D23], LOCAL, runs) == "🌱 живых уроков: 0 → 1 → 2 → 2 📈"


def test_live_line_trend():
    lessons = {"a": _lesson("a", live_since="2026-10-21T10:00:00Z")}

    assert chart.live_line(lessons, [D21, D22], LOCAL).endswith("1 → 1 ➖")
    assert chart.live_line({"a": _lesson("a", status="live", live_since="2026-10-22T10:00:00Z")}, [D21],
                           LOCAL) == "🌱 живых уроков: 0 ➖"
    assert chart.live_line(lessons, [], LOCAL) == ""


def test_live_line_trend_can_fall():
    # the days are read in the order given, so a line that ends lower than it began reads as 📉
    lessons = {"a": _lesson("a", live_since="2026-10-21T10:00:00Z")}

    assert chart.live_line(lessons, [D22, D20], LOCAL) == "🌱 живых уроков: 1 → 0 📉"


def test_memory_gauge_daily_matches_mockup(cfg, tmp_path):
    lesson = _gauge_lesson(tmp_path, lines=113, auto=2, chars=16_209, cyrillic=4_205)

    assert (tmp_path / "memory" / "MEMORY.md").stat().st_size == 20_414
    assert chart.memory_gauges(cfg, {"x": lesson}, detailed=False) == [
        "🗃 MEMORY.md webapp  ████████████████░░░░ 82% ⚠️ осталось ~4,6 тыс. байт"]


def test_memory_gauge_weekly_matches_mockup(cfg, tmp_path):
    lesson = _gauge_lesson(tmp_path, lines=113, auto=2, chars=16_209, cyrillic=4_205)

    assert chart.memory_gauges(cfg, {"x": lesson}, detailed=True) == [
        "🗃 MEMORY.md · webapp",
        "   символы  ██████████████░░░░░░  68%",
        "   байты    ████████████████░░░░  82%  ⚠️ осталось ~4,6 тыс. байт",
        "   строки   ████████████░░░░░░░░  59%",
        "   уроки    █░░░░░░░░░░░░░░░░░░░  2 из 40",
    ]


def test_no_warning_below_80_percent(cfg, tmp_path):
    lesson = _gauge_lesson(tmp_path, lines=100, auto=3, chars=1_000)  # lines are the fullest: 53%

    [line] = chart.memory_gauges(cfg, {"x": lesson}, detailed=False)
    detailed = chart.memory_gauges(cfg, {"x": lesson}, detailed=True)

    assert line == "🗃 MEMORY.md webapp  ███████████░░░░░░░░░ 53%"
    assert "⚠️" not in "".join(detailed)


def test_warning_starts_at_80_percent_and_says_what_is_left(cfg, tmp_path):
    lesson = _gauge_lesson(tmp_path, lines=152, auto=2, chars=2_000)  # 152 of 190 lines is exactly 80%

    [line] = chart.memory_gauges(cfg, {"x": lesson}, detailed=False)

    assert line == "🗃 MEMORY.md webapp  ████████████████░░░░ 80% ⚠️ осталось 38 строк"


def test_warning_names_each_limit(cfg, tmp_path):
    chars = _gauge_lesson(tmp_path / "a", lines=10, auto=1, chars=20_400)
    lessons = _gauge_lesson(tmp_path / "b", lines=40, auto=34, chars=2_000)

    assert chart.memory_gauges(cfg, {"x": chars}, detailed=False)[0].endswith("85% ⚠️ осталось ~3,6 тыс. символов")
    assert chart.memory_gauges(cfg, {"x": lessons}, detailed=False)[0].endswith("85% ⚠️ осталось 6 мест для уроков")
    assert chart.memory_gauges(cfg, {"x": lessons}, detailed=True)[-1] == (
        "   уроки    █████████████████░░░  34 из 40  ⚠️ осталось 6 мест для уроков")


def test_what_is_left_agrees_with_its_number(cfg, tmp_path):
    one = _gauge_lesson(tmp_path / "a", lines=189, auto=1, chars=2_000)
    two = _gauge_lesson(tmp_path / "b", lines=188, auto=1, chars=2_000)

    assert chart.memory_gauges(cfg, {"x": one}, detailed=False)[0].endswith("99% ⚠️ осталось 1 строка")
    assert chart.memory_gauges(cfg, {"x": two}, detailed=False)[0].endswith("99% ⚠️ осталось 2 строки")


def test_gauge_clamps_an_index_over_its_limit(cfg, tmp_path):
    lesson = _gauge_lesson(tmp_path, lines=20, auto=1, chars=26_000)

    [line] = chart.memory_gauges(cfg, {"x": lesson}, detailed=False)

    assert line == "🗃 MEMORY.md webapp  ████████████████████ 108% ⚠️ сверх лимита на ~2,0 тыс. символов"


def test_a_limit_overrun_says_by_how_much(cfg, tmp_path):
    size = _gauge_lesson(tmp_path / "a", lines=10, auto=1, chars=14_000, cyrillic=12_000)  # 26 000 bytes
    rows = _gauge_lesson(tmp_path / "b", lines=200, auto=1, chars=2_000)
    row = _gauge_lesson(tmp_path / "c", lines=191, auto=1, chars=2_000)
    places = _gauge_lesson(tmp_path / "d", lines=50, auto=42, chars=2_000)

    assert chart.memory_gauges(cfg, {"x": size}, detailed=False)[0].endswith(
        "104% ⚠️ сверх лимита на ~1,0 тыс. байт")
    assert chart.memory_gauges(cfg, {"x": rows}, detailed=False)[0].endswith("105% ⚠️ сверх лимита на 10 строк")
    assert chart.memory_gauges(cfg, {"x": row}, detailed=False)[0].endswith("101% ⚠️ сверх лимита на 1 строку")
    assert chart.memory_gauges(cfg, {"x": places}, detailed=True)[-1] == (
        "   уроки    ████████████████████  42 из 40  ⚠️ сверх лимита на 2 места для уроков")


def test_a_daily_warning_too_wide_for_its_line_goes_under_the_bar(cfg, tmp_path):
    lesson = _gauge_lesson(tmp_path, lines=152, auto=2, chars=2_000)
    lesson.project_root = "/Users/me/git/a-project-with-a-rather-long-name"

    lines = chart.memory_gauges(cfg, {"x": lesson}, detailed=False, width=73)

    assert lines == ["🗃 MEMORY.md a-project-with-a-rather-long-name  ████████████████░░░░ 80%",
                     " " * 48 + "⚠️ осталось 38 строк"]
    assert chart.columns(lines[0]) == 72 and lines[1].index("⚠️") == chart.columns(lines[0].split("█")[0])
    assert chart.memory_gauges(cfg, {"x": lesson}, detailed=False, width=93) == chart.memory_gauges(
        cfg, {"x": lesson}, detailed=False) == [
        "🗃 MEMORY.md a-project-with-a-rather-long-name  ████████████████░░░░ 80% ⚠️ осталось 38 строк"]


def test_gauge_without_memory_md_is_empty(cfg, tmp_path):
    lesson = _lesson(memory_dir=str(tmp_path / "none"))

    assert chart.memory_gauges(cfg, {"x": lesson}, detailed=False) == ["🗃 MEMORY.md webapp  ░░░░░░░░░░░░░░░░░░░░ 0%"]


def test_gauge_for_unreadable_memory_md(cfg, tmp_path):
    memory_dir = tmp_path / "memory"
    memory_dir.mkdir()
    (memory_dir / "MEMORY.md").write_bytes(b"\xff\xfe broken")
    lesson = _lesson(memory_dir=str(memory_dir))

    assert chart.memory_gauges(cfg, {"x": lesson}, detailed=False) == ["🗃 MEMORY.md webapp  не читается"]
    assert chart.memory_gauges(cfg, {"x": lesson}, detailed=True) == ["🗃 MEMORY.md webapp  не читается"]


def test_gauge_per_project_with_live_or_staged_lessons_only(cfg, tmp_path):
    webapp = _gauge_lesson(tmp_path / "n", lines=10, auto=1, chars=500)
    other = _lesson("o", status="staged", memory_dir=str(tmp_path / "o"), project_root="/Users/me/git/agent-reflect")
    done = _lesson("d", status="removed", memory_dir=str(tmp_path / "d"), project_root="/Users/me/git/old")
    second = _lesson("n2", memory_dir=webapp.memory_dir)

    lines = chart.memory_gauges(cfg, {"1": webapp, "2": other, "3": done, "4": second}, detailed=False)

    assert [ln.split("  ")[0] for ln in lines] == ["🗃 MEMORY.md agent-reflect", "🗃 MEMORY.md webapp"]
    assert chart.memory_gauges(cfg, {"3": done}, detailed=False) == []


def test_plural_and_period_phrases():
    forms = ("изменение", "изменения", "изменений")

    assert [i18n.plural(n, ("change", "changes"), forms) for n in (1, 2, 5, 11, 21, 22, 12, 0, 111, 101)] == [
        "изменение", "изменения", "изменений", "изменений", "изменение", "изменения", "изменений", "изменений",
        "изменений", "изменение"]
    assert chart.period_phrase(D20, date(2026, 10, 27)) == "20–27 октября"
    assert chart.period_phrase(date(2026, 10, 28), date(2026, 11, 3)) == "28 октября – 3 ноября"
    assert chart.period_phrase(D23, D23) == "23 октября"
    assert chart.period_phrase(date(2025, 12, 30), date(2026, 1, 2)) == "30 декабря – 2 января"


def test_dates_and_plurals_in_english():
    i18n.use("en")
    assert [i18n.plural(n, ("change", "changes"), ("изменение", "изменения", "изменений")) for n in (0, 1, 2, 21)] == [
        "changes", "change", "changes", "changes"]
    assert chart.period_phrase(D20, date(2026, 10, 27)) == "20–27 October"
    assert chart.period_phrase(date(2025, 12, 30), date(2026, 1, 2)) == "30 December – 2 January"
    assert chart.day_label(date(2026, 11, 17)) == "Tu 17"
    assert chart.short_date(date(2026, 11, 7)) == "7 Nov"
    i18n.use("ru")
    assert chart.short_date(date(2026, 11, 7)) == "07.11"


def test_day_label_and_date_phrase():
    assert [chart.day_label(d) for d in (D20, D21, D22, D23, D24, date(2026, 10, 25), date(2026, 10, 26))] == [
        "вт 20", "ср 21", "чт 22", "пт 23", "сб 24", "вс 25", "пн 26"]
    assert chart.date_phrase(D23) == "23 октября"
    assert [chart.date_phrase(date(2026, m, 1)) for m in range(1, 13)] == [
        "1 января", "1 февраля", "1 марта", "1 апреля", "1 мая", "1 июня",
        "1 июля", "1 августа", "1 сентября", "1 октября", "1 ноября", "1 декабря"]
