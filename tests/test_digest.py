from __future__ import annotations

import subprocess
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest

from agent_reflect import chart, digest
from agent_reflect.config import Config
from agent_reflect.ledger import Lesson


@pytest.fixture
def cfg(tmp_path):
    return Config.under(tmp_path)


def make_lesson(**kw) -> Lesson:
    base = dict(id="ab12cd34", project_root="/Users/me/git/webapp", memory_dir="/m", file="auto_ab12cd34.md",
                title="Не трогать .env", hook="h", cue="правишь .env", lesson="спроси сначала", why="w",
                confidence="high", repeats=3,
                evidence=[{"session": "s1", "ts": "t", "text": "x"}, {"session": "s2", "ts": "t", "text": "y"}],
                status="live", sha256="", created_at="", updated_at="")
    base.update(kw)
    return Lesson(**base)


def full_summary() -> dict:
    return {
        "new_lessons": ["ab12cd34"],
        "promoted": [
            {"target": "personal:a", "summary": "Sum A", "lesson_ids": ["ab12cd34"], "user_rule_files": ["/r/x.md"],
             "backup_run_id": "run1", "pr_url": None, "notes": ["note one"]},
            {"target": "repo:webapp", "summary": "Sum B", "lesson_ids": [], "user_rule_files": [],
             "backup_run_id": None, "pr_url": "https://github.com/o/r/pull/1", "notes": []},
        ],
        "prs": ["https://github.com/o/r/pull/1"],
        "removed": [{"file": "/m/auto_x.md", "reason": "дубль"}],
        "suggestions": [{"file": "/r/CLAUDE.md", "suggestion": "уточни"}],
        "errors": [{"ts": "2026-10-20T10:00:00", "kind": "llm_error", "error": "boom"}],
    }


def empty_summary() -> dict:
    return {"new_lessons": [], "promoted": [], "prs": [], "removed": [], "suggestions": [], "errors": []}


def test_week_tag():
    assert digest.week_tag(datetime(2026, 10, 25)) == "2026-W43"


def test_render_sections_in_order_and_empty_dash():
    text = digest.render(empty_summary(), {}, "2026-W43")
    lines = text.splitlines()
    assert lines[0] == "# agent-reflect — неделя 2026-W43"
    heads = ["## 🆕 Новые уроки · 0", "## 🛠 Перенесено в скиллы · 0", "## 🔀 Draft PR · 0", "## 🗑 Убрано из памяти · 0",
             "## 💡 Предложения по твоим записям · 0", "## 🔥 Ошибки прогонов · 0"]
    idx = [lines.index(h) for h in heads]
    assert idx == sorted(idx)
    for i in idx:
        assert lines[i + 2] == "—"
    assert any(line.startswith("Итого:") for line in lines)


def test_render_lesson_line():
    text = digest.render(full_summary(), {"ab12cd34": make_lesson()}, "2026-W43")
    assert ("**Не трогать .env** (webapp, high, повторов: 3) — Когда: правишь .env → спроси сначала. "
            "Сессии: s1, s2") in text
    assert "note one" in text
    assert "/r/x.md" in text
    assert "https://github.com/o/r/pull/1" in text
    assert "llm_error" in text and "boom" in text
    assert "/r/CLAUDE.md" in text and "дубль" in text


def test_render_scrubs_secrets():
    s = full_summary()
    s["errors"][0]["error"] = "token ghp_" + "a" * 30
    assert "ghp_" not in digest.render(s, {}, "2026-W43")


def test_notice_names_the_period_and_the_sections_that_have_items():
    assert digest.notice(full_summary(), date(2026, 10, 20), date(2026, 10, 26)) == (
        "📅 agent-reflect · 20–26 октября",
        "🧠 1 урок → 🛠 2 скилла",
        "🆕 Уроки 1 · 🛠 В скиллы 2 · 🔀 PR 1 · 🗑 Убрано 1 · 💡 Предложения 1 · 🔥 Ошибки 1",
    )


def test_notice_of_an_empty_week():
    assert digest.notice(empty_summary(), date(2026, 10, 26), date(2026, 10, 26)) == (
        "📅 agent-reflect · 26 октября", "🧠 0 уроков → 🛠 0 скиллов", "😴 без изменений")


def test_notice_counts_what_the_headings_count(cfg):
    summary = {**empty_summary(), "violated": ["a", "b"], "upstream": [{}] * 5, "advisories": [{}],
               "health": [{}, {}, {}], "promoted": [{"target": "personal:a"}, {"target": "personal:a"}]}
    _, subtitle, totals = digest.notice(summary, date(2026, 10, 20), date(2026, 10, 26))
    assert subtitle == "🧠 0 уроков → 🛠 1 скилл"  # two promotions into one skill
    assert totals == "⚠️ Нарушены 2 · 🛠 В скиллы 2 · 🧰 Кандидаты 5 · 📣 Предупреждения 1 · 🩺 Сбои 3"
    text = digest.render(summary, {}, "2026-W44", cfg)  # the candidates line names the tool's files
    for heading in ("## ⚠️ Нарушены после публикации · 2", "## 🛠 Перенесено в скиллы · 2",
                    "## 🧰 Кандидаты в dev-process-toolkit · 5", "## 📣 Предупреждения · 1", "## 🩺 Тихие сбои · 3"):
        assert heading in text


def test_write(tmp_path: Path):
    cfg = Config.under(tmp_path)
    path = digest.write(cfg, "текст\n", "2026-W43")
    assert path == cfg.digests_dir / "2026-W43.md"
    assert path.read_text(encoding="utf-8") == "текст\n"
    digest.write(cfg, "новый", "2026-W43")
    assert path.read_text(encoding="utf-8") == "новый"
    assert [p.name for p in cfg.digests_dir.iterdir()] == ["2026-W43.md"]


def test_notify_terminal_notifier_with_open(tmp_path: Path):
    calls = []
    digest.notify("msg", tmp_path / "d.md", run=lambda cmd, **kw: calls.append(cmd),
                  which=lambda name: "/bin/terminal-notifier")
    assert calls == [["terminal-notifier", "-title", "agent-reflect", "-message", "msg",
                      "-open", (tmp_path / "d.md").as_uri()]]


def test_notify_with_title_subtitle_and_group(tmp_path: Path):
    calls = []
    digest.notify("✍️ Память 1", tmp_path / "2026-10-23.md", title="🌙 agent-reflect · ночь на 23 октября",
                  subtitle="✅ без ошибок", group=digest.NIGHT_GROUP, run=lambda cmd, **kw: calls.append(cmd),
                  which=lambda name: "/bin/terminal-notifier")
    assert calls == [["terminal-notifier", "-title", "🌙 agent-reflect · ночь на 23 октября",
                      "-subtitle", "✅ без ошибок", "-message", "✍️ Память 1", "-group", "agent-reflect.night",
                      "-open", (tmp_path / "2026-10-23.md").as_uri()]]


def test_notify_osascript_carries_title_and_subtitle():
    calls = []
    digest.notify("m", title='t "1"', subtitle="s", run=lambda cmd, **kw: calls.append(cmd), which=lambda name: None)
    assert calls == [["osascript", "-e", 'display notification "m" with title "t \\"1\\"" subtitle "s"']]


def test_notify_falls_back_to_osascript():
    calls = []
    digest.notify('say "hi" \\ there', run=lambda cmd, **kw: calls.append(cmd), which=lambda name: None)
    assert calls == [["osascript", "-e",
                      'display notification "say \\"hi\\" \\\\ there" with title "agent-reflect"']]


def test_notify_never_raises():
    def boom(cmd, **kw):
        raise subprocess.TimeoutExpired(cmd, 1)

    digest.notify("m", run=boom, which=lambda name: None)
    digest.notify("m", run=boom, which=lambda name: "/x")


def test_skipped_no_project_is_one_line_not_an_error():
    s = empty_summary()
    s["skipped_no_project"] = 4
    text = digest.render(s, {}, "2026-W43")
    assert text.count("Отрезков без проекта пропущено: 4") == 1
    errors = text.split("## 🔥 Ошибки прогонов")[1]
    assert "без проекта" not in errors
    assert "без проекта" not in digest.render(empty_summary(), {}, "2026-W43")


def test_promoted_item_shows_its_diff_in_a_fenced_block():
    s = full_summary()
    s["promoted"][0]["diff"] = "--- a/a/SKILL.md\n+++ b/a/SKILL.md\n@@ -1 +1,2 @@\n rule\n+new rule"
    lines = digest.render(s, {}, "2026-W43").splitlines()
    start = lines.index("  ```diff")
    assert lines[start - 1] == "  - note one"  # under the promoted item, after its notes
    assert lines[start + 1:start + 6] == ["  --- a/a/SKILL.md", "  +++ b/a/SKILL.md", "  @@ -1 +1,2 @@",
                                          "   rule", "  +new rule"]
    assert lines[start + 6] == "  ```"
    assert start < lines.index("- `repo:webapp` — Sum B (https://github.com/o/r/pull/1)")


def test_diff_with_backticks_gets_a_longer_fence():
    s = full_summary()
    s["promoted"][0]["diff"] = "+```bash\n+echo hi\n+```"
    lines = digest.render(s, {}, "2026-W43").splitlines()
    assert "  ````diff" in lines and lines[lines.index("  ````diff") + 4] == "  ````"


def test_write_night_keeps_the_report_as_text(tmp_path: Path):
    cfg = Config.under(tmp_path)
    path = digest.write_night(cfg, "🌙 ночь\n┌─ 1/1 карточка", date(2026, 10, 23))
    assert path == cfg.reports_dir / "2026-10-23.md"
    assert path.read_text(encoding="utf-8") == "```text\n🌙 ночь\n┌─ 1/1 карточка\n```\n"
    digest.write_night(cfg, "token ghp_" + "a" * 30, date(2026, 10, 23))
    assert "ghp_" not in path.read_text(encoding="utf-8")
    assert [p.name for p in cfg.reports_dir.iterdir()] == ["2026-10-23.md"]


def test_health_section_comes_after_the_errors():
    s = full_summary()
    s["health"] = [{"key": "PreToolUse|~/hooks/play-sound.sh|127", "kind": "hook_failure", "count": 3823,
                    "first_ts": "t0", "last_ts": "2026-10-20T10:00:00Z", "sessions": ["s1"],
                    "detail": "jq: command not found token=abcdef123456", "notified_at": None}]
    text = digest.render(s, {}, "2026-W43")
    lines = text.splitlines()
    assert lines.index("## 🩺 Тихие сбои · 1") > lines.index("## 🔥 Ошибки прогонов · 1")
    part = text.split("## 🩺 Тихие сбои", 1)[1]
    assert "play-sound.sh" in part and "3823" in part and "127" in part and "jq: command not found" in part
    assert "abcdef123456" not in text
    assert digest.notice(s, date(2026, 10, 20), date(2026, 10, 26))[2].endswith("🔥 Ошибки 1 · 🩺 Сбои 1")


def test_health_section_is_a_dash_when_empty():
    lines = digest.render(empty_summary(), {}, "2026-W43").splitlines()
    assert lines[lines.index("## 🩺 Тихие сбои · 0") + 2] == "—"


def test_violated_lessons_get_their_own_section_with_the_count():
    summary = {**empty_summary(), "violated": ["ab12cd34"]}
    text = digest.render(summary, {"ab12cd34": make_lesson(violations=2)}, "2026-W43")
    lines = text.splitlines()
    i = lines.index("## ⚠️ Нарушены после публикации · 1")
    assert "Не трогать .env" in lines[i + 2] and "нарушен после публикации: 2 раз" in lines[i + 2]
    assert lines.index("## 🆕 Новые уроки · 0") < i < lines.index("## 🛠 Перенесено в скиллы · 0")


def test_new_lesson_line_mentions_violations_only_when_there_are_some():
    text = digest.render(full_summary(), {"ab12cd34": make_lesson()}, "2026-W43")
    assert "нарушен после публикации" not in text
    text = digest.render(full_summary(), {"ab12cd34": make_lesson(violations=1)}, "2026-W43")
    assert "нарушен после публикации: 1 раз" in text


def test_skill_size_advisories_get_their_own_section():
    summary = {**empty_summary(), "advisories": [{"ts": "t", "kind": "skill_large", "target": "personal:a",
                                                  "chars": 25000}]}
    lines = digest.render(summary, {}, "2026-W43").splitlines()
    i = lines.index("## 📣 Предупреждения · 1")
    assert "personal:a" in lines[i + 2] and "25000" in lines[i + 2]
    assert lines.index("## 🔥 Ошибки прогонов · 0") < i


def _usage(**kw):
    return {"reads": 4, "with_reads": 1, "live": 2, "stalled": False, **kw}


def test_reads_section_lists_the_total_and_every_live_lesson():
    lessons = {"ab12cd34": make_lesson(reads=4), "ef56ab78": make_lesson(id="ef56ab78", title="Другой"),
               "00000000": make_lesson(id="00000000", title="Ждёт", status="staged", reads=9)}
    text = digest.render({**empty_summary(), "usage": _usage()}, lessons, "2026-W43")
    part = text.split("## Чтения уроков")[1].split("\n## ")[0]
    assert "Чтения уроков агентом: 4 (уроков с чтениями: 1 из 2 живых)" in part
    assert "Не трогать .env" in part and "прочитан: 4 раз" in part
    assert "Другой" in part and "прочитан: 0 раз" in part
    assert "Ждёт" not in part
    assert "UserPromptSubmit" not in text


def test_reads_section_names_the_next_experiment_only_when_stalled():
    text = digest.render({**empty_summary(), "usage": _usage(reads=0, with_reads=0, stalled=True)},
                         {"ab12cd34": make_lesson()}, "2026-W43")
    assert "UserPromptSubmit" in text.split("## Чтения уроков")[1]


LOCAL = timezone(timedelta(hours=4))
WEEK_HEAD = ["## 📊 Неделя · 20–27 октября", "", "```text", "x", "```"]


def _run(ended_at: str, outcomes: dict) -> dict:
    return {"type": "run", "run": ended_at, "kind": "reflect", "trigger": "schedule", "started_at": ended_at,
            "ended_at": ended_at, "outcomes": outcomes}


def _block(cfg, lessons=None, records=(), first=date(2026, 10, 20), last=date(2026, 10, 27),
           installed=date(2026, 10, 1)) -> list[str]:
    return digest.week_block(cfg, lessons or {}, list(records), first, last, installed, LOCAL)


def test_digest_starts_with_week_chart():
    lines = digest.render(full_summary(), {}, "2026-W44", chart=WEEK_HEAD).splitlines()
    start = lines.index(WEEK_HEAD[0])
    assert lines[start:start + len(WEEK_HEAD)] == WEEK_HEAD
    assert lines.index("Итого: 1 урок, 2 скилла, 1 PR") < start < lines.index("## 🆕 Новые уроки · 1")


def test_digest_without_a_chart_has_none():
    assert "Неделя" not in digest.render(full_summary(), {}, "2026-W44")


def test_chart_stays_below_the_total_and_the_skipped_segments():
    lines = digest.render({**empty_summary(), "skipped_no_project": 4}, {}, "2026-W44", chart=WEEK_HEAD).splitlines()
    assert lines.index("Отрезков без проекта пропущено: 4") < lines.index(WEEK_HEAD[0])


def test_week_block_covers_since_previous_consolidate(cfg):
    block = _block(cfg)
    assert block[:3] == ["## 📊 Неделя · 20–27 октября", "", "```text"]
    assert block[-1] == "```"
    rows = [line for line in block if "прогона не было" in line]
    assert [line.split()[0:2] for line in rows] == [["вт", "20"], ["ср", "21"], ["чт", "22"], ["пт", "23"],
                                                    ["сб", "24"], ["вс", "25"], ["пн", "26"], ["вт", "27"]]


def test_week_block_lays_out_chart_notes_live_lessons_and_memory(cfg):
    lesson = make_lesson(memory_dir=str(cfg.state_dir / "m"), created_at="2026-10-20T10:00:00Z")
    records = [_run("2026-10-21T10:00:00Z", {"no_entry": 2, "repeat": 1}),
               _run("2026-10-23T10:00:00Z", {"new_live": 1, "llm_error": 2, "no_project": 5})]
    block = _block(cfg, {"ab12cd34": lesson}, records, last=date(2026, 10, 24))
    assert block[:-6] == [
        "## 📊 Неделя · 20–24 октября", "", "```text",
        "   вт 20  💤 прогона не было",
        "   ср 21  ⬜⬜🟦  3",
        "   чт 22  💤 прогона не было",
        "   пт 23  🟩🟥🟥  3",
        "   сб 24  💤 прогона не было",
        "          ⬜ ничего · 🟦 повтор · 🟩 новый урок · 🟥 ошибка",
        "          🟥 пт 23: ошибок: 2. This is fine 🔥🐶",
        "",
        "🌱 живых уроков: 1 → 1 → 1 → 1 → 1 ➖",
        "",
    ]
    assert block[-6:-1] == chart.memory_gauges(cfg, {"ab12cd34": lesson}, detailed=True)
    assert block[-1] == "```"


def test_week_block_takes_live_lessons_from_the_runs_that_counted_them(cfg):
    lesson = make_lesson(memory_dir=str(cfg.state_dir / "m"), created_at="2026-10-20T10:00:00Z")
    records = [{**_run("2026-10-21T10:00:00Z", {"repeat": 1}), "live": {"/x/alpha": 3}}]
    block = _block(cfg, {"ab12cd34": lesson}, records, last=date(2026, 10, 22))
    assert "🌱 живых уроков: 1 → 3 → 1 ➖" in block


def test_week_block_leaves_out_days_before_installation_and_empty_lines(cfg):
    block = _block(cfg, first=date(2026, 10, 20), last=date(2026, 10, 23), installed=date(2026, 10, 22))
    assert block == ["## 📊 Неделя · 20–23 октября", "", "```text",
                     "   чт 22  💤 прогона не было",
                     "   пт 23  💤 прогона не было",
                     "",
                     "🌱 живых уроков: 0 → 0 ➖",
                     "```"]


def test_week_block_puts_a_blank_line_between_projects(cfg):
    lessons = {"a": make_lesson(id="a", project_root="/x/alpha", memory_dir=str(cfg.state_dir / "a")),
               "b": make_lesson(id="b", project_root="/x/beta", memory_dir=str(cfg.state_dir / "b"))}
    block = _block(cfg, lessons)
    alpha, beta = block.index("🗃 MEMORY.md · alpha"), block.index("🗃 MEMORY.md · beta")
    assert block[alpha - 1] == "" and block[beta - 1] == "" and beta - alpha == 6
    assert block[-2] != "" and block[-1] == "```"


def test_section_headings_have_counts(cfg):
    text = digest.render(full_summary(), {"ab12cd34": make_lesson()}, "2026-W44", cfg)
    lines = text.splitlines()
    for heading in ("## 🆕 Новые уроки · 1", "## 🛠 Перенесено в скиллы · 2", "## 🔀 Draft PR · 1",
                    "## 🗑 Убрано из памяти · 1", "## 💡 Предложения по твоим записям · 1", "## 🧰 Кандидаты в dev-process-toolkit · 0",
                    "## 🔥 Ошибки прогонов · 1", "## 📣 Предупреждения · 0", "## 🩺 Тихие сбои · 0"):
        assert heading in lines
    assert "## 🗑 Убрано из памяти · 0" in digest.render({**full_summary(), "removed": []}, {}, "2026-W44").splitlines()


def test_a_heading_counts_items_not_their_lines(cfg):
    summary = {**empty_summary(), "upstream": [
        {"repo": "webapp", "sha": "bbbbbbb", "date": "2026-10-21", "subject": "s", "rule": "r", "why": "w",
         "place": ""}] * 2}
    lines = digest.render(summary, {}, "2026-W44", cfg).splitlines()
    assert "## 🧰 Кандидаты в dev-process-toolkit · 2" in lines  # the two candidates, not the three lines


def test_second_digest_in_same_week_gets_suffix(cfg):
    assert digest.free_path(cfg, "2026-W44") == cfg.digests_dir / "2026-W44.md"
    digest.write(cfg, "первая", "2026-W44")
    second = digest.free_path(cfg, "2026-W44")
    assert second == cfg.digests_dir / "2026-W44-2.md"
    assert digest.write(cfg, "вторая", "2026-W44", second) == second
    assert digest.free_path(cfg, "2026-W44") == cfg.digests_dir / "2026-W44-3.md"
    assert (cfg.digests_dir / "2026-W44.md").read_text(encoding="utf-8") == "первая"
    assert second.read_text(encoding="utf-8") == "вторая"


def test_upstream_line_points_to_written_path(cfg):
    summary = {**empty_summary(), "upstream": [
        {"repo": "webapp", "sha": "bbbbbbb", "date": "2026-10-21", "subject": "s", "rule": "r", "why": "w",
         "place": "docs/patterns.md"}]}
    written = cfg.digests_dir / "2026-W44-2.md"
    text = digest.render(summary, {}, "2026-W44", cfg, path=written)
    assert f"перенеси кандидатов из {written} в dev-process-toolkit" in text
    assert str(cfg.digests_dir / "2026-W44.md") not in text


def _memory_reads() -> dict:
    def f(name, reads=0):
        return {"name": name, "reads": reads, "last_read": "2026-10-20T10:00:00.000Z" if reads else None}
    return {"days": 30, "locations": [
        {"name": "notes", "files": [f("feedback_x.md", 2), f("reference_y.md"), f("a_unread.md"),
                                    f("project_z.md", 7)]},
        {"name": "webapp", "files": [f(n, r) for n, r in (("a.md", 1), ("b.md", 9), ("c.md", 3), ("d.md", 3),
                                                          ("e.md", 5), ("f.md", 2))]},
    ]}


def _memory_part(text: str) -> str:
    return text.split("## Память: не открывали за 30 дней\n")[1].split("\n## ")[0]


def test_memory_section_lists_the_unread_files_and_the_most_read_per_location():
    text = digest.render({**empty_summary(), "memory_reads": _memory_reads()}, {}, "2026-W43")
    part = _memory_part(text)
    assert "не открывали" in part.splitlines()[1] and "CLAUDE.md" in part.splitlines()[1]
    assert "повод" in part.splitlines()[1]
    notes, webapp = part.split("### notes")[1].split("### webapp")
    assert "2 из 4 файлов не открывали: a_unread.md, reference_y.md" in notes
    assert "Чаще всего открывали: project_z.md (7), feedback_x.md (2)" in notes
    assert "0 из 6 файлов не открывали" in webapp
    # the five most read, ties by name; the sixth (a.md, 1) is left out
    assert "Чаще всего открывали: b.md (9), e.md (5), c.md (3), d.md (3), f.md (2)" in webapp
    assert "a.md (1)" not in webapp


def test_memory_section_only_when_the_summary_carries_it():
    assert "## Память" not in digest.render(empty_summary(), {}, "2026-W43")
    data = {"days": 30, "locations": [{"name": "notes", "files": [{"name": "x.md", "reads": 0, "last_read": None}]}]}
    part = _memory_part(digest.render({**empty_summary(), "memory_reads": data}, {}, "2026-W43"))
    assert "1 из 1 файла не открывали: x.md" in part
    assert "Чаще всего открывали" not in part  # nothing was read
