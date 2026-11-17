"""English output: every report, digest, notification and CLI line agent-reflect writes itself.

The fixtures are the Russian suite's, with every Cyrillic letter in their data turned into "x": whatever
Cyrillic the English output still has came from the code, not from a session title or a lesson.
"""
from __future__ import annotations

import re
from dataclasses import fields, is_dataclass, replace
from datetime import date

import pytest

from agent_reflect import chart, digest, health, i18n, recent, usage
from test_consolidate import FakeLlm, _plan, env  # noqa: F401  (env is a fixture used by name)
from test_consolidate import _lesson as consolidate_lesson, _run as consolidate_run
from test_delivery import BASELINE
from test_digest import _memory_reads, full_summary, make_lesson
from test_recent import LATER, LOCAL, NOW, WEBAPP, _event, _lesson, _run, _verdict, night  # noqa: F401

CYRILLIC = re.compile(r"[А-Яа-яЁё]")


def _latin(value):
    if isinstance(value, str):
        return CYRILLIC.sub("x", value)
    if isinstance(value, list):
        return [_latin(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_latin(v) for v in value)
    if isinstance(value, dict):
        return {_latin(k): _latin(v) for k, v in value.items()}
    if is_dataclass(value) and not isinstance(value, type):
        return replace(value, **{f.name: _latin(getattr(value, f.name)) for f in fields(value) if f.init})
    return value


def _cyrillic(text: str) -> list[str]:
    return [line for line in text.split("\n") if CYRILLIC.search(line)]


@pytest.fixture
def english():
    i18n.use("en")
    yield
    i18n.use("ru")


@pytest.fixture
def busy(night):  # noqa: F811
    """The mockup's night with a card of every kind, every counter of the decisions and a silent failure."""
    memory_dir = night.memory_dir
    lessons = night.state.lessons
    lessons["5a6b7c8d"] = _lesson("5a6b7c8d", memory_dir, status="staged", title="Ждёт", created_at=night.run["run"])
    lessons["6b7c8d9e"] = _lesson("6b7c8d9e", memory_dir, title="Нарушен", violations=2)
    for i in range(3):  # live for weeks and never read: the index does not reach the agent
        lessons[f"0000000{i}"] = _lesson(f"0000000{i}", memory_dir, live_since="2026-09-01T00:00:00Z")
    night.state.reads_since = "2026-09-01T00:00:00Z"
    night.state.health = {"PreToolUse|~/hooks/play-sound.sh|127": {
        "kind": "hook_failure", "count": 3, "first_ts": "2026-10-23T01:00:00Z", "last_ts": "2026-10-23T01:00:00Z",
        "sessions": ["s1"], "detail": "jq: command not found", "notified_at": None}}
    night.run.update(deferred=2, outcomes={**night.run["outcomes"], "no_project": 1},
                     episodes={"intent": 2, "self_report": 1, "constraint": 1, "operational": 1, "other": 1},
                     spent={"calls": 3, "in": 12_000, "out": 800, "cost_usd": 0.21})
    night.records += [
        _event("lesson_new", project=WEBAPP, lesson_id="5a6b7c8d", status="staged", title="Ждёт"),
        _event("lesson_violated", project=WEBAPP, lesson_id="6b7c8d9e", title="Нарушен"),
        _event("lesson_rejected", project=WEBAPP, lesson_id="7c8d9e0f", title="Урок", reason="file_deleted"),
        _event("lesson_rejected", project=WEBAPP, lesson_id="8d9e0f1a", title="Урок", reason="index_line_deleted"),
        _event("lesson_unlisted", project=WEBAPP, lesson_id="9e0f1a2b", title="Урок"),
        _event("lesson_user_owned", project=WEBAPP, lesson_id="0f1a2b3c", title="Урок"),
        _verdict("deadbeef-0000-4000-8000-000000000000", "llm_error", title="Упала", notes="timeout"),
        _event("transcript_failed", transcript="/p/x.jsonl", error="boom"),
        _event("guard_rules_unwritten", error="disk full"),
        _run(LATER, failed="RuntimeError: boom", trigger="manual"),
        _run("2026-10-23T02:00:00Z", kind="consolidate", trigger="schedule", digest="/d/2026-W43.md"),
        _run("2026-10-23T02:30:00Z", kind="consolidate", trigger="manual", failed="ValueError: plan"),
    ]
    night.state.lessons = _latin(night.state.lessons)
    night.state.user_rule_repeats = _latin(night.state.user_rule_repeats)
    night.records[:] = _latin(night.records)
    return night


def test_the_night_report_in_english_has_every_section_and_no_russian(busy, english):
    text = busy.render()
    assert _cyrillic(text) == []
    flat = " ".join(text.split())  # a long line is wrapped onto the next
    for seen in ("night of 23 October", "✍️ memory: ", "📊 LAST 7 DAYS", "✍️ MEMORY · ", "🚨 NEEDS YOU · ",
                 "🗂 REVIEWED · ", "🩺 SILENT FAILURES · 1", "Fr 23", "this night", "🟦 repeat", "🌱 live lessons: ",
                 "🆕 new lesson", "⏳ waits for a second case", "accept now: agent-reflect approve 5a6b7c8d",
                 "⚠️ broken after it went live: 2 times", "🗑 lesson rejected: you deleted its file",
                 "➖ lesson taken out of MEMORY.md", "✋ the lesson is yours now", "🧹 the weekly clean-up ran: ",
                 "🤨 a note looks out of date", "🔁 here we go again: your rule was broken",
                 "🔥 the model did not answer", "🔥 transcript not read", "🔥 hook rules not written",
                 "🔥 the run crashed", "🔥 the weekly clean-up failed", "👉 see logs/",
                 "📭 agents do not read the lessons", "and 16 more “nothing”", "you sent work back: 6",
                 "misunderstood: 2", "💸 model: 3 calls · 12k tokens in, 800 out · $0.21 at API prices",
                 "outside projects: 1 segment", "put off to the next night: 2 segments",
                 "the model did not look", "3 times · code 127"):
        assert seen in flat, seen


def test_the_verbose_report_and_the_notification_in_english(busy, english):
    assert _cyrillic(busy.render(verbose=True)) == []
    title, status, totals = recent.notice(busy.cfg, busy.state, busy.records, NOW, tz=LOCAL)
    assert title == "🌙 agent-reflect · night of 23 October"
    assert status.startswith("🔥 errors: ")
    assert totals.startswith("✍️ Memory ") and "🚨 Needs you " in totals and " of 28" in totals
    assert _cyrillic(" ".join((title, status, totals))) == []


def test_a_window_with_no_nightly_run_in_english(tmp_path, english):
    from agent_reflect.config import Config
    from agent_reflect.ledger import State
    text = recent.render(Config.under(tmp_path), State(), [], NOW, tz=LOCAL)
    assert "last 24 h · 🦗 the nightly run did not start" in text and _cyrillic(text) == []
    assert recent.render(Config.under(tmp_path), State(), [], NOW, hours=6, tz=LOCAL).startswith(
        "🌙 agent-reflect · last 6 h")


def _week_summary() -> dict:
    return _latin({
        **full_summary(),
        "violated": ["ab12cd34"],
        "upstream": [{"repo": "webapp", "sha": "abc1234", "date": "2026-10-21", "subject": "Тесты",
                      "rule": "правило", "place": "tdd/SKILL.md", "why": "потому что"}],
        "advisories": [{"ts": "t", "kind": "skill_large", "target": "personal:a", "chars": 9000, "limit": 8000},
                       {"ts": "t", "kind": "upstream_rebaselined", "repo": "webapp", "old": "a" * 40, "new": "b" * 40}],
        "health": [{"key": "api_error|529", "kind": "api_error", "count": 2, "last_ts": "t", "detail": "overloaded"}],
        "clusters": [{"lesson_id": "ab12cd34", "outcome": "repeat", "sessions": 2, "count": 3,
                      "said": ["я же говорил"], "note": "заметка"},
                     {"title": "Новое", "outcome": "new", "sessions": 1, "count": 1, "said": []}],
        "broken_rules": [{"rule": "на телефоне", "count": 3, "sessions": 2, "said": "я же говорил"}],
        "care": [{"lesson_id": "ab12cd34", "reason": "stale", "days": 50},
                 {"lesson_id": "ab12cd34", "reason": "unhelped", "violations": 2},
                 {"lesson_id": "ab12cd34", "reason": "ignored", "violations": 2, "fires": 3},
                 {"lesson_id": "ab12cd34", "reason": "recheck", "days": 40}],
        "effects": [{"lesson_id": "ab12cd34", "target": "personal:a", "verdict": verdict, "before": 2, "after": 1,
                     "days_left": 9}
                    for verdict in ("helped", "not_helped", "confounded", "too_little_data", "not_merged", "watching")],
        "usage": {"reads": 0, "with_reads": 0, "live": 1, "stalled": True},
        "memory_reads": _memory_reads(),
        "skipped_no_project": 4,
    })


def test_the_weekly_digest_in_english_has_every_section_and_no_russian(busy, english):
    lessons = {"ab12cd34": _latin(make_lesson(violations=2, trigger="timeout", fires=3))}
    block = digest.week_block(busy.cfg, busy.state.lessons, busy.records, date(2026, 10, 20), date(2026, 10, 23),
                              date(2026, 10, 20), LOCAL)
    text = digest.render(_week_summary(), lessons, "2026-W43", busy.cfg, chart=block)
    assert _cyrillic(text) == []
    for seen in ("# agent-reflect — week 2026-W43", "Total: 1 lesson, 2 skills, 1 PR", "## 📊 Week · 20–23 October",
                 "Segments with no project skipped: 4", "## 🆕 New lessons · 1", "repeats: 3", "When: ",
                 "broken after going live: 2 times", "(backup: run1)", "Lessons and notes: ",
                 "## 🧰 Candidates for dev-process-toolkit · 1", "To move them: ask an agent",
                 "grew to 9000 characters (limit 8000)", "is gone from the history of origin/main",
                 "2 times · code 529", "## 🧩 Repeated corrections · 2",
                 "lesson confirmed; 3 corrections in 2 sessions", "new lesson; 1 correction in 1 session",
                 "## 📜 Broken rules · 1", "3 times in 2 sessions; for example: ", "can be shown to the agent as it acts",
                 "## 🧹 Lesson care · 4", "50 days in the index", "and it has no hook pattern",
                 "although the hook showed it (3 times)", "`agent-reflect guard block ab12cd34`",
                 "a fact last confirmed 40 days ago", "## 📈 Did the moves into skills help · 6", "— helped: ",
                 "— did not help: ", "— cannot tell: ", "— too little data: ", "closed without a merge",
                 "the verdict in 9 days",
                 "## Lesson reads", "Lesson reads by the agent: 0", "shown by the hook: 3 times", "UserPromptSubmit",
                 "## Memory: not opened in 30 days", "2 of 4 files never opened", "Opened most: ",
                 "🔁 work sent back: ", "💸 model: 3 calls", "🗃 MEMORY.md · webapp", "chars    ", "~4.6k bytes left", "lessons  ", " 2 of 40"):
        assert seen in text, seen


def test_the_digest_notice_baseline_and_failures_in_english(english):
    title, subtitle, totals = digest.notice(_week_summary(), date(2026, 10, 20), date(2026, 10, 27))
    assert title == "📅 agent-reflect · 20–27 October"
    assert subtitle == "🧠 1 lesson → 🛠 2 skills"
    assert "🆕 Lessons 1" in totals and "🧹 Care 4" in totals and "📈 Moves 6" in totals
    assert digest.notice({}, date(2026, 10, 20), date(2026, 10, 27))[2] == "😴 no changes"
    assert digest.baseline_line({**BASELINE, "detector": 1}, 2) == (
        "📏 baseline 20 Oct–9 Nov: 9.1 per 100 of your replies (77 of 846, sessions: 31)"
        " — counted by another detector, not to be compared")
    store = {"PreToolUse|~/h/x.sh|1": {"kind": "hook_failure", "count": 1, "last_ts": "t"}}
    assert health.notify_message(store, list(store)) == (
        "agent-reflect: silent failures — hook x.sh fails (code 1), 1 time; see agent-reflect recent")
    assert _cyrillic(usage.stalled_note() + digest.guard_hint() + digest.memory_note()) == []
    assert [chart.form_name(f) for f in ("self_report", "constraint", "unknown")] == [
        "said “done” when it was not", "broke a rule", "unknown"]


# -- text a run writes into the journal and the digest, read back by the reports later --------


def test_a_subagent_title_is_marked_in_the_runs_language(tmp_path, english):
    from agent_reflect.transcript import session_title
    from conftest import write_jsonl
    write_jsonl(tmp_path / "-w-app" / "s1.jsonl", [{"type": "ai-title", "aiTitle": "Autotests"}])
    sub = tmp_path / "-w-app" / "s1" / "subagents" / "agent-a1.jsonl"
    write_jsonl(sub, [{"type": "user"}])
    assert session_title(sub) == "Autotests (subagent)"


def test_a_cut_diff_says_so_in_the_runs_language(english):
    from agent_reflect.edits import DIFF_MAX_LINES, unified_diff
    before = "\n".join(f"line {i}" for i in range(DIFF_MAX_LINES * 2))
    diff = unified_diff(before, before.replace("line", "row"), "SKILL.md")
    assert diff.splitlines()[-1].startswith("… diff cut, ") and _cyrillic(diff) == []


def test_a_merge_reason_speaks_the_runs_language(env, english):  # noqa: F811
    one, two = (consolidate_lesson(env, f"aaaa000{i}", status="staged", confidence="medium") for i in (1, 2))
    summary = consolidate_run(env, FakeLlm(_plan(merges=[{"keep": one.id, "absorb": [two.id], "reason": "same"}])))
    assert [r["reason"] for r in summary["removed"]] == [f"merged into {one.file}: same"]


def test_a_stopped_consolidate_says_why_in_the_runs_language(env, english):  # noqa: F811
    import json
    env.cfg.claude_settings.write_text(json.dumps({"autoDreamEnabled": True}), encoding="utf-8")
    stopped = consolidate_run(env, FakeLlm())
    assert stopped["aborted"].startswith("autoDreamEnabled is not off") and _cyrillic(stopped["aborted"]) == []
