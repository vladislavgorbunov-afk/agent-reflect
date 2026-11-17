"""What the review of the corrections work found: each test is one finding, red before its fix."""
from __future__ import annotations

import json
import os
from dataclasses import replace
from datetime import date, datetime, timezone
from pathlib import Path

import pytest

from agent_reflect import clusters, consolidate, digest, guard, recent, reflect, transcript
from agent_reflect.cli import main
from agent_reflect.ledger import State, load_state
from agent_reflect.reflect import check_refs, episodes_of, run, trigger_hits, valid_trigger, worth_asking
from agent_reflect.transcript import read_segment
from conftest import write_jsonl
from test_delivery import (  # noqa: F401  (fixtures are used by name)
    CLUSTER, NOW, NOW_ISO, PROJECT, TS, WEEK, Counting, Model, Weekly, _baseline, _ep, _journal, _step, _talk,
    _verdict, call, cli, lesson, said, segment_of, user, week, world,
)

UTC = timezone.utc


def _result(tool_id, text, error=False):
    return {"type": "user", "timestamp": TS, "sessionId": "s1", "cwd": "/proj",
            "message": {"content": [{"type": "tool_result", "tool_use_id": tool_id, "is_error": error,
                                     "content": text}]}}


def _offset_after(path: Path, entries: list) -> int:
    """Where a run that read `entries` (the head of the file) would have stopped."""
    return sum(len(json.dumps(e, ensure_ascii=False).encode("utf-8")) + 1 for e in entries)


# -- 1. an answer that comes after the night's read ------------------------------------


def test_an_answer_to_what_the_agent_said_before_the_last_read_is_still_an_answer(tmp_path):
    head = [user("сделай кнопку"), said("Готово, кнопка на месте.")]
    tail = [user("кнопки нет, ничего не работает после перезагрузки")]
    p = tmp_path / "s.jsonl"
    write_jsonl(p, head + tail)
    seg = read_segment(p, _offset_after(p, head))
    assert seg.turns[0].before == "Готово, кнопка на месте."
    assert [t.n for t in seg.replies] == [1] and worth_asking(seg)


def test_what_the_agent_said_before_an_earlier_message_of_the_user_is_not_carried(tmp_path):
    head = [said("Готово."), user("а где вторая кнопка, я её не вижу?")]
    p = tmp_path / "s.jsonl"
    write_jsonl(p, head + [user("и первая пропала совсем")])
    seg = read_segment(p, _offset_after(p, head))
    assert seg.turns[0].before == "" and seg.replies == ()


def test_tool_results_between_the_agents_words_and_the_read_do_not_hide_them(tmp_path):
    head = [said("Проверяю и сообщу."), call("t1", "Bash", {"command": "ls"}), _result("t1", "ok"),
            said("Готово, всё на месте.")]
    p = tmp_path / "s.jsonl"
    write_jsonl(p, head + [user("нет, файла там нет, посмотри ещё раз")])
    assert read_segment(p, _offset_after(p, head)).turns[0].before == "Готово, всё на месте."


def test_the_agents_words_are_looked_for_only_near_the_read(tmp_path, monkeypatch):
    head = [said("Готово."), call("t1", "Bash", {"command": "echo " + "x" * 5000}), _result("t1", "y" * 5000)]
    p = tmp_path / "s.jsonl"
    write_jsonl(p, head + [user("нет, это не то, что я просил сделать")])
    monkeypatch.setattr(transcript, "CARRY_BYTES", 2000)
    assert read_segment(p, _offset_after(p, head)).turns[0].before == ""


def test_a_subagent_carries_nothing(tmp_path):
    head = [said("Готово.")]
    p = tmp_path / "s9" / "subagents" / "agent-a1.jsonl"
    write_jsonl(p, head + [user("ещё задача от родителя, сделай её тоже")])
    assert read_segment(p, _offset_after(p, head)).turns == ()


# -- 2. corrections and answers are counted over the same messages ----------------------


def test_an_episode_says_whether_its_message_answered_the_agent(tmp_path):
    seg = segment_of(tmp_path, [said("Готово."), user("кнопки нет, ничего не работает"),
                                user("и первая пропала совсем")])
    first, second = episodes_of(seg, [{"ref": "u1", "form": "self_report"}, {"ref": "u2", "form": "self_report"}])
    assert (first["reply"], second["reply"]) == (True, False)


def test_the_rate_counts_only_corrections_on_answers_and_shows_the_fraction():
    reply = lambda n, form: {"turn": n, "form": form, "reply": True}  # noqa: E731
    records = [
        _verdict("s1", "2026-11-10T03:30:00Z", [reply(2, "self_report"), reply(5, "intent"),
                                                {"turn": 6, "form": "intent", "reply": False}], replies=20),
        _verdict("s2", "2026-11-11T03:30:00Z", replies=10, run="r2"),
    ]
    line = digest.episodes_line(records, *WEEK, UTC)
    assert line.startswith("🔁 возвраты работы: 3 · 6,7 на 100 твоих ответов (2 из 30) · в 1 из 2 сессий")


def test_the_run_counts_the_answers_the_model_was_shown_not_the_ones_left_out(world):
    cfg = replace(world.cfg, segment_max_chars=700, segment_max_parts=1)
    world.add("a", _talk("2026-11-09", 14))
    model = Model([])
    out = run(cfg, world.state, now=NOW, llm=model)
    shown = len(reflect.shown_turns(model.reflect[0]))
    assert 0 < shown < 14 and out["replies"] == shown == out["verdicts"][0]["replies"]


# -- 3. the settings file keeps its mode and its link -----------------------------------


def test_the_hook_goes_into_a_linked_settings_file_without_replacing_the_link(world):
    real = world.cfg.claude_settings.parent / "dotfiles" / "settings.json"
    real.parent.mkdir()
    real.write_text(json.dumps({"autoDreamEnabled": False}), encoding="utf-8")
    os.chmod(real, 0o644)
    world.cfg.claude_settings.unlink()
    world.cfg.claude_settings.symlink_to(real)
    assert guard.install_hook(world.cfg) is True
    assert world.cfg.claude_settings.is_symlink()
    assert "PreToolUse" in json.loads(real.read_text(encoding="utf-8"))["hooks"]
    assert real.stat().st_mode & 0o777 == 0o644
    assert guard.uninstall_hook(world.cfg) is True
    assert world.cfg.claude_settings.is_symlink() and real.stat().st_mode & 0o777 == 0o644


# -- 4. a heredoc's body is no command at night either ----------------------------------


def _failed(command):
    return [call("t1", "Bash", {"command": command}), _result("t1", "Exit code 1\nboom", error=True)]


def test_a_trigger_that_only_matches_a_heredocs_body_is_refused(tmp_path):
    seg = segment_of(tmp_path, _failed("python3 - <<'PY'\nimport foo_bar\nPY"))
    assert valid_trigger(seg, r"import\s+foo_bar") == ("", "matches no failing command of the segment")
    assert valid_trigger(seg, r"python3\s+-\s+<<")[0] == r"python3\s+-\s+<<"


def test_a_failing_command_whose_heredoc_names_the_trigger_is_no_repeat(world, tmp_path):
    seg = segment_of(tmp_path, _failed("cat > notes.md <<EOF\nзапусти flutter test потом\nEOF"))
    world.state.lessons["a1"] = lesson("a1", status="staged", project_root=str(world.root),
                                       trigger=r"\bflutter\s+test\b")
    assert trigger_hits(world.state, world.cfg, seg, world.root, NOW_ISO) == []


# -- 5. a rule's project is stored the way sessions name it -----------------------------


def test_a_rules_project_loses_its_trailing_slash_and_a_relative_one_is_made_absolute(cli, tmp_path, monkeypatch):
    assert main(["guard", "add", "fmt", r"\bdart\s+format\b", "--text", "t", "--project", PROJECT + "/"]) == 0
    monkeypatch.chdir(tmp_path)
    assert main(["guard", "add", "here", r"\bdart\s+format\b", "--text", "t", "--project", "."]) == 0
    first, second = load_state(cli.state_file).user_guards
    assert first["project"] == PROJECT
    assert second["project"] == os.path.abspath(str(tmp_path)) and not second["project"].endswith("/")
    rules = guard.rules(load_state(cli.state_file))
    assert rules[0]["slug"] == "-Users-u-git-app"


# -- 6. a correction is put into a group once -------------------------------------------


def test_corrections_a_group_used_are_not_offered_again(week):
    _step(week, Weekly([CLUSTER]))
    again = Weekly()
    assert _step(week, again) == [] and again.prompts == []  # one correction is left: nothing to group
    assert len(week.state.clustered) == 3


def test_a_group_that_names_a_lesson_the_user_took_over_writes_nothing_and_is_not_offered_again(week):
    week.state.lessons["own00001"] = lesson("own00001", status="user_owned", project_root=str(week.project),
                                            memory_dir=str(week.memory), cue="правка панели готова")
    model = Weekly([{"members": ["e1", "e2", "e3"], "lesson_id": "own00001"}])
    assert _step(week, model) == []
    assert "own00001 [user_owned]" in model.prompts[0]
    assert list(week.state.lessons) == ["own00001"] and len(week.state.clustered) == 3
    assert _step(week, Weekly()) == []


def test_marks_of_used_corrections_older_than_the_window_are_dropped(week):
    week.state.clustered = {"/p|s0|1": "2026-09-01T00:00:00Z", "/p|s0|2": "2026-11-15T00:00:00Z"}
    _step(week, Weekly([]))
    assert list(week.state.clustered) == ["/p|s0|2"]


# -- 7. the baseline's cache outlives a change of the days ------------------------------


def test_a_baseline_over_other_days_asks_again_only_about_what_changed(world):
    world.add("a", _talk("2026-10-21", 3) + _talk("2026-11-09", 2))
    first = _baseline(world, Counting(), until=date(2026, 11, 8))
    assert (first["calls"], first["replies"], first["episodes"]) == (1, 3, 3)
    again = Counting()
    second = _baseline(world, again, until=date(2026, 11, 9))
    assert again.prompts == [] and (second["calls"], second["replies"], second["episodes"]) == (0, 5, 5)


# -- 8. a hook that stopped being fed is an error the reports show ----------------------


def test_rules_of_the_hook_that_could_not_be_written_are_an_error_of_the_night(world, monkeypatch):
    world.add("a", [said("Готово."), user("релоад сделал но стили не поменялись")])

    def refuse(cfg, state):
        raise OSError("read-only file system")
    monkeypatch.setattr(guard, "sync", refuse)
    out = run(world.cfg, world.state, now=NOW, llm=Model([]))
    kinds = [e["kind"] for e in world.state.events]
    assert out["errors"] == 1 and "guard_rules_unwritten" in kinds
    for kind in ("guard_rules_unwritten", "guard_fires_unread"):
        assert kind in consolidate.ERROR_KINDS and kind in recent._ERROR_EVENTS


# -- 9. the user's words quoted through their signal are still the user's ---------------


def test_a_quote_of_the_user_under_its_correction_signal_counts_as_the_users_words(tmp_path):
    seg = segment_of(tmp_path, [said("Сделал, запушил в main."), user("нет, я же просил не трогать main"),
                                *_failed("git push")])
    assert [s.kind for s in seg.signals] == ["user_correction", "tool_error"]
    assert check_refs(seg, [{"ref": "1", "quote": "я же просил не трогать main"}]).user_turns == (1,)
    assert check_refs(seg, [{"ref": "2", "quote": "Exit code 1"}]).user_turns == ()


# -- 10. a group says how many corrections it holds -------------------------------------


def test_a_large_group_is_counted_whole_in_the_digest(week):
    week.records = _journal(week.project, ("s1", [_ep(n, f"ошибка {n}", f"слова {n}") for n in range(1, 6)]),
                            ("s2", [_ep(n, f"ошибка {n}", f"слова {n}") for n in range(6, 10)]))
    (item,) = _step(week, Weekly([{**CLUSTER, "members": [f"e{n}" for n in range(1, 10)]}]))
    assert item["count"] == 9 and len(item["said"]) == clusters.SAID_SHOWN
    text = digest.render({"clusters": [item]}, week.state.lessons, "2026-W47")
    assert "9 поправок в 2 сессиях" in text
