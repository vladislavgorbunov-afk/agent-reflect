"""Reversible memory mutations: the weekly run archives memory, journals each change, and can be rolled back."""
from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import os

import pytest

from agent_reflect import cli, consolidate
from agent_reflect.config import Config
from agent_reflect.consolidate import EDIT_SCHEMA, PLAN_SCHEMA
from agent_reflect.ledger import Lesson, State, load_state, run_lock, save_state
from agent_reflect.memory import add_index_line, lesson_filename, remove_lesson, write_lesson
from agent_reflect.projects import memory_dir_for

NOW = datetime(2026, 10, 25, 4, 30, 0, tzinfo=timezone.utc)
NOW_ISO = "2026-10-25T04:30:00Z"
RUN_ID = "20261025T043000"
USER_FILE = "feedback_one_branch.md"
USER_LINE = f"- [{USER_FILE}]({USER_FILE}) — одна задача = одна ветка"
USER_ENTRY = "---\nname: one-branch\ndescription: одна задача — одна ветка\n---\n\nОдна задача — одна ветка.\n"
SECRET = "ghp_A1b2C3d4E5f6G7h8I9j0K1l2M3n4O5p6Q7r8"
SKILL = "---\nname: alpha\ndescription: Does alpha things\n---\n\n# alpha\n\nUse the old rule here.\n"
EMPTY_PLAN = {"promotions": [], "merges": [], "removals": [], "user_entry_suggestions": []}


class FakeLlm:
    def __init__(self, plan: dict, edit: dict | None = None):
        self.plan, self.edit = plan, edit

    def __call__(self, cfg, model, prompt, schema, *, effort):
        assert schema in (PLAN_SCHEMA, EDIT_SCHEMA)
        return self.plan if schema is PLAN_SCHEMA else self.edit


class NoGit:
    def __getattr__(self, name):
        raise AssertionError(f"git.{name} must not be called")


@pytest.fixture
def env(tmp_path, monkeypatch):
    cfg = replace(Config.under(tmp_path), repos={})
    monkeypatch.setattr(Config, "default", classmethod(lambda cls: cfg))
    cfg.claude_settings.parent.mkdir(parents=True)
    cfg.claude_settings.write_text(json.dumps({"autoDreamEnabled": False}))
    skill = cfg.personal_skills_dir / "alpha" / "SKILL.md"
    skill.parent.mkdir(parents=True)
    skill.write_text(SKILL, encoding="utf-8")
    project = tmp_path / "home" / "git" / "proj"
    memory = memory_dir_for(cfg.projects_dir, project)
    memory.mkdir(parents=True)
    (memory / "MEMORY.md").write_text(f"# Memory Index\n\n{USER_LINE}\n", encoding="utf-8")
    (memory / USER_FILE).write_text(USER_ENTRY, encoding="utf-8")
    state = State(installed_at="2026-10-01T00:00:00Z")
    return SimpleNamespace(cfg=cfg, state=state, project=project, memory=memory, skill=skill, tmp=tmp_path)


def _lesson(env, lid: str, *, status="live", confidence="high", sessions=("s1", "s2")) -> Lesson:
    item = Lesson(
        id=lid, project_root=str(env.project), memory_dir=str(env.memory), file=lesson_filename(lid),
        title=f"Урок {lid}", hook=f"крючок {lid}", cue=f"ситуация {lid}", lesson=f"запусти `make {lid}`",
        why="потому что", confidence=confidence, repeats=len(sessions),
        evidence=[{"session": s, "ts": "2026-10-20T00:00:00Z", "text": f"[1] {s}"} for s in sessions],
        status=status, sha256="", created_at="2026-10-02T00:00:00Z", updated_at="2026-10-02T00:00:00Z",
    )
    item.sha256 = write_lesson(env.memory, item)
    if status == "live":
        add_index_line(env.memory, item.file, item.hook)
    env.state.lessons[lid] = item
    return item


def _fill(env, count: int) -> None:
    """Staged medium lessons nobody touches, so the removal cap leaves room for the plan."""
    for i in range(count):
        _lesson(env, f"ffff000{i}", status="staged", confidence="medium", sessions=("s9",))


def _run(env, plan: dict, edit: dict | None = None) -> dict:
    return consolidate.run(env.cfg, env.state, now=NOW, llm=FakeLlm(plan, edit), git=NoGit())


def _archive(env) -> Path:
    return env.cfg.backups_dir / "memory" / RUN_ID / env.memory.parent.name


def _journal(env) -> list[dict]:
    path = env.cfg.state_dir / "journal.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _index(env) -> str:
    return (env.memory / "MEMORY.md").read_text(encoding="utf-8")


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# -- archive ------------------------------------------------------------------------

def test_archive_is_written_before_memory_changes(env, monkeypatch):
    gone = _lesson(env, "aaaa0001")
    staged = _lesson(env, "aaaa0002", status="staged", confidence="medium", sessions=("s1",))
    _fill(env, 2)
    original = (env.memory / gone.file).read_bytes()
    seen = []
    real_remove = consolidate.remove_lesson

    def remove(memory_dir, lesson):
        seen.append((_archive(env) / lesson.file).read_bytes())  # archived before the file goes
        return real_remove(memory_dir, lesson)

    monkeypatch.setattr(consolidate, "remove_lesson", remove)
    _run(env, {**EMPTY_PLAN, "removals": [{"lesson_id": gone.id, "reason": "устарел"}]})

    assert seen == [original] and not (env.memory / gone.file).exists()
    folder = _archive(env)
    assert sorted(p.name for p in folder.iterdir()) == sorted(
        [gone.file, staged.file, "auto_ffff0000.md", "auto_ffff0001.md", "MEMORY.auto.md", "manifest.json"])
    assert (folder / staged.file).read_bytes() == (env.memory / staged.file).read_bytes()
    assert (folder / "MEMORY.auto.md").read_text(encoding="utf-8") == (
        f"- [{gone.file}]({gone.file}) — {gone.hook}\n")  # only our lines, not the user's
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["run_id"] == RUN_ID and manifest["memory_dir"] == str(env.memory)
    assert manifest["project_root"] == str(env.project)
    assert {f["name"]: f["sha256"] for f in manifest["files"]}[gone.file] == _sha(original)
    assert manifest["lessons"][gone.id]["status"] == "live"
    assert manifest["lessons"][staged.id]["status"] == "staged"
    assert USER_FILE not in json.dumps(manifest)


def test_run_that_changes_no_memory_keeps_no_archive(env):
    _lesson(env, "aaaa0001")
    summary = _run(env, EMPTY_PLAN)
    assert summary["errors"] == []
    assert not (env.cfg.backups_dir / "memory" / RUN_ID).exists()


def test_memory_archives_are_pruned_to_the_four_newest(env):
    root = env.cfg.backups_dir / "memory"
    old = ["20260920T043000", "20260927T043000", "20261004T043000", "20261011T043000", "20261018T043000"]
    for run_id in old:
        (root / run_id / "slug").mkdir(parents=True)
        (root / run_id / "slug" / "manifest.json").write_text("{}")
    skill_backup = env.cfg.backups_dir / "20260920T043000"
    skill_backup.mkdir()  # skill backups are not memory archives and stay
    gone = _lesson(env, "aaaa0001")
    _fill(env, 3)
    _run(env, {**EMPTY_PLAN, "removals": [{"lesson_id": gone.id, "reason": "r"}]})
    assert sorted(p.name for p in root.iterdir()) == [*old[2:], RUN_ID]
    assert skill_backup.is_dir()


def test_archive_failure_aborts_before_any_change(env):
    gone = _lesson(env, "aaaa0001")
    _fill(env, 3)
    env.cfg.backups_dir.mkdir(parents=True)
    (env.cfg.backups_dir / "memory").write_text("not a directory")
    summary = _run(env, {**EMPTY_PLAN, "removals": [{"lesson_id": gone.id, "reason": "r"}]})
    assert set(summary) == {"aborted"}
    assert gone.status == "live" and (env.memory / gone.file).exists()


# -- journal ------------------------------------------------------------------------

def test_every_memory_mutation_is_journaled_append_only(env):
    journal = env.cfg.state_dir / "journal.jsonl"
    journal.parent.mkdir(parents=True)
    earlier = json.dumps({"ts": "2026-10-18T04:30:00Z", "run_id": "earlier", "kind": "remove"}) + "\n"
    journal.write_text(earlier, encoding="utf-8")

    keep = _lesson(env, "aaaa0001", status="staged", confidence="medium", sessions=("s1",))
    absorbed = _lesson(env, "aaaa0002", status="staged", confidence="medium", sessions=("s2",))
    removed = _lesson(env, "aaaa0003")
    promoted = _lesson(env, "aaaa0004")
    rejected = _lesson(env, "aaaa0005")
    (env.memory / rejected.file).unlink()  # the user deleted it
    waiting = _lesson(env, "aaaa0006", status="staged", confidence="high")  # rebalance puts it live
    _fill(env, 6)
    plan = {**EMPTY_PLAN,
            "merges": [{"keep": keep.id, "absorb": [absorbed.id], "reason": f"одно и то же {SECRET}"}],
            "removals": [{"lesson_id": removed.id, "reason": "устарел"}],
            "promotions": [{"target": "personal:alpha", "lesson_ids": [promoted.id], "user_rule_files": [],
                            "reason": "r"}]}
    edit = {"skip": False, "reason": "", "old_string": "Use the old rule here.",
            "new_string": "Use the old rule here.\n- Run `make` first.", "summary": "Add a rule",
            "why_en": "Agents kept skipping make."}
    _run(env, plan, edit)

    text = journal.read_text(encoding="utf-8")
    assert text.startswith(earlier)
    lines = _journal(env)[1:]
    assert {(e["kind"], e["lesson_id"], e["previous_status"]) for e in lines} == {
        ("reject", rejected.id, "live"),
        ("promote", waiting.id, "staged"),
        ("merge", keep.id, "staged"),
        ("merge", absorbed.id, "staged"),
        ("remove", removed.id, "live"),
        ("retire", promoted.id, "live"),
    }
    for entry in lines:
        assert entry["ts"] == NOW_ISO and entry["run_id"] == RUN_ID
        assert entry["project_root"] == str(env.project)
    assert SECRET not in text and "[REDACTED]" in text


def test_rebalance_only_run_is_journaled_and_keeps_no_archive(env):
    lesson = _lesson(env, "aaaa0001", confidence="low")
    env.cfg = replace(env.cfg, index_line_limit=3)  # header, blank, user line: no room for ours
    _run(env, EMPTY_PLAN)
    assert lesson.status == "staged"
    [entry] = _journal(env)
    assert (entry["kind"], entry["lesson_id"], entry["previous_status"]) == ("demote", lesson.id, "live")
    assert not (env.cfg.backups_dir / "memory" / RUN_ID).exists()  # rollback cannot undo index-only moves


def test_unreadable_auto_file_is_skipped_and_the_run_goes_on(env):
    gone = _lesson(env, "aaaa0001")
    _fill(env, 3)
    odd = env.memory / "auto_zzzz9999.md"
    odd.write_text("x", encoding="utf-8")
    os.chmod(odd, 0)
    try:
        summary = _run(env, {**EMPTY_PLAN, "removals": [{"lesson_id": gone.id, "reason": "r"}]})
    finally:
        os.chmod(odd, 0o644)
    assert "aborted" not in summary and gone.status == "removed"
    assert (_archive(env) / gone.file).exists() and not (_archive(env) / odd.name).exists()
    [event] = [e for e in summary["errors"] if e["kind"] == "archive_file_skipped"]
    assert event["file"] == str(odd)


# -- rollback -----------------------------------------------------------------------

def _removed_by_a_run(env) -> tuple[Lesson, bytes]:
    gone = _lesson(env, "aaaa0001")
    _fill(env, 3)
    original = (env.memory / gone.file).read_bytes()
    _run(env, {**EMPTY_PLAN, "removals": [{"lesson_id": gone.id, "reason": "r"}]})
    assert gone.status == "removed" and not (env.memory / gone.file).exists()
    save_state(env.state, env.cfg.state_file)
    return gone, original


def test_memory_rollback_restores_a_removed_lesson_and_leaves_user_lines(env, capsys):
    other = _lesson(env, "bbbb0001")
    gone, original = _removed_by_a_run(env)
    edited_line = f"- [{other.file}]({other.file}) — мой собственный текст"
    index = _index(env).replace(f"- [{other.file}]({other.file}) — {other.hook}", edited_line)
    index += "- [feedback_new.md](feedback_new.md) — новая запись пользователя\n"
    (env.memory / "MEMORY.md").write_text(index, encoding="utf-8")
    user_file_before = (env.memory / USER_FILE).read_bytes()

    assert cli.main(["rollback", "--memory", RUN_ID]) == 0

    assert (env.memory / gone.file).read_bytes() == original
    line = f"- [{gone.file}]({gone.file}) — {gone.hook}"
    assert _index(env) == index + line + "\n"  # appended; nothing else rewritten
    assert (env.memory / USER_FILE).read_bytes() == user_file_before
    restored = load_state(env.cfg.state_file).lessons[gone.id]
    assert restored.status == "live" and restored.sha256 == _sha(original)
    assert restored.live_since is not None and restored.live_since > gone.created_at  # its line went in again now
    out = capsys.readouterr().out
    first_write = out.index(str(env.memory / gone.file))
    assert RUN_ID in out[:first_write] and line in out


def test_memory_rollback_never_overwrites_a_file_changed_after_the_run(env, capsys):
    keep = _lesson(env, "aaaa0001", status="staged", confidence="medium", sessions=("s1",))
    absorbed = _lesson(env, "aaaa0002", status="staged", confidence="medium", sessions=("s2",))
    _fill(env, 6)
    _run(env, {**EMPTY_PLAN, "merges": [{"keep": keep.id, "absorb": [absorbed.id], "reason": "r"}]})
    save_state(env.state, env.cfg.state_file)
    mine = (env.memory / keep.file).read_text(encoding="utf-8") + "\nМоя правка.\n"
    (env.memory / keep.file).write_text(mine, encoding="utf-8")

    assert cli.main(["rollback", "--memory", RUN_ID]) == 0

    assert (env.memory / keep.file).read_text(encoding="utf-8") == mine
    assert (env.memory / absorbed.file).exists()  # the absorbed one comes back
    state = load_state(env.cfg.state_file)
    assert state.lessons[absorbed.id].status == "staged"
    assert state.lessons[keep.id].repeats == 2  # its record stays with the file left in place
    assert f"skip {env.memory / keep.file}" in capsys.readouterr().out


def test_memory_rollback_ignores_non_auto_files_in_the_archive(env):
    _removed_by_a_run(env)
    folder = _archive(env)
    (folder / USER_FILE).write_text("подмена", encoding="utf-8")
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    manifest["files"].append({"name": USER_FILE, "sha256": "x"})
    manifest["files"].append({"name": "../escape.md", "sha256": "x"})
    (folder / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (env.memory / USER_FILE).unlink()
    assert cli.main(["rollback", "--memory", RUN_ID]) == 1  # a tampered manifest is a refusal
    assert not (env.memory / USER_FILE).exists()
    assert not (env.memory.parent / "escape.md").exists()


def test_memory_rollback_respects_the_run_lock(env, capsys):
    gone, _ = _removed_by_a_run(env)
    with run_lock(env.cfg.lock_file) as held:
        assert held
        assert cli.main(["rollback", "--memory", RUN_ID]) == 0
    assert not (env.memory / gone.file).exists()
    assert load_state(env.cfg.state_file).lessons[gone.id].status == "removed"
    assert "already running" in capsys.readouterr().out


def test_memory_rollback_of_an_unknown_run_restores_nothing(env, capsys):
    assert cli.main(["rollback", "--memory", "20990101T000000"]) == 1
    assert "nothing to restore" in capsys.readouterr().out


def test_rollback_lists_skill_and_memory_run_ids(env, capsys):
    _removed_by_a_run(env)
    (env.cfg.backups_dir / "20261018T043000" / "personal__alpha").mkdir(parents=True)
    assert cli.main(["rollback", "--list"]) == 0
    assert capsys.readouterr().out.splitlines() == ["20261018T043000", f"--memory {RUN_ID}"]
    assert cli.main(["rollback"]) == 2
    captured = capsys.readouterr()
    assert captured.out.splitlines() == ["20261018T043000", f"--memory {RUN_ID}"]
    assert "--memory RUN_ID" in captured.err


def _merged_by_a_run(env) -> tuple[Lesson, Lesson, bytes]:
    keep = _lesson(env, "aaaa0001", status="staged", confidence="medium", sessions=("s1",))
    absorbed = _lesson(env, "aaaa0002", status="staged", confidence="medium", sessions=("s2",))
    _fill(env, 6)
    before = (env.memory / keep.file).read_bytes()
    _run(env, {**EMPTY_PLAN, "merges": [{"keep": keep.id, "absorb": [absorbed.id], "reason": "r"}]})
    assert env.state.lessons[keep.id].status == "live" and env.state.lessons[keep.id].repeats == 2
    save_state(env.state, env.cfg.state_file)
    return keep, absorbed, before


def test_memory_rollback_puts_a_merged_lesson_back_as_it_was(env):
    keep, absorbed, before = _merged_by_a_run(env)
    index = _index(env)
    assert cli.main(["rollback", "--memory", RUN_ID]) == 0
    assert (env.memory / keep.file).read_bytes() == before
    state = load_state(env.cfg.state_file)
    restored = state.lessons[keep.id]
    assert restored.repeats == 1 and restored.sha256 == _sha(before)
    assert restored.status == "live"  # the run gave it a line; rollback never takes a line away
    assert _index(env) == index
    assert state.lessons[absorbed.id].status == "staged" and (env.memory / absorbed.file).exists()


def test_memory_rollback_leaves_a_lesson_the_user_rejected_since(env, capsys):
    keep, _, _ = _merged_by_a_run(env)
    state = load_state(env.cfg.state_file)
    state.lessons[keep.id].status = "rejected"  # the user deleted it after the run
    save_state(state, env.cfg.state_file)
    (env.memory / keep.file).unlink()
    assert cli.main(["rollback", "--memory", RUN_ID]) == 0
    assert not (env.memory / keep.file).exists()
    assert load_state(env.cfg.state_file).lessons[keep.id].status == "rejected"
    assert "the user's decision stands" in capsys.readouterr().out



def test_memory_rollback_never_brings_back_a_lesson_the_user_deleted_after_the_run(env, capsys):
    untouched = _lesson(env, "bbbb0001")  # the run never touches it
    only_file = _lesson(env, "bbbb0002")
    gone, original = _removed_by_a_run(env)
    (env.memory / untouched.file).unlink()  # the user deletes the lesson and its line
    index = "\n".join(l for l in _index(env).split("\n") if untouched.file not in l)
    (env.memory / "MEMORY.md").write_text(index, encoding="utf-8")
    (env.memory / only_file.file).unlink()  # and only the file of another one

    assert cli.main(["rollback", "--memory", RUN_ID]) == 0

    for lesson in (untouched, only_file):
        assert not (env.memory / lesson.file).exists()
        assert lesson.file not in _index(env)
    state = load_state(env.cfg.state_file)
    assert state.lessons[untouched.id].status == "rejected"
    assert state.lessons[only_file.id].status == "rejected"
    assert (env.memory / gone.file).read_bytes() == original  # what the run did is still undone
    assert state.lessons[gone.id].status == "live"
    out = capsys.readouterr().out
    assert f"{untouched.id}: live → rejected" in out


def test_memory_rollback_does_not_re_add_a_line_the_user_deleted_from_a_merged_lesson(env):
    keep, absorbed, _ = _merged_by_a_run(env)
    index = "\n".join(l for l in _index(env).split("\n") if keep.file not in l)
    (env.memory / "MEMORY.md").write_text(index, encoding="utf-8")
    assert cli.main(["rollback", "--memory", RUN_ID]) == 0
    assert keep.file not in _index(env)
    state = load_state(env.cfg.state_file)
    assert state.lessons[keep.id].status == "rejected"
    assert state.lessons[absorbed.id].status == "staged" and (env.memory / absorbed.file).exists()


def test_memory_rollback_leaves_a_lesson_the_run_did_not_change(env):
    later = _lesson(env, "bbbb0001", status="staged", confidence="medium", sessions=("s1",))
    gone, _ = _removed_by_a_run(env)
    # a reflect night after the run repeats the untouched lesson: its file and record move on
    state = load_state(env.cfg.state_file)
    record = state.lessons[later.id]
    record.repeats = 2
    record.evidence = [*record.evidence, {"session": "s5", "ts": "2026-10-26T00:00:00Z", "text": "e"}]
    record.sha256 = write_lesson(env.memory, record)
    save_state(state, env.cfg.state_file)
    after_repeat = (env.memory / later.file).read_bytes()
    assert cli.main(["rollback", "--memory", RUN_ID]) == 0
    assert (env.memory / later.file).read_bytes() == after_repeat
    assert load_state(env.cfg.state_file).lessons[later.id].repeats == 2
    assert (env.memory / gone.file).exists()


def test_memory_rollback_without_a_journal_restores_only_what_state_shows_the_run_did(env):
    later = _lesson(env, "bbbb0001", status="staged", confidence="medium", sessions=("s1",))
    gone, original = _removed_by_a_run(env)
    (env.cfg.state_dir / "journal.jsonl").unlink()
    before = (env.memory / later.file).read_bytes()
    assert cli.main(["rollback", "--memory", RUN_ID]) == 0
    assert (env.memory / gone.file).read_bytes() == original
    assert (env.memory / later.file).read_bytes() == before


def _promoted_by_a_run(env) -> Lesson:
    lesson = _lesson(env, "aaaa0004")
    _fill(env, 3)
    edit = {"skip": False, "reason": "", "old_string": "Use the old rule here.",
            "new_string": "Use the old rule here.\n- Run `make` first.", "summary": "Add a rule",
            "why_en": "Agents kept skipping make."}
    plan = {**EMPTY_PLAN, "promotions": [{"target": "personal:alpha", "lesson_ids": [lesson.id],
                                          "user_rule_files": [], "reason": "r"}]}
    summary = _run(env, plan, edit)
    assert [p["target"] for p in summary["promoted"]] == ["personal:alpha"]
    save_state(env.state, env.cfg.state_file)
    return lesson


def test_rolled_back_promotion_is_held_out_of_promotion_while_the_skill_edit_stays(env, capsys):
    lesson = _promoted_by_a_run(env)
    assert cli.main(["rollback", "--memory", RUN_ID]) == 0
    out = capsys.readouterr().out
    assert f"skill edit stays — run `agent-reflect rollback {RUN_ID}` to undo it" in out
    assert "Run `make` first." in env.skill.read_text(encoding="utf-8")
    state = load_state(env.cfg.state_file)
    restored = state.lessons[lesson.id]
    assert restored.status == "staged" and restored.rollback_hold == RUN_ID
    assert (env.memory / lesson.file).exists() and lesson.file not in _index(env)
    assert consolidate.eligible(state, NOW)[0] == []

    assert cli.main(["rollback", RUN_ID]) == 0  # the skill edit goes too: the hold is lifted
    assert env.skill.read_text(encoding="utf-8") == SKILL
    state = load_state(env.cfg.state_file)
    assert state.lessons[lesson.id].rollback_hold is None
    assert [l.id for l in consolidate.eligible(state, NOW)[0]] == [lesson.id]


def test_memory_rollback_refuses_a_manifest_that_points_elsewhere(env, capsys):
    gone, _ = _removed_by_a_run(env)
    elsewhere = env.tmp / "elsewhere"
    elsewhere.mkdir()
    manifest_path = _archive(env) / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["memory_dir"] = str(elsewhere)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    assert cli.main(["rollback", "--memory", RUN_ID]) == 1
    assert list(elsewhere.iterdir()) == [] and not (env.memory / gone.file).exists()
    assert "refused" in capsys.readouterr().out


def test_memory_rollback_without_room_in_the_index_leaves_the_lesson_staged(env, monkeypatch, capsys):
    gone, original = _removed_by_a_run(env)
    full = replace(env.cfg, auto_index_cap=0)
    monkeypatch.setattr(Config, "default", classmethod(lambda cls: full))
    index = _index(env)
    assert cli.main(["rollback", "--memory", RUN_ID]) == 0
    assert (env.memory / gone.file).read_bytes() == original
    assert _index(env) == index
    assert load_state(env.cfg.state_file).lessons[gone.id].status == "staged"
    assert "no room" in capsys.readouterr().out



# -- fix round 2 -------------------------------------------------------------------

ROLLBACK_AT = datetime(2026, 10, 27, 9, 0, 0, tzinfo=timezone.utc)
ROLLBACK_RUN = "20261027T090000"
OLD_RUN = "20261018T043000"


def _append_journal(env, entry: dict) -> None:
    with (env.cfg.state_dir / "journal.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry) + "\n")


def _personal_and_repo_promoted_by_a_run(env) -> tuple[Lesson, Lesson]:
    """One run: A goes into personal:alpha, B into a draft PR (its retire as promote_repo leaves it)."""
    repo = _lesson(env, "bbbb0004")
    personal = _promoted_by_a_run(env)
    record = env.state.lessons[repo.id]
    remove_lesson(env.memory, record)
    record.status, record.updated_at = "promoted", NOW_ISO
    save_state(env.state, env.cfg.state_file)
    _append_journal(env, {"ts": NOW_ISO, "run_id": RUN_ID, "kind": "retire", "lesson_id": repo.id,
                          "project_root": str(env.project), "previous_status": "live", "status": "promoted",
                          "file": repo.file, "target": "repo:webapp:check-logic", "pr_url": "https://x/pr/9"})
    return personal, repo


def test_skill_rollback_lifts_only_the_holds_of_the_skill_edits_it_undid(env, capsys):
    personal, repo = _personal_and_repo_promoted_by_a_run(env)
    assert cli.main(["rollback", "--memory", RUN_ID]) == 0
    out = capsys.readouterr().out
    assert "draft PR https://x/pr/9" in out and "until you remove the lesson by hand" in out
    assert "closing that PR" not in out
    state = load_state(env.cfg.state_file)
    assert state.lessons[personal.id].rollback_hold == RUN_ID and state.lessons[repo.id].rollback_hold == RUN_ID

    assert cli.main(["rollback", RUN_ID]) == 0  # undoes the personal:alpha edit only
    state = load_state(env.cfg.state_file)
    assert state.lessons[personal.id].rollback_hold is None
    assert state.lessons[repo.id].rollback_hold == RUN_ID  # its edit still stands in the PR
    assert [l.id for l in consolidate.eligible(state, NOW)[0]] == [personal.id]


def test_no_hold_when_the_skill_edit_was_already_rolled_back(env, capsys):
    lesson = _promoted_by_a_run(env)
    assert cli.main(["rollback", RUN_ID]) == 0
    assert env.skill.read_text(encoding="utf-8") == SKILL
    capsys.readouterr()
    assert cli.main(["rollback", "--memory", RUN_ID]) == 0
    out = capsys.readouterr().out
    state = load_state(env.cfg.state_file)
    assert state.lessons[lesson.id].rollback_hold is None and state.lessons[lesson.id].status == "staged"
    assert "skill edit stays" not in out and "already undone" in out
    assert [l.id for l in consolidate.eligible(state, NOW)[0]] == [lesson.id]


def test_restore_takes_the_hold_from_state_not_from_the_archive(env):
    held = _lesson(env, "aaaa0001")
    held.rollback_hold = OLD_RUN  # held by an earlier memory rollback
    _fill(env, 3)
    _run(env, {**EMPTY_PLAN, "removals": [{"lesson_id": held.id, "reason": "r"}]})
    assert held.status == "removed"
    held.rollback_hold = None  # `rollback OLD_RUN` lifted it since
    save_state(env.state, env.cfg.state_file)
    assert cli.main(["rollback", "--memory", RUN_ID]) == 0
    assert load_state(env.cfg.state_file).lessons[held.id].rollback_hold is None


def test_no_journal_fallback_never_reverts_a_later_repeat(env):
    later = _lesson(env, "bbbb0001", status="staged", confidence="medium", sessions=("s1",))
    gone, original = _removed_by_a_run(env)
    (env.cfg.state_dir / "journal.jsonl").unlink()
    state = load_state(env.cfg.state_file)
    record = state.lessons[later.id]
    record.repeats, record.updated_at = 2, "2026-10-26T00:00:00Z"
    record.evidence = [*record.evidence, {"session": "s5", "ts": "2026-10-26T00:00:00Z", "text": "e"}]
    record.sha256 = write_lesson(env.memory, record)
    save_state(state, env.cfg.state_file)
    assert cli.main(["rollback", "--memory", RUN_ID]) == 0
    state = load_state(env.cfg.state_file)
    assert state.lessons[later.id].repeats == 2
    assert (env.memory / gone.file).read_bytes() == original


def test_a_removal_missing_from_a_partial_journal_is_reported(env, capsys):
    one = _lesson(env, "aaaa0001")
    two = _lesson(env, "aaaa0002")
    _fill(env, 6)
    _run(env, {**EMPTY_PLAN, "removals": [{"lesson_id": one.id, "reason": "r"},
                                          {"lesson_id": two.id, "reason": "r"}]})
    save_state(env.state, env.cfg.state_file)
    path = env.cfg.state_dir / "journal.jsonl"
    kept = [line for line in path.read_text(encoding="utf-8").splitlines() if two.id not in line]
    path.write_text("".join(line + "\n" for line in kept), encoding="utf-8")  # its append failed
    assert cli.main(["rollback", "--memory", RUN_ID]) == 0
    out = capsys.readouterr().out
    assert (env.memory / one.file).exists() and not (env.memory / two.file).exists()
    assert f"skip {env.memory / two.file}: not in the journal of {RUN_ID}" in out


def test_rejections_found_by_rollback_are_printed_and_journaled(env, monkeypatch, capsys):
    monkeypatch.setattr(cli, "_now", lambda: ROLLBACK_AT)
    by_file = _lesson(env, "bbbb0001")
    by_line = _lesson(env, "bbbb0002")
    _removed_by_a_run(env)
    (env.memory / by_file.file).unlink()  # the user deleted one file and another line
    index = "\n".join(l for l in _index(env).split("\n") if by_line.file not in l)
    (env.memory / "MEMORY.md").write_text(index, encoding="utf-8")
    before = len(_journal(env))

    assert cli.main(["rollback", "--memory", RUN_ID]) == 0

    out = capsys.readouterr().out
    assert f"user    {by_file.id}: live → rejected" in out and "its index line is removed" in out
    assert f"user    {by_line.id}: live → rejected" in out and f"{env.memory / by_line.file} is deleted" in out
    assert not (env.memory / by_line.file).exists()
    added = _journal(env)[before:]
    assert {(e["kind"], e["lesson_id"], e["previous_status"]) for e in added} == {
        ("reject", by_file.id, "live"), ("reject", by_line.id, "live")}
    assert {e["run_id"] for e in added} == {ROLLBACK_RUN}
