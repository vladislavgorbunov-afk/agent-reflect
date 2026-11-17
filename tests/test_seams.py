"""Seam tests on hand-built transcripts shaped like the CLI's real ones (no real content)."""
from __future__ import annotations

import json
import os
import subprocess
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import pytest

from agent_reflect import consolidate, digest, reflect
from agent_reflect.config import Config
from agent_reflect.ledger import State
from agent_reflect.projects import memory_dir_for, sanitize
from agent_reflect.transcript import read_segment

NOW = datetime(2026, 10, 20, 3, 0, 0, tzinfo=timezone.utc)
SESSION = "0f8e2c1a-5b7d-4e3f-9a10-2b3c4d5e6f70"
NEW_HIGH = {"verdict": "NEW", "notes": "n", "title": "Хук блокирует коммит", "hook": "сначала /gate-check",
            "cue": "коммит блокирует хук гейта", "lesson": "Запусти /gate-check, не повторяй коммит.",
            "why": "Хук отказал дважды.", "evidence": "[1] hook error", "confidence": "high"}


def _base(cwd: str, n: int, kind: str) -> dict:
    return {"parentUuid": f"u{n - 1}" if n else None, "isSidechain": False, "userType": "external",
            "cwd": cwd, "sessionId": SESSION, "version": "2.1.304", "gitBranch": "main", "type": kind,
            "uuid": f"u{n}", "timestamp": f"2026-10-19T10:{n:02d}:00.000Z"}


def user_text(cwd: str, n: int, text: str, **extra) -> dict:
    return {**_base(cwd, n, "user"), "message": {"role": "user", "content": text}, **extra}


def tool_use(cwd: str, n: int, tool_id: str, command: str) -> dict:
    return {**_base(cwd, n, "assistant"), "requestId": f"req_{n}",
            "message": {"id": f"msg_{n}", "type": "message", "role": "assistant", "model": "claude-opus-5-5",
                        "content": [{"type": "tool_use", "id": tool_id, "name": "Bash",
                                     "input": {"command": command, "description": "Commit"}}]}}


def hook_block(cwd: str, n: int, tool_id: str) -> dict:
    text = ("PreToolUse:Bash hook error: [/h/hooks/pre-commit-gate-check.sh]: Refusing: this commit needs "
            "/dev-process-toolkit:gate-check in this session")
    return {**_base(cwd, n, "user"), "toolUseResult": "Error: " + text,
            "message": {"role": "user", "content": [{"type": "tool_result", "tool_use_id": tool_id,
                                                     "is_error": True, "content": text}]}}


def assistant_text(cwd: str, n: int, text: str) -> dict:
    return {**_base(cwd, n, "assistant"),
            "message": {"role": "assistant", "content": [{"type": "text", "text": text}]}}


def compact_summary(cwd: str, n: int) -> dict:
    text = ("This session is being continued from a previous conversation that ran out of context. "
            "Summary: the user said stop twice and called the first approach wrong; "
            "стой и не так were quoted from earlier messages.")
    return user_text(cwd, n, text, isCompactSummary=True, isVisibleInTranscriptOnly=True)


def _git_repo(path: Path) -> Path:
    path.mkdir(parents=True)
    subprocess.run(["git", "-C", str(path), "init", "-q"], check=True, capture_output=True)
    return path.resolve()


def _write(path: Path, entries: list[dict], *, append: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab" if append else "wb") as f:
        for e in entries:
            f.write((json.dumps(e, ensure_ascii=False) + "\n").encode("utf-8"))
    t = NOW.timestamp() - 36000
    os.utime(path, (t, t))


class Llm:
    def __init__(self, result: dict):
        self.result, self.prompts = result, []

    def __call__(self, cfg, model, prompt, schema, **kw):
        self.prompts.append(prompt)
        return self.result


@pytest.fixture
def cfg(tmp_path) -> Config:
    c = Config.under(tmp_path)
    c.claude_settings.parent.mkdir(parents=True, exist_ok=True)
    c.claude_settings.write_text(json.dumps({"autoDreamEnabled": False}))
    return c


def test_cwd_drift_mid_session_keeps_the_starting_project(cfg, tmp_path):
    start = _git_repo(tmp_path / "work" / "alpha")
    other = _git_repo(tmp_path / "work" / "beta")
    path = cfg.projects_dir / sanitize(str(start)) / f"{SESSION}.jsonl"
    _write(path, [user_text(str(start), 0, "почини сборку"), assistant_text(str(start), 1, "Смотрю.")])
    state = State(installed_at="2026-10-01T00:00:00Z")
    reflect.run(cfg, state, now=NOW, llm=Llm(NEW_HIGH))
    assert state.lessons == {}
    _write(path, [tool_use(str(other), 2, "toolu_1", "git commit -m fix"), hook_block(str(other), 3, "toolu_1"),
                  assistant_text(str(other), 4, "Запускаю gate-check.")], append=True)
    llm = Llm(NEW_HIGH)
    result = reflect.run(cfg, state, now=NOW, llm=llm)
    assert len(llm.prompts) == 1 and result["outcomes"]["new_live"] == 1
    (lesson,) = state.lessons.values()
    assert lesson.project_root == str(start)
    assert (memory_dir_for(cfg.projects_dir, start) / lesson.file).is_file()
    assert not memory_dir_for(cfg.projects_dir, other).exists()


def test_compaction_summary_is_not_a_correction(cfg, tmp_path):
    root = _git_repo(tmp_path / "work" / "alpha")
    path = cfg.projects_dir / sanitize(str(root)) / f"{SESSION}.jsonl"
    _write(path, [compact_summary(str(root), 0), assistant_text(str(root), 1, "Продолжаю с того места.")])
    seg = read_segment(path, 0)
    assert seg.signals == () and seg.user_messages == ()
    llm = Llm(NEW_HIGH)
    reflect.run(cfg, State(installed_at="2026-10-01T00:00:00Z"), now=NOW, llm=llm)
    assert llm.prompts == []


def test_hook_block_is_a_signal_with_its_command_and_next_step(tmp_path):
    path = tmp_path / "s.jsonl"
    _write(path, [tool_use("/w", 0, "toolu_1", "git commit -m x"), hook_block("/w", 1, "toolu_1"),
                  assistant_text("/w", 2, "Запускаю gate-check.")])
    (signal,) = read_segment(path, 0).signals
    assert signal.kind == "hook_block" and signal.tool == "Bash"
    assert "git commit" in signal.tool_input and "gate-check" in signal.after


def test_origin_human_versus_task_notification(tmp_path):
    path = tmp_path / "s.jsonl"
    _write(path, [
        user_text("/w", 0, "Background task finished: wrong exit code, stop the watcher",
                  origin={"kind": "task-notification"}),
        user_text("/w", 1, "стой, это не тот репозиторий", origin={"kind": "human"}),
    ])
    seg = read_segment(path, 0)
    assert seg.user_messages == ("стой, это не тот репозиторий",)
    assert [s.kind for s in seg.signals] == ["user_correction"]


SKILL = "---\nname: alpha\ndescription: Commits and gates\n---\n\n# alpha\n\nCommit only after the gate.\n"


class NoGit:
    """consolidate must not need git when only a personal skill is edited."""

    def __getattr__(self, name):
        raise AssertionError(f"git.{name} called")


class ConsolidateLlm:
    def __init__(self, lesson_id: str):
        self.lesson_id, self.efforts = lesson_id, []

    def __call__(self, cfg, model, prompt, schema, *, effort):
        self.efforts.append(effort)
        if schema is consolidate.PLAN_SCHEMA:
            return {"promotions": [{"target": "personal:alpha", "lesson_ids": [self.lesson_id],
                                    "user_rule_files": [], "reason": "про коммиты"}],
                    "merges": [], "removals": [], "user_entry_suggestions": []}
        return {"skip": False, "reason": "", "old_string": "Commit only after the gate.",
                "new_string": "Commit only after the gate.\n\n- A blocked commit: run /gate-check, do not retry.",
                "summary": "Run gate-check after a blocked commit", "why_en": "A hook blocked commits twice."}


def test_reflect_lesson_is_promoted_by_consolidate_and_shown_in_the_digest(cfg, tmp_path):
    cfg = replace(cfg, repos={})
    skill = cfg.personal_skills_dir / "alpha" / "SKILL.md"
    skill.parent.mkdir(parents=True)
    skill.write_text(SKILL, encoding="utf-8")
    root = _git_repo(tmp_path / "work" / "alpha")
    path = cfg.projects_dir / sanitize(str(root)) / f"{SESSION}.jsonl"
    _write(path, [tool_use(str(root), 0, "toolu_1", "git commit -m fix"), hook_block(str(root), 1, "toolu_1"),
                  assistant_text(str(root), 2, "Запускаю gate-check.")])
    state = State(installed_at="2026-10-01T00:00:00Z")

    result = reflect.run(cfg, state, now=NOW, llm=Llm(NEW_HIGH))
    assert result["outcomes"]["new_live"] == 1
    (lesson,) = state.lessons.values()
    memory = memory_dir_for(cfg.projects_dir, root)
    assert (memory / lesson.file).is_file() and lesson.file in (memory / "MEMORY.md").read_text()

    # one session is not enough for a skill: the same mistake comes back in another one
    later = datetime(2026, 10, 25, 4, 30, tzinfo=timezone.utc)
    assert consolidate.eligible(state, later)[0] == []
    second = "1a2b3c4d-5e6f-4a7b-8c9d-0e1f2a3b4c5d"
    entries = [tool_use(str(root), 0, "toolu_2", "git commit -m again"), hook_block(str(root), 1, "toolu_2"),
               assistant_text(str(root), 2, "Запускаю gate-check.")]
    _write(path.with_name(f"{second}.jsonl"), [{**e, "sessionId": second} for e in entries])
    repeat = {"verdict": "REPEAT", "notes": "n", "lesson_id": lesson.id, "evidence": "[1] hook error again"}
    assert reflect.run(cfg, state, now=NOW, llm=Llm(repeat))["outcomes"]["repeat"] == 1
    lesson = state.lessons[lesson.id]  # a repeat stores an updated record
    assert consolidate.eligible(state, later)[0] == [lesson]

    llm = ConsolidateLlm(lesson.id)
    summary = consolidate.run(cfg, state, now=later, llm=llm, git=NoGit())
    assert llm.efforts == ["high", "high"]
    assert "- A blocked commit: run /gate-check, do not retry." in skill.read_text(encoding="utf-8")
    assert lesson.status == "promoted" and not (memory / lesson.file).exists()
    assert summary["new_lessons"] == [lesson.id] and summary["errors"] == []

    text = digest.render(summary, state.lessons, digest.week_tag(later))
    assert "Хук блокирует коммит" in text
    assert "`personal:alpha` — Run gate-check after a blocked commit" in text
    assert "  ```diff" in text and "  +- A blocked commit: run /gate-check, do not retry." in text
