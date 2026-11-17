from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent_reflect import memory_reads
from agent_reflect.config import Config
from agent_reflect.projects import sanitize

from conftest import write_jsonl

NOW = datetime(2026, 10, 25, 4, 30, tzinfo=timezone.utc)
TS = "2026-10-20T10:00:00.000Z"
LATER = "2026-10-21T09:00:00.000Z"
OLD = "2026-09-10T10:00:00.000Z"  # more than 30 days before NOW


def _write(path: Path, text: str = "x\n") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def call(name: str, inp, ts: str = TS) -> dict:
    return {"type": "assistant", "timestamp": ts, "sessionId": "s1",
            "message": {"role": "assistant", "content": [{"type": "tool_use", "id": "t1", "name": name, "input": inp}]}}


def read(path: Path | str, ts: str = TS) -> dict:
    return call("Read", {"file_path": str(path)}, ts)


def bash(command: str, ts: str = TS) -> dict:
    return call("Bash", {"command": command, "description": "Look"}, ts)


@pytest.fixture
def env(tmp_path):
    cfg = Config.under(tmp_path)
    project = cfg.projects_dir / sanitize(str(cfg.repos["webapp"]))
    memory = project / "memory"
    for name in ("MEMORY.md", "auto_ab12cd34.md", "feedback_a.md", "feedback_b.md", "reference_c.md"):
        _write(memory / name)
    notes = cfg.home / ".claude" / "notes"
    for name in ("CLAUDE.md", "feedback_x.md", "reference_y.md", "project_z.md"):
        _write(notes / name)
    return SimpleNamespace(cfg=cfg, project=project, memory=memory, notes=notes)


def _scan(env) -> dict:
    return memory_reads.scan(env.cfg, now=NOW)


def _files(result: dict, location: str) -> dict[str, dict]:
    [loc] = [loc for loc in result["locations"] if loc["name"] == location]
    return {f["name"]: f for f in loc["files"]}


def _reads(result: dict, location: str) -> dict[str, int]:
    return {name: f["reads"] for name, f in _files(result, location).items()}


def test_a_read_of_a_memory_file_counts_once_per_call_and_keeps_the_latest_time(env):
    write_jsonl(env.project / "s1.jsonl", [read(env.memory / "feedback_a.md", LATER),
                                           read(env.memory / "feedback_a.md", TS)])
    result = _scan(env)
    assert result["days"] == 30
    files = _files(result, "webapp")
    assert files["feedback_a.md"]["reads"] == 2
    assert files["feedback_a.md"]["last_read"] == LATER


def test_bash_and_grep_mentions_count_each_file_once_per_call_tilde_and_absolute_alike(env):
    notes_abs = env.notes / "reference_y.md"
    write_jsonl(env.project / "s1.jsonl", [
        bash(f"cat ~/.claude/notes/feedback_x.md && head ~/.claude/notes/feedback_x.md {notes_abs}"),
        bash('sed -n 1,5p "$HOME/.claude/notes/project_z.md"'),
        call("Grep", {"pattern": "gate", "path": str(env.memory / "reference_c.md")}),
        bash("grep -n gate ~/.claude/projects/" + env.project.name + "/memory/feedback_b.md"),
    ])
    result = _scan(env)
    assert _reads(result, "notes") == {"feedback_x.md": 1, "reference_y.md": 1, "project_z.md": 1}
    assert _reads(result, "webapp") == {"feedback_a.md": 0, "feedback_b.md": 1, "reference_c.md": 1}


def test_a_subagent_transcript_counts(env):
    write_jsonl(env.project / "s1" / "subagents" / "agent-a1.jsonl", [read(env.notes / "project_z.md")])
    files = _files(_scan(env), "notes")
    assert files["project_z.md"]["reads"] == 1 and files["project_z.md"]["last_read"] == TS


def test_indexes_and_auto_lessons_are_never_listed(env):
    write_jsonl(env.project / "s1.jsonl", [
        read(env.memory / "MEMORY.md"), read(env.memory / "auto_ab12cd34.md"), read(env.notes / "CLAUDE.md"),
        bash(f"cat {env.memory / 'auto_ab12cd34.md'} ~/.claude/notes/CLAUDE.md"),
    ])
    result = _scan(env)
    assert set(_files(result, "webapp")) == {"feedback_a.md", "feedback_b.md", "reference_c.md"}
    assert set(_files(result, "notes")) == {"feedback_x.md", "reference_y.md", "project_z.md"}
    assert all(f["reads"] == 0 for loc in result["locations"] for f in loc["files"])


def test_a_transcript_older_than_the_window_does_not_count(env):
    old = env.project / "old.jsonl"
    write_jsonl(old, [read(env.notes / "feedback_x.md")])
    stamp = (NOW - timedelta(days=40)).timestamp()
    os.utime(old, (stamp, stamp))
    assert _reads(_scan(env), "notes")["feedback_x.md"] == 0


def test_an_entry_older_than_the_window_does_not_count_in_a_fresh_transcript(env):
    write_jsonl(env.project / "s1.jsonl", [read(env.notes / "feedback_x.md", OLD),
                                           read(env.notes / "reference_y.md", TS)])
    reads = _reads(_scan(env), "notes")
    assert (reads["feedback_x.md"], reads["reference_y.md"]) == (0, 1)


def test_malformed_lines_are_skipped(env):
    unfinished = json.dumps(read(env.notes / "project_z.md"))[:-5]
    write_jsonl(env.project / "s1.jsonl", [
        ["tool_use", ".md"],
        {"type": "assistant", "message": {"content": "tool_use .md"}},
        {"type": "assistant", "message": {"content": [{"type": "tool_use", "name": "Read", "input": "x.md"}]}},
        {"type": "assistant", "timestamp": 5, "message": [{"type": "tool_use", "name": "Read"}], "x": ".md"},
        read(env.notes / "feedback_x.md"),
    ], tail="{not json tool_use .md\n" + unfinished)
    assert _reads(_scan(env), "notes") == {"feedback_x.md": 1, "reference_y.md": 0, "project_z.md": 0}


def test_other_tools_globs_and_lookalikes_do_not_count(env):
    write_jsonl(env.project / "s1.jsonl", [
        call("Edit", {"file_path": str(env.notes / "feedback_x.md"), "old_string": "a", "new_string": "b"}),
        call("Write", {"file_path": str(env.memory / "feedback_a.md"), "content": "x"}),
        bash("ls ~/.claude/notes/*.md ~/.claude/notes/feedback_*.md"),
        bash("cp ~/.claude/notes/reference_y.md.bak /tmp/ && cat /elsewhere/.claude/notes/project_z.md"),
        read(env.cfg.home / "git" / "notes" / "feedback_x.md"),
    ])
    result = _scan(env)
    assert all(f["reads"] == 0 for loc in result["locations"] for f in loc["files"])


def test_locations_are_the_notes_then_each_project_memory_with_files_on_disk(env):
    other = env.cfg.projects_dir / sanitize(str(env.cfg.home / "git" / "agent-reflect"))
    _write(other / "memory" / "project_r.md")
    _write(env.cfg.projects_dir / sanitize(str(env.cfg.home / "git" / "only-index")) / "memory" / "MEMORY.md")
    (env.cfg.projects_dir / sanitize(str(env.cfg.home / "git" / "no-memory"))).mkdir(parents=True)
    write_jsonl(env.project / "s1.jsonl", [read(env.notes / "gone.md"), read(other / "memory" / "project_r.md")])
    result = _scan(env)
    assert [loc["name"] for loc in result["locations"]] == ["notes", "agent-reflect", "webapp"]
    files = _files(result, "webapp")
    assert list(files) == ["feedback_a.md", "feedback_b.md", "reference_c.md"]  # sorted by name
    assert files["feedback_a.md"] == {"name": "feedback_a.md", "reads": 0, "last_read": None}
    assert "gone.md" not in _files(result, "notes")  # read, but no longer on disk
    assert _reads(result, "agent-reflect") == {"project_r.md": 1}


def test_nothing_on_disk_is_no_location(tmp_path):
    assert memory_reads.scan(Config.under(tmp_path), now=NOW) == {"days": 30, "locations": []}
