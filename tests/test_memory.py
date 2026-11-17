from __future__ import annotations

import hashlib
import json
from dataclasses import replace

from agent_reflect.config import Config
from agent_reflect.ledger import Lesson, State, normalize_cue
from agent_reflect.memory import (
    MemoryIndexError,
    MemoryWriteError,
    add_index_line,
    autodream_disabled,
    go_live_if_room,
    has_capacity,
    index_line,
    is_user_entry,
    lesson_filename,
    read_index,
    rebalance,
    remove_index_line,
    remove_lesson,
    render_lesson,
    sync_ownership,
    write_lesson,
)


def _lesson(memory_dir, status="live", lid="ab12cd34") -> Lesson:
    return Lesson(
        id=lid,
        project_root="/r",
        memory_dir=str(memory_dir),
        file=lesson_filename(lid),
        title="Заголовок",
        hook="Одна ветка: на задачу — всегда",
        cue="Когда коммит блокирует хук.",
        lesson="Не обходить хук.",
        why="Потому что.",
        confidence="high",
        repeats=2,
        evidence=[{"session": "s1", "ts": "2026-10-20T00:00:00Z", "text": "т"}],
        status=status,
        sha256="",
        created_at="2026-10-20T00:00:00Z",
        updated_at="2026-10-20T01:00:00Z",
    )


def _cfg(tmp_path, **kw) -> Config:
    return replace(Config.under(tmp_path), **kw)


def test_write_lesson_format(tmp_path):
    lesson = _lesson(tmp_path)
    digest = write_lesson(tmp_path, lesson)
    path = tmp_path / "auto_ab12cd34.md"
    assert path.exists()
    text = path.read_text(encoding="utf-8")
    assert digest == hashlib.sha256(text.encode("utf-8")).hexdigest()
    assert text.startswith("---\nname: auto-ab12cd34\n")
    assert 'description: "Одна ветка: на задачу — всегда"' in text
    for needle in ("type: feedback", "source: agent-reflect", "lesson_id: ab12cd34",
                   "confidence: high", "repeats: 2", "evidence_sessions: [s1@2026-10-20T00:00:00Z]",
                   "**Why:** Потому что.", "**How to apply:** Когда: Когда коммит блокирует хук. Делать: Не обходить хук."):
        assert needle in text
    assert render_lesson(lesson) == text


def test_index_add_idempotent_and_remove(tmp_path):
    add_index_line(tmp_path, "auto_x.md", "хук")
    add_index_line(tmp_path, "auto_x.md", "хук")
    text = (tmp_path / "MEMORY.md").read_text(encoding="utf-8")
    assert text.count("- [auto_x.md](auto_x.md) — хук") == 1
    remove_index_line(tmp_path, "auto_x.md")
    assert "auto_x.md" not in (tmp_path / "MEMORY.md").read_text(encoding="utf-8")
    remove_index_line(tmp_path, "auto_x.md")  # no-op


def test_index_created_when_missing(tmp_path):
    add_index_line(tmp_path, "auto_x.md", "хук")
    assert (tmp_path / "MEMORY.md").read_text(encoding="utf-8") == (
        "# Memory Index\n\n- [auto_x.md](auto_x.md) — хук\n"
    )


def test_index_line_is_what_add_index_line_writes(tmp_path):
    line = index_line("auto_x.md", "хук\n  на две строки")
    assert line == "- [auto_x.md](auto_x.md) — хук на две строки"
    add_index_line(tmp_path, "auto_x.md", "хук\n  на две строки")
    assert (tmp_path / "MEMORY.md").read_text(encoding="utf-8").splitlines()[-1] == line


def test_read_index_text_and_errors(tmp_path):
    import pytest

    assert read_index(tmp_path) == ""
    (tmp_path / "MEMORY.md").write_text("# Memory Index\n", encoding="utf-8")
    assert read_index(tmp_path) == "# Memory Index\n"
    (tmp_path / "MEMORY.md").write_bytes(b"\xff\xfe")
    with pytest.raises(MemoryIndexError):
        read_index(tmp_path)


def test_index_keeps_user_lines(tmp_path):
    (tmp_path / "MEMORY.md").write_text(
        "# Memory Index\n\n- [user_language.md](user_language.md) — язык\n", encoding="utf-8"
    )
    add_index_line(tmp_path, "auto_x.md", "хук")
    remove_index_line(tmp_path, "auto_x.md")
    assert (tmp_path / "MEMORY.md").read_text(encoding="utf-8") == (
        "# Memory Index\n\n- [user_language.md](user_language.md) — язык\n"
    )


def test_capacity_auto_cap(tmp_path):
    cfg = _cfg(tmp_path, auto_index_cap=2)
    add_index_line(tmp_path, "auto_a.md", "a")
    assert has_capacity(tmp_path, cfg, "- [auto_c.md](auto_c.md) — c")
    add_index_line(tmp_path, "auto_b.md", "b")
    assert not has_capacity(tmp_path, cfg, "- [auto_c.md](auto_c.md) — c")


def test_capacity_total_line_limit(tmp_path):
    cfg = _cfg(tmp_path)
    body = "".join(f"- [u{i}.md](u{i}.md) — x\n" for i in range(188))
    (tmp_path / "MEMORY.md").write_text("# Memory Index\n\n" + body, encoding="utf-8")  # 190 lines
    assert not has_capacity(tmp_path, cfg, "- [auto_c.md](auto_c.md) — c")


def test_capacity_missing_index_ok(tmp_path):
    assert has_capacity(tmp_path, _cfg(tmp_path), "- [auto_c.md](auto_c.md) — c")


def test_capacity_char_limit(tmp_path):
    cfg = _cfg(tmp_path, index_char_limit=100)
    (tmp_path / "MEMORY.md").write_text("# Memory Index\n\n" + "x" * 70 + "\n", encoding="utf-8")
    assert not has_capacity(tmp_path, cfg, "y" * 30)
    assert has_capacity(tmp_path, cfg, "y" * 5)
    assert has_capacity(tmp_path, cfg, "y" * 12)  # 87 + 12 + its newline = 100
    assert not has_capacity(tmp_path, cfg, "y" * 13)


def test_is_user_entry(tmp_path):
    user = tmp_path / "user.md"
    user.write_text("---\nname: u\nmetadata:\n  type: user\n---\n\nbody\n", encoding="utf-8")
    write_lesson(tmp_path, _lesson(tmp_path))
    assert is_user_entry(user)
    assert not is_user_entry(tmp_path / "auto_ab12cd34.md")
    assert not is_user_entry(tmp_path / "missing.md")


def test_sync_ownership_deleted_becomes_rejected(tmp_path):
    lesson = _lesson(tmp_path)
    lesson.sha256 = write_lesson(tmp_path, lesson)
    add_index_line(tmp_path, lesson.file, lesson.hook)
    state = State(lessons={lesson.id: lesson})
    (tmp_path / lesson.file).unlink()
    assert sync_ownership(state) == [("/r", str(tmp_path))]
    assert lesson.status == "rejected"
    assert state.rejected_cues["/r"] == [normalize_cue(lesson.cue)]
    assert lesson.file not in (tmp_path / "MEMORY.md").read_text(encoding="utf-8")


def test_sync_ownership_edited_becomes_user_owned(tmp_path):
    lesson = _lesson(tmp_path, status="staged")
    lesson.sha256 = write_lesson(tmp_path, lesson)
    state = State(lessons={lesson.id: lesson})
    assert sync_ownership(state) == []
    assert lesson.status == "staged"
    with (tmp_path / lesson.file).open("a", encoding="utf-8") as fh:
        fh.write("\nправка\n")
    assert sync_ownership(state) == [("/r", str(tmp_path))]
    assert lesson.status == "user_owned"


def test_sync_ownership_ignores_other_statuses(tmp_path):
    lesson = _lesson(tmp_path, status="promoted")
    state = State(lessons={lesson.id: lesson})
    assert sync_ownership(state) == []
    assert lesson.status == "promoted"


def test_autodream_disabled(tmp_path):
    p = tmp_path / "settings.json"
    p.write_text(json.dumps({"autoDreamEnabled": False}), encoding="utf-8")
    assert autodream_disabled(p)
    p.write_text(json.dumps({"other": 1}), encoding="utf-8")
    assert not autodream_disabled(p)
    p.write_text(json.dumps({"autoDreamEnabled": True}), encoding="utf-8")
    assert not autodream_disabled(p)
    p.write_text("{broken", encoding="utf-8")
    assert not autodream_disabled(p)
    assert not autodream_disabled(tmp_path / "missing.json")


def test_index_crlf_preserved_on_add(tmp_path):
    original = b"# Memory Index\r\n\r\n- [u.md](u.md) \xe2\x80\x94 x\r\nlone\rcr\r\n"
    (tmp_path / "MEMORY.md").write_bytes(original)
    add_index_line(tmp_path, "auto_x.md", "h")
    out = (tmp_path / "MEMORY.md").read_bytes()
    assert out.startswith(original)
    assert out.endswith("- [auto_x.md](auto_x.md) \u2014 h\r\n".encode())
    remove_index_line(tmp_path, "auto_x.md")
    assert (tmp_path / "MEMORY.md").read_bytes() == original


def test_index_no_trailing_newline_gets_one(tmp_path):
    (tmp_path / "MEMORY.md").write_bytes(b"# Memory Index\n\n- [u.md](u.md) - x")
    add_index_line(tmp_path, "auto_x.md", "h")
    assert (tmp_path / "MEMORY.md").read_text(encoding="utf-8") == (
        "# Memory Index\n\n- [u.md](u.md) - x\n- [auto_x.md](auto_x.md) \u2014 h\n"
    )


def test_index_write_is_atomic_no_temp_left(tmp_path, monkeypatch):
    add_index_line(tmp_path, "auto_x.md", "h")
    assert [p.name for p in tmp_path.iterdir()] == ["MEMORY.md"]
    import agent_reflect.memory as m

    def boom(src, dst):
        raise OSError("boom")

    monkeypatch.setattr(m.os, "replace", boom)
    before = (tmp_path / "MEMORY.md").read_bytes()
    try:
        add_index_line(tmp_path, "auto_y.md", "h")
    except OSError:
        pass
    assert (tmp_path / "MEMORY.md").read_bytes() == before
    assert [p.name for p in tmp_path.iterdir()] == ["MEMORY.md"]


def test_non_utf8_index_fails_closed(tmp_path):
    (tmp_path / "MEMORY.md").write_bytes(b"\xff\xfe bad")
    import pytest

    with pytest.raises(MemoryIndexError):
        add_index_line(tmp_path, "auto_x.md", "h")
    with pytest.raises(MemoryIndexError):
        remove_index_line(tmp_path, "auto_x.md")
    assert not has_capacity(tmp_path, _cfg(tmp_path), "- [auto_c.md](auto_c.md) - c")
    assert (tmp_path / "MEMORY.md").read_bytes() == b"\xff\xfe bad"


def test_is_user_entry_fails_safe(tmp_path):
    bad = tmp_path / "bad.md"
    bad.write_bytes(b"\xff\xfe")
    assert is_user_entry(bad)
    top = tmp_path / "top.md"
    top.write_text("---\nsource: agent-reflect\nname: x\n---\nbody\n", encoding="utf-8")
    assert is_user_entry(top)
    body = tmp_path / "body.md"
    body.write_text("---\nname: x\n---\n  source: agent-reflect\n", encoding="utf-8")
    assert is_user_entry(body)
    nofm = tmp_path / "nofm.md"
    nofm.write_text("metadata:\n  source: agent-reflect\n", encoding="utf-8")
    assert is_user_entry(nofm)


def test_write_lesson_refuses_bad_name_and_user_file(tmp_path):
    import pytest

    lesson = _lesson(tmp_path)
    lesson.file = "../evil.md"
    with pytest.raises(MemoryWriteError):
        write_lesson(tmp_path, lesson)
    lesson.file = "user.md"
    with pytest.raises(MemoryWriteError):
        write_lesson(tmp_path, lesson)
    lesson = _lesson(tmp_path)
    target = tmp_path / lesson.file
    target.write_text("---\nname: mine\n---\nuser text\n", encoding="utf-8")
    with pytest.raises(MemoryWriteError):
        write_lesson(tmp_path, lesson)
    assert target.read_text(encoding="utf-8").endswith("user text\n")
    target.unlink()
    write_lesson(tmp_path, lesson)
    write_lesson(tmp_path, lesson)  # our own file may be rewritten


def test_remove_index_line_refuses_non_auto_and_keeps_prefix_siblings(tmp_path):
    import pytest

    with pytest.raises(MemoryWriteError):
        remove_index_line(tmp_path, "user_language.md")
    text = (
        "# Memory Index\n\n- [auto_x.md](auto_x.md) - a\n"
        "- [auto_x.md.bak](auto_x.md.bak) - b\n- [auto_xy.md](auto_xy.md) - c\n"
    )
    (tmp_path / "MEMORY.md").write_text(text, encoding="utf-8")
    remove_index_line(tmp_path, "auto_x.md")
    assert (tmp_path / "MEMORY.md").read_text(encoding="utf-8") == (
        "# Memory Index\n\n- [auto_x.md.bak](auto_x.md.bak) - b\n- [auto_xy.md](auto_xy.md) - c\n"
    )


def test_hook_and_cue_normalized(tmp_path):
    lesson = _lesson(tmp_path)
    lesson.hook = "строка один\n  строка   два"
    lesson.cue = "когда хук!"
    text = render_lesson(lesson)
    assert 'description: "строка один строка два"' in text
    assert "Когда: когда хук. Делать:" in text
    add_index_line(tmp_path, lesson.file, lesson.hook)
    assert (tmp_path / "MEMORY.md").read_text(encoding="utf-8").endswith("\u2014 строка один строка два\n")


def test_sync_ownership_missing_dir_skipped(tmp_path):
    lesson = _lesson(tmp_path / "gone")
    state = State(lessons={lesson.id: lesson})
    assert sync_ownership(state) == []
    assert lesson.status == "live"
    assert state.rejected_cues == {}


def test_sync_ownership_unreadable_and_byte_edit_are_user_owned(tmp_path):
    a = _lesson(tmp_path, lid="aaaaaaaa")
    a.sha256 = write_lesson(tmp_path, a)
    (tmp_path / a.file).write_bytes((tmp_path / a.file).read_bytes() + b"\xff")
    b = _lesson(tmp_path, lid="bbbbbbbb")
    b.sha256 = write_lesson(tmp_path, b)
    (tmp_path / b.file).write_bytes(b"\xff\xfe")
    state = State(lessons={a.id: a, b.id: b})
    assert sync_ownership(state) == [("/r", str(tmp_path))]
    assert a.status == "user_owned" and b.status == "user_owned"


def test_sync_ownership_survives_bad_index(tmp_path):
    lesson = _lesson(tmp_path)
    lesson.sha256 = write_lesson(tmp_path, lesson)
    (tmp_path / "MEMORY.md").write_bytes(b"\xff")
    (tmp_path / lesson.file).unlink()
    state = State(lessons={lesson.id: lesson})
    assert sync_ownership(state) == [("/r", str(tmp_path))]
    assert lesson.status == "rejected"


def _written(tmp_path, lid="ab12cd34"):
    lesson = _lesson(tmp_path, lid=lid)
    lesson.sha256 = write_lesson(tmp_path, lesson)
    (tmp_path / "MEMORY.md").write_text(
        "# Memory Index\n\n- [user.md](user.md) — u\n" + index_line(lesson.file, lesson.hook) + "\n",
        encoding="utf-8",
    )
    return lesson


def test_remove_lesson_deletes_our_unchanged_file_and_line(tmp_path):
    lesson = _written(tmp_path)
    remove_lesson(tmp_path, lesson)
    assert not (tmp_path / lesson.file).exists()
    assert (tmp_path / "MEMORY.md").read_text(encoding="utf-8") == "# Memory Index\n\n- [user.md](user.md) — u\n"


def test_remove_lesson_refuses_edited_user_misnamed_or_missing(tmp_path):
    import pytest

    lesson = _written(tmp_path)
    index = (tmp_path / "MEMORY.md").read_text(encoding="utf-8")
    path = tmp_path / lesson.file
    ours = path.read_bytes()

    path.write_bytes(ours + b"\nmy note\n")  # edited, still carries source: agent-reflect
    with pytest.raises(MemoryWriteError):
        remove_lesson(tmp_path, lesson)
    assert path.read_bytes() == ours + b"\nmy note\n"

    user = b"---\nname: mine\n---\nuser text\n"  # hash matches, but it is a user entry
    path.write_bytes(user)
    with pytest.raises(MemoryWriteError):
        remove_lesson(tmp_path, replace(lesson, sha256=hashlib.sha256(user).hexdigest()))
    assert path.read_bytes() == user

    path.write_bytes(ours)
    with pytest.raises(MemoryWriteError):
        remove_lesson(tmp_path, replace(lesson, file="auto_other.md"))
    with pytest.raises(MemoryWriteError):
        remove_lesson(tmp_path, replace(lesson, id="../x", file="auto_../x.md"))
    assert path.read_bytes() == ours

    path.unlink()
    with pytest.raises(MemoryWriteError):
        remove_lesson(tmp_path, lesson)
    assert (tmp_path / "MEMORY.md").read_text(encoding="utf-8") == index


def test_go_live_if_room(tmp_path):
    import pytest

    lesson = _lesson(tmp_path, status="staged")
    lesson.sha256 = write_lesson(tmp_path, lesson)
    assert go_live_if_room(_cfg(tmp_path, auto_index_cap=0), lesson) is False
    assert lesson.status == "staged" and not (tmp_path / "MEMORY.md").exists()
    assert go_live_if_room(_cfg(tmp_path), lesson) is True
    assert lesson.status == "live"
    assert (tmp_path / "MEMORY.md").read_text(encoding="utf-8").endswith(index_line(lesson.file, lesson.hook) + "\n")
    (tmp_path / "MEMORY.md").write_bytes(b"\xff")
    with pytest.raises(MemoryIndexError):
        go_live_if_room(_cfg(tmp_path), _lesson(tmp_path, status="staged", lid="bbbbbbbb"))


def _live(tmp_path, lid="ab12cd34", **kw):
    lesson = replace(_lesson(tmp_path, lid=lid), **kw)
    lesson.sha256 = write_lesson(tmp_path, lesson)
    add_index_line(tmp_path, lesson.file, lesson.hook)
    return lesson


def test_sync_ownership_deleted_index_line_is_a_rejection(tmp_path):
    lesson = _live(tmp_path)
    other = _live(tmp_path, lid="bbbbbbbb", cue="Другая ситуация.")
    (tmp_path / "MEMORY.md").write_text(
        "# Memory Index\n\n- [user.md](user.md) — u\n" + index_line(other.file, other.hook) + "\n",
        encoding="utf-8")
    state = State(lessons={lesson.id: lesson, other.id: other})
    assert sync_ownership(state) == [("/r", str(tmp_path))]
    assert lesson.status == "rejected" and other.status == "live"
    assert state.rejected_cues["/r"] == [normalize_cue(lesson.cue)]
    assert not (tmp_path / lesson.file).exists() and (tmp_path / other.file).exists()
    assert "user.md" in (tmp_path / "MEMORY.md").read_text(encoding="utf-8")


def test_sync_ownership_deleted_line_of_edited_lesson_hands_it_over(tmp_path):
    lesson = _live(tmp_path)
    (tmp_path / "MEMORY.md").write_text("# Memory Index\n", encoding="utf-8")
    with (tmp_path / lesson.file).open("a", encoding="utf-8") as fh:
        fh.write("\nмоя правка\n")
    state = State(lessons={lesson.id: lesson})
    assert sync_ownership(state) == [("/r", str(tmp_path))]
    assert lesson.status == "user_owned" and (tmp_path / lesson.file).exists()
    assert state.rejected_cues == {}


def test_sync_ownership_unreadable_or_absent_index_changes_nothing(tmp_path):
    lesson = _live(tmp_path)
    state = State(lessons={lesson.id: lesson})
    (tmp_path / "MEMORY.md").write_bytes(b"\xff\xfe")
    assert sync_ownership(state) == []
    (tmp_path / "MEMORY.md").unlink()
    assert sync_ownership(state) == []
    assert lesson.status == "live" and (tmp_path / lesson.file).exists()
    assert state.rejected_cues == {}


def _user_index(tmp_path, n_user: int) -> list[str]:
    lines = [f"- [user_{i}.md](user_{i}.md) — запись пользователя {i}" for i in range(n_user)]
    (tmp_path / "MEMORY.md").write_text("# Memory Index\n\n" + "\n".join(lines) + "\n", encoding="utf-8")
    return lines


def _ours(tmp_path, lid, *, status="live", confidence="high", repeats=1, created_at="2026-10-01T00:00:00Z"):
    lesson = replace(_lesson(tmp_path, lid=lid, status=status), confidence=confidence, repeats=repeats,
                     created_at=created_at, hook=f"крючок {lid}")
    lesson.sha256 = write_lesson(tmp_path, lesson)
    if status == "live":
        add_index_line(tmp_path, lesson.file, lesson.hook)
    return lesson


def test_rebalance_demotes_auto_lines_until_the_index_fits(tmp_path):
    cfg = _cfg(tmp_path, index_line_limit=10)
    user = _user_index(tmp_path, 4)  # header 2 + 4 user lines
    old_high = _ours(tmp_path, "aaaaaaa1", created_at="2026-10-01T00:00:00Z")
    low = _ours(tmp_path, "aaaaaaa2", confidence="low", created_at="2026-10-05T00:00:00Z")
    new_high = _ours(tmp_path, "aaaaaaa3", created_at="2026-10-06T00:00:00Z")
    medium = _ours(tmp_path, "aaaaaaa4", confidence="medium", created_at="2026-10-02T00:00:00Z")
    user += _user_index_append(tmp_path, ["- [later.md](later.md) — пользователь дописал", "- [x.md](x.md) — ещё"])
    state = State(lessons={x.id: x for x in (old_high, low, new_high, medium)})
    assert len((tmp_path / "MEMORY.md").read_text(encoding="utf-8").splitlines()) == 12
    assert rebalance(state, cfg, "/r", tmp_path) is True
    text = (tmp_path / "MEMORY.md").read_text(encoding="utf-8")
    assert len(text.splitlines()) == 10
    assert [ln for ln in text.splitlines() if not ln.startswith("- [auto_")][2:] == user
    assert low.status == "staged" and medium.status == "staged"
    assert old_high.status == "live" and new_high.status == "live"
    assert low.file not in text and medium.file not in text
    assert (tmp_path / low.file).exists() and (tmp_path / medium.file).exists()
    assert rebalance(state, cfg, "/r", tmp_path) is False


def _user_index_append(tmp_path, lines: list[str]) -> list[str]:
    with (tmp_path / "MEMORY.md").open("a", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    return lines


def test_rebalance_char_limit_demotes_oldest_high_after_low_and_medium(tmp_path):
    user = _user_index(tmp_path, 1)
    old = _ours(tmp_path, "aaaaaaa1", created_at="2026-10-01T00:00:00Z")
    new = _ours(tmp_path, "aaaaaaa2", created_at="2026-10-09T00:00:00Z")
    text = (tmp_path / "MEMORY.md").read_text(encoding="utf-8")
    cfg = _cfg(tmp_path, index_char_limit=len(text) - 1)
    state = State(lessons={old.id: old, new.id: new})
    assert rebalance(state, cfg, "/r", tmp_path) is True
    assert old.status == "staged" and new.status == "live"
    assert user[0] in (tmp_path / "MEMORY.md").read_text(encoding="utf-8")


def test_rebalance_never_touches_user_or_user_owned_lines(tmp_path):
    cfg = _cfg(tmp_path, index_line_limit=3)
    _user_index(tmp_path, 3)
    taken = _ours(tmp_path, "aaaaaaa1")
    taken.status = "user_owned"
    before = (tmp_path / "MEMORY.md").read_bytes()
    state = State(lessons={taken.id: taken})
    assert rebalance(state, cfg, "/r", tmp_path) is False
    assert (tmp_path / "MEMORY.md").read_bytes() == before


def test_rebalance_promotes_eligible_staged_lessons_when_room_returns(tmp_path):
    _user_index(tmp_path, 2)  # 4 lines
    cfg = _cfg(tmp_path, index_line_limit=6)
    weak = _ours(tmp_path, "aaaaaaa1", status="staged", confidence="medium", repeats=1)
    high = _ours(tmp_path, "aaaaaaa2", status="staged", confidence="high", repeats=1)
    repeated = _ours(tmp_path, "aaaaaaa3", status="staged", confidence="medium", repeats=3)
    other_dir = tmp_path / "other"
    other_dir.mkdir()
    foreign = _ours(other_dir, "aaaaaaa4", status="staged", repeats=5)
    state = State(lessons={x.id: x for x in (weak, high, repeated, foreign)})
    assert rebalance(state, cfg, "/r", tmp_path) is True
    text = (tmp_path / "MEMORY.md").read_text(encoding="utf-8")
    assert repeated.status == "live" and high.status == "live" and weak.status == "staged"
    assert text.index(repeated.file) < text.index(high.file)
    assert weak.file not in text and foreign.status == "staged"
    assert len(text.splitlines()) == 6


def test_rebalance_unreadable_index_changes_nothing(tmp_path):
    staged = _ours(tmp_path, "aaaaaaa1", status="staged")
    (tmp_path / "MEMORY.md").write_bytes(b"\xff")
    state = State(lessons={staged.id: staged})
    assert rebalance(state, _cfg(tmp_path), "/r", tmp_path) is False
    assert staged.status == "staged" and (tmp_path / "MEMORY.md").read_bytes() == b"\xff"


def test_going_live_records_when(tmp_path):
    from datetime import datetime, timezone

    lesson = _lesson(tmp_path, status="staged")
    lesson.sha256 = write_lesson(tmp_path, lesson)
    assert go_live_if_room(_cfg(tmp_path), lesson, "2026-10-20T03:00:00Z") is True
    assert lesson.live_since == "2026-10-20T03:00:00Z"
    other = _lesson(tmp_path, status="staged", lid="bbbbbbbb")
    other.sha256 = write_lesson(tmp_path, other)
    before = datetime.now(timezone.utc).replace(microsecond=0)
    go_live_if_room(_cfg(tmp_path), other)
    assert datetime.fromisoformat(other.live_since) >= before


def test_rebalance_promotion_records_when(tmp_path):
    _user_index(tmp_path, 1)
    staged = _ours(tmp_path, "aaaaaaa1", status="staged", confidence="high")
    state = State(lessons={staged.id: staged})
    assert rebalance(state, _cfg(tmp_path), "/r", tmp_path, "2026-10-21T03:00:00Z") is True
    assert staged.status == "live" and staged.live_since == "2026-10-21T03:00:00Z"


def _events(state: State) -> list[dict]:
    return [{k: v for k, v in event.items() if k != "ts"} for event in state.events]


def test_go_live_records_lesson_live_event(tmp_path):
    state = State()
    lesson = _lesson(tmp_path, status="staged")
    lesson.sha256 = write_lesson(tmp_path, lesson)
    assert go_live_if_room(_cfg(tmp_path, auto_index_cap=0), lesson, state=state) is False
    assert state.events == []  # no room, no event
    assert go_live_if_room(_cfg(tmp_path), lesson, state=state) is True
    assert _events(state) == [{"kind": "lesson_live", "project": "/r", "lesson_id": lesson.id, "title": "Заголовок"}]
    other = _lesson(tmp_path, status="staged", lid="bbbbbbbb")
    other.sha256 = write_lesson(tmp_path, other)
    assert go_live_if_room(_cfg(tmp_path), other) is True  # without a state nothing is recorded
    assert other.status == "live" and len(state.events) == 1


def test_rebalance_records_unlisted(tmp_path):
    _user_index(tmp_path, 1)
    old = _ours(tmp_path, "aaaaaaa1", created_at="2026-10-01T00:00:00Z")
    new = _ours(tmp_path, "aaaaaaa2", created_at="2026-10-09T00:00:00Z")
    cfg = _cfg(tmp_path, index_char_limit=len((tmp_path / "MEMORY.md").read_text(encoding="utf-8")) - 1)
    state = State(lessons={old.id: old, new.id: new})
    assert rebalance(state, cfg, "/r", tmp_path) is True
    assert _events(state) == [{"kind": "lesson_unlisted", "project": "/r", "lesson_id": old.id,
                               "title": "Заголовок", "reason": "no_room"}]


def test_rebalance_records_live_for_promoted_staged(tmp_path):
    _user_index(tmp_path, 1)
    staged = _ours(tmp_path, "aaaaaaa1", status="staged", confidence="medium", repeats=2)
    state = State(lessons={staged.id: staged})
    assert rebalance(state, _cfg(tmp_path), "/r", tmp_path) is True
    assert _events(state) == [{"kind": "lesson_live", "project": "/r", "lesson_id": staged.id,
                               "title": "Заголовок"}]


def test_sync_ownership_records_rejection_and_takeover(tmp_path):
    gone = _live(tmp_path, lid="aaaaaaa1", cue="Первая ситуация.")
    unlisted = _live(tmp_path, lid="aaaaaaa2", cue="Вторая ситуация.")
    edited = _live(tmp_path, lid="aaaaaaa3", cue="Третья ситуация.")
    (tmp_path / gone.file).unlink()
    (tmp_path / "MEMORY.md").write_text(
        "# Memory Index\n\n" + index_line(gone.file, gone.hook) + "\n" + index_line(edited.file, edited.hook) + "\n",
        encoding="utf-8")
    with (tmp_path / edited.file).open("a", encoding="utf-8") as fh:
        fh.write("\nправка\n")
    state = State(lessons={x.id: x for x in (gone, unlisted, edited)})
    assert sync_ownership(state) == [("/r", str(tmp_path))]
    assert _events(state) == [
        {"kind": "lesson_rejected", "project": "/r", "lesson_id": gone.id, "title": "Заголовок",
         "reason": "file_deleted"},
        {"kind": "lesson_rejected", "project": "/r", "lesson_id": unlisted.id, "title": "Заголовок",
         "reason": "index_line_deleted"},
        {"kind": "lesson_user_owned", "project": "/r", "lesson_id": edited.id, "title": "Заголовок",
         "reason": "edited"}]
    assert sync_ownership(state) == []
    assert len(state.events) == 3  # a handled lesson is not recorded twice


def test_sync_ownership_records_an_unreadable_file_as_takeover(tmp_path):
    lesson = _live(tmp_path)
    (tmp_path / lesson.file).write_bytes(b"\xff\xfe")
    state = State(lessons={lesson.id: lesson})
    assert sync_ownership(state) == [("/r", str(tmp_path))]
    assert _events(state) == [{"kind": "lesson_user_owned", "project": "/r", "lesson_id": lesson.id,
                               "title": "Заголовок", "reason": "edited"}]


def test_rebalance_never_takes_a_blocking_lesson_out_of_the_index(tmp_path):
    # a staged lesson leaves the hook's rules, and its block with them: only the user lifts a block
    cfg = _cfg(tmp_path, index_line_limit=10)
    _user_index(tmp_path, 4)
    low = _ours(tmp_path, "aaaaaaa2", confidence="low", created_at="2026-10-05T00:00:00Z")
    low.block = True
    medium = _ours(tmp_path, "aaaaaaa4", confidence="medium", created_at="2026-10-02T00:00:00Z")
    others = [_ours(tmp_path, f"aaaaaab{n}", created_at=f"2026-10-0{n}T00:00:00Z") for n in (1, 6)]
    _user_index_append(tmp_path, ["- [later.md](later.md) — пользователь дописал", "- [x.md](x.md) — ещё"])
    state = State(lessons={x.id: x for x in (low, medium, *others)})
    assert rebalance(state, cfg, "/r", tmp_path) is True
    assert low.status == "live" and medium.status == "staged"
