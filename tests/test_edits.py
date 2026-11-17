from __future__ import annotations

import pytest

from agent_reflect.edits import DIFF_MAX_LINES, REPLACE_MAX, incident_error, size_error, unified_diff


def test_insertion_of_any_size_is_accepted():
    old = "x" * 2000
    assert size_error(old, old + "\n- one more rule") is None
    assert size_error(old, "- first\n" + old) is None


def test_small_replacement_is_accepted_large_one_refused():
    assert size_error("a" * REPLACE_MAX, "b") is None
    reason = size_error("a" * (REPLACE_MAX + 1), "b")
    assert reason is not None and str(REPLACE_MAX) in reason


def test_unified_diff_shows_the_change():
    before = "# skill\n\nUse the old rule here.\n"
    after = "# skill\n\nUse the old rule here.\n\n- Run /gate-check first.\n"
    diff = unified_diff(before, after, "alpha/SKILL.md")
    lines = diff.splitlines()
    assert lines[0] == "--- a/alpha/SKILL.md" and lines[1] == "+++ b/alpha/SKILL.md"
    assert "+- Run /gate-check first." in lines
    assert not any(line.startswith("-") and not line.startswith("---") for line in lines)


def test_unified_diff_is_truncated_with_a_marker():
    before = "".join(f"line {i}\n" for i in range(200))
    after = "".join(f"changed {i}\n" for i in range(200))
    lines = unified_diff(before, after, "f.md").splitlines()
    assert len(lines) == DIFF_MAX_LINES
    assert "обрезан" in lines[-1]


def test_unified_diff_is_scrubbed():
    token = "ghp_" + "a" * 36
    assert token not in unified_diff("a\n", f"a\ntoken {token}\n", "f.md")



OLD = "## Commits\n\nKeep commits small.\n"
CLEAN = OLD + "\n- When the gate hook blocks `git commit`, run `/gate-check` first, then commit once.\n"


@pytest.mark.parametrize("added", [
    "- On 2026-10-20 the agent committed without the gate.",
    "- See PR #263 for the fix.",
    "- Fixed in pull #12.",
    "- Tracked as issue #7.",
    "- #263 broke this.",
    "- This bit us in ABC-483.",
    "- Seen in run 6b0c9283ab.",
    "- Сессия a1b2c3: агент повторял коммит.",
    "- In session 94b42366 the agent retried.",
    "- In session 1f2e3d4c-0000-4abc-8def-123456789abc the agent retried.",
])
def test_an_edit_that_logs_an_incident_is_refused(added):
    reason = incident_error(OLD, OLD + added + "\n")
    assert reason is not None and reason.startswith("incident_log")


def test_a_rule_stated_as_a_procedure_passes():
    assert incident_error(OLD, CLEAN) is None


@pytest.mark.parametrize("text", [
    "Files are UTF-8 and hashed with SHA-256.",
    "Use `#000000` for black.",  # a colour, not a PR
    "The deadbeefcafe pattern is plain prose.",  # hex letters alone are words, not an id
    "Run the session tests.",
    "Pick step #1 from the list.",
    "Reuse the session cookie2fa flow.",  # a word after session, not an id
])
def test_ordinary_text_is_no_incident(text):
    assert incident_error(OLD, OLD + text + "\n") is None


def test_a_reference_already_in_the_file_is_not_counted_again():
    old = OLD + "History: ABC-1 on 2026-01-01 (#5).\n"
    assert incident_error(old, old.replace("Keep commits small.", "Keep commits small and focused.")) is None



def test_an_edit_that_names_a_promoted_lesson_id_is_refused():
    reason = incident_error(OLD, OLD + "- Rule from ab12cd34: run the gate first.\n", lesson_ids=("ab12cd34",))
    assert reason is not None and reason.startswith("incident_log") and "ab12cd34" in reason
    assert incident_error(OLD, CLEAN, lesson_ids=("ab12cd34",)) is None
