from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent_reflect import i18n


@pytest.fixture(autouse=True)
def _russian_by_default(monkeypatch, tmp_path_factory):
    """The suite was written against the Russian texts; a test that checks English sets it itself.

    "auto" never reads the machine running the tests: no macOS preferences, no locale variables.
    """
    missing = tmp_path_factory.mktemp("no-prefs") / "GlobalPreferences.plist"
    monkeypatch.setattr(i18n, "global_prefs", lambda: missing)
    for var in ("LC_ALL", "LC_MESSAGES", "LANG"):
        monkeypatch.delenv(var, raising=False)
    # a suite run from a Claude Code shell is still the user's: a test that needs the agent sets it itself
    monkeypatch.delenv("CLAUDECODE", raising=False)
    i18n.use("ru")
    yield
    i18n.use(i18n.DEFAULT)


def write_jsonl(path: Path, entries: list, tail: str = "") -> None:
    """One JSON entry per line, then the raw tail (no trailing newline)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = "".join(json.dumps(e, ensure_ascii=False) + "\n" for e in entries)
    path.write_bytes((lines + tail).encode("utf-8"))
