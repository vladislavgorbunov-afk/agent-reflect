# agent-reflect

Local macOS tool that reflects on Claude Code session transcripts and writes lessons into memory.
What it does and how it is configured: `README.md`.

## Commands

```bash
uv sync
uv run pytest -q
```

## Gating rule

Every commit must pass `uv run pytest -q`.

## Tests

Tests never call real `claude`, `gh`, `launchctl` or `terminal-notifier`; all external calls go through an injected `run` / `runner` / `llm`.
Tests never read the developer's own config: `Config.under(tmp_path)` for paths, `Config.default(<path>)` with a file the test writes; `tests/conftest.py` hides the machine's macOS language preferences and locale variables, so `language = "auto"` resolves the same everywhere.

## User-facing text

Reports, the digest, notifications and CLI output speak the configured language (`agent_reflect/i18n.py`).
Write each phrase at its call site as `t("English", "Русский")`, counts with `i18n.count`, fractions with `i18n.decimal`.
The suite runs in Russian by default (`tests/conftest.py`). `tests/test_english.py` renders every report from fixtures whose Cyrillic was replaced, so any Cyrillic left came from the code; text a run writes into the journal and is shown later (titles, notes, reasons, diffs) is checked where it is written.
Prompts, lesson files and the guard hook's text are for the model and the agent, not the user, and stay Russian.
