# agent-reflect

Local macOS tool that reflects on Claude Code session transcripts and writes lessons into memory.
What it does and how it is configured: `README.md`.
The README's diagrams describe the code: change them with it. Its sample of the digest's memory map is what
`agent_reflect/memory_map.py` draws, and `tests/test_memory_map.py` fails when the two drift apart.

## Commands

```bash
uv sync
uv run --locked pytest -q
```

## Gating rule

Every commit must pass `uv run --locked pytest -q` (`--locked`: a dependency change that did not update `uv.lock` fails here, as it does in CI).

Changes land through a pull request into `master`, never a direct push: the ruleset on `master` requires the `tests` check (`.github/workflows/pr-checks.yml`, the same `uv run --locked pytest -q` on macOS) on a branch that carries the latest `master`.

If GitHub Actions is down, the check never starts and nothing can be merged. For an urgent merge the owner turns the ruleset off (Settings → Rules → Rulesets → `master` → Enforcement status: Disabled, or `gh api -X PUT repos/vladislavgorbunov-afk/agent-reflect/rulesets/27454875 -f enforcement=disabled`, which keeps its rules), merges, and turns it back on at once (`-f enforcement=active`): while it is off, nothing stops a direct push, a force push or deleting `master`.

## Tests

Tests never call real `claude`, `gh`, `launchctl` or `terminal-notifier`; all external calls go through an injected `run` / `runner` / `llm`.
Tests never read the developer's own config: `Config.under(tmp_path)` for paths, `Config.default(<path>)` with a file the test writes; `tests/conftest.py` hides the machine's macOS language preferences and locale variables, so `language = "auto"` resolves the same everywhere.

## User-facing text

Reports, the digest, notifications and CLI output speak the configured language (`agent_reflect/i18n.py`).
Write each phrase at its call site as `t("English", "Русский")`, counts with `i18n.count`, fractions with `i18n.decimal`.
The suite runs in Russian by default (`tests/conftest.py`). `tests/test_english.py` renders every report from fixtures whose Cyrillic was replaced, so any Cyrillic left came from the code; text a run writes into the journal and is shown later (titles, notes, reasons, diffs) is checked where it is written.
Prompts, lesson files and the guard hook's text are for the model and the agent, not the user, and stay Russian.
