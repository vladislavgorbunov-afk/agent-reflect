# agent-reflect

agent-reflect reads your Claude Code session transcripts every night, finds the mistakes the agent made and the work you had to send back, and writes them down as short lessons in Claude Code's memory, so the next session does not repeat them. Once a week it moves the lessons that keep coming up into your skills.

It runs locally on macOS under launchd and calls the model through your own `claude` CLI. Nothing is sent anywhere else.

## What it does

**Every night (03:30, and at login).** It reads the parts of your sessions it has not seen yet, including subagent transcripts. It looks for:

- failed commands, including quiet failures (`command not found`, a traceback, `Permission denied`);
- your corrections: "you said it was done, it is not", a rule from your `CLAUDE.md` that was broken, "that is not what I meant".

The model writes at most a few lessons a night, as `auto_<id>.md` files in the project's memory folder. A second call checks each lesson against the transcript: the quoted words must be in it. Only lessons that are important or have happened twice get a line in `MEMORY.md`. The rest wait for a second case, and `agent-reflect staged` lists them.

The night's report goes to `reports/<date>.md`, and a macOS notification sums it up.

**Every week.** It moves confirmed lessons into skills:

- personal skills in `~/.claude/skills` are edited directly, with a backup;
- skills and `CLAUDE.md` in the repositories you list are changed through draft PRs.

It also groups similar corrections from the last 30 days into new lessons. The weekly digest (`digests/<week>.md`) shows how often you sent work back, the number the lessons are there to bring down. It also lists which memory files no agent opened in 30 days.

**While you work.** A `PreToolUse` hook on Bash, Edit and Write shows a lesson to the agent at the moment it runs a command or edits a file that matches the lesson's pattern. The hook never blocks anything.

**What it leaves alone.** Memory files you wrote yourself are never touched. Delete an `auto_*.md` file or its index line and that lesson is rejected for good. Edit one and it becomes yours.

## Requirements

- macOS (launchd, notifications)
- [Claude Code](https://claude.com/claude-code) CLI at `~/.local/bin/claude`
- Python 3.12 and [uv](https://docs.astral.sh/uv/)
- Optional: [terminal-notifier](https://github.com/julienXX/terminal-notifier) for clickable notifications (falls back to `osascript`)
- Optional: an authenticated [`gh`](https://cli.github.com/) for the weekly draft PRs

## Install

```bash
git clone https://github.com/vladislavgorbunov-afk/agent-reflect.git ~/git/agent-reflect
cd ~/git/agent-reflect
uv sync
.venv/bin/agent-reflect install
```

`install` writes two launchd agents (`local.agent-reflect.reflect` and `local.agent-reflect.consolidate`) and adds the guard hook to `~/.claude/settings.json`. On its first run it marks what is already in your transcripts as read, so it only learns from sessions after the install.

Turn off Claude Code's built-in memory consolidation, or agent-reflect will not touch memory:

```json
// ~/.claude/settings.json
{ "autoDreamEnabled": false }
```

## Configuration

Everything is optional. Without a config file agent-reflect still writes lessons and edits personal skills, and edits no repository.

`~/.config/agent-reflect/config.toml` (always this path: the launchd runs see only `HOME`):

```toml
# Language of reports, the digest, notifications and CLI output: "auto", "en" or "ru".
# "auto" (the default) takes the system's first language and falls back to English.
language = "auto"

# Keep .cursor/skills copies in step with the skills (off by default).
# Top-level keys go above the first [table]: TOML puts anything below it inside that table.
cursor_sync = false
cursor_repos = ["app"]

# Repositories whose skills (.claude/skills/*/SKILL.md, plugins/*/skills/*/SKILL.md)
# and CLAUDE.md the weekly run may change, through draft PRs.
[repos]
app = "~/git/app"
tools = "~/git/tools"

# Files where a repository keeps its test-writing rules: the weekly digest lists
# their new commits as candidates for dev-process-toolkit.
[[upstream]]
repo = "app"
paths = ["docs/TESTING.md"]
```

A config file that cannot be read stops every command with the reason, rather than running without your repositories.

The `language` setting covers what agent-reflect writes for you to read. The prompts, and so the lessons and the text the guard hook shows the agent, are in Russian for now.

## Commands

```bash
agent-reflect recent [--hours N] [-v]   # what the last night did
agent-reflect staged                    # lessons waiting for a second case
agent-reflect approve <id>              # put a waiting lesson into MEMORY.md now
agent-reflect retire <id>               # take a lesson's line out, keep its file
agent-reflect guard list|add|remove|set|unset|test|on|off
agent-reflect baseline --since YYYY-MM-DD [--dry-run]
agent-reflect rollback <RUN_ID> | --list
agent-reflect reflect --dry-run         # a night's run without writing
agent-reflect status
agent-reflect uninstall
```

State, logs, reports and digests stay inside the checkout (`state/`, `logs/`, `reports/`, `digests/`, `backups/`), and all of them are git-ignored.

## Cost

A model call reads about 50k tokens. The night's report and the weekly digest show the calls and their price at API rates. On a Claude subscription, these calls count against your usage limits instead of being billed.

## Development

```bash
uv sync
uv run pytest -q
```

Tests never call the real `claude`, `gh`, `launchctl` or `terminal-notifier`.

## License

MIT
