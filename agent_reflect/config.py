from __future__ import annotations

import re
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from agent_reflect import i18n
from agent_reflect.projects import sanitize

# A consolidate run's id: its start time, also the name of its backup folders.
RUN_ID_FORMAT = "%Y%m%dT%H%M%S"
_RUN_ID_RE = re.compile(r"\d{8}T\d{6}")


def is_run_id(value: str | None) -> bool:
    return isinstance(value, str) and _RUN_ID_RE.fullmatch(value) is not None


class ConfigError(ValueError):
    """The user's config file exists but cannot be used: a run stops rather than guess."""


@dataclass(frozen=True)
class UserSettings:
    """What only the user knows, read from their config file (see README): no file, nothing set."""

    # repositories whose skills and CLAUDE.md the weekly run may change through draft PRs
    repos: Mapping[str, Path]
    # where a project works out its test-writing rules: the weekly run lists what changed there
    # as candidates for dev-process-toolkit. (repo name in `repos`, watched paths in that repo)
    upstream_sources: tuple[tuple[str, tuple[str, ...]], ...] = ()
    # repos whose .cursor/skills copies are kept in step, while cursor_sync is on
    cursor_repos: frozenset[str] = frozenset()
    cursor_sync: bool = False
    # what reports, the digest, notifications and CLI output speak: "auto" is the system's language
    language: str = "auto"
    # the weekly run's worktree stays after it opens a PR (or fails to): see Config.keep_worktrees
    keep_worktrees: bool = True


_USER_KEYS = {"repos", "upstream", "cursor_repos", "cursor_sync", "language", "keep_worktrees"}


def user_config_path(home: Path) -> Path:
    """One place under HOME: launchd sets HOME and PATH only, so the night reads what the shell reads."""
    return home / ".config" / "agent-reflect" / "config.toml"


def _repo_path(path: Path, name: str, value: object, home: Path) -> Path:
    if isinstance(value, str) and (value == "~" or value.startswith("~/")):
        return home / value[2:]
    if isinstance(value, str) and Path(value).is_absolute():
        return Path(value)
    raise ConfigError(f"{path}: repos.{name} must be an absolute path or start with ~/")


def _names(path: Path, key: str, value: object) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(v, str) and v for v in value):
        raise ConfigError(f"{path}: {key} must be a list of non-empty strings")
    return value


def load_user_settings(path: Path, home: Path) -> UserSettings:
    if not path.exists():
        return UserSettings(repos={})
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise ConfigError(f"{path}: {exc}") from exc
    unknown = sorted(set(data) - _USER_KEYS)
    if unknown:
        raise ConfigError(f"{path}: unknown keys: {', '.join(unknown)}")
    raw_repos = data.get("repos", {})
    if not isinstance(raw_repos, dict):
        raise ConfigError(f"{path}: [repos] must be a table of name = path")
    repos = {name: _repo_path(path, name, value, home) for name, value in raw_repos.items()}
    raw_upstream = data.get("upstream", [])
    if not isinstance(raw_upstream, list):
        raise ConfigError(f"{path}: upstream must be an array of tables ([[upstream]])")
    upstream = []
    for n, entry in enumerate(raw_upstream, 1):
        repo = entry.get("repo") if isinstance(entry, dict) else None
        if not isinstance(repo, str) or repo not in repos or set(entry) != {"repo", "paths"}:
            raise ConfigError(f"{path}: upstream #{n} needs exactly a repo from [repos] and its paths")
        paths = _names(path, f"upstream #{n} paths", entry["paths"])
        if not paths:
            raise ConfigError(f"{path}: upstream #{n} watches no paths")
        upstream.append((repo, tuple(paths)))
    cursor_repos = _names(path, "cursor_repos", data.get("cursor_repos", []))
    stray = sorted(set(cursor_repos) - set(repos))
    if stray:
        raise ConfigError(f"{path}: cursor_repos names repos missing from [repos]: {', '.join(stray)}")
    cursor_sync = data.get("cursor_sync", False)
    if not isinstance(cursor_sync, bool):
        raise ConfigError(f"{path}: cursor_sync must be true or false")
    language = data.get("language", "auto")
    if language != "auto" and language not in i18n.LANGUAGES:
        raise ConfigError(f"{path}: language must be auto or one of {', '.join(i18n.LANGUAGES)}")
    keep_worktrees = data.get("keep_worktrees", True)
    if not isinstance(keep_worktrees, bool):
        raise ConfigError(f"{path}: keep_worktrees must be true or false")
    return UserSettings(repos=repos, upstream_sources=tuple(upstream),
                        cursor_repos=frozenset(cursor_repos), cursor_sync=cursor_sync, language=language,
                        keep_worktrees=keep_worktrees)


# Scratch places a session may run in; a repository there is never a project.
TEMP_DIRS = (Path("/tmp"), Path("/private/tmp"), Path("/var/folders"), Path("/private/var/folders"))


@dataclass(frozen=True)
class Config:
    repo_dir: Path
    claude_bin: Path
    projects_dir: Path
    claude_settings: Path
    personal_skills_dir: Path
    cursor_skills_dir: Path
    repos: dict[str, Path]
    reflect_model: str = "claude-sonnet-5-5"
    consolidate_model: str = "claude-opus-5-5"
    # the second look at a new lesson: another model than the one that wrote it, since two
    # samples of one model share its biases and agree for the same wrong reason
    verify_model: str = "claude-opus-5-5"
    # passed as --effort so the user's own global effort never applies to background calls
    reflect_effort: str = "medium"
    consolidate_effort: str = "high"
    quiet_seconds: int = 7200
    max_segments_per_run: int = 30
    segment_max_chars: int = 40_000
    # a long segment goes to the model in this many texts at most, each a call of its own
    segment_max_parts: int = 3
    auto_index_cap: int = 40
    index_line_limit: int = 190
    index_char_limit: int = 24_000
    consolidate_interval_days: int = 7
    llm_timeout_s: int = 600
    # a silent-failure key that fires this often in one night raises a notification
    health_notify_threshold: int = 20
    # hook failures whose command (quotes stripped, home as ~/) or hook name holds one of these
    # are not tracked at all; remove an entry to get its alerts back. The CLI stores plugin hook
    # commands unexpanded ("${CLAUDE_PLUGIN_ROOT}"/templates/hooks/process/x.sh), so the plugin's
    # name never shows up in them: the second entry is that plugin's own hook folder.
    health_ignore: tuple[str, ...] = ("dev-process-toolkit", "/templates/hooks/process/")
    temp_dirs: tuple[Path, ...] = TEMP_DIRS
    # empty: the weekly upstream scan is off (tests opt in with dataclasses.replace)
    upstream_sources: tuple[tuple[str, tuple[str, ...]], ...] = ()
    # repos (keys of `repos`) whose .cursor/skills copies are kept in step while cursor_sync is on
    cursor_repos: frozenset[str] = frozenset()
    # keep the Cursor copies (skills, .cursor/rules/claude-memory.mdc) in step; off unless the
    # user's config turns it on, Config.under turns it on to cover those paths
    cursor_sync: bool = False
    # resolved: one of i18n.LANGUAGES, never "auto"
    language: str = i18n.DEFAULT
    # the weekly run's worktree holds what it applied: the PR's checkout, or edits a failed publish
    # left behind. On, it stays: a PR's goes once the PR is merged or closed and nobody worked in it
    # since, an unpublished one is replaced by the next run. Off, it is removed at once. A worktree
    # the run applied nothing in goes either way: it holds nothing.
    keep_worktrees: bool = True

    @property
    def home(self) -> Path:
        return self.projects_dir.parents[1]

    @property
    def no_project_roots(self) -> tuple[Path, ...]:
        """A project root under any of these is no project: scratch dirs and Claude's own tree."""
        return (*self.temp_dirs, self.home / ".claude")

    @property
    def launch_agents_dir(self) -> Path:
        return self.home / "Library" / "LaunchAgents"

    @property
    def state_dir(self) -> Path:
        return self.repo_dir / "state"

    @property
    def state_file(self) -> Path:
        return self.state_dir / "state.json"

    @property
    def upstream_candidates_file(self) -> Path:
        """Every candidate the weekly upstream scan proposed, one JSON line each; only appended to."""
        return self.state_dir / "upstream-candidates.jsonl"

    @property
    def journal_file(self) -> Path:
        """What reflect, backfill and consolidate did, one JSON line each: the reports read it."""
        return self.state_dir / "journal.jsonl"

    @property
    def baseline_file(self) -> Path:
        """How often the user sent work back before the lessons could help: what later weeks are held against."""
        return self.state_dir / "baseline.json"

    @property
    def lock_file(self) -> Path:
        return self.state_dir / "run.lock"

    @property
    def sandbox_dir(self) -> Path:
        """The model's cwd: outside every git tree, so no repo's CLAUDE.md or settings load."""
        return self.home / "Library/Caches/agent-reflect/sandbox"

    @property
    def own_project_prefixes(self) -> tuple[str, ...]:
        """Project-folder slugs of our own sessions: this repo (and its worktrees) and the sandbox."""
        return (sanitize(str(self.repo_dir)), sanitize(str(self.sandbox_dir)))

    @property
    def isolation_settings(self) -> Path:
        return self.state_dir / "isolation-settings.json"

    @property
    def worktrees_dir(self) -> Path:
        return self.state_dir / "worktrees"

    @property
    def logs_dir(self) -> Path:
        return self.repo_dir / "logs"

    @property
    def digests_dir(self) -> Path:
        return self.repo_dir / "digests"

    @property
    def reports_dir(self) -> Path:
        return self.repo_dir / "reports"

    @property
    def backups_dir(self) -> Path:
        return self.repo_dir / "backups"

    @property
    def venv_bin(self) -> Path:
        return self.repo_dir / ".venv" / "bin" / "agent-reflect"

    @classmethod
    def _build(cls, repo_dir: Path, home: Path, user: UserSettings,
               temp_dirs: tuple[Path, ...] = TEMP_DIRS) -> "Config":
        return cls(
            repo_dir=repo_dir,
            claude_bin=home / ".local/bin/claude",
            projects_dir=home / ".claude/projects",
            claude_settings=home / ".claude/settings.json",
            personal_skills_dir=home / ".claude/skills",
            cursor_skills_dir=home / ".cursor/skills",
            repos=dict(user.repos),
            temp_dirs=temp_dirs,
            upstream_sources=user.upstream_sources,
            cursor_repos=user.cursor_repos,
            cursor_sync=user.cursor_sync,
            language=i18n.resolve(user.language),
            keep_worktrees=user.keep_worktrees,
        )

    @classmethod
    def default(cls, settings_path: Path | None = None) -> "Config":
        """The real run's config: home-based paths plus the user's config file (ConfigError if unusable)."""
        home = Path.home()
        user = load_user_settings(settings_path or user_config_path(home), home)
        return cls._build(Path(__file__).resolve().parents[1], home, user)

    @classmethod
    def under(cls, root: Path) -> "Config":
        # pytest's tmp dirs live under /var/folders: tests get scratch dirs of their own
        home = root / "home"
        user = UserSettings(repos={"webapp": home / "git/webapp", "toolbox": home / "git/toolbox"},
                            cursor_repos=frozenset({"webapp"}), cursor_sync=True, language="ru")
        return cls._build(root / "agent-reflect", home, user, (root / "tmp",))
