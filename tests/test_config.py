from pathlib import Path

import pytest

from agent_reflect import i18n
from agent_reflect.config import Config, ConfigError, load_user_settings, user_config_path

NO_FILE = Path("/nonexistent/agent-reflect/config.toml")


def test_default_paths():
    cfg = Config.default(NO_FILE)
    home = Path.home()
    assert cfg.claude_bin == home / ".local/bin/claude"
    assert cfg.projects_dir == home / ".claude/projects"
    assert cfg.state_dir == cfg.repo_dir / "state"
    assert cfg.reflect_model == "claude-sonnet-5-5" and cfg.consolidate_model == "claude-opus-5-5"


def test_under_keeps_everything_in_root(tmp_path):
    cfg = Config.under(tmp_path)
    for p in [cfg.repo_dir, cfg.claude_bin, cfg.projects_dir, cfg.claude_settings,
              cfg.personal_skills_dir, cfg.cursor_skills_dir, *cfg.repos.values(), cfg.state_dir]:
        assert p.is_relative_to(tmp_path)


def test_no_project_roots():
    cfg = Config.default(NO_FILE)
    for d in ("/tmp", "/private/tmp", "/var/folders", "/private/var/folders"):
        assert Path(d) in cfg.no_project_roots
    assert Path.home() / ".claude" in cfg.no_project_roots


def test_under_no_project_roots_stay_in_root(tmp_path):
    cfg = Config.under(tmp_path)
    assert cfg.no_project_roots and all(p.is_relative_to(tmp_path) for p in cfg.no_project_roots)
    assert cfg.home / ".claude" in cfg.no_project_roots


def test_efforts_and_sandbox_outside_any_repo():
    cfg = Config.default(NO_FILE)
    assert cfg.reflect_effort == "medium" and cfg.consolidate_effort == "high"
    assert cfg.sandbox_dir == Path.home() / "Library/Caches/agent-reflect/sandbox"
    assert not cfg.sandbox_dir.is_relative_to(cfg.repo_dir)


def test_under_sandbox_stays_in_root(tmp_path):
    cfg = Config.under(tmp_path)
    assert cfg.sandbox_dir == cfg.home / "Library/Caches/agent-reflect/sandbox"
    assert cfg.sandbox_dir.is_relative_to(tmp_path) and not cfg.sandbox_dir.is_relative_to(cfg.repo_dir)


def test_cursor_sync_is_off_for_real_runs_and_on_in_tests(tmp_path):
    # off unless the user's config turns it on; tests keep the Cursor paths covered
    assert Config.default(NO_FILE).cursor_sync is False
    assert Config.under(tmp_path).cursor_sync is True
    fields = {f.name: f.default for f in __import__("dataclasses").fields(Config)}
    assert fields["cursor_sync"] is False  # a Config built field by field never syncs by accident


def test_keep_worktrees_is_on_unless_the_config_turns_it_off(tmp_path):
    # the weekly run's worktree holds the PR it opened: kept by default, for the review
    assert load_user_settings(tmp_path / "missing.toml", tmp_path).keep_worktrees is True
    assert Config.default(NO_FILE).keep_worktrees is True and Config.under(tmp_path).keep_worktrees is True
    fields = {f.name: f.default for f in __import__("dataclasses").fields(Config)}
    assert fields["keep_worktrees"] is True
    off = _settings(tmp_path, "keep_worktrees = false\n")
    assert load_user_settings(off, tmp_path).keep_worktrees is False
    assert Config.default(off).keep_worktrees is False


# -- the user's config file ----------------------------------------------------------

FULL = """\
language = "ru"
cursor_sync = true
cursor_repos = ["app"]

[repos]
app = "~/git/app"
tools = "/opt/src/tools"

[[upstream]]
repo = "app"
paths = ["docs/TESTING.md", "specs/testing.md"]
"""


def _settings(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "config.toml"
    path.write_text(text, encoding="utf-8")
    return path


def test_no_config_file_sets_nothing(tmp_path):
    user = load_user_settings(tmp_path / "missing.toml", tmp_path)
    assert (user.repos, user.upstream_sources, user.cursor_repos) == ({}, (), frozenset())
    assert user.cursor_sync is False and user.language == "auto"
    assert Config.default(tmp_path / "missing.toml").repos == {}


def test_config_file_sets_repos_upstream_cursor_and_language(tmp_path):
    user = load_user_settings(_settings(tmp_path, FULL), Path("/home/u"))
    assert user.repos == {"app": Path("/home/u/git/app"), "tools": Path("/opt/src/tools")}
    assert user.upstream_sources == (("app", ("docs/TESTING.md", "specs/testing.md")),)
    assert user.cursor_repos == frozenset({"app"}) and user.cursor_sync is True
    assert user.language == "ru"


def test_default_takes_the_user_settings(tmp_path):
    cfg = Config.default(_settings(tmp_path, FULL))
    assert cfg.repos["app"] == Path.home() / "git/app"
    assert cfg.upstream_sources == (("app", ("docs/TESTING.md", "specs/testing.md")),)
    assert cfg.cursor_repos == frozenset({"app"}) and cfg.cursor_sync is True
    assert cfg.language == "ru"


@pytest.mark.parametrize("text, says", [
    ("[repos\n", "config.toml"),
    ("colour = 1\n", "unknown keys: colour"),
    ('[repos]\napp = "git/app"\n', "repos.app"),
    ('[[upstream]]\nrepo = "app"\npaths = ["a"]\n', "upstream #1"),
    ('[repos]\napp = "/a"\n[[upstream]]\nrepo = "app"\npaths = []\n', "watches no paths"),
    ('[repos]\napp = "/a"\n[[upstream]]\nrepo = "app"\npaths = ["a"]\nbranch = "x"\n', "upstream #1"),
    ('cursor_repos = ["app"]\n', "cursor_repos names repos missing"),
    ('cursor_sync = "yes"\n', "cursor_sync"),
    ('language = "de"\n', "language must be"),
    ('keep_worktrees = "no"\n', "keep_worktrees"),
])
def test_an_unusable_config_file_is_refused_with_the_reason(tmp_path, text, says):
    with pytest.raises(ConfigError, match=says):
        load_user_settings(_settings(tmp_path, text), tmp_path)


def test_config_path_is_under_home_whatever_the_environment(tmp_path, monkeypatch):
    # launchd sets HOME and PATH only: an XDG_CONFIG_HOME of the shell must not split the two readers
    monkeypatch.setenv("XDG_CONFIG_HOME", "/x")
    assert user_config_path(tmp_path) == tmp_path / ".config/agent-reflect/config.toml"


def _broken_home(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    (tmp_path / ".config/agent-reflect").mkdir(parents=True)
    (tmp_path / ".config/agent-reflect/config.toml").write_text('language = "de"\n', encoding="utf-8")


def test_the_cli_stops_on_an_unusable_config_file(tmp_path, monkeypatch, capsys):
    from agent_reflect import cli, digest
    _broken_home(tmp_path, monkeypatch)
    sent = []
    monkeypatch.setattr(digest, "notify", lambda message, *a, **kw: sent.append(message))
    assert cli.main(["staged"]) == 2
    assert "language must be" in capsys.readouterr().err
    assert sent == []  # the user typed it and reads stderr


@pytest.mark.parametrize("cmd", ["reflect", "consolidate"])
def test_a_scheduled_run_on_an_unusable_config_file_notifies(tmp_path, monkeypatch, cmd):
    from agent_reflect import cli, digest
    _broken_home(tmp_path, monkeypatch)
    sent = []
    monkeypatch.setattr(digest, "notify", lambda message, *a, **kw: sent.append(message))
    assert cli.main([cmd]) == 2
    assert len(sent) == 1 and cmd in sent[0] and "language must be" in sent[0]


def test_a_list_where_a_repo_name_goes_is_refused_not_crashed(tmp_path):
    text = '[repos]\napp = "/a"\n[[upstream]]\nrepo = ["app"]\npaths = ["x"]\n'
    with pytest.raises(ConfigError, match="upstream #1"):
        load_user_settings(_settings(tmp_path, text), tmp_path)


def test_the_readme_example_is_a_config_the_loader_takes(tmp_path):
    readme = (Path(__file__).parents[1] / "README.md").read_text(encoding="utf-8")
    example = readme.split("```toml\n", 1)[1].split("```", 1)[0]
    user = load_user_settings(_settings(tmp_path, example), Path("/home/u"))
    assert set(user.repos) == {"app", "tools"} and user.upstream_sources[0][0] == "app"
    assert user.cursor_repos == frozenset({"app"}) and user.cursor_sync is False and user.language == "auto"


def test_the_cli_speaks_the_configured_language(tmp_path, monkeypatch, capsys):
    from agent_reflect import cli
    cfg = Config.under(tmp_path)
    for lang, says in (("en", "no lessons are waiting"), ("ru", "уроков в ожидании нет")):
        monkeypatch.setattr(Config, "default", classmethod(lambda cls, c=replace_language(cfg, lang): c))
        assert cli.main(["staged"]) == 0
        assert says in capsys.readouterr().out
        assert i18n.current() == lang


def replace_language(cfg: Config, lang: str) -> Config:
    from dataclasses import replace
    return replace(cfg, language=lang)


# -- which language "auto" means -----------------------------------------------------


def _prefs(tmp_path: Path, langs) -> Path:
    import plistlib
    path = tmp_path / "GlobalPreferences.plist"
    path.write_bytes(plistlib.dumps({"AppleLanguages": langs} if langs is not None else {}))
    return path


@pytest.mark.parametrize("langs, environ, want", [
    (["ru-RU", "en-US"], {"LANG": "en_US.UTF-8"}, "ru"),  # the system's first language wins over LANG
    (["en-US", "ru-GE"], {}, "en"),
    (["de-DE"], {"LANG": "ru_RU.UTF-8"}, "en"),  # an unsupported system language is English
    (None, {"LANG": "ru_RU.UTF-8"}, "ru"),  # no macOS preferences: the locale variables
    (None, {"LC_ALL": "C.UTF-8", "LANG": "ru_RU"}, "ru"),  # the C locale names no language
    (None, {}, "en"),
])
def test_auto_is_the_system_language_when_supported(tmp_path, langs, environ, want):
    assert i18n.resolve("auto", environ, _prefs(tmp_path, langs)) == want


def test_a_set_language_is_kept_whatever_the_system_says(tmp_path):
    assert i18n.resolve("ru", {}, _prefs(tmp_path, ["en-US"])) == "ru"
    assert i18n.resolve("en", {}, _prefs(tmp_path, ["ru-RU"])) == "en"


def test_unreadable_preferences_fall_back_to_the_locale(tmp_path):
    bad = tmp_path / "broken.plist"
    bad.write_bytes(b"not a plist")
    assert i18n.resolve("auto", {"LANG": "ru_RU.UTF-8"}, bad) == "ru"


def test_plural_and_decimal_follow_the_language():
    i18n.use("en")
    assert [i18n.count(n, ("change", "changes"), ("изменение", "изменения", "изменений")) for n in (1, 2, 5)] == [
        "1 change", "2 changes", "5 changes"]
    assert i18n.decimal(4.56) == "4.6"
    i18n.use("ru")
    assert [i18n.count(n, ("change", "changes"), ("изменение", "изменения", "изменений")) for n in (1, 2, 5, 11, 21)] == [
        "1 изменение", "2 изменения", "5 изменений", "11 изменений", "21 изменение"]
    assert i18n.decimal(4.56) == "4,6"
