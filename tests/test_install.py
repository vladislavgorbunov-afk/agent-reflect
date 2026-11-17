from __future__ import annotations

import os
import plistlib
from types import SimpleNamespace

from conftest import write_jsonl

from agent_reflect import install as inst
from agent_reflect.config import Config
from agent_reflect.ledger import load_state


class FakeRun:
    def __init__(self):
        self.calls = []

    def __call__(self, argv, **kw):
        self.calls.append(argv)
        return SimpleNamespace(returncode=0, stdout="", stderr="")


def test_plist_contents(tmp_path):
    cfg = Config.under(tmp_path)
    for cmd in ("reflect", "consolidate"):
        d = inst.plist_dict(cfg, cmd)
        assert d["Label"] == inst.LABELS[cmd]
        assert d["ProgramArguments"] == [str(cfg.venv_bin), cmd]
        assert d["RunAtLoad"] is True
        assert d["EnvironmentVariables"]["PATH"].startswith("/opt/homebrew/bin")
        assert d["EnvironmentVariables"]["HOME"] == str(cfg.home)
        assert d["WorkingDirectory"] == str(cfg.repo_dir)
        assert d["StandardOutPath"].startswith(str(cfg.logs_dir))
        assert d["StandardErrorPath"].startswith(str(cfg.logs_dir))


def test_install_bootstraps_both_and_snapshots(tmp_path):
    cfg = Config.under(tmp_path)
    t = cfg.projects_dir / "-proj" / "s1.jsonl"
    write_jsonl(t, [{"type": "user"}])
    run = FakeRun()
    paths = inst.install(cfg, run=run)
    assert [p.name for p in paths] == [f"{l}.plist" for l in inst.LABELS.values()]
    for cmd, p in zip(inst.LABELS, paths):
        assert p.parent == cfg.launch_agents_dir
        assert plistlib.loads(p.read_bytes())["Label"] == inst.LABELS[cmd]
    uid = os.getuid()
    for label, p in zip(inst.LABELS.values(), paths):
        assert ["launchctl", "bootout", f"gui/{uid}/{label}"] in run.calls
        assert ["launchctl", "bootstrap", f"gui/{uid}", str(p)] in run.calls
    assert cfg.logs_dir.is_dir()
    state = load_state(cfg.state_file)
    assert state.installed_at
    assert state.sessions[str(t)]["offset"] == t.stat().st_size


def test_uninstall_boots_out_and_removes(tmp_path):
    cfg = Config.under(tmp_path)
    run = FakeRun()
    paths = inst.install(cfg, run=run)
    run.calls.clear()
    inst.uninstall(cfg, run=run)
    assert not any(p.exists() for p in paths)
    assert len([c for c in run.calls if c[1] == "bootout"]) == 2
    inst.uninstall(cfg, run=run)  # idempotent


def test_reflect_runs_nightly_and_consolidate_daily_at_0430(tmp_path):
    cfg = Config.under(tmp_path)
    reflect = inst.plist_dict(cfg, "reflect")
    assert reflect["StartCalendarInterval"] == {"Hour": 3, "Minute": 30} and "StartInterval" not in reflect
    consolidate = inst.plist_dict(cfg, "consolidate")
    assert consolidate["StartCalendarInterval"] == {"Hour": 4, "Minute": 30}
    assert "StartInterval" not in consolidate
    assert consolidate["RunAtLoad"] is True
