"""launchd agents: the nightly reflect and the consolidate (launched nightly, weekly work)."""
from __future__ import annotations

import os
import plistlib
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from .config import Config
from .ledger import load_state, run_lock, save_state, snapshot_baseline
from .transcript import list_transcripts

LABELS = {"reflect": "local.agent-reflect.reflect", "consolidate": "local.agent-reflect.consolidate"}
# consolidate is launched daily (and at login); is_due gates the real work to weekly, so
# neither a closed lid nor a reflect holding the lock pushes it back a week, and an aborted
# run retries the next night instead of every hour.
SCHEDULE = {"reflect": {"StartCalendarInterval": {"Hour": 3, "Minute": 30}},
            "consolidate": {"StartCalendarInterval": {"Hour": 4, "Minute": 30}}}


def plist_dict(cfg: Config, cmd: str) -> dict:
    return {
        "Label": LABELS[cmd],
        "ProgramArguments": [str(cfg.venv_bin), cmd],
        **SCHEDULE[cmd],
        "RunAtLoad": True,
        "StandardOutPath": str(cfg.logs_dir / f"{cmd}.out.log"),
        "StandardErrorPath": str(cfg.logs_dir / f"{cmd}.err.log"),
        "EnvironmentVariables": {"PATH": "/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin",
                                 "HOME": str(cfg.home)},
        "WorkingDirectory": str(cfg.repo_dir),
    }


def _plist_path(cfg: Config, cmd: str) -> Path:
    return cfg.launch_agents_dir / f"{LABELS[cmd]}.plist"


def _snapshot_if_new(cfg: Config) -> None:
    with run_lock(cfg.lock_file) as acquired:
        if not acquired:
            return
        state = load_state(cfg.state_file)
        if state.installed_at:
            return
        now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        transcripts = list_transcripts(cfg.projects_dir, cfg.own_project_prefixes)
        snapshot_baseline(state, transcripts, now_iso)
        save_state(state, cfg.state_file)


def install(cfg: Config, run=subprocess.run) -> list[Path]:
    cfg.logs_dir.mkdir(parents=True, exist_ok=True)
    cfg.launch_agents_dir.mkdir(parents=True, exist_ok=True)
    _snapshot_if_new(cfg)  # before bootstrap: RunAtLoad starts a run at once
    domain = f"gui/{os.getuid()}"
    paths: list[Path] = []
    for cmd, label in LABELS.items():
        path = _plist_path(cfg, cmd)
        path.write_bytes(plistlib.dumps(plist_dict(cfg, cmd)))
        run(["launchctl", "bootout", f"{domain}/{label}"], check=False, capture_output=True, text=True)
        run(["launchctl", "bootstrap", domain, str(path)], check=True, capture_output=True, text=True)
        paths.append(path)
    return paths


def uninstall(cfg: Config, run=subprocess.run) -> None:
    domain = f"gui/{os.getuid()}"
    for cmd, label in LABELS.items():
        run(["launchctl", "bootout", f"{domain}/{label}"], check=False, capture_output=True, text=True)
        _plist_path(cfg, cmd).unlink(missing_ok=True)
