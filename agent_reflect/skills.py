from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from agent_reflect.config import Config
from agent_reflect.projects import GIT_TIMEOUT_S

_SKILL_RES = (
    re.compile(r"^\.claude/skills/([^/]+)/SKILL\.md$"),
    re.compile(r"^plugins/[^/]+/skills/([^/]+)/SKILL\.md$"),
)


@dataclass(frozen=True)
class Target:
    key: str  # "personal:<name>" | "repo:<repo>:<skill>" | "repo:<repo>:CLAUDE.md"
    kind: str  # "personal" | "repo"
    path: Path  # file to edit (in repo: path inside the user's checkout, read-only)
    cursor_copy: Path | None
    repo: str | None  # key of cfg.repos
    rel_path: str | None  # path inside the repo
    description: str  # frontmatter description or "project rules"


class EditError(Exception):
    pass


def _is_plain_file(path: Path) -> bool:
    return path.is_file() and not path.is_symlink() and not path.parent.is_symlink()


def _frontmatter(text: str) -> list[str] | None:
    lines = text.splitlines()
    if not lines or lines[0] != "---":
        return None
    for i in range(1, len(lines)):
        if lines[i] == "---":
            return lines[1:i]
    return None


def _description(path: Path) -> str:
    try:
        fm = _frontmatter(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError):
        return ""
    for line in fm or []:
        if line.startswith("description:"):
            value = line[len("description:"):].strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                value = value[1:-1]
            return value
    return ""


def _cursor_copy(directory: Path, name: str) -> Path | None:
    candidate = directory / name / "SKILL.md"
    return candidate if _is_plain_file(candidate) else None


def _personal_targets(cfg: Config) -> list[Target]:
    out: list[Target] = []
    try:
        dirs = sorted(cfg.personal_skills_dir.iterdir())
    except OSError:
        return out
    for d in dirs:
        if d.name == "synced" or d.is_symlink() or not d.is_dir():
            continue
        skill = d / "SKILL.md"
        if not _is_plain_file(skill):
            continue
        out.append(
            Target(
                key=f"personal:{d.name}",
                kind="personal",
                path=skill,
                cursor_copy=_cursor_copy(cfg.cursor_skills_dir, d.name) if cfg.cursor_sync else None,
                repo=None,
                rel_path=None,
                description=_description(skill),
            )
        )
    return out


def _repo_targets(cfg: Config, name: str, root: Path, run) -> list[Target]:
    try:
        proc = run(
            ["git", "-C", str(root), "ls-files", "-z"],
            capture_output=True,
            text=True,
            check=False,
            timeout=GIT_TIMEOUT_S,
        )
        if proc.returncode != 0:
            return []
        tracked = sorted(p for p in proc.stdout.split("\0") if p)
    except Exception:  # a failed, missing or hung git (TimeoutExpired) offers no repo targets
        return []
    out: list[Target] = []
    for rel in tracked:
        path = root / rel
        if rel == "CLAUDE.md":
            if _is_plain_file(path):
                out.append(Target(f"repo:{name}:CLAUDE.md", "repo", path, None, name, rel, "project rules"))
            continue
        for rx in _SKILL_RES:
            m = rx.match(rel)
            if m and _is_plain_file(path):
                skill = m.group(1)
                cursor = _cursor_copy(root / ".cursor/skills", skill) if cfg.cursor_sync and name in cfg.cursor_repos else None
                out.append(Target(f"repo:{name}:{skill}", "repo", path, cursor, name, rel, _description(path)))
                break
    return out


def registry(cfg: Config, run=subprocess.run) -> list[Target]:
    targets = _personal_targets(cfg)
    for name, root in cfg.repos.items():
        targets.extend(_repo_targets(cfg, name, root, run))
    return targets


def apply_edit_text(text: str, old: str, new: str, *, skill: bool) -> str:
    if not old:
        raise EditError("empty 'old' text")
    count = text.count(old)
    if count != 1:
        raise EditError(f"'old' must occur exactly once, found {count}")
    result = text.replace(old, new, 1)
    if skill:
        fm = _frontmatter(result)
        if fm is None:
            raise EditError("result has no valid frontmatter")
        for field in ("name:", "description:"):
            if not any(line.startswith(field) for line in fm):
                raise EditError(f"frontmatter lost '{field}'")
    return result


def _read(path: Path) -> str:
    return path.read_bytes().decode("utf-8")


def write_atomic(path: Path, text: str) -> None:
    """Replace `path` with `text` atomically, keeping its mode."""
    mode = os.stat(path).st_mode & 0o7777
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(text.encode("utf-8"))
        os.chmod(tmp, mode)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _backup_dir(cfg: Config, run_id: str, key: str) -> Path:
    return cfg.backups_dir / run_id / key.replace(":", "__")


def backup_folder(cfg: Config, run_id: str, key: str) -> Path:
    """Where `backup` keeps the run's copy of one target's files."""
    return _backup_dir(cfg, run_id, key)


def backup_restored(cfg: Config, run_id: str, key: str) -> bool:
    """True when every file the run backed up for `key` is back to its backed-up bytes."""
    try:
        entries = json.loads((backup_folder(cfg, run_id, key) / "manifest.json").read_text(encoding="utf-8"))
        return bool(entries) and all(Path(e["src"]).read_bytes() == Path(e["saved"]).read_bytes()
                                     for e in entries)
    except (OSError, ValueError, KeyError, TypeError):
        return False


def backup(cfg: Config, run_id: str, target: Target) -> None:
    folder = _backup_dir(cfg, run_id, target.key)
    folder.mkdir(parents=True, exist_ok=True)
    manifest_path = folder / "manifest.json"
    manifest: list[dict] = json.loads(manifest_path.read_text()) if manifest_path.exists() else []
    seen = {m["src"] for m in manifest}
    for i, src in enumerate(p for p in (target.path, target.cursor_copy) if p is not None):
        if str(src) in seen:
            continue  # keep the first (original) copy of this run
        saved = folder / f"{len(manifest)}_{i}_{src.parent.name}_{src.name}"
        shutil.copy2(src, saved)
        manifest.append({"src": str(src), "saved": str(saved)})
    tmp = manifest_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    os.replace(tmp, manifest_path)


def rollback(cfg: Config, run_id: str) -> list[Path]:
    run_dir = cfg.backups_dir / run_id
    restored: list[Path] = []
    if not run_dir.is_dir():
        return restored
    for manifest_path in sorted(run_dir.glob("*/manifest.json")):
        for m in json.loads(manifest_path.read_text()):
            src, saved = Path(m["src"]), Path(m["saved"])
            _write_atomic_bytes(src, saved.read_bytes(), saved)
            restored.append(src)
    return restored


def _write_atomic_bytes(dest: Path, data: bytes, mode_from: Path) -> None:
    mode = os.stat(dest).st_mode & 0o7777 if dest.exists() else os.stat(mode_from).st_mode & 0o7777
    fd, tmp = tempfile.mkstemp(dir=dest.parent, prefix=dest.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        os.chmod(tmp, mode)
        os.replace(tmp, dest)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def list_backups(cfg: Config) -> list[str]:
    try:
        # backups/memory holds the memory archives (memory_archive.py), not a skill run
        return sorted(p.name for p in cfg.backups_dir.iterdir() if p.is_dir() and p.name != "memory")
    except OSError:
        return []


def apply_personal(cfg: Config, run_id: str, target: Target, old: str, new: str) -> list[str]:
    if target.kind != "personal":
        raise EditError(f"refusing to edit non-personal target {target.key}")
    skill = target.path.name == "SKILL.md"
    new_text = apply_edit_text(_read(target.path), old, new, skill=skill)
    cursor_text: str | None = None
    notes: list[str] = []
    if target.cursor_copy is not None:
        try:
            cursor_text = apply_edit_text(_read(target.cursor_copy), old, new, skill=skill)
        except (EditError, OSError, UnicodeDecodeError):
            notes.append(f"cursor copy diverged: {target.cursor_copy}")
    backup(cfg, run_id, target)
    write_atomic(target.path, new_text)
    if cursor_text is not None:
        try:
            write_atomic(target.cursor_copy, cursor_text)
        except OSError as exc:  # the skill itself is edited: report the copy, don't fail the edit
            notes.append(f"cursor copy not written: {target.cursor_copy}: {exc}")
    return notes
