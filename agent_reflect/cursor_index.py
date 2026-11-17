from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

from agent_reflect.memory import _atomic_write

if TYPE_CHECKING:
    from agent_reflect.config import Config

MDC_RELATIVE = Path(".cursor/rules/claude-memory.mdc")
ENTRY_PREFIX = "- ["
MEMORY_LINK_RE = re.compile(r"\]\(([A-Za-z0-9_.\-]+\.md)\)")
CURSOR_LINK_TEMPLATE = r"](.cursor/memory/\1)"


def sync(cfg: "Config", project_root: Path, memory_dir: Path) -> bool:
    """Regenerate the Cursor index only while the Cursor copies are kept (cfg.cursor_sync)."""
    return cfg.cursor_sync and regenerate(project_root, memory_dir)


def regenerate(project_root: Path, memory_dir: Path) -> bool:
    """Rewrite Cursor's static memory index from MEMORY.md, keeping the .mdc header.

    Returns False, touching nothing, when the .mdc is absent or MEMORY.md cannot be
    read as UTF-8. Never raises.
    """
    mdc = Path(project_root) / MDC_RELATIVE
    try:
        if not mdc.is_file():
            return False
        header = _header(mdc.read_bytes())
        index = (Path(memory_dir) / "MEMORY.md").read_bytes().decode("utf-8")
        body = "".join(_index_entries(index))
        if header and not header.endswith((b"\n", b"\r")):
            header += b"\n"
        _atomic_write(mdc, header + body.encode("utf-8"))
    except (OSError, ValueError):
        return False
    return True


def _header(raw: bytes) -> bytes:
    head = []
    for line in raw.splitlines(keepends=True):
        if line.startswith(ENTRY_PREFIX.encode("utf-8")):
            break
        head.append(line)
    return b"".join(head)


def _index_entries(text: str) -> list[str]:
    entries = []
    for line in text.split("\n"):
        line = line.rstrip("\r")
        if line.startswith(ENTRY_PREFIX):
            entries.append(MEMORY_LINK_RE.sub(CURSOR_LINK_TEMPLATE, line) + "\n")
    return entries
