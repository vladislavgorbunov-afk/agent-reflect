"""Shared helpers for the model prompts and for text that comes back from the model."""

from __future__ import annotations

import re
import secrets
from importlib import resources
from string import Template
from typing import Iterable

from agent_reflect.scrub import scrub

ERROR_MAX = 300
_ATTRIBUTION_RE = re.compile(r"\A<!--.*?-->\n")


def nonce() -> str:
    """A fresh id for one prompt's data blocks: only tags carrying it close a block."""
    return secrets.token_hex(4)


def load_template(name: str) -> Template:
    """prompts/<name> as a Template, without a leading attribution comment."""
    raw = resources.files("agent_reflect").joinpath("prompts", name).read_text(encoding="utf-8")
    return Template(_ATTRIBUTION_RE.sub("", raw, count=1))


class DataTags:
    """The data-block tags of one prompt, and how to keep data from faking them."""

    def __init__(self, tags: Iterable[str]):
        self._re = re.compile(r"<(/?)(" + "|".join(tags) + r")\b", re.IGNORECASE)

    def defuse(self, text: str) -> str:
        """Defuse anything inside a data block that looks like one of the block tags."""
        return self._re.sub(r"&lt;\1\2", text)

    def one_per_line(self, items: Iterable[str]) -> str:
        lines = [line for line in (self.defuse(squash(item)) for item in items) if line]
        return "\n".join(lines) if lines else "—"


def squash(text: str) -> str:
    return " ".join(text.split())


def model_text(value) -> str:
    """A string the model returned, safe to store and show: scrubbed, no lone surrogates."""
    if not isinstance(value, str):
        return ""
    value = value.encode("utf-8", "replace").decode("utf-8")  # JSON can carry lone surrogates
    return scrub(value).strip()


def error_text(exc: BaseException) -> str:
    return scrub(str(exc))[:ERROR_MAX]
