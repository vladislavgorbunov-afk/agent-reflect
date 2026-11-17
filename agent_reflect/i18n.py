"""The language of what agent-reflect tells the user: reports, the digest, notifications, CLI output.

One run speaks one language, set once at start (`use`). Each phrase is written at its call site in
both languages, `t("night of 3 May", "ночь на 3 мая")`, so a module needs no catalog of keys.
What the model or the agent reads (prompts, lesson files, the guard hook) is not covered here.
"""
from __future__ import annotations

import os
import plistlib
from collections.abc import Mapping
from pathlib import Path

LANGUAGES = ("en", "ru")
DEFAULT = "en"


def global_prefs() -> Path:
    """Where macOS keeps the user's preferred languages, first the one the system speaks."""
    return Path.home() / "Library/Preferences/.GlobalPreferences.plist"


_current = DEFAULT


def use(lang: str) -> None:
    global _current
    if lang not in LANGUAGES:
        raise ValueError(f"unsupported language: {lang}")
    _current = lang


def current() -> str:
    return _current


def is_ru() -> bool:
    return _current == "ru"


def t(en: str, ru: str) -> str:
    return ru if _current == "ru" else en


def quoted(text: str) -> str:
    return f"«{text}»" if _current == "ru" else f"“{text}”"


def plural(n: int, en: tuple[str, str], ru: tuple[str, str, str]) -> str:
    """The form of a noun that goes with n: 1 change / 2 changes; 1 изменение, 2 изменения, 5 изменений."""
    if _current == "ru":
        one, few, many = ru
        if n % 100 in range(11, 15):
            return many
        return one if n % 10 == 1 else few if n % 10 in (2, 3, 4) else many
    return en[0] if n == 1 else en[1]


def count(n: int, en: tuple[str, str], ru: tuple[str, str, str]) -> str:
    return f"{n} {plural(n, en, ru)}"


def times(n: int) -> str:
    """How often: 1 time, 3 times; 1 раз, 3 раза."""
    return count(n, ("time", "times"), ("раз", "раза", "раз"))


def decimal(x: float, digits: int = 1) -> str:
    """A fraction as the language writes it: 4.6 / 4,6."""
    text = f"{x:.{digits}f}"
    return text.replace(".", ",") if _current == "ru" else text


def _language_of(tag: object) -> str | None:
    if not isinstance(tag, str) or not tag.strip():
        return None
    return tag.strip().replace("-", "_").split("_")[0].split(".")[0].lower() or None


def system_language(environ: Mapping[str, str] | None = None, prefs: Path | None = None) -> str | None:
    """The system's first language (macOS preferences, else the locale variables), whatever it is."""
    environ = os.environ if environ is None else environ
    prefs = global_prefs() if prefs is None else prefs
    try:
        langs = plistlib.loads(prefs.read_bytes()).get("AppleLanguages")
    except (OSError, plistlib.InvalidFileException, ValueError, AttributeError):
        langs = None
    if isinstance(langs, list) and langs:
        found = _language_of(langs[0])
        if found:
            return found
    for var in ("LC_ALL", "LC_MESSAGES", "LANG"):
        found = _language_of(environ.get(var))
        if found and found not in ("c", "posix"):
            return found
    return None


def resolve(setting: str, environ: Mapping[str, str] | None = None, prefs: Path | None = None) -> str:
    """A configured language as is; "auto" is the system's language when supported, else English."""
    if setting != "auto":
        return setting
    found = system_language(environ, prefs)
    return found if found in LANGUAGES else DEFAULT
