from __future__ import annotations

import re

_REDACTED = "[REDACTED]"

# (pattern, replacement). Replacement templates keep the context group
# (the keyword / the Bearer word) and replace only the secret value.
_RULES: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"gh[pousr]_[A-Za-z0-9]{20,}"), _REDACTED),
    (re.compile(r"github_pat_[A-Za-z0-9_]{20,}"), _REDACTED),
    (re.compile(r"\bglpat-[A-Za-z0-9_-]{20,}"), _REDACTED),
    (re.compile(r"\bsk-ant-[A-Za-z0-9_-]{10,}"), _REDACTED),
    (re.compile(r"\bsk-[A-Za-z0-9_-]{20,}"), _REDACTED),
    (re.compile(r"AKIA[0-9A-Z]{16}"), _REDACTED),
    (re.compile(r"xox[abposr]-[A-Za-z0-9-]{10,}"), _REDACTED),
    (re.compile(r"eyJ[\w-]{10,}\.[\w-]{10,}\.[\w-]{10,}"), _REDACTED),
    (re.compile(r"(?i)\b(bearer)\s+[A-Za-z0-9._~+/=-]{8,}"), r"\1 " + _REDACTED),
    (
        re.compile(
            r"(?i)((?:token|secret|api[_-]?key)(?:_[A-Za-z0-9]*)?"
            r"[\"']?[ \t]*[:=][ \t]*([\"'])?)"
            r"(?(2)(?:(?!\2)[^\n])*|[^\s\"']+)"
        ),
        r"\1" + _REDACTED,
    ),
    (
        re.compile(
            r"(?i)((?:password|passwd|пароль)(?:_[A-Za-z0-9]*)?"
            r"(?:[\"']?[ \t]*[:=][ \t]*|[ \t]+)([\"'])?)"
            r"(?(2)(?:(?!\2)[^\n])*|[^\s\"']+)"
        ),
        r"\1" + _REDACTED,
    ),
    (re.compile(r"(?i)\b(otp|code|код)\b(\D{0,12})(\d{4,8})\b"), r"\1\2" + _REDACTED),
)


def scrub(text: str) -> str:
    for pattern, replacement in _RULES:
        text = pattern.sub(replacement, text)
    return text
