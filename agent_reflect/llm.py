from __future__ import annotations

import json
import subprocess
import threading
from collections import Counter
from pathlib import Path
from typing import Callable

from agent_reflect.config import Config
from agent_reflect.scrub import scrub

Runner = Callable[[list[str], str, Path, int], subprocess.CompletedProcess]  # (argv, stdin, cwd, timeout_s)

_STDERR_LIMIT = 500
_STDOUT_HEAD = 200  # of stdout that is neither stderr nor a JSON envelope


# What the calls since the last `take_spent` cost: their number, the tokens the CLI reported and its
# price for them in dollars (what the API would charge; a subscription pays nothing per call).
SPENT: Counter = Counter()
_SPENT_LOCK = threading.Lock()  # the baseline asks from several threads
_TOKEN_FIELDS = ("input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens")


class LlmError(Exception):
    pass


def _meter(envelope: dict) -> None:
    """Add one answered call to SPENT; an envelope without usage counts as a call that reported nothing."""
    usage = envelope.get("usage")
    cost = envelope.get("total_cost_usd")
    with _SPENT_LOCK:
        SPENT["calls"] += 1
        for name in _TOKEN_FIELDS if isinstance(usage, dict) else ():
            value = usage.get(name)
            if isinstance(value, int) and not isinstance(value, bool):
                SPENT[name] += value
        if isinstance(cost, (int, float)) and not isinstance(cost, bool):
            SPENT["cost_usd"] += cost


def take_spent() -> dict:
    """What was spent since the last take, and start counting again."""
    with _SPENT_LOCK:
        spent = {name: value for name, value in SPENT.items() if value}
        SPENT.clear()
    return spent


def default_runner(argv: list[str], stdin: str, cwd: Path, timeout: int) -> subprocess.CompletedProcess:
    return subprocess.run(argv, input=stdin, cwd=cwd, timeout=timeout, capture_output=True, text=True)


def ensure_isolation_files(cfg: Config) -> None:
    cfg.state_dir.mkdir(parents=True, exist_ok=True)
    cfg.sandbox_dir.mkdir(parents=True, exist_ok=True)
    cfg.isolation_settings.write_text(json.dumps({"disableAllHooks": True, "autoMemoryEnabled": False}),
                                      encoding="utf-8")


def build_argv(cfg: Config, model: str, schema: dict, effort: str) -> list[str]:
    return [
        str(cfg.claude_bin),
        "-p",
        "--model", model,
        "--effort", effort,
        "--output-format", "json",
        "--json-schema", json.dumps(schema, ensure_ascii=False),
        "--tools", "",
        "--strict-mcp-config",
        "--disable-slash-commands",
        "--no-session-persistence",
        "--settings", str(cfg.isolation_settings),
    ]


def _envelope_reason(stdout: str) -> str:
    """The `result` or `error` of the JSON envelope claude printed, if it printed one."""
    try:
        envelope = json.loads(stdout)
    except (json.JSONDecodeError, TypeError):
        return ""
    if not isinstance(envelope, dict):
        return ""
    for name in ("result", "error"):
        value = envelope.get(name)
        if isinstance(value, dict):
            value = value.get("message") or json.dumps(value, ensure_ascii=False)
        if isinstance(value, str) and value.strip():
            return value.strip()
    errors = envelope.get("errors")
    if isinstance(errors, list):
        errors = "; ".join(e if isinstance(e, str) else json.dumps(e, ensure_ascii=False) for e in errors)
    errors = errors.strip() if isinstance(errors, str) else ""
    subtype = envelope.get("subtype") if isinstance(envelope.get("subtype"), str) else ""
    return ": ".join(part for part in (subtype.strip(), errors) if part)


def _exit_reason(stderr: str | None, stdout: str | None) -> str:
    """Why claude exited non-zero: stderr, else the envelope's result/error, else the start of stdout.

    Scrubbed first, then clipped, so a secret cut at the limit cannot slip through. The
    prompt goes in on stdin and is never part of it.
    """
    reason = (stderr or "").strip() or _envelope_reason(stdout or "")
    if not reason:
        reason = scrub((stdout or "").strip())[:_STDOUT_HEAD]
    return " ".join(scrub(reason).split())


def run_structured(cfg: Config, model: str, prompt: str, schema: dict, runner: Runner = default_runner,
                   *, effort: str) -> dict:
    ensure_isolation_files(cfg)
    argv = build_argv(cfg, model, schema, effort)
    try:
        proc = runner(argv, prompt, cfg.sandbox_dir, cfg.llm_timeout_s)
    except subprocess.TimeoutExpired:
        raise LlmError(f"claude timed out after {cfg.llm_timeout_s}s") from None
    except OSError as e:
        raise LlmError(f"claude could not be started: {e}") from None
    if proc.returncode != 0:
        head = f"claude exited with code {proc.returncode}"
        reason = _exit_reason(proc.stderr, proc.stdout)
        if not reason:
            raise LlmError(f"{head}, no output")
        raise LlmError(f"{head}: {reason}"[:_STDERR_LIMIT])
    try:
        envelope = json.loads(proc.stdout)
    except json.JSONDecodeError:
        raise LlmError("claude output is not valid JSON") from None
    if not isinstance(envelope, dict):
        raise LlmError("claude output is not a JSON object")
    if envelope.get("is_error"):
        raise LlmError(f"claude reported an error: {str(envelope.get('result', ''))[:_STDERR_LIMIT]}")
    _meter(envelope)
    structured = envelope.get("structured_output")
    if isinstance(structured, dict):
        return structured
    try:
        parsed = json.loads(envelope.get("result") or "")
    except (json.JSONDecodeError, TypeError):
        raise LlmError("claude result has no structured_output and is not JSON") from None
    if not isinstance(parsed, dict):
        raise LlmError("claude result is not a JSON object")
    return parsed
