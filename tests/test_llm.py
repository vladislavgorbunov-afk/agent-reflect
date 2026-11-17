from __future__ import annotations

import json
import subprocess

import pytest

from agent_reflect.config import Config
from agent_reflect.llm import LlmError, build_argv, ensure_isolation_files, run_structured

SCHEMA = {"type": "object", "properties": {"a": {"type": "string"}}}


def completed(stdout="", returncode=0, stderr=""):
    return subprocess.CompletedProcess([], returncode, stdout, stderr)


def fake(result):
    calls = []

    def runner(argv, stdin, cwd, timeout):
        calls.append({"argv": argv, "stdin": stdin, "cwd": cwd, "timeout": timeout})
        if isinstance(result, Exception):
            raise result
        return result

    runner.calls = calls
    return runner


def test_argv_isolation_flags(tmp_path):
    cfg = Config.under(tmp_path)
    argv = build_argv(cfg, "m1", SCHEMA, "medium")
    assert argv[0] == str(cfg.claude_bin)
    assert "-p" in argv
    pairs = list(zip(argv, argv[1:]))
    assert ("--model", "m1") in pairs
    assert ("--output-format", "json") in pairs
    assert ("--tools", "") in pairs
    assert ("--settings", str(cfg.isolation_settings)) in pairs
    assert ("--json-schema", json.dumps(SCHEMA, ensure_ascii=False)) in pairs
    assert ("--effort", "medium") in pairs
    for flag in ("--strict-mcp-config", "--disable-slash-commands", "--no-session-persistence"):
        assert flag in argv
    assert "--bare" not in argv


def test_isolation_settings_disable_hooks(tmp_path):
    cfg = Config.under(tmp_path)
    ensure_isolation_files(cfg)
    assert json.loads(cfg.isolation_settings.read_text()) == {"disableAllHooks": True,
                                                              "autoMemoryEnabled": False}
    assert cfg.sandbox_dir.is_dir()


def test_parses_structured_output(tmp_path):
    cfg = Config.under(tmp_path)
    out = json.dumps({"is_error": False, "result": "", "structured_output": {"a": "x"}})
    assert run_structured(cfg, "m", "p", SCHEMA, fake(completed(out)), effort="high") == {"a": "x"}


def test_falls_back_to_result_string(tmp_path):
    cfg = Config.under(tmp_path)
    out = json.dumps({"is_error": False, "result": json.dumps({"a": "y"})})
    assert run_structured(cfg, "m", "p", SCHEMA, fake(completed(out)), effort="high") == {"a": "y"}


@pytest.mark.parametrize(
    "result",
    [
        completed("", returncode=1, stderr="boom"),
        completed(json.dumps({"is_error": True, "result": "bad"})),
        completed("not json"),
        completed(json.dumps({"is_error": False, "result": "not json either"})),
        subprocess.TimeoutExpired(["claude"], 5),
    ],
)
def test_error_paths_raise(tmp_path, result):
    cfg = Config.under(tmp_path)
    with pytest.raises(LlmError):
        run_structured(cfg, "m", "SECRET PROMPT", SCHEMA, fake(result), effort="high")


def test_error_message_has_no_prompt_and_truncates_stderr(tmp_path):
    cfg = Config.under(tmp_path)
    with pytest.raises(LlmError) as e:
        run_structured(cfg, "m", "SECRET PROMPT", SCHEMA, fake(completed("", 3, "x" * 5000)),
                       effort="high")
    msg = str(e.value)
    assert "SECRET PROMPT" not in msg and "3" in msg and len(msg) < 700


def test_runs_in_sandbox_with_prompt_on_stdin(tmp_path):
    cfg = Config.under(tmp_path)
    out = json.dumps({"is_error": False, "structured_output": {}})
    runner = fake(completed(out))
    run_structured(cfg, "m", "hello", SCHEMA, runner, effort="high")
    call = runner.calls[0]
    assert ("--effort", "high") in list(zip(call["argv"], call["argv"][1:]))
    assert call["cwd"] == cfg.sandbox_dir
    assert call["stdin"] == "hello"
    assert call["timeout"] == cfg.llm_timeout_s


def _exit_message(tmp_path, proc) -> str:
    with pytest.raises(LlmError) as e:
        run_structured(Config.under(tmp_path), "m", "SECRET PROMPT", SCHEMA, fake(proc), effort="high")
    return str(e.value)


def test_nonzero_exit_reports_stderr_first(tmp_path):
    out = json.dumps({"is_error": True, "result": "from stdout"})
    msg = _exit_message(tmp_path, completed(out, 1, "  stderr says why  \n"))
    assert msg == "claude exited with code 1: stderr says why"


@pytest.mark.parametrize("envelope, reason", [
    ({"type": "result", "is_error": True, "result": "Credit balance is too low"}, "Credit balance is too low"),
    ({"type": "result", "is_error": True, "result": "", "error": "rate_limit_error"}, "rate_limit_error"),
    ({"is_error": True, "error": {"type": "overloaded_error", "message": "Overloaded"}}, "Overloaded"),
])
def test_nonzero_exit_without_stderr_reads_the_json_envelope(tmp_path, envelope, reason):
    msg = _exit_message(tmp_path, completed(json.dumps(envelope), 1, ""))
    assert msg.startswith("claude exited with code 1: ") and reason in msg


def test_nonzero_exit_with_plain_stdout_reports_its_start(tmp_path):
    msg = _exit_message(tmp_path, completed("Not logged in · Please run /login " + "x" * 1000, 1, ""))
    assert msg.startswith("claude exited with code 1: Not logged in · Please run /login")
    assert len(msg) <= len("claude exited with code 1: ") + 200


def test_nonzero_exit_with_no_output_says_so(tmp_path):
    msg = _exit_message(tmp_path, completed("", 2, ""))
    assert msg.startswith("claude exited with code 2") and "SECRET PROMPT" not in msg


@pytest.mark.parametrize("stdout, stderr", [
    ("", "token ghp_" + "a" * 36 + " " + "y" * 2000),
    (json.dumps({"is_error": True, "result": "key sk-ant-" + "b" * 30 + " " + "z" * 2000}), ""),
    ("Bearer abcdefghijklmnop " + "w" * 2000, ""),
], ids=["stderr", "envelope", "plain-stdout"])
def test_exit_message_is_scrubbed_and_bounded(tmp_path, stdout, stderr):
    msg = _exit_message(tmp_path, completed(stdout, 1, stderr))
    assert len(msg) <= 500
    assert "ghp_" not in msg and "sk-ant-" not in msg and "abcdefghijklmnop" not in msg


def test_a_secret_straddling_the_stdout_cut_is_scrubbed(tmp_path):
    stdout = "x" * 190 + " ghp_" + "c" * 36
    msg = _exit_message(tmp_path, completed(stdout, 1, ""))
    assert "ghp_" not in msg


@pytest.mark.parametrize("errors, expected", [
    (["Prompt is too long", "second"], "error_during_execution: Prompt is too long; second"),
    ("Prompt is too long", "error_during_execution: Prompt is too long"),
])
def test_nonzero_exit_reads_the_envelope_subtype_and_errors(tmp_path, errors, expected):
    envelope = {"type": "result", "subtype": "error_during_execution", "is_error": True, "errors": errors}
    msg = _exit_message(tmp_path, completed(json.dumps(envelope), 1, ""))
    assert msg == f"claude exited with code 1: {expected}"


def test_envelope_errors_are_scrubbed_and_bounded(tmp_path):
    envelope = {"type": "result", "subtype": "error_during_execution", "is_error": True,
                "errors": ["token ghp_" + "d" * 36, "x" * 3000]}
    msg = _exit_message(tmp_path, completed(json.dumps(envelope), 1, ""))
    assert len(msg) <= 500 and "ghp_" not in msg and "error_during_execution" in msg
