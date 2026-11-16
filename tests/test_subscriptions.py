"""Tests for subscription-backed LM providers (`claude-code`, `codex`, `cursor`, `subscription`)."""

from __future__ import annotations

import json
import os
import stat
import sys
from pathlib import Path

import dspy
import pytest

import tracepress as tp
from tracepress import lm as lm_config
from tracepress import subscriptions


def _write_executable(path: Path, body: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    script = f"#!{sys.executable}\n{body}"
    path.write_text(script, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return path


@pytest.fixture(autouse=True)
def isolated_subscriptions(monkeypatch, tmp_path):
    """Isolate HOME, PATH, env vars, LM cache and DSPy disk cache per test."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(home / ".claude"))
    monkeypatch.setenv("CODEX_HOME", str(home / ".codex"))
    monkeypatch.setenv("CURSOR_CONFIG_DIR", str(home / ".cursor"))
    monkeypatch.setenv("PATH", str(tmp_path / "bin"))
    for var in (
        "TRACEPRESS_LM",
        "CLAUDECODE",
        "CLAUDE_CODE_OAUTH_TOKEN",
        "CODEX_THREAD_ID",
        "CODEX_CI",
        "CURSOR_AGENT",
        "CURSOR_API_KEY",
        "ANTHROPIC_API_KEY",
        "OPENAI_API_KEY",
        "CODEX_API_KEY",
    ):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(lm_config, "_env_loaded", False)
    monkeypatch.setattr(lm_config, "_lm_cache", {})
    prev_lm = tp.configure().lm
    prev_adapter = tp.configure().adapter
    tp.configure(lm=None, adapter=dspy.JSONAdapter())
    dspy.configure_cache(
        enable_disk_cache=True,
        enable_memory_cache=False,
        disk_cache_dir=str(tmp_path / "dspy-cache"),
    )
    yield tmp_path
    tp.configure(lm=prev_lm, adapter=prev_adapter)


def test_claude_cli_end_to_end_and_disk_cache(isolated_subscriptions, monkeypatch):
    log_file = isolated_subscriptions / "claude_calls.jsonl"
    _write_executable(
        isolated_subscriptions / "bin" / "claude",
        f"""
import json, os, sys

stdin_text = sys.stdin.read()
rec = {{"argv": sys.argv[1:], "stdin": stdin_text, "has_claudecode": "CLAUDECODE" in os.environ, "has_api_key": "ANTHROPIC_API_KEY" in os.environ}}
with open({str(log_file)!r}, "a", encoding="utf-8") as f:
    f.write(json.dumps(rec) + "\\n")

print(json.dumps({{
    "type": "result",
    "subtype": "success",
    "is_error": False,
    "session_id": "sess-1",
    "result": json.dumps({{"summary": "fixed build", "ok": True}}),
    "usage": {{"input_tokens": 20, "cache_read_input_tokens": 5, "cache_creation_input_tokens": 0, "output_tokens": 10}},
    "total_cost_usd": 0.001,
}}))
""",
    )
    # Parent session vars that must be stripped before invoking `claude -p`.
    monkeypatch.setenv("CLAUDECODE", "1")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-should-not-leak")

    tp.configure(lm="claude-code:sonnet")
    rows = (
        tp.from_rows([tp.Row((("step", 0),), {"text": "make test failed"})])
        .lm_map("text -> summary, ok: bool", on_error="raise")
        .collect()
    )
    assert rows[0].values == {"text": "make test failed", "summary": "fixed build", "ok": True}
    stats1 = tp.last_run()
    assert (stats1.calls, stats1.cached_calls, stats1.prompt_tokens, stats1.completion_tokens) == (1, 0, 25, 10)
    assert stats1.unpriced_calls == 1 and stats1.cost_usd == 0.0

    logged = [json.loads(line) for line in log_file.read_text().splitlines()]
    assert len(logged) == 1
    assert not logged[0]["has_claudecode"] and not logged[0]["has_api_key"]
    assert "--tools" in logged[0]["argv"] and "--no-session-persistence" in logged[0]["argv"]
    assert "make test failed" in logged[0]["stdin"]

    # Second identical run hits DSPy's disk cache without spawning `claude` again.
    rows2 = (
        tp.from_rows([tp.Row((("step", 0),), {"text": "make test failed"})])
        .lm_map("text -> summary, ok: bool", on_error="raise")
        .collect()
    )
    assert rows2[0].values == rows[0].values
    assert tp.last_run().cached_calls == 1
    assert len(log_file.read_text().splitlines()) == 1


def test_codex_cli_end_to_end_and_env_stripping(isolated_subscriptions, monkeypatch):
    log_file = isolated_subscriptions / "codex_calls.jsonl"
    _write_executable(
        isolated_subscriptions / "bin" / "codex",
        f"""
import json, os, sys
from pathlib import Path

argv = sys.argv[1:]
stdin_text = sys.stdin.read()
out_idx = argv.index("-o") + 1
out_path = Path(argv[out_idx])
answer = json.dumps({{"verdict": "codex-ok"}})
out_path.write_text(answer, encoding="utf-8")

with open({str(log_file)!r}, "a", encoding="utf-8") as f:
    f.write(json.dumps({{"argv": argv, "has_openai_key": "OPENAI_API_KEY" in os.environ, "stdin": stdin_text}}) + "\\n")

print(json.dumps({{"type": "thread.started", "thread_id": "thr-1"}}))
print(json.dumps({{"type": "item.completed", "item": {{"id": "i1", "type": "agent_message", "text": answer}}}}))
print(json.dumps({{"type": "turn.completed", "usage": {{"input_tokens": 100, "cached_input_tokens": 40, "output_tokens": 15, "reasoning_output_tokens": 5}}}}))
""",
    )
    monkeypatch.setenv("OPENAI_API_KEY", "sk-should-not-override-codex-sub")

    tp.configure(lm="codex-cli:gpt-5.4")
    rows = tp.from_rows([tp.Row((("row", 0),), {"q": "status?"})]).lm_map("q -> verdict", on_error="raise").collect()
    assert rows[0].verdict == "codex-ok"
    stats = tp.last_run()
    # input_tokens (100) already includes cached_input_tokens (40); output_tokens (15) includes reasoning (5).
    assert (stats.prompt_tokens, stats.completion_tokens) == (100, 15)

    logged = json.loads(log_file.read_text().splitlines()[0])
    assert not logged["has_openai_key"]
    assert "--ephemeral" in logged["argv"] and "--sandbox" in logged["argv"]


def test_cursor_cli_isolated_workspace_and_tool_denial(isolated_subscriptions):
    log_file = isolated_subscriptions / "cursor_calls.jsonl"
    _write_executable(
        isolated_subscriptions / "bin" / "agent",
        f"""
import json, sys
from pathlib import Path

argv = sys.argv[1:]
ws = Path(argv[argv.index("--workspace") + 1])
cli_cfg = json.loads((ws / ".cursor" / "cli.json").read_text(encoding="utf-8"))
stdin_text = sys.stdin.read()

with open({str(log_file)!r}, "a", encoding="utf-8") as f:
    f.write(json.dumps({{"argv": argv, "deny": cli_cfg["permissions"]["deny"], "stdin": stdin_text}}) + "\\n")

print(json.dumps({{
    "type": "result",
    "subtype": "success",
    "is_error": False,
    "result": json.dumps({{"tag": "cursor-verified"}}),
    "session_id": "cur-1",
}}))
""",
    )

    tp.configure(lm="cursor")
    rows = tp.from_rows([tp.Row((("row", 0),), {"item": "check"})]).lm_map("item -> tag", on_error="raise").collect()
    assert rows[0].tag == "cursor-verified"

    logged = json.loads(log_file.read_text().splitlines()[0])
    assert "--mode=ask" in logged["argv"] and "--trust" in logged["argv"]
    assert "Shell(*)" in logged["deny"] and "Write(**)" in logged["deny"]


def test_oauth_files_select_direct_http_engines_by_default(isolated_subscriptions):
    home = Path(os.environ["HOME"])
    creds = home / ".claude" / ".credentials.json"
    creds.parent.mkdir(parents=True, exist_ok=True)
    creds.write_text(json.dumps({"claudeAiOauth": {"accessToken": "tok-1", "refreshToken": "ref-1"}}))

    codex_auth = home / ".codex" / "auth.json"
    codex_auth.parent.mkdir(parents=True, exist_ok=True)
    codex_auth.write_text(json.dumps({"tokens": {"access_token": "ctok", "refresh_token": "cref"}}))

    c_lm = lm_config.make_lm("claude-code:sonnet")
    assert c_lm.model == "claude-code/claude-sonnet-4-6"
    assert lm_config.model_label(c_lm) == ("claude-code", "claude-sonnet-4-6")

    x_lm = lm_config.make_lm("codex")
    assert x_lm.model == "openai-codex/gpt-5.4"
    assert lm_config.model_label(x_lm) == ("codex", "gpt-5.4")


def test_subscription_auto_detection_and_doctor(isolated_subscriptions, monkeypatch):
    empty_doc = tp.doctor()
    assert empty_doc["recommended"] is None
    assert all(not p["available"] for p in empty_doc["providers"].values())

    with pytest.raises(dspy.LMAuthError, match="No AI coding subscription found"):
        lm_config.make_lm("subscription")

    _write_executable(isolated_subscriptions / "bin" / "agent", "import sys; sys.exit(0)\n")
    _write_executable(isolated_subscriptions / "bin" / "claude", "import sys; sys.exit(0)\n")

    # Default priority picks claude-cli over cursor when no host env var is set.
    assert subscriptions.detect_subscription() == "claude-cli:sonnet"

    # Active Cursor session env var prefers Cursor when its CLI is present.
    monkeypatch.setenv("CURSOR_AGENT", "1")
    assert subscriptions.detect_subscription() == "cursor:auto"
    lm = lm_config.make_lm("subscription")
    assert lm_config.model_label(lm) == ("cursor", "auto")

    info = tp.doctor()
    assert info["recommended"] == "cursor:auto"
    assert info["providers"]["cursor"]["available"] is True
    assert info["providers"]["codex"]["available"] is False


def test_cli_auth_error_stops_run_immediately(isolated_subscriptions):
    _write_executable(
        isolated_subscriptions / "bin" / "agent",
        'import sys; sys.stderr.write("Error: Not logged in. Please run `agent login`.\\n"); sys.exit(1)\n',
    )
    tp.configure(lm="cursor:auto")
    # Even with default on_error="skip", LMAuthError is in STOP_RUN and halts immediately.
    table = tp.from_rows([tp.Row((("row", i),), {"x": str(i)}) for i in range(5)]).lm_map("x -> y")
    with pytest.raises(dspy.LMAuthError, match="agent login"):
        table.collect()
