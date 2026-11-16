"""Tests for the zero-dependency `/eval` subscription skill & runner (`sub_eval.py`)."""

from __future__ import annotations

import importlib.util
import json
import stat
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT_PATH = Path(__file__).resolve().parent.parent / "skills" / "eval" / "scripts" / "sub_eval.py"
EXPORT_SCRIPT = Path(__file__).resolve().parent.parent / "skills" / "eval" / "scripts" / "export_standalone.sh"

spec = importlib.util.spec_from_file_location("sub_eval", SCRIPT_PATH)
assert spec and spec.loader
sub_eval = importlib.util.module_from_spec(spec)
sys.modules["sub_eval"] = sub_eval
spec.loader.exec_module(sub_eval)


def _make_bin(path: Path, code: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"#!{sys.executable}\n{code}", encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return path


@pytest.fixture(autouse=True)
def isolated_env(monkeypatch, tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("PATH", str(tmp_path / "bin"))
    for v in ("CLAUDECODE", "CURSOR_AGENT", "CODEX_THREAD_ID", "CODEX_CI", "ANTHROPIC_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(v, raising=False)
    return tmp_path


def test_doctor_subagent_fallback_and_cli_detection(isolated_env, monkeypatch):
    info = sub_eval.doctor_info()
    assert info["recommended_mode"] == "subagent"
    assert info["recommended_backend"] is None

    _make_bin(isolated_env / "bin" / "claude", "import sys; sys.exit(0)\n")
    _make_bin(isolated_env / "bin" / "agent", "import sys; sys.exit(0)\n")
    assert sub_eval.doctor_info()["recommended_backend"] == "claude"

    monkeypatch.setenv("CURSOR_AGENT", "1")
    assert sub_eval.doctor_info()["recommended_backend"] == "cursor"


def test_run_caching_and_ab_comparison_across_claude_cli(isolated_env, monkeypatch):
    calls_log = isolated_env / "calls.jsonl"
    _make_bin(
        isolated_env / "bin" / "claude",
        f"""
import json, os, sys
stdin_text = sys.stdin.read()
argv = sys.argv[1:]
sys_prompt = argv[argv.index("--system-prompt") + 1] if "--system-prompt" in argv else ""
assert "CLAUDECODE" not in os.environ and "ANTHROPIC_API_KEY" not in os.environ
with open({str(calls_log)!r}, "a", encoding="utf-8") as f:
    f.write(json.dumps({{"sys": sys_prompt, "stdin": stdin_text}}) + "\\n")

# Variant B prompt contains 'STRICT_V2' and gets all 3 right; Variant A misses case 2.
if "STRICT_V2" in sys_prompt or "STRICT_V2" in stdin_text:
    ans = "42" if "6*7" in stdin_text else ("paris" if "France" in stdin_text else "even")
else:
    ans = "42" if "6*7" in stdin_text else ("lyon" if "France" in stdin_text else "even")

print(json.dumps({{
    "type": "result",
    "subtype": "success",
    "is_error": False,
    "result": ans,
    "usage": {{"input_tokens": 12, "output_tokens": 3}},
}}))
""",
    )
    monkeypatch.setenv("CLAUDECODE", "1")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-ignore")

    suite = isolated_env / "cases.jsonl"
    suite.write_text(
        "\n".join(
            [
                json.dumps({"id": "math", "input": "What is 6*7?", "expected": "42"}),
                json.dumps({"id": "geo", "input": "Capital of France?", "expected": "Paris"}),
                json.dumps({"id": "parity", "input": "Is 8 even?", "expected": "even"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    p1 = isolated_env / "v1.txt"
    p2 = isolated_env / "v2.txt"
    p1.write_text("Answer concisely.", encoding="utf-8")
    p2.write_text("STRICT_V2: Answer concisely and accurately.", encoding="utf-8")

    cache_dir = isolated_env / ".eval-cache"
    report_path = isolated_env / "report.md"
    rc = sub_eval.main(
        [
            "run",
            str(suite),
            "--prompt",
            str(p1),
            "--compare",
            str(p2),
            "--cache-dir",
            str(cache_dir),
            "--out",
            str(report_path),
        ]
    )
    assert rc == 0
    report = report_path.read_text(encoding="utf-8")
    assert "66.7%" in report and "100.0%" in report
    assert "+33.3 pts" in report and "`geo`" in report
    assert len(calls_log.read_text().splitlines()) == 6

    # Re-running hits SQLite cache for all 6 calls (0 new CLI invocations).
    rc2 = sub_eval.main(["run", str(suite), "--prompt", str(p1), "--compare", str(p2), "--cache-dir", str(cache_dir)])
    assert rc2 == 0
    assert len(calls_log.read_text().splitlines()) == 6


def test_local_cmd_sut_with_cursor_llm_judge_and_codex_json_match(isolated_env):
    _make_bin(
        isolated_env / "bin" / "agent",
        """
import json, sys
from pathlib import Path
argv = sys.argv[1:]
ws = Path(argv[argv.index("--workspace") + 1])
assert "Shell(*)" in json.loads((ws / ".cursor" / "cli.json").read_text())["permissions"]["deny"]
prompt = sys.stdin.read()
passed = "42" in prompt
print(json.dumps({
    "type": "result",
    "subtype": "success",
    "is_error": False,
    "result": json.dumps({"pass": passed, "score": 1.0 if passed else 0.0, "reason": "checked answer"}),
}))
""",
    )
    suite = isolated_env / "cmd_suite.jsonl"
    suite.write_text(
        json.dumps({"id": "c1", "input": "compute", "rubric": "Must output 42"}) + "\n",
        encoding="utf-8",
    )
    json_out = isolated_env / "cmd_run.json"
    rc = sub_eval.main(
        [
            "run",
            str(suite),
            "--cmd",
            f"{sys.executable} -c 'print(42)'",
            "--backend",
            "cursor",
            "--cache-dir",
            str(isolated_env / "cache"),
            "--json-out",
            str(json_out),
        ]
    )
    assert rc == 0
    saved = json.loads(json_out.read_text(encoding="utf-8"))
    assert saved["pass_rate"] == 1.0

    # Also verify codex backend + json_match grader.
    _make_bin(
        isolated_env / "bin" / "codex",
        """
import json, sys
from pathlib import Path
argv = sys.argv[1:]
out_file = Path(argv[argv.index("-o") + 1])
ans = "```json\\n{\\"severity\\": \\"P0\\"}\\n```"
out_file.write_text(ans, encoding="utf-8")
print(json.dumps({"type": "turn.completed", "usage": {"input_tokens": 10, "output_tokens": 4}}))
""",
    )
    j_suite = isolated_env / "json_suite.jsonl"
    j_suite.write_text(json.dumps({"id": "j1", "input": "db down", "expected": {"severity": "P0"}}) + "\n")
    j_out = isolated_env / "codex_run.json"
    assert sub_eval.main(["run", str(j_suite), "--backend", "codex", "--cache-dir", str(isolated_env / "c2"), "--json-out", str(j_out)]) == 0
    assert json.loads(j_out.read_text())["pass_rate"] == 1.0


def test_prepare_and_ingest_subagent_workflow(isolated_env):
    suite = isolated_env / "sub_cases.jsonl"
    suite.write_text(
        "\n".join(
            [
                json.dumps({"id": "s1", "input": "2+2", "expected": "4"}),
                json.dumps({"id": "s2", "input": "Explain ACID", "rubric": "Mentions atomicity"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    work_dir = isolated_env / "batch"
    assert sub_eval.main(["prepare", str(suite), "--shard-size", "2", "--work-dir", str(work_dir)]) == 0

    shard = json.loads((work_dir / "shard_00.json").read_text(encoding="utf-8"))
    assert len(shard["items"]) == 2
    Path(shard["expected_output_file"]).write_text(
        json.dumps(
            [
                {"id": "s1", "output": "4"},
                {"id": "s2", "output": "Atomicity, consistency...", "judge": {"pass": True, "score": 0.95, "reason": "solid"}},
            ]
        ),
        encoding="utf-8",
    )

    out_json = isolated_env / "ingested.json"
    assert sub_eval.main(["ingest", str(work_dir), "--cache-dir", str(isolated_env / "icache"), "--json-out", str(out_json)]) == 0
    ingested = json.loads(out_json.read_text(encoding="utf-8"))
    assert ingested["passed"] == 2 and ingested["mean_score"] == pytest.approx(0.975)


def test_export_standalone_creates_working_project(tmp_path):
    dest = tmp_path / "sub-eval-standalone"
    sys_env = {**dict(__import__("os").environ), "PATH": "/usr/local/bin:/usr/bin:/bin"}
    subprocess.run(["/bin/bash", str(EXPORT_SCRIPT), str(dest)], env=sys_env, check=True)
    assert (dest / "README.md").is_file()
    assert (dest / ".claude-plugin" / "plugin.json").is_file()
    assert (dest / ".cursor-plugin" / "plugin.json").is_file()
    assert (dest / "skills" / "eval" / "SKILL.md").is_file()
    assert (dest / ".agents" / "skills" / "eval" / "SKILL.md").is_file()
    subprocess.run(
        [sys.executable, "-m", "pytest", str(dest / "tests" / "test_sub_eval.py"), "-k", "not export_standalone"],
        env=sys_env,
        check=True,
    )
