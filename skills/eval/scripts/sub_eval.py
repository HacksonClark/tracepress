#!/usr/bin/env python3
"""Zero-dependency evaluation runner powered by existing AI coding subscriptions.

Uses your logged-in Claude Code (`claude`), OpenAI Codex (`codex`), or Cursor (`agent` /
`cursor-agent`) subscription — or shards work for in-session coding-agent subagents — with
content-addressed SQLite caching, deterministic + LLM-judge graders, and paired bootstrap CIs.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import json
import os
import random
import re
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_CLAUDE_STRIP_ENV = (
    "CLAUDECODE",
    "CLAUDE_CODE_ENTRYPOINT",
    "CLAUDE_CODE_CHILD_SESSION",
    "CLAUDE_CODE_SESSION_ID",
    "CLAUDE_CODE_MESSAGING_SOCKET",
    "CLAUDE_PID",
    "ANTHROPIC_API_KEY",
)
_CODEX_STRIP_ENV = ("OPENAI_API_KEY", "CODEX_API_KEY")

JUDGE_SYSTEM = (
    "You are an impartial evaluation grader. Grade the candidate output against the input, "
    "expected reference (if provided), and rubric. Respond ONLY with a single JSON object: "
    '{"pass": true or false, "score": number between 0.0 and 1.0, "reason": "concise explanation"}.'
)


# --- CLI Discovery & Doctor ------------------------------------------------------------


def _search_path() -> str:
    current = os.environ.get("PATH", "")
    extras = [str(Path.home() / ".local" / "bin"), "/usr/local/bin", "/opt/homebrew/bin"]
    parts = [p for p in current.split(os.pathsep) if p]
    for extra in extras:
        if extra not in parts:
            parts.append(extra)
    return os.pathsep.join(parts)


def find_cli(*names: str) -> str | None:
    path = _search_path()
    for name in names:
        found = shutil.which(name, path=path)
        if found:
            return found
    return None


def doctor_info() -> dict[str, Any]:
    claude_bin = find_cli("claude")
    codex_bin = find_cli("codex")
    cursor_bin = find_cli("agent", "cursor-agent")

    host_agent = (
        "claude" if os.environ.get("CLAUDECODE")
        else "cursor" if os.environ.get("CURSOR_AGENT")
        else "codex" if (os.environ.get("CODEX_THREAD_ID") or os.environ.get("CODEX_CI"))
        else None
    )

    available = {
        "claude": {"available": bool(claude_bin), "binary": claude_bin},
        "codex": {"available": bool(codex_bin), "binary": codex_bin},
        "cursor": {"available": bool(cursor_bin), "binary": cursor_bin},
    }

    preferred_cli: str | None = None
    if host_agent and available[host_agent]["available"]:
        preferred_cli = host_agent
    else:
        for name in ("claude", "codex", "cursor"):
            if available[name]["available"]:
                preferred_cli = name
                break

    return {
        "host_agent": host_agent,
        "recommended_mode": "cli" if preferred_cli else "subagent",
        "recommended_backend": preferred_cli,
        "backends": available,
    }


def resolve_backend(backend: str) -> tuple[str, str]:
    info = doctor_info()
    if backend == "auto":
        chosen = info["recommended_backend"]
        if not chosen:
            raise RuntimeError(
                "No subscription CLI (`claude`, `codex`, `agent`) found on PATH. "
                "Use `sub_eval.py prepare` + subagents + `sub_eval.py ingest`, or log in to a CLI."
            )
        return chosen, info["backends"][chosen]["binary"]
    if backend not in info["backends"]:
        raise ValueError(f"Unknown backend {backend!r}; choose from auto, claude, codex, cursor")
    binary = info["backends"][backend]["binary"]
    if not binary:
        raise RuntimeError(f"Backend {backend!r} CLI binary not found on PATH.")
    return backend, binary


# --- Content-Addressed SQLite Cache ----------------------------------------------------


class EvalCache:
    def __init__(self, cache_dir: Path, enabled: bool = True):
        self.cache_dir = cache_dir
        self.enabled = enabled
        self.db_path = cache_dir / "cache.sqlite3"
        if enabled:
            cache_dir.mkdir(parents=True, exist_ok=True)
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS responses (
                        key TEXT PRIMARY KEY,
                        kind TEXT NOT NULL,
                        backend TEXT NOT NULL,
                        model TEXT NOT NULL,
                        text TEXT NOT NULL,
                        input_tokens INTEGER NOT NULL,
                        output_tokens INTEGER NOT NULL,
                        created_at TEXT NOT NULL
                    )
                    """
                )

    @staticmethod
    def make_key(kind: str, backend: str, model: str, system: str, prompt: str) -> str:
        payload = json.dumps(
            {"kind": kind, "backend": backend, "model": model or "default", "system": system, "prompt": prompt},
            sort_keys=True,
            ensure_ascii=False,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def get(self, key: str) -> dict[str, Any] | None:
        if not self.enabled or not self.db_path.exists():
            return None
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT text, input_tokens, output_tokens FROM responses WHERE key = ?",
                (key,),
            ).fetchone()
        if row is None:
            return None
        return {"text": row[0], "input_tokens": row[1], "output_tokens": row[2], "cached": True}

    def put(self, key: str, kind: str, backend: str, model: str, text: str, in_tok: int, out_tok: int) -> None:
        if not self.enabled:
            return
        now = datetime.now(timezone.utc).isoformat()
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO responses
                (key, kind, backend, model, text, input_tokens, output_tokens, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (key, kind, backend, model or "default", text, int(in_tok), int(out_tok), now),
            )


# --- Subscription CLI Execution --------------------------------------------------------


def _clean_env(strip: tuple[str, ...]) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if k not in strip}
    env["PATH"] = _search_path()
    return env


async def _exec_proc(
    argv: list[str], stdin_text: str, *, env: dict[str, str], cwd: Path | None = None, timeout: float = 120.0
) -> tuple[int, str, str]:
    proc = await asyncio.create_subprocess_exec(
        *argv,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=env,
        cwd=str(cwd) if cwd else None,
    )
    try:
        out_b, err_b = await asyncio.wait_for(proc.communicate(stdin_text.encode("utf-8")), timeout=timeout)
    except BaseException:
        if proc.returncode is None:
            try:
                proc.kill()
            except ProcessLookupError:
                pass
            await proc.wait()
        raise
    return (
        proc.returncode or 0,
        out_b.decode("utf-8", errors="replace"),
        err_b.decode("utf-8", errors="replace"),
    )


async def call_subscription(
    backend: str,
    binary: str,
    model: str,
    system: str,
    prompt: str,
    *,
    kind: str,
    cache: EvalCache,
    timeout: float = 120.0,
) -> dict[str, Any]:
    cache_key = EvalCache.make_key(kind, backend, model, system, prompt)
    hit = cache.get(cache_key)
    if hit is not None:
        return hit

    t0 = time.monotonic()
    if backend == "claude":
        argv = [
            binary,
            "-p",
            "--output-format",
            "json",
            "--tools",
            "",
            "--disallowedTools",
            "mcp__*",
            "--disable-slash-commands",
            "--no-session-persistence",
        ]
        if model:
            argv.extend(["--model", model])
        if system:
            argv.extend(["--system-prompt", system])
        env = _clean_env(_CLAUDE_STRIP_ENV)
        env["CLAUDE_CODE_SKIP_PROMPT_HISTORY"] = "1"
        code, out, err = await _exec_proc(argv, prompt, env=env, timeout=timeout)
        payload = _last_json_obj(out)
        if code != 0 or not isinstance(payload, dict) or payload.get("is_error") or payload.get("subtype", "success") != "success":
            detail = (payload.get("result") if isinstance(payload, dict) else None) or err or out or f"exit {code}"
            raise RuntimeError(f"claude CLI failed: {str(detail).strip()}")
        text = str(payload.get("result") or "")
        u = payload.get("usage") if isinstance(payload.get("usage"), dict) else {}
        in_tok = int(u.get("input_tokens") or 0) + int(u.get("cache_read_input_tokens") or 0) + int(u.get("cache_creation_input_tokens") or 0)
        out_tok = int(u.get("output_tokens") or 0)

    elif backend == "codex":
        with tempfile.TemporaryDirectory(prefix="subeval-codex-") as tmp:
            workdir = Path(tmp)
            out_file = workdir / "last.txt"
            inst_file = workdir / "instructions.md"
            inst = f"{system}\n\nAnswer directly without running tools." if system else "Answer directly without running tools."
            inst_file.write_text(inst, encoding="utf-8")
            argv = [
                binary,
                "exec",
                "-",
                "--sandbox",
                "read-only",
                "--ephemeral",
                "--skip-git-repo-check",
                "--json",
                "-o",
                str(out_file),
                "-c",
                f'model_instructions_file="{inst_file.as_posix()}"',
            ]
            if model:
                argv.extend(["-m", model])
            code, out, err = await _exec_proc(argv, prompt, env=_clean_env(_CODEX_STRIP_ENV), cwd=workdir, timeout=timeout)
            text_candidate: str | None = out_file.read_text(encoding="utf-8").strip() if out_file.exists() else None
            usage_d: dict[str, Any] = {}
            for line in out.splitlines():
                if not line.strip().startswith("{"):
                    continue
                try:
                    ev = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if ev.get("type") == "item.completed" and (ev.get("item") or {}).get("type") == "agent_message":
                    if text_candidate is None:
                        text_candidate = str(ev["item"].get("text") or "")
                elif ev.get("type") == "turn.completed" and isinstance(ev.get("usage"), dict):
                    usage_d = ev["usage"]
            if code != 0 or text_candidate is None:
                raise RuntimeError(f"codex CLI failed: {(err or out or f'exit {code}').strip()}")
            text = text_candidate
            in_tok = int(usage_d.get("input_tokens") or 0)
            out_tok = int(usage_d.get("output_tokens") or 0)

    elif backend == "cursor":
        with tempfile.TemporaryDirectory(prefix="subeval-cursor-") as tmp:
            workdir = Path(tmp)
            (workdir / ".cursor").mkdir(parents=True, exist_ok=True)
            (workdir / ".cursor" / "cli.json").write_text(
                json.dumps({"permissions": {"allow": [], "deny": ["Read(**)", "Write(**)", "Shell(*)", "WebFetch(*)", "Mcp(*:*)"]}}),
                encoding="utf-8",
            )
            argv = [binary, "-p", "--mode=ask", "--trust", "--output-format", "json", "--workspace", str(workdir)]
            if model:
                argv.extend(["--model", model])
            combined = f"System instructions:\n{system}\n\n{prompt}" if system else prompt
            code, out, err = await _exec_proc(argv, combined, env=_clean_env(()), cwd=workdir, timeout=timeout)
            payload = _last_json_obj(out)
            if code != 0 or not isinstance(payload, dict) or payload.get("is_error") or payload.get("subtype", "success") != "success":
                detail = (payload.get("result") if isinstance(payload, dict) else None) or err or out or f"exit {code}"
                raise RuntimeError(f"cursor CLI failed: {str(detail).strip()}")
            text = str(payload.get("result") or "")
            u = payload.get("usage") if isinstance(payload.get("usage"), dict) else {}
            in_tok = int(u.get("input_tokens") or u.get("prompt_tokens") or 0)
            out_tok = int(u.get("output_tokens") or u.get("completion_tokens") or 0)
    else:
        raise ValueError(f"Unsupported backend {backend!r}")

    elapsed = time.monotonic() - t0
    cache.put(cache_key, kind, backend, model, text, in_tok, out_tok)
    return {"text": text, "input_tokens": in_tok, "output_tokens": out_tok, "cached": False, "latency_s": elapsed}


async def run_local_cmd(cmd_template: str, case_input: Any, timeout: float = 60.0) -> tuple[str, float]:
    rendered_in = case_input if isinstance(case_input, str) else json.dumps(case_input, ensure_ascii=False)
    t0 = time.monotonic()
    proc = await asyncio.create_subprocess_shell(
        cmd_template,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    out_b, err_b = await asyncio.wait_for(proc.communicate(rendered_in.encode("utf-8")), timeout=timeout)
    elapsed = time.monotonic() - t0
    if proc.returncode != 0:
        raise RuntimeError(f"SUT command exited {proc.returncode}: {err_b.decode('utf-8', errors='replace').strip()}")
    return out_b.decode("utf-8", errors="replace").strip(), elapsed


# --- Dataset Loading & Prompt Rendering ------------------------------------------------


@dataclass
class EvalCase:
    id: str
    input: Any
    expected: Any = None
    rubric: str | None = None
    grader: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def load_cases(path: Path, limit: int | None = None) -> list[EvalCase]:
    suffix = path.suffix.lower()
    raw_rows: list[dict[str, Any]] = []
    if suffix == ".jsonl":
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                raw_rows.append(json.loads(line))
    elif suffix == ".json":
        loaded = json.loads(path.read_text(encoding="utf-8"))
        raw_rows = loaded["cases"] if isinstance(loaded, dict) and "cases" in loaded else list(loaded)
    elif suffix == ".csv":
        with path.open(encoding="utf-8", newline="") as f:
            raw_rows = list(csv.DictReader(f))
    else:
        raise ValueError(f"Unsupported suite format {path}; expected .jsonl, .json, or .csv")

    cases: list[EvalCase] = []
    for i, r in enumerate(raw_rows):
        if limit is not None and len(cases) >= limit:
            break
        cid = str(r.get("id") or f"case_{i + 1:03d}")
        inp = r.get("input", r.get("question", r.get("prompt", "")))
        exp = r.get("expected", r.get("target", r.get("ground_truth")))
        rubric = r.get("rubric")
        grader = r.get("grader")
        reserved = {"id", "input", "question", "prompt", "expected", "target", "ground_truth", "rubric", "grader"}
        meta = {k: v for k, v in r.items() if k not in reserved}
        cases.append(EvalCase(id=cid, input=inp, expected=exp, rubric=rubric, grader=grader, metadata=meta))
    return cases


def load_prompt_spec(spec: str | None) -> str:
    if not spec:
        return ""
    p = Path(spec)
    if p.is_file():
        return p.read_text(encoding="utf-8").strip()
    return spec.strip()


def render_case_prompt(template: str, case: EvalCase) -> tuple[str, str]:
    inp_text = case.input if isinstance(case.input, str) else json.dumps(case.input, indent=2, ensure_ascii=False)
    if not template:
        return "", inp_text
    if "{input}" in template:
        rendered = template.replace("{input}", inp_text)
        if isinstance(case.input, dict):
            for k, v in case.input.items():
                rendered = rendered.replace(f"{{{k}}}", str(v))
        return "", rendered
    return template, inp_text


# --- Graders ---------------------------------------------------------------------------


def _last_json_obj(text: str) -> dict[str, Any] | None:
    stripped = text.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", stripped, re.S | re.I)
    candidates = []
    if fence:
        candidates.append(fence.group(1))
    candidates.append(stripped)
    brace = re.search(r"(\{.*\})", stripped, re.S)
    if brace:
        candidates.append(brace.group(1))
    for c in candidates:
        if not c:
            continue
        try:
            parsed = json.loads(c)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            continue
    return None


def pick_grader(requested: str, case: EvalCase, default_rubric: str | None) -> str:
    if case.grader:
        return case.grader
    if requested != "auto":
        return requested
    if case.rubric or default_rubric:
        return "llm_judge"
    if isinstance(case.expected, dict):
        return "json_match"
    return "exact"


def grade_deterministic(grader: str, output: str, expected: Any) -> tuple[bool, float, str]:
    out_clean = output.strip()
    if grader == "exact":
        exp_str = "" if expected is None else str(expected).strip()
        ok = " ".join(out_clean.lower().split()) == " ".join(exp_str.lower().split())
        return ok, (1.0 if ok else 0.0), ("exact match" if ok else f"expected {exp_str!r}, got {out_clean[:120]!r}")
    if grader == "contains":
        exp_str = "" if expected is None else str(expected).strip()
        ok = exp_str.lower() in out_clean.lower()
        return ok, (1.0 if ok else 0.0), ("substring present" if ok else f"missing {exp_str!r}")
    if grader == "regex":
        pattern = str(expected or "")
        ok = bool(re.search(pattern, out_clean, re.S))
        return ok, (1.0 if ok else 0.0), ("regex matched" if ok else f"did not match /{pattern}/")
    if grader == "json_match":
        parsed = _last_json_obj(out_clean)
        if parsed is None:
            return False, 0.0, "output is not valid JSON object"
        exp_dict = expected if isinstance(expected, dict) else json.loads(str(expected))
        mismatches = [k for k, v in exp_dict.items() if parsed.get(k) != v]
        if not mismatches:
            return True, 1.0, "all JSON fields matched"
        frac = max(0.0, 1.0 - len(mismatches) / max(1, len(exp_dict)))
        return False, frac, f"mismatched keys: {mismatches}"
    raise ValueError(f"Unknown deterministic grader {grader!r}")


def build_judge_prompt(case: EvalCase, output: str, default_rubric: str | None) -> str:
    inp_str = case.input if isinstance(case.input, str) else json.dumps(case.input, ensure_ascii=False)
    rubric = case.rubric or default_rubric or "The output accurately and completely satisfies the user request."
    exp_block = (
        f"\nReference / Expected Answer:\n{case.expected if isinstance(case.expected, str) else json.dumps(case.expected, ensure_ascii=False)}\n"
        if case.expected is not None
        else ""
    )
    return (
        f"Task Input:\n{inp_str}\n"
        f"{exp_block}\n"
        f"Grading Rubric:\n{rubric}\n\n"
        f"Candidate Output:\n{output}\n\n"
        'Return ONLY JSON: {"pass": bool, "score": float, "reason": str}'
    )


def parse_judge_response(raw_text: str) -> tuple[bool, float, str]:
    obj = _last_json_obj(raw_text)
    if obj is None:
        raise ValueError(f"LLM judge did not return parseable JSON: {raw_text[:200]!r}")
    passed = bool(obj.get("pass"))
    score = float(obj.get("score", 1.0 if passed else 0.0))
    score = max(0.0, min(1.0, score))
    reason = str(obj.get("reason") or "")
    return passed, score, reason


# --- Bootstrap Statistics & Formatting -------------------------------------------------


def bootstrap_ci(values: list[float], n_boot: int = 2000, seed: int = 0) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    rng = random.Random(seed)
    k = len(values)
    means = sorted(sum(rng.choices(values, k=k)) / k for _ in range(n_boot))
    return means[int(0.025 * n_boot)], means[int(0.975 * n_boot)]


def paired_bootstrap_diff(a: list[float], b: list[float], n_boot: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    if not a or len(a) != len(b):
        return 0.0, 0.0, 0.0
    rng = random.Random(seed)
    k = len(a)
    indices = range(k)
    diffs = sorted(
        sum(b[i] - a[i] for i in rng.choices(indices, k=k)) / k
        for _ in range(n_boot)
    )
    delta = sum(b) / k - sum(a) / k
    return delta, diffs[int(0.025 * n_boot)], diffs[int(0.975 * n_boot)]


def summarize_results(records: list[dict[str, Any]], label: str, backend: str, model: str) -> dict[str, Any]:
    n = len(records)
    passes = [1.0 if r["passed"] else 0.0 for r in records]
    scores = [float(r["score"]) for r in records]
    pass_rate = sum(passes) / n if n else 0.0
    mean_score = sum(scores) / n if n else 0.0
    ci_lo, ci_hi = bootstrap_ci(passes)
    cached = sum(1 for r in records if r.get("cached"))
    in_tok = sum(int(r.get("input_tokens") or 0) for r in records)
    out_tok = sum(int(r.get("output_tokens") or 0) for r in records)
    return {
        "label": label,
        "backend": backend,
        "model": model or "default",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "n_cases": n,
        "passed": int(sum(passes)),
        "pass_rate": pass_rate,
        "pass_ci95": [ci_lo, ci_hi],
        "mean_score": mean_score,
        "cached_cases": cached,
        "input_tokens": in_tok,
        "output_tokens": out_tok,
        "results": records,
    }


def format_markdown_report(run_data: dict[str, Any], comparison: dict[str, Any] | None = None) -> str:
    lo, hi = run_data["pass_ci95"]
    lines = [
        f"## Eval Report: `{run_data['label']}`",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Backend | `{run_data['backend']}` (`{run_data['model']}`) |",
        f"| Cases | {run_data['passed']} / {run_data['n_cases']} passed (**{100 * run_data['pass_rate']:.1f}%**) |",
        f"| 95% Bootstrap CI | [{100 * lo:.1f}%, {100 * hi:.1f}%] |",
        f"| Mean Score | {run_data['mean_score']:.3f} |",
        f"| Cache Hits | {run_data['cached_cases']} / {run_data['n_cases']} |",
        f"| Tokens (in / out) | {run_data['input_tokens']:,} / {run_data['output_tokens']:,} |",
    ]
    failures = [r for r in run_data["results"] if not r["passed"]]
    if failures:
        lines.extend(["", "### Failing Cases", "", "| Case | Score | Reason | Output (preview) |", "|---|---:|---|---|"])
        for r in failures[:15]:
            preview = " ".join(str(r.get("output", "")).split())[:80].replace("|", "\\|")
            reason = " ".join(str(r.get("reason", "")).split())[:80].replace("|", "\\|")
            lines.append(f"| `{r['id']}` | {r['score']:.2f} | {reason} | {preview} |")

    if comparison is not None:
        lines.extend(["", format_comparison_markdown(run_data, comparison)])
    return "\n".join(lines) + "\n"


def format_comparison_markdown(run_a: dict[str, Any], run_b: dict[str, Any]) -> str:
    by_id_a = {r["id"]: r for r in run_a["results"]}
    by_id_b = {r["id"]: r for r in run_b["results"]}
    shared = [cid for cid in by_id_a if cid in by_id_b]
    passes_a = [1.0 if by_id_a[c]["passed"] else 0.0 for c in shared]
    passes_b = [1.0 if by_id_b[c]["passed"] else 0.0 for c in shared]
    scores_a = [float(by_id_a[c]["score"]) for c in shared]
    scores_b = [float(by_id_b[c]["score"]) for c in shared]

    d_pass, p_lo, p_hi = paired_bootstrap_diff(passes_a, passes_b)
    d_score, s_lo, s_hi = paired_bootstrap_diff(scores_a, scores_b)
    wins = [c for c in shared if not by_id_a[c]["passed"] and by_id_b[c]["passed"]]
    regs = [c for c in shared if by_id_a[c]["passed"] and not by_id_b[c]["passed"]]

    lines = [
        f"## Paired Comparison: `{run_a['label']}` (A) vs `{run_b['label']}` (B)",
        "",
        "| Metric | Variant A | Variant B | Delta (B - A) | Paired 95% CI |",
        "|---|---:|---:|---:|---|",
        (
            f"| Pass Rate ({len(shared)} shared) | {100 * (sum(passes_a) / max(1, len(shared))):.1f}% "
            f"| {100 * (sum(passes_b) / max(1, len(shared))):.1f}% "
            f"| **{100 * d_pass:+.1f} pts** | [{100 * p_lo:+.1f}, {100 * p_hi:+.1f}] pts |"
        ),
        (
            f"| Mean Score | {sum(scores_a) / max(1, len(shared)):.3f} "
            f"| {sum(scores_b) / max(1, len(shared)):.3f} "
            f"| **{d_score:+.3f}** | [{s_lo:+.3f}, {s_hi:+.3f}] |"
        ),
        "",
        f"- **Improvements (A fail -> B pass, {len(wins)}):** {', '.join(f'`{c}`' for c in wins) or 'none'}",
        f"- **Regressions (A pass -> B fail, {len(regs)}):** {', '.join(f'`{c}`' for c in regs) or 'none'}",
    ]
    return "\n".join(lines) + "\n"


# --- Core Async Suite Evaluation -------------------------------------------------------


async def evaluate_suite_async(
    cases: list[EvalCase],
    *,
    label: str,
    prompt_template: str,
    sut_cmd: str | None,
    requested_grader: str,
    default_rubric: str | None,
    backend_spec: str,
    model: str,
    concurrency: int,
    cache: EvalCache,
) -> dict[str, Any]:
    needs_sub = sut_cmd is None or any(
        pick_grader(requested_grader, c, default_rubric) == "llm_judge" for c in cases
    )
    if needs_sub:
        backend, binary = resolve_backend(backend_spec)
    else:
        backend, binary = "local-cmd", ""

    sem = asyncio.Semaphore(max(1, concurrency))

    async def eval_one(case: EvalCase) -> dict[str, Any]:
        async with sem:
            in_tok = 0
            out_tok = 0
            cached = True
            if sut_cmd:
                output, latency = await run_local_cmd(sut_cmd, case.input)
                cached = False
            else:
                sys_p, usr_p = render_case_prompt(prompt_template, case)
                gen = await call_subscription(
                    backend, binary, model, sys_p, usr_p, kind="generate", cache=cache
                )
                output = gen["text"]
                in_tok += gen["input_tokens"]
                out_tok += gen["output_tokens"]
                cached = cached and bool(gen.get("cached"))
                latency = float(gen.get("latency_s") or 0.0)

            grader = pick_grader(requested_grader, case, default_rubric)
            if grader == "llm_judge":
                j_prompt = build_judge_prompt(case, output, default_rubric)
                j_res = await call_subscription(
                    backend, binary, model, JUDGE_SYSTEM, j_prompt, kind="judge", cache=cache
                )
                in_tok += j_res["input_tokens"]
                out_tok += j_res["output_tokens"]
                cached = cached and bool(j_res.get("cached"))
                passed, score, reason = parse_judge_response(j_res["text"])
            else:
                passed, score, reason = grade_deterministic(grader, output, case.expected)

            return {
                "id": case.id,
                "input": case.input,
                "expected": case.expected,
                "output": output,
                "grader": grader,
                "passed": passed,
                "score": score,
                "reason": reason,
                "cached": cached,
                "input_tokens": in_tok,
                "output_tokens": out_tok,
                "latency_s": round(latency, 3),
            }

    results = await asyncio.gather(*(eval_one(c) for c in cases))
    return summarize_results(list(results), label=label, backend=backend, model=model)


def save_run_artifact(cache_dir: Path, run_data: dict[str, Any], explicit_out: Path | None = None) -> Path:
    runs_dir = cache_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", run_data["label"]).strip("_") or "eval"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    dest = explicit_out or (runs_dir / f"{stamp}_{slug}.json")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(run_data, indent=2, ensure_ascii=False), encoding="utf-8")
    return dest


# --- In-Session Subagent Sharding (`prepare` / `ingest`) -------------------------------


async def prepare_shards_async(args: argparse.Namespace) -> dict[str, Any]:
    suite_path = Path(args.cases)
    cases = load_cases(suite_path, limit=args.limit)
    prompt_template = load_prompt_spec(args.prompt)
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    prepared_items: list[dict[str, Any]] = []
    for case in cases:
        grader = pick_grader(args.grader, case, args.rubric)
        item: dict[str, Any] = {
            "id": case.id,
            "input": case.input,
            "expected": case.expected,
            "rubric": case.rubric or args.rubric,
            "grader": grader,
        }
        if args.cmd:
            out_text, _ = await run_local_cmd(args.cmd, case.input)
            item["precomputed_output"] = out_text
            if grader == "llm_judge":
                item["judge_prompt"] = build_judge_prompt(case, out_text, args.rubric)
        else:
            sys_p, usr_p = render_case_prompt(prompt_template, case)
            item["system_prompt"] = sys_p
            item["user_prompt"] = usr_p
        prepared_items.append(item)

    shard_size = max(1, args.shard_size)
    shards: list[str] = []
    for idx in range(0, len(prepared_items), shard_size):
        chunk = prepared_items[idx : idx + shard_size]
        shard_name = f"shard_{idx // shard_size:02d}.json"
        shard_path = work_dir / shard_name
        shard_payload = {
            "instructions": (
                "For each item in `items`: (1) if `precomputed_output` is absent, generate `output` answering "
                "`user_prompt` under `system_prompt`; (2) if `grader == 'llm_judge'`, grade `output` against "
                "`expected` and `rubric` as `judge: {'pass': bool, 'score': float, 'reason': str}`. "
                f"Write JSON list to `{shard_name.replace('.json', '.out.json')}`."
            ),
            "expected_output_file": str(work_dir / shard_name.replace(".json", ".out.json")),
            "items": chunk,
        }
        shard_path.write_text(json.dumps(shard_payload, indent=2, ensure_ascii=False), encoding="utf-8")
        shards.append(str(shard_path))

    manifest = {
        "label": args.label or suite_path.stem,
        "cases_path": str(suite_path),
        "prompt": prompt_template,
        "sut_cmd": args.cmd,
        "grader": args.grader,
        "rubric": args.rubric,
        "shards": shards,
    }
    (work_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def ingest_shards(args: argparse.Namespace) -> dict[str, Any]:
    work_dir = Path(args.work_dir)
    manifest = json.loads((work_dir / "manifest.json").read_text(encoding="utf-8"))
    cache = EvalCache(Path(args.cache_dir), enabled=not args.no_cache)

    records: list[dict[str, Any]] = []
    for shard_file in manifest["shards"]:
        shard_path = Path(shard_file)
        shard_data = json.loads(shard_path.read_text(encoding="utf-8"))
        out_path = Path(shard_data["expected_output_file"])
        if not out_path.exists():
            raise FileNotFoundError(f"Missing subagent output file: {out_path}")
        raw_out = json.loads(out_path.read_text(encoding="utf-8"))
        out_list = raw_out["results"] if isinstance(raw_out, dict) and "results" in raw_out else list(raw_out)
        by_id = {str(x["id"]): x for x in out_list}

        for item in shard_data["items"]:
            cid = item["id"]
            if cid not in by_id:
                raise KeyError(f"Case {cid!r} missing from {out_path}")
            ans = by_id[cid]
            output = str(item.get("precomputed_output") if "precomputed_output" in item else ans.get("output", ""))
            if "user_prompt" in item:
                gk = EvalCache.make_key("generate", "subagent", "session", item.get("system_prompt") or "", item["user_prompt"])
                cache.put(gk, "generate", "subagent", "session", output, 0, 0)

            grader = item["grader"]
            if grader == "llm_judge":
                j = ans.get("judge")
                if not isinstance(j, dict):
                    raise ValueError(f"Case {cid!r} requires `judge` object in {out_path}")
                passed = bool(j.get("pass"))
                score = max(0.0, min(1.0, float(j.get("score", 1.0 if passed else 0.0))))
                reason = str(j.get("reason") or "")
            else:
                passed, score, reason = grade_deterministic(grader, output, item.get("expected"))

            records.append(
                {
                    "id": cid,
                    "input": item["input"],
                    "expected": item.get("expected"),
                    "output": output,
                    "grader": grader,
                    "passed": passed,
                    "score": score,
                    "reason": reason,
                    "cached": False,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "latency_s": 0.0,
                }
            )

    run_data = summarize_results(records, label=manifest["label"], backend="subagent", model="session")
    save_run_artifact(Path(args.cache_dir), run_data, Path(args.json_out) if args.json_out else None)
    report = format_markdown_report(run_data)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(report, encoding="utf-8")
    print(report)
    return run_data


# --- CLI Entrypoint --------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="sub_eval", description="Run evaluations using your existing AI subscription.")
    sub = p.add_subparsers(dest="command", required=True)

    p_doc = sub.add_parser("doctor", help="Check available subscription CLIs and recommended mode")
    p_doc.add_argument("--json", action="store_true")

    p_run = sub.add_parser("run", help="Run an eval suite via subscription CLI (with SQLite cache)")
    p_run.add_argument("cases", help="Path to .jsonl, .json, or .csv eval suite")
    p_run.add_argument("--prompt", help="Prompt file path or inline prompt template")
    p_run.add_argument("--compare", help="Second prompt file/template for paired A/B comparison")
    p_run.add_argument("--cmd", help="Optional local shell command under test (receives case input on stdin)")
    p_run.add_argument("--grader", choices=("auto", "exact", "contains", "regex", "json_match", "llm_judge"), default="auto")
    p_run.add_argument("--rubric", help="Grading rubric for llm_judge")
    p_run.add_argument("--backend", choices=("auto", "claude", "codex", "cursor"), default="auto")
    p_run.add_argument("--model", default="", help="Optional model alias/slug for the subscription CLI")
    p_run.add_argument("--concurrency", type=int, default=6)
    p_run.add_argument("--limit", type=int)
    p_run.add_argument("--label", help="Run label")
    p_run.add_argument("--cache-dir", default=".eval-cache")
    p_run.add_argument("--no-cache", action="store_true")
    p_run.add_argument("--out", help="Write Markdown report to file")
    p_run.add_argument("--json-out", help="Write run JSON to file")

    p_prep = sub.add_parser("prepare", help="Prepare JSON shards for parallel in-session subagents")
    p_prep.add_argument("cases")
    p_prep.add_argument("--prompt")
    p_prep.add_argument("--cmd")
    p_prep.add_argument("--grader", choices=("auto", "exact", "contains", "regex", "json_match", "llm_judge"), default="auto")
    p_prep.add_argument("--rubric")
    p_prep.add_argument("--shard-size", type=int, default=10)
    p_prep.add_argument("--limit", type=int)
    p_prep.add_argument("--label")
    p_prep.add_argument("--work-dir", default=".eval-cache/batch")

    p_ing = sub.add_parser("ingest", help="Score completed subagent shards and emit report")
    p_ing.add_argument("work_dir")
    p_ing.add_argument("--cache-dir", default=".eval-cache")
    p_ing.add_argument("--no-cache", action="store_true")
    p_ing.add_argument("--out")
    p_ing.add_argument("--json-out")

    p_cmp = sub.add_parser("compare", help="Compare two saved run JSON artifacts with paired bootstrap CIs")
    p_cmp.add_argument("run_a")
    p_cmp.add_argument("run_b")
    p_cmp.add_argument("--out")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "doctor":
        info = doctor_info()
        if args.json:
            print(json.dumps(info, indent=2))
        else:
            print(f"Recommended mode: {info['recommended_mode']} (backend: {info['recommended_backend'] or 'subagent'})")
            for name, b in info["backends"].items():
                status = f"READY ({b['binary']})" if b["available"] else "MISSING"
                print(f"  {name:8s}: {status}")
        return 0

    if args.command == "run":
        suite_path = Path(args.cases)
        cases = load_cases(suite_path, limit=args.limit)
        cache_dir = Path(args.cache_dir)
        cache = EvalCache(cache_dir, enabled=not args.no_cache)
        prompt_a = load_prompt_spec(args.prompt)
        label_a = args.label or (Path(args.prompt).stem if args.prompt and Path(args.prompt).is_file() else suite_path.stem)

        run_a = asyncio.run(
            evaluate_suite_async(
                cases,
                label=label_a,
                prompt_template=prompt_a,
                sut_cmd=args.cmd,
                requested_grader=args.grader,
                default_rubric=args.rubric,
                backend_spec=args.backend,
                model=args.model,
                concurrency=args.concurrency,
                cache=cache,
            )
        )
        save_run_artifact(cache_dir, run_a, Path(args.json_out) if args.json_out else None)

        run_b = None
        if args.compare:
            prompt_b = load_prompt_spec(args.compare)
            label_b = Path(args.compare).stem if Path(args.compare).is_file() else f"{label_a}_variant_b"
            run_b = asyncio.run(
                evaluate_suite_async(
                    cases,
                    label=label_b,
                    prompt_template=prompt_b,
                    sut_cmd=args.cmd,
                    requested_grader=args.grader,
                    default_rubric=args.rubric,
                    backend_spec=args.backend,
                    model=args.model,
                    concurrency=args.concurrency,
                    cache=cache,
                )
            )
            save_run_artifact(cache_dir, run_b)

        report = format_markdown_report(run_a, comparison=run_b)
        if args.out:
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out).write_text(report, encoding="utf-8")
        print(report)
        return 0

    if args.command == "prepare":
        manifest = asyncio.run(prepare_shards_async(args))
        print(json.dumps(manifest, indent=2))
        return 0

    if args.command == "ingest":
        ingest_shards(args)
        return 0

    if args.command == "compare":
        run_a = json.loads(Path(args.run_a).read_text(encoding="utf-8"))
        run_b = json.loads(Path(args.run_b).read_text(encoding="utf-8"))
        md = format_comparison_markdown(run_a, run_b)
        if args.out:
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out).write_text(md, encoding="utf-8")
        print(md)
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
