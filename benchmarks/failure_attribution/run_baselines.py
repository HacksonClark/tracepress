#!/usr/bin/env python3
"""Run upstream Who&When `Automated_FA/Lib/utils.py` baselines (`all_at_once`, `step_by_step`, `binary_search`)
against either an OpenAI-compatible HTTP endpoint OR any TracePress/DSPy LM via `TracePressOpenAIShim`.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import json
import os
import random
import re
import sys
import tempfile
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import tracepress as tp

from benchmarks.failure_attribution.datasets.whowhen import resolve_whowhen_dir

MARKER_WRAP_RE = re.compile(r"^\s*\[\[\s*##\s*\w+\s*##\s*\]\]\s*\n(.*?)(?:\n\s*\[\[\s*##\s*completed\s*##\s*\]\]\s*)?$", re.DOTALL)


@dataclass
class _ShimMessage:
    content: str
    role: str = "assistant"


@dataclass
class _ShimChoice:
    message: _ShimMessage
    index: int = 0


@dataclass
class _ShimCompletionResponse:
    choices: list[_ShimChoice]
    model: str = ""


class _ShimChatCompletions:
    def __init__(self, shim: TracePressOpenAIShim):
        self._shim = shim

    def create(
        self,
        *,
        model: str = "",
        messages: list[dict[str, str]],
        max_tokens: int | None = 1024,
        **kwargs: Any,
    ) -> _ShimCompletionResponse:
        return self._shim._complete(model=model, messages=messages, max_tokens=max_tokens, **kwargs)


class _ShimChat:
    def __init__(self, shim: TracePressOpenAIShim):
        self.completions = _ShimChatCompletions(shim)


class TracePressOpenAIShim:
    """Minimal OpenAI-client drop-in (`client.chat.completions.create(...)`) backed by a TracePress/DSPy LM.

    Allows unmodified OpenAI-SDK baseline code (such as `Automated_FA/Lib/utils.py`) to run on
    any TracePress model spec (`openrouter:...`, `portkey:...`, `ollama_chat/...`, or a `dspy.BaseLM`).
    """

    def __init__(self, lm_spec_or_obj: Any):
        tp.configure(telemetry=False)
        self.lm = tp.lm.resolve_lm(lm_spec_or_obj)
        self.chat = _ShimChat(self)
        self.calls = 0

    def _complete(
        self,
        *,
        model: str = "",
        messages: list[dict[str, str]],
        max_tokens: int | None = 1024,
        **kwargs: Any,
    ) -> _ShimCompletionResponse:
        self.calls += 1
        call_kwargs: dict[str, Any] = dict(kwargs)
        if max_tokens is not None:
            call_kwargs["max_tokens"] = max_tokens

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            raw = self.lm(messages=messages, **call_kwargs)

        if isinstance(raw, list) and raw:
            first = raw[0]
            text = first if isinstance(first, str) else str(first.get("text", first) if isinstance(first, dict) else first)
        else:
            text = str(raw or "")

        # Unwrap single-field DSPy chat marker output when running under DummyLM / FuncLM in tests.
        m = MARKER_WRAP_RE.match(text)
        if m:
            text = m.group(1)

        model_name = model or str(getattr(self.lm, "model", "tracepress-lm"))
        return _ShimCompletionResponse(choices=[_ShimChoice(message=_ShimMessage(content=text.strip()))], model=model_name)


def _find_upstream_utils(data_path: Path, explicit_dir: str | Path | None = None) -> Path | None:
    candidates: list[Path] = []
    if explicit_dir is not None:
        p = Path(explicit_dir)
        candidates.extend([p / "Lib" / "utils.py", p / "utils.py", p / "Automated_FA" / "Lib" / "utils.py"])
    for anc in [data_path, *data_path.parents]:
        candidates.append(anc / "Automated_FA" / "Lib" / "utils.py")
    candidates.append(Path("/tmp/fa-research/Agents_Failure_Attribution/Automated_FA/Lib/utils.py"))
    for c in candidates:
        if c.is_file():
            return c
    return None


def _load_upstream_module(utils_path: Path):
    spec = importlib.util.spec_from_file_location("_automated_fa_utils", utils_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load upstream module from {utils_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --- Faithful Standalone Fallback (when `Automated_FA/Lib/utils.py` isn't cloned nearby) --


def _sorted_json_files(directory: Path) -> list[Path]:
    files = [f for f in directory.iterdir() if f.is_file() and f.suffix == ".json"]
    return sorted(files, key=lambda p: (int("".join(filter(str.isdigit, p.stem)) or 0), p.name))


def _call_chat(client: Any, model: str, messages: list[dict[str, str]], max_tokens: int) -> str | None:
    resp = client.chat.completions.create(model=model, messages=messages, max_tokens=max_tokens)
    return resp.choices[0].message.content.strip()


def _fallback_all_at_once(client: Any, directory_path: str, is_handcrafted: bool, model: str, max_tokens: int) -> None:
    index_agent = "role" if is_handcrafted else "name"
    for fp in _sorted_json_files(Path(directory_path)):
        data = json.loads(fp.read_text(encoding="utf-8"))
        history = data.get("history") or []
        if not history:
            continue
        chat_content = "\n".join(f"{e.get(index_agent, 'Unknown Agent')}: {e.get('content', '')}" for e in history)
        prompt = (
            "You are an AI assistant tasked with analyzing a multi-agent conversation history when solving a real world problem. "
            f"The problem is:  {data.get('question', '')}\n"
            f"The Answer for the problem is: {data.get('ground_truth', '')}\n"
            "Identify which agent made an error, at which step, and explain the reason for the error. "
            "Here's the conversation:\n\n"
            + chat_content
            + "\n\nBased on this conversation, please predict the following:\n"
            "1. The name of the agent who made a mistake that should be directly responsible for the wrong solution.\n"
            "2. In which step (0-indexed) the mistake agent first made mistake.\n"
            "3. The reason for your prediction."
            "Please answer in the format: Agent Name: (Your prediction)\n Step Number: (Your prediction)\n Reason for Mistake: \n"
        )
        messages = [
            {"role": "system", "content": "You are a helpful assistant skilled in analyzing conversations."},
            {"role": "user", "content": prompt},
        ]
        res = _call_chat(client, model, messages, max_tokens)
        print(f"Prediction for {fp.name}:")
        print(res or "Failed to get prediction.")
        print("\n" + "=" * 50 + "\n")


def _fallback_step_by_step(client: Any, directory_path: str, is_handcrafted: bool, model: str, max_tokens: int) -> None:
    index_agent = "role" if is_handcrafted else "name"
    for fp in _sorted_json_files(Path(directory_path)):
        data = json.loads(fp.read_text(encoding="utf-8"))
        history = data.get("history") or []
        if not history:
            continue
        prefix = ""
        for idx, entry in enumerate(history):
            agent_name = entry.get(index_agent, "Unknown Agent")
            prefix += f"Step {idx} - {agent_name}: {entry.get('content', '')}\n"
            prompt = (
                f"You are an AI assistant tasked with evaluating the correctness of each step in an ongoing multi-agent conversation. "
                f"The problem being addressed is: {data.get('question', '')}. "
                f"The Answer for the problem is: {data.get('ground_truth', '')}\n"
                f"Here is the conversation history up to the current step:\n{prefix}\n"
                f"The most recent step ({idx}) was by '{agent_name}'.\n"
                "Respond ONLY in the format: 1. Yes/No.\n2. Reason: [Your explanation here]"
            )
            messages = [
                {"role": "system", "content": "You are a precise step-by-step conversation evaluator."},
                {"role": "user", "content": prompt},
            ]
            ans = _call_chat(client, model, messages, max_tokens)
            if not ans:
                break
            if ans.lower().strip().startswith("1. yes"):
                reason = ans.split("Reason:", 1)[-1].strip() if "Reason:" in ans else ans
                print(f"\nPrediction for {fp.name}: Error found.")
                print(f"Agent Name: {agent_name}")
                print(f"Step Number: {idx}")
                print(f"Reason provided by LLM: {reason}")
                break
        print("\n" + "=" * 50 + "\n")


def _fallback_binary_search(client: Any, directory_path: str, is_handcrafted: bool, model: str, max_tokens: int) -> None:
    index_agent = "role" if is_handcrafted else "name"
    rng = random.Random(0)
    for fp in _sorted_json_files(Path(directory_path)):
        data = json.loads(fp.read_text(encoding="utf-8"))
        history = data.get("history") or []
        if not history:
            continue
        lo, hi = 0, len(history) - 1
        while lo < hi:
            mid = lo + (hi - lo) // 2
            seg = "\n".join(f"{e.get(index_agent, 'Unknown Agent')}: {e.get('content', '')}" for e in history[lo : hi + 1])
            prompt = (
                f"The problem to address is as follows: {data.get('question', '')}\n"
                f"The Answer for the problem is: {data.get('ground_truth', '')}\n"
                f"Review the following conversation segment from step {lo} to step {hi}:\n\n{seg}\n\n"
                f"Predict whether the most critical error is in the upper half (from step {lo} to step {mid}) "
                f"or the lower half (from step {mid + 1} to step {hi}). Respond with ONLY 'upper half' or 'lower half'."
            )
            messages = [
                {"role": "system", "content": "You are an AI assistant specializing in localizing errors in conversation segments."},
                {"role": "user", "content": prompt},
            ]
            ans = (_call_chat(client, model, messages, max_tokens) or "").lower()
            if "upper half" in ans:
                hi = mid
            elif "lower half" in ans:
                lo = min(mid + 1, hi)
            else:
                if rng.randint(0, 1) == 0:
                    hi = mid
                else:
                    lo = min(mid + 1, hi)
        agent_name = history[lo].get(index_agent, "Unknown Agent")
        print(f"\nPrediction for {fp.name}:")
        print(f"Agent Name: {agent_name}")
        print(f"Step Number: {lo}")
        print("\n" + "=" * 50 + "\n")


def _prepare_working_dir(src_dir: Path, *, limit: int | None, no_gt: bool) -> tempfile.TemporaryDirectory | None:
    if limit is None and not no_gt:
        return None
    tmp = tempfile.TemporaryDirectory(prefix="whowhen_baseline_")
    files = _sorted_json_files(src_dir)
    if limit is not None:
        files = files[:limit]
    tmp_path = Path(tmp.name)
    for fp in files:
        if no_gt:
            data = json.loads(fp.read_text(encoding="utf-8"))
            data["ground_truth"] = ""
            (tmp_path / fp.name).write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        else:
            (tmp_path / fp.name).write_bytes(fp.read_bytes())
    return tmp


def run_baseline(
    *,
    method: str,
    data_dir: str | Path,
    subset: str = "algo",
    out_path: str | Path,
    client: Any,
    model: str = "gpt-4o-mini",
    max_tokens: int = 1024,
    limit: int | None = None,
    no_gt: bool = False,
    automated_fa_dir: str | Path | None = None,
) -> Path:
    resolved_dir, _, is_handcrafted = resolve_whowhen_dir(data_dir, subset)
    utils_path = _find_upstream_utils(resolved_dir, automated_fa_dir)

    tmp_ctx = _prepare_working_dir(resolved_dir, limit=limit, no_gt=no_gt)
    active_dir = tmp_ctx.name if tmp_ctx is not None else str(resolved_dir)

    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            if utils_path is not None:
                mod = _load_upstream_module(utils_path)
                fn = getattr(mod, method)
                fn(
                    client=client,
                    directory_path=active_dir,
                    is_handcrafted=is_handcrafted,
                    model=model,
                    max_tokens=max_tokens,
                )
            else:
                fallback_map = {
                    "all_at_once": _fallback_all_at_once,
                    "step_by_step": _fallback_step_by_step,
                    "binary_search": _fallback_binary_search,
                }
                fallback_map[method](client, active_dir, is_handcrafted, model, max_tokens)
    finally:
        if tmp_ctx is not None:
            tmp_ctx.cleanup()

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(buf.getvalue(), encoding="utf-8")
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run upstream Who&When baselines (`Automated_FA/Lib/utils.py`).")
    parser.add_argument("--data", "--data-path", dest="data", required=True, help="Who&When directory path.")
    parser.add_argument("--subset", default="algo", help="Subset (`hand` or `algo`, default: `algo`).")
    parser.add_argument("--method", required=True, choices=["all_at_once", "step_by_step", "binary_search"])
    parser.add_argument("--out", required=True, help="Output log file path (parsed by `whowhen_tp.py score`).")
    parser.add_argument("--tp-lm", default=None, help="TracePress LM spec (runs via `TracePressOpenAIShim`).")
    parser.add_argument("--model", default="openai/gpt-6-luna", help="Model name for OpenAI-compatible client.")
    parser.add_argument("--base-url", default="https://openrouter.ai/api/v1", help="OpenAI-compatible base URL.")
    parser.add_argument("--api-key-env", default="OPENROUTER_API_KEY", help="Env var holding API key.")
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--limit", type=int, default=None, help="Run only the first N traces.")
    parser.add_argument("--no-gt", action="store_true", help="Strip `ground_truth` from upstream baseline prompts.")
    parser.add_argument("--automated-fa-dir", default=None, help="Optional path to `Automated_FA` directory.")
    args = parser.parse_args(argv)

    if args.tp_lm:
        client: Any = TracePressOpenAIShim(args.tp_lm)
        model_id = args.tp_lm
    else:
        from openai import OpenAI

        api_key = os.environ.get(args.api_key_env)
        if not api_key:
            raise SystemExit(f"Missing API key in env var {args.api_key_env!r} (or pass --tp-lm <spec>)")
        client = OpenAI(base_url=args.base_url, api_key=api_key)
        model_id = args.model

    out_path = run_baseline(
        method=args.method,
        data_dir=args.data,
        subset=args.subset,
        out_path=args.out,
        client=client,
        model=model_id,
        max_tokens=args.max_tokens,
        limit=args.limit,
        no_gt=args.no_gt,
        automated_fa_dir=args.automated_fa_dir,
    )
    print(f"Saved baseline output ({args.method}) to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
