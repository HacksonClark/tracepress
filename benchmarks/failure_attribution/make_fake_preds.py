#!/usr/bin/env python3
"""Generate `oracle`, `random`, and trivial constant prediction files for scorer verification & CI baselines."""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.failure_attribution.datasets import (
    TraceRecord,
    load_agentrx_traces,
    load_longrca_traces,
    load_trail_traces,
    load_whowhen_traces,
    write_whowhen_predictions,
)


def build_fake_predictions(
    traces: Sequence[TraceRecord],
    mode: str,
    *,
    seed: int = 0,
    const_agent: str | None = None,
    const_step: int | None = None,
) -> list[dict[str, Any]]:
    """Build synthetic predictions (`oracle`, `random`, `constant`, `always_websurfer`, `always_step_8`, etc.)."""
    rng = random.Random(seed)
    m = mode.strip().lower()
    preds: list[dict[str, Any]] = []

    for tr in traces:
        if tr.dataset == "trail":
            if m == "oracle":
                preds.append({"trace_id": tr.trace_id, "errors": list(tr.trail_errors)})
            elif m == "random" and tr.steps:
                sp = rng.choice(tr.steps)
                preds.append({
                    "trace_id": tr.trace_id,
                    "errors": [{"location": sp.span_id, "category": "Goal Deviation", "description": "random"}],
                })
            else:
                preds.append({"trace_id": tr.trace_id, "errors": []})
            continue

        if m == "oracle":
            preds.append({
                "trace_id": tr.trace_id,
                "predicted_agent": tr.mistake_agent,
                "predicted_step": tr.mistake_step,
                "failure_category": tr.failure_category,
                "reason": tr.mistake_reason or "Oracle ground-truth label.",
            })
        elif m == "random":
            if tr.steps:
                chosen = rng.choice(tr.steps)
                s_idx, ag = chosen.step_idx, chosen.agent
            else:
                s_idx, ag = 0, "Unknown"
            preds.append({
                "trace_id": tr.trace_id,
                "predicted_agent": ag,
                "predicted_step": s_idx,
                "failure_category": "Inconclusive",
                "reason": f"Uniform random step (seed={seed}).",
            })
        elif m in ("always_websurfer", "websurfer"):
            preds.append({
                "trace_id": tr.trace_id,
                "predicted_agent": "WebSurfer",
                "predicted_step": 0 if const_step is None else const_step,
                "reason": "Trivial constant agent baseline: WebSurfer.",
            })
        elif m in ("always_verification_expert", "verification_expert"):
            preds.append({
                "trace_id": tr.trace_id,
                "predicted_agent": "Verification_Expert",
                "predicted_step": 0 if const_step is None else const_step,
                "reason": "Trivial constant agent baseline: Verification_Expert.",
            })
        elif m == "always_step_8":
            preds.append({
                "trace_id": tr.trace_id,
                "predicted_agent": tr.speaker_at(8) or const_agent or "Unknown",
                "predicted_step": 8,
                "reason": "Trivial constant step baseline: step 8.",
            })
        elif m == "always_step_1":
            preds.append({
                "trace_id": tr.trace_id,
                "predicted_agent": tr.speaker_at(1) or const_agent or "Unknown",
                "predicted_step": 1,
                "reason": "Trivial constant step baseline: step 1.",
            })
        elif m == "constant":
            step_val = 0 if const_step is None else const_step
            agent_val = const_agent or tr.speaker_at(step_val) or "Unknown"
            preds.append({
                "trace_id": tr.trace_id,
                "predicted_agent": agent_val,
                "predicted_step": step_val,
                "reason": f"Constant baseline ({agent_val}, step {step_val}).",
            })
        else:
            raise ValueError(f"Unknown fake prediction mode {mode!r}")

    return preds


def _load_traces(dataset: str, data_path: str | Path, subset: str | None, limit: int | None) -> list[TraceRecord]:
    ds = dataset.strip().lower()
    if ds == "whowhen":
        return load_whowhen_traces(data_path, subset=subset or "algo", limit=limit)
    if ds == "longrca":
        return load_longrca_traces(data_path, subset=subset or "mini", limit=limit)
    if ds == "agentrx":
        return load_agentrx_traces(data_path, subset=subset or "auto", limit=limit)
    if ds == "trail":
        return load_trail_traces(data_path, subset=subset, limit=limit)
    raise ValueError(f"Unknown dataset {dataset!r}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate oracle, random, or constant prediction files.")
    parser.add_argument("--dataset", default="whowhen", choices=["whowhen", "longrca", "agentrx", "trail"])
    parser.add_argument("--data", "--data-path", dest="data", required=True)
    parser.add_argument("--subset", default=None)
    parser.add_argument(
        "--mode",
        default="oracle",
        choices=[
            "oracle",
            "random",
            "constant",
            "always_websurfer",
            "always_verification_expert",
            "always_step_8",
            "always_step_1",
        ],
    )
    parser.add_argument("--agent", default=None, help="Constant agent name when --mode=constant.")
    parser.add_argument("--step", type=int, default=None, help="Constant step index when --mode=constant.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", required=True, help="Output file (.txt, .json, or .jsonl).")
    args = parser.parse_args(argv)

    traces = _load_traces(args.dataset, args.data, args.subset, args.limit)
    preds = build_fake_predictions(
        traces,
        args.mode,
        seed=args.seed,
        const_agent=args.agent,
        const_step=args.step,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix == ".json" or args.dataset == "trail":
        out_path.write_text(json.dumps(preds, indent=2) + "\n", encoding="utf-8")
    elif out_path.suffix == ".jsonl":
        out_path.write_text("\n".join(json.dumps(p) for p in preds) + "\n", encoding="utf-8")
    else:
        write_whowhen_predictions(preds, out_path, ensure_json_ext=(args.dataset == "whowhen"))

    print(f"Generated {len(preds)} '{args.mode}' predictions -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
