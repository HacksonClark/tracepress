#!/usr/bin/env python3
"""Empirical LM call, prompt-token, USD cost, and sequential-depth estimator across failure-attribution methods."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.failure_attribution.datasets.common import TraceRecord
from benchmarks.failure_attribution.datasets.longrca import load_longrca_traces
from benchmarks.failure_attribution.datasets.whowhen import load_whowhen_traces


def _make_token_counter():
    try:
        import tiktoken

        enc = tiktoken.get_encoding("o200k_base")
        return lambda text: len(enc.encode(text or "", disallowed_special=()))
    except Exception:
        return lambda text: max(1, len(text or "") // 4) if text else 0


def estimate_suite_costs(
    traces: Sequence[TraceRecord],
    *,
    flag_rate: float = 0.25,
    vote_k: int = 1,
    chunk_tokens: int = 30_000,
    input_price_per_mtok: float = 0.10,
    output_price_per_mtok: float = 0.50,
) -> dict[str, Any]:
    """Compute exact call counts, prompt tokens, completion tokens, USD cost, and sequential depth
    for 7 methods on a concrete list of `TraceRecord`s.
    """
    count_tok = _make_token_counter()
    n_traces = len(traces)
    if n_traces == 0:
        return {"n_traces": 0, "methods": {}}

    totals: dict[str, dict[str, float]] = {
        m: {"calls": 0.0, "prompt_tokens": 0.0, "completion_tokens": 0.0, "depth_sum": 0.0, "max_depth": 0.0}
        for m in (
            "all_at_once",
            "step_by_step (early_stop)",
            "step_by_step (full)",
            "binary_search",
            "tp map_reduce",
            "tp fold",
            f"tp staged (k={vote_k})",
        )
    }

    total_steps = 0
    step_toks_all: list[int] = []

    for tr in traces:
        q_tok = count_tok(tr.question) + count_tok(tr.ground_truth) + 120
        s_toks = [count_tok(f"Step {s.step_idx} - {s.raw_agent}: {s.content}\n") for s in tr.steps]
        ctx_toks = [count_tok(s.prev_context) + count_tok(s.upstream_handoff) for s in tr.steps]
        t_len = max(len(s_toks), 1)
        total_steps += len(s_toks)
        full_trace_tok = sum(s_toks)
        step_toks_all.append(full_trace_tok)

        # 1. `all_at_once` (tree-chunks when trace > chunk_tokens)
        n_chunks = max(1, math.ceil(full_trace_tok / chunk_tokens))
        aao_calls = n_chunks + (n_chunks - 1 if n_chunks > 1 else 0)
        aao_depth = 1 + (math.ceil(math.log2(n_chunks)) if n_chunks > 1 else 0)
        aao_prompt = q_tok * aao_calls + full_trace_tok + (n_chunks - 1) * 350
        _acc(totals["all_at_once"], aao_calls, aao_prompt, aao_calls * 180, aao_depth)

        # 2. `step_by_step (early_stop)` (stops at `mistake_step`, or full trace if unlabeled)
        stop_idx = tr.mistake_step if (tr.mistake_step is not None and 0 <= tr.mistake_step < t_len) else (t_len - 1)
        # Adjust 1-based AgentRx step_idx if needed:
        if tr.dataset == "agentrx" and tr.mistake_step is not None:
            stop_idx = min(max(tr.mistake_step - 1, 0), t_len - 1)
        sbs_early_calls = stop_idx + 1
        prefix_sum_early = 0
        running = 0
        for idx in range(sbs_early_calls):
            running += s_toks[idx] if idx < len(s_toks) else 0
            prefix_sum_early += q_tok + running + 90
        _acc(totals["step_by_step (early_stop)"], sbs_early_calls, prefix_sum_early, sbs_early_calls * 90, sbs_early_calls)

        # 3. `step_by_step (full)` (evaluates all T steps sequentially with growing prefix)
        prefix_sum_full = 0
        running = 0
        for tok in s_toks:
            running += tok
            prefix_sum_full += q_tok + running + 90
        _acc(totals["step_by_step (full)"], t_len, prefix_sum_full, t_len * 90, t_len)

        # 4. `binary_search` (halves segment `[lo, hi]` down to `stop_idx`)
        lo, hi = 0, len(s_toks) - 1
        bs_calls = 0
        bs_prompt = 0
        while lo < hi:
            bs_calls += 1
            seg_tok = sum(s_toks[lo : hi + 1])
            bs_prompt += q_tok + seg_tok + 140
            mid = lo + (hi - lo) // 2
            if stop_idx <= mid:
                hi = mid
            else:
                lo = min(mid + 1, hi)
        bs_calls = max(bs_calls, 1)
        _acc(totals["binary_search"], bs_calls, bs_prompt, bs_calls * 20, bs_calls)

        # 5. `tp map_reduce` (parallel per-step digest + 1 trace reduce; depth = 2)
        mr_map_prompt = sum(q_tok + s_toks[i] + ctx_toks[i] + 100 for i in range(len(s_toks)))
        mr_red_prompt = q_tok + len(s_toks) * 65 + 150
        mr_calls = len(s_toks) + 1
        _acc(totals["tp map_reduce"], mr_calls, mr_map_prompt + mr_red_prompt, len(s_toks) * 60 + 200, 2)

        # 6. `tp fold` (sequential bounded-state fold; depth = T)
        fold_prompt = sum(q_tok + tok + 280 for tok in s_toks)
        _acc(totals["tp fold"], t_len, fold_prompt, t_len * 140, t_len)

        # 7. `tp staged (k=vote_k)` (Stage 1 story + k * [Stage 2 screen + Stage 3 trace_back on flagged + Stage 4 decide])
        story_calls = n_chunks + (n_chunks - 1 if n_chunks > 1 else 0)
        story_depth = 1 + (math.ceil(math.log2(n_chunks)) if n_chunks > 1 else 0)
        story_prompt = q_tok * story_calls + full_trace_tok
        story_comp = story_calls * 220

        n_flagged = max(1, round(len(s_toks) * flag_rate))
        avg_step_plus_ctx = (sum(s_toks) + sum(ctx_toks)) / t_len
        screen_prompt = sum(q_tok + 220 + s_toks[i] + ctx_toks[i] + 120 for i in range(len(s_toks)))
        tb_prompt = n_flagged * (q_tok + 220 + n_flagged * 55 + avg_step_plus_ctx + 150)
        decide_prompt = q_tok + 220 + n_flagged * 110 + 180

        branch_calls = len(s_toks) + n_flagged + 1
        branch_prompt = screen_prompt + tb_prompt + decide_prompt
        branch_comp = len(s_toks) * 45 + n_flagged * 85 + 220

        st_calls = story_calls + vote_k * branch_calls
        st_prompt = story_prompt + vote_k * branch_prompt
        st_comp = story_comp + vote_k * branch_comp
        st_depth = story_depth + 3  # screen (1) -> trace_back (1) -> decide (1)
        _acc(totals[f"tp staged (k={vote_k})"], st_calls, st_prompt, st_comp, st_depth)

    summary_methods: dict[str, dict[str, float]] = {}
    for name, d in totals.items():
        p_mtok = d["prompt_tokens"] / 1e6
        c_mtok = d["completion_tokens"] / 1e6
        usd = p_mtok * input_price_per_mtok + c_mtok * output_price_per_mtok
        summary_methods[name] = {
            "calls": int(round(d["calls"])),
            "calls_per_trace": d["calls"] / n_traces,
            "prompt_mtok": p_mtok,
            "completion_mtok": c_mtok,
            "est_cost_usd": usd,
            "mean_seq_depth": d["depth_sum"] / n_traces,
            "max_seq_depth": int(round(d["max_depth"])),
        }

    sorted_toks = sorted(step_toks_all)
    median_toks = sorted_toks[len(sorted_toks) // 2]
    sorted_steps = sorted(tr.n_steps for tr in traces)
    median_steps = sorted_steps[len(sorted_steps) // 2]

    return {
        "n_traces": n_traces,
        "total_steps": total_steps,
        "mean_steps": total_steps / n_traces,
        "median_steps": median_steps,
        "max_steps": max(sorted_steps),
        "mean_trace_tokens": sum(step_toks_all) / n_traces,
        "median_trace_tokens": median_toks,
        "max_trace_tokens": max(sorted_toks),
        "pricing_per_mtok": {"input": input_price_per_mtok, "output": output_price_per_mtok},
        "methods": summary_methods,
    }


def _acc(bucket: dict[str, float], calls: float, prompt: float, comp: float, depth: float) -> None:
    bucket["calls"] += calls
    bucket["prompt_tokens"] += prompt
    bucket["completion_tokens"] += comp
    bucket["depth_sum"] += depth
    bucket["max_depth"] = max(bucket["max_depth"], depth)


def format_markdown_table(title: str, report: dict[str, Any]) -> str:
    lines = [
        f"### {title} (`N={report['n_traces']}`, total steps={report['total_steps']:,}, "
        f"mean/median/max steps={report['mean_steps']:.1f}/{report['median_steps']}/{report['max_steps']}, "
        f"mean/max trace tok={report['mean_trace_tokens']:,.0f}/{report['max_trace_tokens']:,.0f})",
        "",
        "| Method | LM Calls | Calls/Trace | Prompt Mtok | Output Mtok | Est. USD ($0.10/$0.50) | Mean Seq Depth | Max Seq Depth |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for mname, stats in report["methods"].items():
        lines.append(
            f"| `{mname}` | {stats['calls']:,} | {stats['calls_per_trace']:.1f} | "
            f"{stats['prompt_mtok']:.3f} | {stats['completion_mtok']:.3f} | "
            f"${stats['est_cost_usd']:.3f} | {stats['mean_seq_depth']:.1f} | {stats['max_seq_depth']} |"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Estimate failure-attribution LM calls, tokens, cost, and depth.")
    parser.add_argument("--whowhen-dir", default=None, help="Path to Who&When root directory.")
    parser.add_argument("--longrca-path", default=None, help="Path to LongRCA parquet/JSON directory.")
    parser.add_argument("--flag-rate", type=float, default=0.25, help="Assumed Stage-2 flag rate for `staged` (default: 0.25).")
    parser.add_argument("--vote-k", type=int, default=1, help="Vote rounds for `tp staged` (default: 1).")
    parser.add_argument("--input-price", type=float, default=0.10, help="USD per 1M input tokens (default: $0.10).")
    parser.add_argument("--output-price", type=float, default=0.50, help="USD per 1M output tokens (default: $0.50).")
    parser.add_argument("--json-out", default=None, help="Optional JSON output file.")
    args = parser.parse_args(argv)

    results: dict[str, Any] = {}

    if args.whowhen_dir:
        for sub_key, label in (("hand", "Who&When Hand-Crafted"), ("algo", "Who&When Algorithm-Generated")):
            traces = load_whowhen_traces(args.whowhen_dir, subset=sub_key)
            rep = estimate_suite_costs(
                traces,
                flag_rate=args.flag_rate,
                vote_k=args.vote_k,
                input_price_per_mtok=args.input_price,
                output_price_per_mtok=args.output_price,
            )
            results[f"whowhen_{sub_key}"] = rep
            print(format_markdown_table(label, rep))
            print()

    if args.longrca_path:
        traces = load_longrca_traces(args.longrca_path, subset="mini")
        rep = estimate_suite_costs(
            traces,
            flag_rate=args.flag_rate,
            vote_k=args.vote_k,
            input_price_per_mtok=args.input_price,
            output_price_per_mtok=args.output_price,
        )
        results["longrca_mini"] = rep
        print(format_markdown_table("LongRCA-Mini", rep))
        print()

    if not results:
        parser.error("Provide at least one of --whowhen-dir or --longrca-path")

    if args.json_out:
        out_p = Path(args.json_out)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
