#!/usr/bin/env python3
"""TracePress failure-attribution CLI (`run` and `score`) across Who&When, LongRCA, AgentRx, and TRAIL."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import tracepress as tp

from benchmarks.failure_attribution.datasets import (
    FailureTraceAdapter,
    load_agentrx,
    load_longrca,
    load_trail,
    load_whowhen,
    paired_diff_whowhen,
    score_agentrx,
    score_longrca,
    score_trail,
    score_whowhen,
    trivial_baselines,
    write_whowhen_predictions,
)
from benchmarks.failure_attribution.pipelines import (
    all_at_once,
    fold,
    map_reduce,
    staged,
    trail_pipeline,
)


def _load_adapter(
    dataset: str,
    data_path: str | Path,
    subset: str | None,
    *,
    gt_path: str | Path | None = None,
    window: int = 3,
    limit: int | None = None,
) -> FailureTraceAdapter:
    ds = dataset.strip().lower()
    if ds == "whowhen":
        return load_whowhen(data_path, subset=subset or "algo", window=window, limit=limit)
    if ds == "longrca":
        return load_longrca(data_path, subset=subset or "mini", window=window, limit=limit)
    if ds == "agentrx":
        return load_agentrx(
            data_path,
            subset=subset or "auto",
            ground_truth_path=gt_path,
            window=window,
            limit=limit,
        )
    if ds == "trail":
        return load_trail(data_path, subset=subset, window=window, limit=limit)
    raise ValueError(f"Unknown dataset {dataset!r}; choose from whowhen, longrca, agentrx, trail")


def _write_outputs(dataset: str, rows: list[Any], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ds = dataset.strip().lower()

    if ds == "trail":
        if out_path.is_dir() or out_path.suffix == "":
            out_path.mkdir(parents=True, exist_ok=True)
            for r in rows:
                tid = str(r.trace)
                fname = tid if tid.endswith(".json") else f"{tid}.json"
                (out_path / fname).write_text(
                    json.dumps({"trace_id": tid, "errors": list(r.errors)}, indent=2) + "\n",
                    encoding="utf-8",
                )
        elif out_path.suffix == ".jsonl":
            lines = [
                json.dumps({"trace_id": str(r.trace), "errors": list(r.errors)})
                for r in rows
            ]
            out_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        else:
            payload = {str(r.trace): {"trace_id": str(r.trace), "errors": list(r.errors)} for r in rows}
            out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        return

    pred_records = []
    for r in rows:
        item: dict[str, Any] = {
            "trace_id": str(r.trace),
            "predicted_agent": str(r.mistake_agent),
            "predicted_step": int(r.mistake_step),
            "reason": str(r.get("reason", "") or ""),
        }
        if "failure_category" in r.values:
            item["failure_category"] = str(r.failure_category)
        if "vote_agreement" in r.values:
            item["vote_agreement"] = float(r.vote_agreement)
        pred_records.append(item)

    if out_path.suffix == ".json":
        out_path.write_text(json.dumps(pred_records, indent=2) + "\n", encoding="utf-8")
    elif out_path.suffix == ".jsonl":
        lines = [json.dumps(item) for item in pred_records]
        out_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    else:
        write_whowhen_predictions(pred_records, out_path, ensure_json_ext=(ds == "whowhen"))


def _print_score_summary(dataset: str, label: str, res: dict[str, Any]) -> None:
    ds = dataset.strip().lower()
    print(f"\n=== {label} ({ds}, n={res.get('n_traces', 0)}) ===")
    if ds == "whowhen":
        ci = res.get("ci95", {})
        for key, title in (
            ("agent", "Agent Exact"),
            ("step", "Step Exact"),
            ("both_exact", "Both Exact"),
            ("agent_from_step", "Agent-from-Step"),
            ("step_pm1", "Step +-1"),
            ("step_pm3", "Step +-3"),
            ("step_pm5", "Step +-5"),
            ("paper_agent", "Paper Substring Agent"),
            ("paper_step", "Paper Substring Step"),
        ):
            if key not in res:
                continue
            val = res[key] * 100.0
            if key in ci:
                lo, hi = ci[key][0] * 100.0, ci[key][1] * 100.0
                print(f"  {title:<24}: {val:6.2f}%  [95% CI: {lo:6.2f}%, {hi:6.2f}%]")
            else:
                print(f"  {title:<24}: {val:6.2f}%")
    elif ds == "longrca":
        ci = res.get("ci95", {})
        for key, title in (
            ("agent_acc", "Role Accuracy"),
            ("step_acc", "Root Exact"),
            ("step_pm5", "Root Within +-5"),
            ("both_exact", "Both Exact"),
        ):
            val = res[key] * 100.0
            lo, hi = ci[key][0] * 100.0, ci[key][1] * 100.0
            print(f"  {title:<24}: {val:6.2f}%  [95% CI: {lo:6.2f}%, {hi:6.2f}%]")
        mae_s = f"{res['root_mae']:.2f}" if res.get("root_mae") is not None else "N/A"
        micro_s = f"{res['micro_mae']:.2f}" if res.get("micro_mae") is not None else "N/A"
        print(f"  {'Source-Weighted MAE':<24}: {mae_s}  (micro MAE: {micro_s}, valid={res['valid_step_n']})")
    elif ds == "agentrx":
        ci = res.get("ci95", {})
        s_lo, s_hi = ci["step_acc"][0] * 100.0, ci["step_acc"][1] * 100.0
        c_lo, c_hi = ci["category_acc"][0] * 100.0, ci["category_acc"][1] * 100.0
        print(f"  {'Step Exact':<24}: {res['step_acc']*100:6.2f}%  [95% CI: {s_lo:6.2f}%, {s_hi:6.2f}%]")
        print(f"  {'Step +-1 / +-3 / +-5':<24}: {res['step_pm1']*100:5.1f}% / {res['step_pm3']*100:5.1f}% / {res['step_pm5']*100:5.1f}%")
        print(f"  {'Root Category Exact':<24}: {res['category_acc']*100:6.2f}%  [95% CI: {c_lo:6.2f}%, {c_hi:6.2f}%]")
        mae_s = f"{res['step_mae']:.2f}" if res.get("step_mae") is not None else "N/A"
        nd_s = f"{res['norm_step_dist']:.4f}" if res.get("norm_step_dist") is not None else "N/A"
        print(f"  {'Step MAE / Norm Dist':<24}: {mae_s} / {nd_s}")
    elif ds == "trail":
        ci = res.get("ci95", {})
        l_lo, l_hi = ci["location_recall"][0] * 100.0, ci["location_recall"][1] * 100.0
        j_lo, j_hi = ci["joint_location_category_recall"][0] * 100.0, ci["joint_location_category_recall"][1] * 100.0
        print(f"  {'Location Recall':<24}: {res['location_recall']*100:6.2f}%  [95% CI: {l_lo:6.2f}%, {l_hi:6.2f}%]")
        print(f"  {'Joint Loc+Cat Recall':<24}: {res['joint_location_category_recall']*100:6.2f}%  [95% CI: {j_lo:6.2f}%, {j_hi:6.2f}%]")
        print(f"  {'Category Weighted F1':<24}: {res['weighted_f1']*100:6.2f}%")


def _score_one(dataset: str, src: FailureTraceAdapter, pred_path: str | Path, *, n_boot: int, seed: int) -> dict[str, Any]:
    ds = dataset.strip().lower()
    if ds == "whowhen":
        return score_whowhen(src, pred_path, n_boot=n_boot, seed=seed)
    if ds == "longrca":
        return score_longrca(src, pred_path, n_boot=n_boot, seed=seed)
    if ds == "agentrx":
        return score_agentrx(src, pred_path, n_boot=n_boot, seed=seed)
    if ds == "trail":
        return score_trail(src, pred_path, n_boot=n_boot, seed=seed)
    raise ValueError(f"Unsupported dataset {dataset!r}")


def cmd_run(args: argparse.Namespace) -> int:
    tp.configure(telemetry=False)
    if args.lm:
        tp.configure(lm=args.lm)

    src = _load_adapter(
        args.dataset,
        args.data,
        args.subset,
        gt_path=args.gt_path,
        window=args.window,
        limit=args.limit,
    )
    with_gt = not args.no_gt
    with_category = args.dataset.lower() == "agentrx"

    method = args.method.strip().lower()
    if args.dataset.lower() == "trail" or method == "trail_pipeline":
        plan = trail_pipeline(src, lm=args.lm, screen_lm=args.screen_lm)
    elif method == "all_at_once":
        plan = all_at_once(src, with_gt=with_gt, with_category=with_category, lm=args.lm)
    elif method == "map_reduce":
        plan = map_reduce(
            src,
            with_gt=with_gt,
            with_category=with_category,
            lm=args.lm,
            screen_lm=args.screen_lm,
        )
    elif method == "fold":
        plan = fold(src, with_gt=with_gt, with_category=with_category, lm=args.lm)
    elif method == "staged":
        plan = staged(
            src,
            with_gt=with_gt,
            with_category=with_category,
            window=args.window,
            vote_k=args.vote_k,
            lm=args.lm,
            screen_lm=args.screen_lm,
            compiled_state=args.compiled_state,
        )
    else:
        raise ValueError(f"Unknown method {args.method!r}")

    rows = plan.collect(max_concurrency=args.max_concurrency)
    stats = tp.last_run()

    out_path = Path(args.out)
    _write_outputs(args.dataset, rows, out_path)
    print(f"Wrote {len(rows)} predictions to {out_path}")
    if stats is not None:
        print(f"Run stats: {stats}")

    has_labels = any(
        (tr.trail_errors if args.dataset.lower() == "trail" else tr.mistake_step is not None)
        for tr in src.trace_records
    )
    if has_labels:
        res = _score_one(args.dataset, src, out_path, n_boot=args.n_boot, seed=args.seed)
        _print_score_summary(args.dataset, f"{method} ({out_path.name})", res)
    return 0


def cmd_score(args: argparse.Namespace) -> int:
    src = _load_adapter(
        args.dataset,
        args.data,
        args.subset,
        gt_path=args.gt_path,
        limit=args.limit,
    )
    pred_files = list(args.preds or [])
    if args.eval_file:
        pred_files.append(args.eval_file)

    all_results: dict[str, Any] = {}

    if args.trivial and args.dataset.lower() == "whowhen":
        triv = trivial_baselines(src, seed=args.seed, n_boot=args.n_boot)
        all_results["trivial_baselines"] = triv
        for bname, bres in triv.items():
            _print_score_summary("whowhen", f"Trivial: {bname}", bres)

    if not pred_files and not args.trivial:
        raise SystemExit("score requires at least one prediction file (or --trivial on whowhen)")

    scored_list: list[tuple[str, dict[str, Any]]] = []
    for pf in pred_files:
        res = _score_one(args.dataset, src, pf, n_boot=args.n_boot, seed=args.seed)
        label = Path(pf).name
        scored_list.append((pf, res))
        all_results[str(pf)] = res
        _print_score_summary(args.dataset, label, res)

    if len(scored_list) >= 2 and args.dataset.lower() == "whowhen":
        ref_path, _ = scored_list[0]
        ref_name = Path(ref_path).name
        all_results["paired_diffs"] = {}
        for other_path, _ in scored_list[1:]:
            other_name = Path(other_path).name
            diffs = paired_diff_whowhen(src, ref_path, other_path, n_boot=args.n_boot, seed=args.seed)
            all_results["paired_diffs"][f"{ref_name}_minus_{other_name}"] = diffs
            print(f"\n=== Paired Diff: {ref_name} - {other_name} ===")
            for k in ("agent", "step", "both_exact", "agent_from_step", "step_pm1", "step_pm5"):
                d = diffs[k]
                print(
                    f"  {k:<20}: {d['diff']*100:+6.2f}%  "
                    f"[95% CI: {d['ci_low']*100:+6.2f}%, {d['ci_high']*100:+6.2f}%]"
                )

    if args.json_out:
        out_p = Path(args.json_out)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps(all_results, indent=2, default=str) + "\n", encoding="utf-8")
        print(f"\nSaved JSON metrics to {out_p}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="TracePress Failure Attribution Runner & Scorer.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="Run a TracePress failure-attribution pipeline.")
    p_run.add_argument("--dataset", default="whowhen", choices=["whowhen", "longrca", "agentrx", "trail"])
    p_run.add_argument("--data", "--data-path", dest="data", required=True, help="Dataset directory or file path.")
    p_run.add_argument("--gt-path", default=None, help="Optional separate ground-truth JSON path (for AgentRx).")
    p_run.add_argument("--subset", default=None, help="Subset (e.g. hand|algo, mini|full, tau|magentic, gaia|swe_bench).")
    p_run.add_argument(
        "--method",
        default="staged",
        choices=["all_at_once", "map_reduce", "fold", "staged", "trail_pipeline"],
    )
    p_run.add_argument("--lm", default=None, help="TracePress LM spec (e.g. openrouter:openai/gpt-6-luna).")
    p_run.add_argument("--screen-lm", default=None, help="Optional screening LM for map_reduce / staged / trail.")
    p_run.add_argument("--vote-k", type=int, default=1, help="Number of majority-vote rounds in staged (default: 1).")
    p_run.add_argument("--window", type=int, default=3, help="Sliding context window size (default: 3).")
    p_run.add_argument("--compiled-state", default=None, help="Optional DSPy compiled state JSON for staged.")
    p_run.add_argument("--no-gt", action="store_true", help="Withhold ground_truth answer from LM prompts.")
    p_run.add_argument("--limit", type=int, default=None, help="Evaluate only the first N traces.")
    p_run.add_argument("--max-concurrency", type=int, default=None, help="Override TracePress max concurrency.")
    p_run.add_argument("--out", required=True, help="Output prediction path (.txt, .json, .jsonl, or directory for trail).")
    p_run.add_argument("--n-boot", type=int, default=2000, help="Bootstrap resamples for 95%% CIs (default: 2000).")
    p_run.add_argument("--seed", type=int, default=0, help="Bootstrap RNG seed (default: 0).")
    p_run.set_defaults(func=cmd_run)

    p_score = sub.add_parser("score", help="Score one or more prediction files with bootstrap 95%% CIs.")
    p_score.add_argument("--dataset", default="whowhen", choices=["whowhen", "longrca", "agentrx", "trail"])
    p_score.add_argument("--data", "--data-path", dest="data", required=True, help="Reference dataset directory or file.")
    p_score.add_argument("--gt-path", default=None, help="Optional separate ground-truth JSON path (for AgentRx).")
    p_score.add_argument("--subset", default=None, help="Subset (e.g. hand|algo, mini|full, tau|magentic, gaia|swe_bench).")
    p_score.add_argument("--limit", type=int, default=None, help="Score against the first N reference traces.")
    p_score.add_argument("--eval-file", default=None, help="Backward-compatible alias for a prediction file.")
    p_score.add_argument("--trivial", action="store_true", help="Also report zero-LM trivial baselines (Who&When).")
    p_score.add_argument("--json-out", default=None, help="Optional path to write full JSON metrics.")
    p_score.add_argument("--n-boot", type=int, default=2000, help="Bootstrap resamples for 95%% CIs (default: 2000).")
    p_score.add_argument("--seed", type=int, default=0, help="Bootstrap RNG seed (default: 0).")
    p_score.add_argument("preds", nargs="*", help="One or more prediction files to evaluate.")
    p_score.set_defaults(func=cmd_score)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
