"""LongRCA (`CLoud5-real/longrca-bench`) dataset loader and official scorer."""

from __future__ import annotations

import json
import re
import unicodedata
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from .common import (
    FailureTraceAdapter,
    StepRecord,
    TraceRecord,
    attach_context,
    bootstrap_ci,
    clip_text,
)
from .whowhen import lookup_prediction, parse_whowhen_predictions

BENCHMARK_EXPECTED_N: dict[str, int] = {
    "swe_bench_pro": 128,
    "terminal_bench_2": 42,
    "travelplanner": 685,
    "vitabench": 108,
    "webarena_verified": 177,
}

ARROW_SUFFIX_RE = re.compile(r"\s*\((?:->|→)\s*[^()]+\)\s*$")


def normalize_role(value: Any) -> str:
    """Official LongRCA role normalization: strip `(-> target)` suffix + NFKC casefold."""
    text = str(value or "").strip()
    text = ARROW_SUFFIX_RE.sub("", text)
    text = unicodedata.normalize("NFKC", text).casefold()
    return " ".join(text.strip().split())


def canonical_role(value: Any) -> str:
    """Display-preserving role string with trailing `(-> target)` stripped."""
    text = str(value or "").strip()
    return ARROW_SUFFIX_RE.sub("", text).strip()


def benchmark_of_qid(qid: str) -> str:
    """Extract the source benchmark prefix from `question_ID` (`<benchmark>__<id>`)."""
    return str(qid or "").split("__", 1)[0]


def _extract_question(row: Mapping[str, Any], history: Sequence[Mapping[str, Any]]) -> str:
    if row.get("question"):
        return str(row["question"])
    for entry in history[:3]:
        content = str(entry.get("content") or "").strip()
        if content and len(content) > 20:
            return clip_text(content, 1600)
    return str(history[0].get("content") or "") if history else ""


def _record_from_row(row: Mapping[str, Any], *, subset_hint: str = "", window: int = 3) -> TraceRecord:
    qid = str(row.get("question_ID") or row.get("qid") or row.get("id") or "")
    bench = benchmark_of_qid(qid)
    raw_hist = row.get("history")
    history = list(raw_hist) if raw_hist is not None else []

    steps: list[StepRecord] = []
    for idx, entry in enumerate(history):
        s_idx = int(entry.get("step", idx))
        raw_spk = str(entry.get("name") or entry.get("role") or "unknown")
        steps.append(
            StepRecord(
                step_idx=s_idx,
                agent=canonical_role(raw_spk) or raw_spk,
                raw_agent=raw_spk,
                content=str(entry.get("content") or ""),
                extra={"role": entry.get("role", ""), "name": entry.get("name", "")},
            )
        )
    attach_context(steps, window=window)

    ms_raw = row.get("mistake_step")
    mistake_step = int(ms_raw) if ms_raw is not None and str(ms_raw).strip() != "" else None

    return TraceRecord(
        trace_id=qid,
        dataset="longrca",
        subset=subset_hint or bench,
        question=_extract_question(row, history),
        ground_truth=str(row.get("ground_truth") or ""),
        steps=steps,
        mistake_step=mistake_step,
        mistake_agent=str(row.get("mistake_agent") or "").strip(),
        mistake_reason=str(row.get("mistake_reason") or ""),
        extra={"benchmark": bench, "question_ID": qid},
    )


def load_longrca_from_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    subset: str | None = None,
    window: int = 3,
    limit: int | None = None,
) -> list[TraceRecord]:
    want = (subset or "").strip().lower()
    filter_bench = want if want in BENCHMARK_EXPECTED_N else None
    out: list[TraceRecord] = []
    for row in rows:
        qid = str(row.get("question_ID") or row.get("qid") or "")
        if filter_bench and benchmark_of_qid(qid) != filter_bench:
            continue
        out.append(_record_from_row(row, subset_hint=want or benchmark_of_qid(qid), window=window))
        if limit is not None and len(out) >= limit:
            break
    return out


def _read_parquet_rows(parquet_path: Path) -> list[dict[str, Any]]:
    import pandas as pd

    try:
        df = pd.read_parquet(parquet_path)
    except ImportError as exc:
        raise ImportError(
            f"Reading {parquet_path} requires pyarrow or fastparquet "
            "(e.g. `uv run --with pyarrow ...`), or pass a JSON/JSONL directory instead."
        ) from exc
    return df.to_dict("records")


def load_longrca_traces(
    data_path: str | Path | Any,
    subset: str | None = "mini",
    *,
    window: int = 3,
    limit: int | None = None,
) -> list[TraceRecord]:
    """Load LongRCA trajectories from a `.parquet` file, JSON directory (`data/{mini,full}/*/*.json`),
    `.jsonl` file, or pandas `DataFrame`.
    """
    if hasattr(data_path, "to_dict") and callable(data_path.to_dict):
        return load_longrca_from_rows(data_path.to_dict("records"), subset=subset, window=window, limit=limit)

    p = Path(data_path)
    want = (subset or "mini").strip().lower()
    split = "full" if want == "full" else "mini"

    if p.is_file():
        if p.suffix == ".parquet":
            return load_longrca_from_rows(_read_parquet_rows(p), subset=subset, window=window, limit=limit)
        if p.suffix == ".jsonl":
            rows = [json.loads(ln) for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]
            return load_longrca_from_rows(rows, subset=subset, window=window, limit=limit)
        if p.suffix == ".json":
            obj = json.loads(p.read_text(encoding="utf-8"))
            rows = obj if isinstance(obj, list) else [obj]
            return load_longrca_from_rows(rows, subset=subset, window=window, limit=limit)

    if p.is_dir():
        # 1. Prefer split JSON directory if present: `data/{mini,full}/*/*.json` or `{mini,full}/*/*.json`
        for candidate_dir in (p / "data" / split, p / split):
            if candidate_dir.is_dir():
                json_files = sorted(candidate_dir.rglob("*.json"))
                if json_files:
                    return _load_json_files(json_files, subset=subset, window=window, limit=limit)

        # 2. Check for `longrca-{mini,full}.parquet` or `.jsonl` in directory
        pq = p / f"longrca-{split}.parquet"
        jl = p / f"longrca-{split}.jsonl"
        if jl.is_file():
            rows = [json.loads(ln) for ln in jl.read_text(encoding="utf-8").splitlines() if ln.strip()]
            return load_longrca_from_rows(rows, subset=subset, window=window, limit=limit)
        if pq.is_file():
            return load_longrca_from_rows(_read_parquet_rows(pq), subset=subset, window=window, limit=limit)

        # 3. Any nested `*.json` trajectory files directly under `p`
        json_files = sorted(f for f in p.rglob("*.json") if ".cache" not in f.parts)
        if json_files:
            return _load_json_files(json_files, subset=subset, window=window, limit=limit)

    raise FileNotFoundError(f"Could not find LongRCA data at {data_path!r} (subset={subset!r})")


def _load_json_files(
    files: Sequence[Path],
    *,
    subset: str | None,
    window: int,
    limit: int | None,
) -> list[TraceRecord]:
    want = (subset or "").strip().lower()
    filter_bench = want if want in BENCHMARK_EXPECTED_N else None
    out: list[TraceRecord] = []
    for fp in files:
        if filter_bench and fp.parent.name != filter_bench and not fp.stem.startswith(filter_bench):
            continue
        row = json.loads(fp.read_text(encoding="utf-8"))
        if not isinstance(row, Mapping) or "history" not in row:
            continue
        out.append(_record_from_row(row, subset_hint=want or benchmark_of_qid(fp.stem), window=window))
        if limit is not None and len(out) >= limit:
            break
    return out


def load_longrca(
    data_path: str | Path | Any,
    subset: str | None = "mini",
    *,
    window: int = 3,
    limit: int | None = None,
) -> FailureTraceAdapter:
    """Load LongRCA as a `FailureTraceAdapter` (`src.traces` and `src.steps`)."""
    return FailureTraceAdapter(load_longrca_traces(data_path, subset=subset, window=window, limit=limit))


# --- Official LongRCA Scorer -------------------------------------------------------------


def score_longrca(
    traces_or_adapter: Sequence[TraceRecord] | FailureTraceAdapter,
    predictions: Mapping[str, Mapping[str, Any]] | str | Path,
    *,
    benchmark_weights: Mapping[str, float] | None = None,
    n_boot: int = 2000,
    seed: int = 0,
) -> dict[str, Any]:
    """Score LongRCA predictions according to the official leaderboard contract:

    - `agent_acc`: independent explicit `predicted_agent` match using `normalize_role`
      (step-emitter roles are intentionally NOT substituted for missing/wrong agent predictions).
    - `step_acc`: exact `predicted_step == mistake_step`.
    - `step_pm5`: `|predicted_step - mistake_step| <= 5`.
    - `both_exact`: joint `agent_acc` and `step_acc`.
    - `root_mae`: source-weighted MAE across the 5 source benchmarks (`question_ID.split("__")[0]`),
      weighted by `BENCHMARK_EXPECTED_N` (128/42/685/108/177 on Full; unbiased stratified estimate on Mini),
      alongside `micro_mae`, `macro_mae`, and `by_benchmark`.
    """
    traces = traces_or_adapter.trace_records if isinstance(traces_or_adapter, FailureTraceAdapter) else list(traces_or_adapter)
    preds = (
        parse_whowhen_predictions(predictions)
        if isinstance(predictions, (str, Path))
        else dict(predictions)
    )
    weights_map = dict(BENCHMARK_EXPECTED_N if benchmark_weights is None else benchmark_weights)

    agent_vec: list[float] = []
    step_vec: list[float] = []
    pm5_vec: list[float] = []
    both_vec: list[float] = []
    all_abs_errs: list[float] = []

    by_bench_rows: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {"agent": [], "step": [], "pm5": [], "both": [], "abs_err": []}
    )

    valid_step_n = 0
    failed_n = 0

    for tr in traces:
        bench = tr.extra.get("benchmark") or benchmark_of_qid(tr.trace_id)
        pred = lookup_prediction(preds, tr.trace_id)
        gt_role = normalize_role(tr.mistake_agent)
        gt_step = tr.mistake_step

        if not pred:
            failed_n += 1
            agent_vec.append(0.0)
            step_vec.append(0.0)
            pm5_vec.append(0.0)
            both_vec.append(0.0)
            for k in ("agent", "step", "pm5", "both"):
                by_bench_rows[bench][k].append(0.0)
            continue

        pred_role = normalize_role(pred.get("predicted_agent") or "")
        raw_step = pred.get("predicted_step")
        if isinstance(raw_step, int):
            pred_step: int | None = raw_step
        elif raw_step is not None and str(raw_step).strip().lstrip("-").isdigit():
            pred_step = int(str(raw_step).strip())
        else:
            pred_step = None

        agent_ok = bool(pred_role and gt_role and pred_role == gt_role)
        step_ok = bool(pred_step is not None and gt_step is not None and pred_step == gt_step)
        pm5_ok = bool(pred_step is not None and gt_step is not None and abs(pred_step - gt_step) <= 5)
        both_ok = agent_ok and step_ok

        if pred_step is not None and gt_step is not None:
            valid_step_n += 1
            err = float(abs(pred_step - gt_step))
            all_abs_errs.append(err)
            by_bench_rows[bench]["abs_err"].append(err)
        else:
            failed_n += 1

        agent_vec.append(1.0 if agent_ok else 0.0)
        step_vec.append(1.0 if step_ok else 0.0)
        pm5_vec.append(1.0 if pm5_ok else 0.0)
        both_vec.append(1.0 if both_ok else 0.0)

        by_bench_rows[bench]["agent"].append(1.0 if agent_ok else 0.0)
        by_bench_rows[bench]["step"].append(1.0 if step_ok else 0.0)
        by_bench_rows[bench]["pm5"].append(1.0 if pm5_ok else 0.0)
        by_bench_rows[bench]["both"].append(1.0 if both_ok else 0.0)

    by_benchmark: dict[str, dict[str, Any]] = {}
    weighted_mae_num = 0.0
    weighted_mae_den = 0.0
    bench_maes: list[float] = []

    for bench, cols in sorted(by_bench_rows.items()):
        n_b = len(cols["step"])
        b_mae = (sum(cols["abs_err"]) / len(cols["abs_err"])) if cols["abs_err"] else None
        by_benchmark[bench] = {
            "n": n_b,
            "agent_acc": sum(cols["agent"]) / n_b if n_b else 0.0,
            "step_acc": sum(cols["step"]) / n_b if n_b else 0.0,
            "step_pm5": sum(cols["pm5"]) / n_b if n_b else 0.0,
            "both_exact": sum(cols["both"]) / n_b if n_b else 0.0,
            "mae": b_mae,
        }
        if b_mae is not None:
            w = float(weights_map.get(bench, n_b))
            weighted_mae_num += w * b_mae
            weighted_mae_den += w
            bench_maes.append(b_mae)

    micro_mae = (sum(all_abs_errs) / len(all_abs_errs)) if all_abs_errs else None
    macro_mae = (sum(bench_maes) / len(bench_maes)) if bench_maes else None
    root_mae = (weighted_mae_num / weighted_mae_den) if weighted_mae_den > 0 else micro_mae

    agent_mean, agent_lo, agent_hi = bootstrap_ci(agent_vec, n_boot=n_boot, seed=seed)
    step_mean, step_lo, step_hi = bootstrap_ci(step_vec, n_boot=n_boot, seed=seed)
    pm5_mean, pm5_lo, pm5_hi = bootstrap_ci(pm5_vec, n_boot=n_boot, seed=seed)
    both_mean, both_lo, both_hi = bootstrap_ci(both_vec, n_boot=n_boot, seed=seed)

    return {
        "n_traces": len(traces),
        "valid_step_n": valid_step_n,
        "failed_n": failed_n,
        "agent_acc": agent_mean,
        "step_acc": step_mean,
        "step_pm5": pm5_mean,
        "both_exact": both_mean,
        "root_mae": root_mae,
        "micro_mae": micro_mae,
        "macro_mae": macro_mae,
        "by_benchmark": by_benchmark,
        "ci95": {
            "agent_acc": (agent_lo, agent_hi),
            "step_acc": (step_lo, step_hi),
            "step_pm5": (pm5_lo, pm5_hi),
            "both_exact": (both_lo, both_hi),
        },
    }
