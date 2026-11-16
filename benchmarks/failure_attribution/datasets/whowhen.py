"""Who&When (`ag2ai/Agents_Failure_Attribution`) dataset loader, writer/parser, and scorer."""

from __future__ import annotations

import json
import os
import random
import re
import unicodedata
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .common import (
    FailureTraceAdapter,
    StepRecord,
    TraceRecord,
    attach_context,
    bootstrap_ci,
    paired_bootstrap_diff,
)

SUBSET_ALIASES = {
    "hand": "Hand-Crafted",
    "hand-crafted": "Hand-Crafted",
    "hand_crafted": "Hand-Crafted",
    "handcrafted": "Hand-Crafted",
    "algo": "Algorithm-Generated",
    "alg": "Algorithm-Generated",
    "algorithm-generated": "Algorithm-Generated",
    "algorithm_generated": "Algorithm-Generated",
}

PAREN_SUFFIX_RE = re.compile(r"\s*\(.*\)\s*$")
BLOCK_HEADER_RE = re.compile(
    r"Prediction for\s+([^\r\n:]+?)\s*:(.*?)(?=Prediction for\s+[^\r\n:]+?\s*:|\Z)",
    re.DOTALL | re.IGNORECASE,
)
AGENT_LINE_RE = re.compile(r"Agent Name\s*:\s*([^\r\n]+)", re.IGNORECASE)
STEP_LINE_RE = re.compile(r"Step Number\s*:\s*(?:Step\s*)?(-?\d+)", re.IGNORECASE)
REASON_LINE_RE = re.compile(
    r"(?:Reason for Mistake|Reason provided by LLM|Reason)\s*:\s*(.*?)(?:\n={10,}|\Z)",
    re.DOTALL | re.IGNORECASE,
)


def canonical_agent_name(raw: Any) -> str:
    """Strip Magentic-One parenthetical suffixes like `(thought)` or `(-> WebSurfer)`."""
    text = str(raw or "").strip()
    text = text.strip("`*\"' ")
    text = PAREN_SUFFIX_RE.sub("", text).strip()
    return text


def normalize_agent_name(raw: Any) -> str:
    """Canonical casefolded agent identifier preserving hyphens and underscores."""
    text = canonical_agent_name(raw)
    text = unicodedata.normalize("NFKC", text).casefold()
    return " ".join(text.split())


def resolve_whowhen_dir(data_path: str | Path, subset: str | None = "algo") -> tuple[Path, str, bool]:
    """Resolve `(directory, canonical_subset_name, is_handcrafted)` for Who&When."""
    p = Path(data_path)
    name_low = p.name.lower()
    if name_low in SUBSET_ALIASES:
        canon = SUBSET_ALIASES[name_low]
        return p, canon, canon == "Hand-Crafted"

    canon = SUBSET_ALIASES.get((subset or "algo").lower(), subset or "Algorithm-Generated")
    candidate = p / canon
    if candidate.is_dir():
        return candidate, canon, canon == "Hand-Crafted"
    nested = p / "Who&When" / canon
    if nested.is_dir():
        return nested, canon, canon == "Hand-Crafted"
    if p.is_dir():
        is_hand = canon == "Hand-Crafted"
        return p, canon, is_hand
    raise FileNotFoundError(f"Could not find Who&When directory under {data_path!r} (subset={subset!r})")


def _sort_json_files(directory: Path) -> list[Path]:
    files = [f for f in directory.iterdir() if f.is_file() and f.suffix == ".json"]
    return sorted(files, key=lambda p: (int("".join(filter(str.isdigit, p.stem)) or 0), p.name))


def load_whowhen_traces(
    data_path: str | Path,
    subset: str | None = "algo",
    *,
    window: int = 3,
    limit: int | None = None,
) -> list[TraceRecord]:
    """Load Who&When JSON files into canonical `TraceRecord`s (0-indexed `step_idx`)."""
    directory, canon_subset, is_handcrafted = resolve_whowhen_dir(data_path, subset)
    files = _sort_json_files(directory)
    if limit is not None:
        files = files[:limit]

    agent_key = "role" if is_handcrafted else "name"
    traces: list[TraceRecord] = []

    for fp in files:
        data = json.loads(fp.read_text(encoding="utf-8"))
        history = data.get("history") or []
        steps: list[StepRecord] = []
        for idx, entry in enumerate(history):
            raw_spk = str(entry.get(agent_key) or entry.get("name") or entry.get("role") or "Unknown")
            steps.append(
                StepRecord(
                    step_idx=idx,
                    agent=canonical_agent_name(raw_spk) or raw_spk,
                    raw_agent=raw_spk,
                    content=str(entry.get("content") or ""),
                )
            )
        attach_context(steps, window=window)

        ms_raw = data.get("mistake_step")
        mistake_step = int(str(ms_raw).strip()) if ms_raw is not None and str(ms_raw).strip() != "" else None
        mistake_agent = str(data.get("mistake_agent") or "").strip()

        traces.append(
            TraceRecord(
                trace_id=fp.name,
                dataset="whowhen",
                subset=canon_subset,
                question=str(data.get("question") or ""),
                ground_truth=str(data.get("ground_truth") or ""),
                steps=steps,
                mistake_step=mistake_step,
                mistake_agent=mistake_agent,
                mistake_reason=str(data.get("mistake_reason") or ""),
                extra={
                    "question_ID": data.get("question_ID", ""),
                    "is_handcrafted": is_handcrafted,
                },
            )
        )
    return traces


def load_whowhen(
    data_path: str | Path,
    subset: str | None = "algo",
    *,
    window: int = 3,
    limit: int | None = None,
) -> FailureTraceAdapter:
    """Load Who&When as a `FailureTraceAdapter` (`src.traces` and `src.steps`)."""
    return FailureTraceAdapter(load_whowhen_traces(data_path, subset=subset, window=window, limit=limit))


# --- Upstream-Compatible Prediction Writer & Parser --------------------------------------


def _ensure_json_key(trace_id: str, *, require_json_suffix: bool = False) -> str:
    tid = str(trace_id).strip()
    if require_json_suffix and not tid.endswith(".json"):
        return f"{tid}.json"
    return tid


def format_prediction_block(
    trace_id: str,
    predicted_agent: str,
    predicted_step: int | str | None,
    reason: str = "",
    *,
    ensure_json_ext: bool = False,
) -> str:
    """Format one upstream-compatible `Prediction for X:` block."""
    header_id = _ensure_json_key(trace_id, require_json_suffix=ensure_json_ext)
    step_str = "" if predicted_step is None else str(predicted_step)
    clean_reason = (reason or "").strip()
    return (
        f"Prediction for {header_id}:\n"
        f"Agent Name: {predicted_agent}\n"
        f"Step Number: {step_str}\n"
        f"Reason for Mistake: {clean_reason}\n\n"
        + "=" * 50
        + "\n"
    )


def write_whowhen_predictions(
    predictions: Sequence[Mapping[str, Any]] | Mapping[str, Mapping[str, Any]],
    out_path: str | Path,
    *,
    ensure_json_ext: bool = False,
) -> Path:
    """Write predictions in the upstream `Automated_FA/evaluate.py` log format."""
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    items: list[tuple[str, Mapping[str, Any]]] = []
    if isinstance(predictions, Mapping):
        items = list(predictions.items())
    else:
        for row in predictions:
            tid = str(row.get("trace_id") or row.get("trace") or row.get("qid") or "")
            items.append((tid, row))

    chunks = []
    for tid, pred in items:
        agent = str(pred.get("predicted_agent") if "predicted_agent" in pred else pred.get("mistake_agent", ""))
        step = pred.get("predicted_step") if "predicted_step" in pred else pred.get("mistake_step")
        reason = str(pred.get("reason") or pred.get("mistake_reason") or "")
        chunks.append(
            format_prediction_block(tid, agent, step, reason, ensure_json_ext=ensure_json_ext)
        )

    path.write_text("\n".join(chunks), encoding="utf-8")
    return path


def _clean_parsed_agent(raw: str) -> str:
    text = raw.strip()
    if text.startswith("(") and text.endswith(")"):
        text = text[1:-1].strip()
    text = text.strip("`*\"' ")
    text = re.sub(r"[.,;:]+\s*$", "", text).strip()
    return text


def parse_whowhen_predictions(source: str | Path) -> dict[str, dict[str, Any]]:
    """Parse upstream text blocks (`Prediction for X:`), JSON dicts/lists, or JSONL files.

    Unlike upstream `Automated_FA/evaluate.py`'s `r"Agent Name:\\s*([\\w_]+)"`, this parser
    preserves hyphens (e.g. `Blu-Ray_Expert` in `91.json`) and spaces/parentheticals.
    """
    if isinstance(source, Path) or (isinstance(source, str) and "\n" not in source and os.path.exists(source)):
        text = Path(source).read_text(encoding="utf-8")
    else:
        text = str(source)

    stripped = text.strip()
    if not stripped:
        return {}

    # Support JSON / JSONL prediction artifacts transparently.
    if stripped[0] in ("{", "["):
        try:
            obj = json.loads(stripped)
            return _normalize_json_preds(obj)
        except json.JSONDecodeError:
            lines = [ln.strip() for ln in stripped.splitlines() if ln.strip()]
            if all(ln.startswith("{") for ln in lines):
                return _normalize_json_preds([json.loads(ln) for ln in lines])

    preds: dict[str, dict[str, Any]] = {}
    for match in BLOCK_HEADER_RE.finditer(text):
        raw_id = match.group(1).strip()
        if raw_id.lower().endswith(": error found."):
            raw_id = raw_id[: -len(": error found.")].strip()
        body = match.group(2)

        agent_m = AGENT_LINE_RE.search(body)
        step_m = STEP_LINE_RE.search(body)
        reason_m = REASON_LINE_RE.search(body)

        if not agent_m and not step_m:
            continue

        raw_agent = _clean_parsed_agent(agent_m.group(1)) if agent_m else ""
        raw_step = step_m.group(1).strip() if step_m else ""
        pred_step = int(raw_step) if raw_step and re.fullmatch(r"-?\d+", raw_step) else None
        reason = reason_m.group(1).strip() if reason_m else ""

        preds[raw_id] = {
            "predicted_agent": raw_agent,
            "predicted_step": pred_step,
            "predicted_step_str": raw_step,
            "reason": reason,
        }
    return preds


def _normalize_json_preds(obj: Any) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    if isinstance(obj, Mapping):
        if "predictions" in obj and isinstance(obj["predictions"], (list, Mapping)):
            return _normalize_json_preds(obj["predictions"])
        items = list(obj.items())
    elif isinstance(obj, list):
        items = [
            (
                str(
                    item.get("trace_id")
                    or item.get("trace")
                    or item.get("qid")
                    or item.get("question_ID")
                    or item.get("trajectory_id")
                    or ""
                ),
                item,
            )
            for item in obj
            if isinstance(item, Mapping)
        ]
    else:
        return out

    for tid, val in items:
        if not tid or not isinstance(val, Mapping):
            continue
        raw_agent = str(
            val.get("predicted_agent")
            if "predicted_agent" in val
            else val.get("mistake_agent", val.get("agent", ""))
        ).strip()
        raw_step = (
            val.get("predicted_step")
            if "predicted_step" in val
            else val.get("mistake_step", val.get("step_idx", val.get("step")))
        )
        pred_step: int | None
        if raw_step is None or str(raw_step).strip() == "":
            pred_step = None
            raw_step_str = ""
        else:
            raw_step_str = str(raw_step).strip()
            try:
                pred_step = int(raw_step_str)
            except ValueError:
                m = re.search(r"-?\d+", raw_step_str)
                pred_step = int(m.group(0)) if m else None
        out[str(tid)] = {
            "predicted_agent": raw_agent,
            "predicted_step": pred_step,
            "predicted_step_str": raw_step_str,
            "reason": str(val.get("reason") or val.get("mistake_reason") or ""),
            "failure_category": str(val.get("failure_category") or val.get("category") or ""),
            "errors": val.get("errors") if isinstance(val.get("errors"), list) else [],
        }
    return out


def lookup_prediction(predictions: Mapping[str, dict[str, Any]], trace_id: str) -> dict[str, Any] | None:
    """Match prediction by exact `trace_id`, or with/without `.json` suffix."""
    if trace_id in predictions:
        return predictions[trace_id]
    if trace_id.endswith(".json") and trace_id[:-5] in predictions:
        return predictions[trace_id[:-5]]
    with_ext = f"{trace_id}.json"
    if with_ext in predictions:
        return predictions[with_ext]
    return None


# --- Scorer, Paired Diffs, and Trivial Baselines -----------------------------------------


def _as_traces(traces_or_adapter: Sequence[TraceRecord] | FailureTraceAdapter) -> list[TraceRecord]:
    if isinstance(traces_or_adapter, FailureTraceAdapter):
        return traces_or_adapter.trace_records
    return list(traces_or_adapter)


def per_trace_whowhen_outcomes(
    traces_or_adapter: Sequence[TraceRecord] | FailureTraceAdapter,
    predictions: Mapping[str, Mapping[str, Any]] | str | Path,
    *,
    tolerances: Sequence[int] = (1, 3, 5),
) -> dict[str, list[float]]:
    """Compute per-trace 0/1 outcome vectors across all reference traces."""
    traces = _as_traces(traces_or_adapter)
    preds = (
        parse_whowhen_predictions(predictions)
        if isinstance(predictions, (str, Path))
        else dict(predictions)
    )

    outcomes: dict[str, list[float]] = {
        "agent": [],
        "step": [],
        "both_exact": [],
        "agent_from_step": [],
        "paper_agent": [],
        "paper_step": [],
        **{f"step_pm{tol}": [] for tol in tolerances},
    }

    for tr in traces:
        pred = lookup_prediction(preds, tr.trace_id)
        gt_agent_norm = normalize_agent_name(tr.mistake_agent)
        gt_agent_raw = (tr.mistake_agent or "").strip().lower()
        gt_step = tr.mistake_step

        if not pred:
            for vec in outcomes.values():
                vec.append(0.0)
            continue

        raw_agent = str(pred.get("predicted_agent") or "").strip()
        pred_agent_norm = normalize_agent_name(raw_agent)

        raw_step_val = pred.get("predicted_step")
        if isinstance(raw_step_val, int):
            pred_step: int | None = raw_step_val
        elif raw_step_val is not None and str(raw_step_val).strip().lstrip("-").isdigit():
            pred_step = int(str(raw_step_val).strip())
        else:
            pred_step = None
        raw_step_str = str(
            pred.get("predicted_step_str")
            if pred.get("predicted_step_str") is not None
            else ("" if pred_step is None else pred_step)
        ).strip()

        agent_ok = bool(pred_agent_norm and gt_agent_norm and pred_agent_norm == gt_agent_norm)
        step_ok = bool(pred_step is not None and gt_step is not None and pred_step == gt_step)
        both_ok = agent_ok and step_ok

        spk = tr.speaker_at(pred_step)
        spk_ok = bool(spk and gt_agent_norm and normalize_agent_name(spk) == gt_agent_norm)

        paper_agent_ok = bool(gt_agent_raw and raw_agent and gt_agent_raw in raw_agent.lower())
        paper_step_ok = bool(gt_step is not None and raw_step_str and str(gt_step) in raw_step_str)

        outcomes["agent"].append(1.0 if agent_ok else 0.0)
        outcomes["step"].append(1.0 if step_ok else 0.0)
        outcomes["both_exact"].append(1.0 if both_ok else 0.0)
        outcomes["agent_from_step"].append(1.0 if spk_ok else 0.0)
        outcomes["paper_agent"].append(1.0 if paper_agent_ok else 0.0)
        outcomes["paper_step"].append(1.0 if paper_step_ok else 0.0)

        for tol in tolerances:
            pm_ok = bool(pred_step is not None and gt_step is not None and abs(pred_step - gt_step) <= tol)
            outcomes[f"step_pm{tol}"].append(1.0 if pm_ok else 0.0)

    return outcomes


def score_whowhen(
    traces_or_adapter: Sequence[TraceRecord] | FailureTraceAdapter,
    predictions: Mapping[str, Mapping[str, Any]] | str | Path,
    *,
    tolerances: Sequence[int] = (1, 3, 5),
    n_boot: int = 2000,
    seed: int = 0,
) -> dict[str, Any]:
    """Score Who&When predictions with exact metrics, upstream substring metrics, and 95% CIs."""
    traces = _as_traces(traces_or_adapter)
    preds = (
        parse_whowhen_predictions(predictions)
        if isinstance(predictions, (str, Path))
        else dict(predictions)
    )
    outcomes = per_trace_whowhen_outcomes(traces, preds, tolerances=tolerances)
    n_total = len(traces)
    n_matched = sum(1 for tr in traces if lookup_prediction(preds, tr.trace_id) is not None)

    metrics: dict[str, float] = {}
    cis: dict[str, tuple[float, float]] = {}
    for k, vec in outcomes.items():
        mean_val, lo, hi = bootstrap_ci(vec, n_boot=n_boot, seed=seed)
        metrics[k] = mean_val
        cis[k] = (lo, hi)

    return {
        "n_traces": n_total,
        "n_predictions": n_matched,
        "coverage": (n_matched / n_total) if n_total else 0.0,
        **metrics,
        "ci95": cis,
    }


def paired_diff_whowhen(
    traces_or_adapter: Sequence[TraceRecord] | FailureTraceAdapter,
    preds_a: Mapping[str, Mapping[str, Any]] | str | Path,
    preds_b: Mapping[str, Mapping[str, Any]] | str | Path,
    *,
    tolerances: Sequence[int] = (1, 3, 5),
    n_boot: int = 2000,
    seed: int = 0,
) -> dict[str, dict[str, float]]:
    """Compute paired bootstrap 95% CIs for `A - B` across all Who&When metrics."""
    out_a = per_trace_whowhen_outcomes(traces_or_adapter, preds_a, tolerances=tolerances)
    out_b = per_trace_whowhen_outcomes(traces_or_adapter, preds_b, tolerances=tolerances)

    diffs: dict[str, dict[str, float]] = {}
    for k in out_a:
        diff_mean, lo, hi = paired_bootstrap_diff(out_a[k], out_b[k], n_boot=n_boot, seed=seed)
        diffs[k] = {"diff": diff_mean, "ci_low": lo, "ci_high": hi}
    return diffs


def trivial_baselines(
    traces_or_adapter: Sequence[TraceRecord] | FailureTraceAdapter,
    *,
    seed: int = 0,
    n_boot: int = 2000,
) -> dict[str, dict[str, Any]]:
    """Evaluate zero-LM trivial baselines on a Who&When trace collection."""
    traces = _as_traces(traces_or_adapter)

    always_ws = {
        tr.trace_id: {"predicted_agent": "WebSurfer", "predicted_step": None}
        for tr in traces
    }
    always_ve = {
        tr.trace_id: {"predicted_agent": "Verification_Expert", "predicted_step": None}
        for tr in traces
    }
    always_s8 = {
        tr.trace_id: {"predicted_agent": tr.speaker_at(8) or "Unknown", "predicted_step": 8}
        for tr in traces
    }
    always_s1 = {
        tr.trace_id: {"predicted_agent": tr.speaker_at(1) or "Unknown", "predicted_step": 1}
        for tr in traces
    }

    rng = random.Random(seed)
    uniform_preds = {}
    for tr in traces:
        s = rng.randrange(tr.n_steps) if tr.n_steps > 0 else 0
        uniform_preds[tr.trace_id] = {
            "predicted_agent": tr.speaker_at(s) or "Unknown",
            "predicted_step": s,
        }

    return {
        "always WebSurfer": score_whowhen(traces, always_ws, n_boot=n_boot, seed=seed),
        "always Verification_Expert": score_whowhen(traces, always_ve, n_boot=n_boot, seed=seed),
        "always step 8": score_whowhen(traces, always_s8, n_boot=n_boot, seed=seed),
        "always step 1": score_whowhen(traces, always_s1, n_boot=n_boot, seed=seed),
        f"uniform_step(seed={seed})": score_whowhen(traces, uniform_preds, n_boot=n_boot, seed=seed),
    }
