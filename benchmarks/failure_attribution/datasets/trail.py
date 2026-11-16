"""TRAIL (`PatronusAI/trail-benchmark`) dataset loader, span flattener, and pure-Python scorer."""

from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping, Sequence
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

TRAIL_CATEGORIES: tuple[str, ...] = (
    "Language-only",
    "Tool-related",
    "Poor Information Retrieval",
    "Incorrect Memory Usage",
    "Tool Output Misinterpretation",
    "Incorrect Problem Identification",
    "Tool Selection Errors",
    "Formatting Errors",
    "Instruction Non-compliance",
    "Tool Definition Issues",
    "Environment Setup Errors",
    "Rate Limiting",
    "Authentication Errors",
    "Service Errors",
    "Resource Not Found",
    "Resource Exhaustion",
    "Timeout Issues",
    "Context Handling Failures",
    "Resource Abuse",
    "Goal Deviation",
    "Task Orchestration",
)

_TYPO_ALIASES: dict[str, str] = {
    "instructionnoncomplience": "Instruction Non-compliance",
    "instructionnoncompliance": "Instruction Non-compliance",
    "noncompliance": "Instruction Non-compliance",
    "goaldeviation": "Goal Deviation",
    "contexthandlingfailure": "Context Handling Failures",
    "contexthandlingfailures": "Context Handling Failures",
    "formattingerror": "Formatting Errors",
    "formattingerrors": "Formatting Errors",
    "taskorchestration": "Task Orchestration",
    "taskorchestrationerror": "Task Orchestration",
    "taskorchestrationerrors": "Task Orchestration",
    "toolselection": "Tool Selection Errors",
    "toolselectionerror": "Tool Selection Errors",
    "toolselectionerrors": "Tool Selection Errors",
    "languageonly": "Language-only",
    "toolrelated": "Tool-related",
    "poorinformationretrieval": "Poor Information Retrieval",
    "incorrectmemoryusage": "Incorrect Memory Usage",
    "tooloutputmisinterpretation": "Tool Output Misinterpretation",
    "incorrectproblemidentification": "Incorrect Problem Identification",
    "tooldefinitionissue": "Tool Definition Issues",
    "tooldefinitionissues": "Tool Definition Issues",
    "environmentsetuperror": "Environment Setup Errors",
    "environmentsetuperrors": "Environment Setup Errors",
    "ratelimit": "Rate Limiting",
    "ratelimiting": "Rate Limiting",
    "authenticationerror": "Authentication Errors",
    "authenticationerrors": "Authentication Errors",
    "serviceerror": "Service Errors",
    "serviceerrors": "Service Errors",
    "resourcenotfound": "Resource Not Found",
    "resourceexhaustion": "Resource Exhaustion",
    "timeoutissue": "Timeout Issues",
    "timeoutissues": "Timeout Issues",
    "resourceabuse": "Resource Abuse",
}

TRAILING_COMMA_RE = re.compile(r",(\s*[\]}])")
BOILERPLATE_ATTR_PREFIXES = ("pat.", "telemetry.", "service.name")


def _compact_key(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (text or "").lower())


def normalize_trail_category(category: Any, all_categories: Sequence[str] = TRAIL_CATEGORIES) -> str:
    """Normalize a TRAIL category string (including human annotation typos) to the 21 official labels."""
    raw = str(category or "").strip()
    if not raw:
        return ""

    key = _compact_key(raw)
    if key in _TYPO_ALIASES and _TYPO_ALIASES[key] in all_categories:
        return _TYPO_ALIASES[key]

    for std in all_categories:
        std_key = _compact_key(std)
        if key == std_key or key.rstrip("s") == std_key.rstrip("s"):
            return std

    for std in all_categories:
        std_key = _compact_key(std)
        if key and (key in std_key or std_key in key):
            return std

    return raw


def strip_trailing_commas(json_text: str) -> str:
    """Strip illegal JSON trailing commas before `]` or `}` outside string literals."""
    out: list[str] = []
    in_str = False
    escape = False
    i = 0
    n = len(json_text)
    while i < n:
        ch = json_text[i]
        if in_str:
            out.append(ch)
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            i += 1
            continue

        if ch == '"':
            in_str = True
            out.append(ch)
            i += 1
            continue

        if ch == ",":
            j = i + 1
            while j < n and json_text[j].isspace():
                j += 1
            if j < n and json_text[j] in ("]", "}"):
                i += 1
                continue

        out.append(ch)
        i += 1
    return "".join(out)


def loads_lenient_json(text: str | Mapping[str, Any]) -> dict[str, Any]:
    """Parse JSON object leniently, handling trailing commas and surrounding prose/fences."""
    if isinstance(text, Mapping):
        return dict(text)
    raw = str(text or "").strip()
    if not raw:
        return {}

    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    cleaned = strip_trailing_commas(raw)
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    m = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if m:
        candidate = strip_trailing_commas(m.group(0))
        while len(candidate) >= 2:
            try:
                parsed = json.loads(candidate)
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                candidate = candidate[:-1]
    raise ValueError("Could not parse valid JSON object from TRAIL text")


def _render_span_content(span: Mapping[str, Any], max_value_chars: int = 1200) -> str:
    span_id = str(span.get("span_id") or "")
    span_name = str(span.get("span_name") or "span")
    service = str(span.get("service_name") or "")
    status_code = str(span.get("status_code") or "")
    status_msg = str(span.get("status_message") or "")

    header = f"[span_id={span_id}] {span_name}"
    if service:
        header += f" (service={service})"
    if status_code and status_code.lower() not in ("unset", "ok", ""):
        header += f" status={status_code}"
    if status_msg:
        header += f": {clip_text(status_msg, 300)}"

    parts = [header]

    attrs = span.get("span_attributes") or {}
    if isinstance(attrs, Mapping):
        kept_attrs = {
            k: v
            for k, v in attrs.items()
            if not any(str(k).startswith(p) for p in BOILERPLATE_ATTR_PREFIXES)
            and v not in (None, "", [], {})
        }
        if kept_attrs:
            parts.append("attributes: " + clip_text(json.dumps(kept_attrs, default=str), max_value_chars))

    events = span.get("events") or []
    if isinstance(events, list) and events:
        parts.append("events: " + clip_text(json.dumps(events, default=str), max_value_chars))

    logs = span.get("logs") or []
    if isinstance(logs, list):
        for log_entry in logs[:6]:
            if not isinstance(log_entry, Mapping):
                continue
            body = log_entry.get("body")
            sev = log_entry.get("severity_text") or ""
            if body not in (None, "", {}):
                body_str = body if isinstance(body, str) else json.dumps(body, default=str)
                prefix = f"log[{sev}]: " if sev else "log: "
                parts.append(prefix + clip_text(body_str, max_value_chars))

    return "\n".join(parts)


def flatten_trail_spans(spans: Sequence[Mapping[str, Any]]) -> list[StepRecord]:
    """Flatten recursive OpenTelemetry `spans` / `child_spans` into timestamp/DFS-ordered steps."""
    collected: list[tuple[str, int, Mapping[str, Any]]] = []
    counter = 0

    def visit(node: Mapping[str, Any]) -> None:
        nonlocal counter
        ts = str(node.get("timestamp") or "")
        collected.append((ts, counter, node))
        counter += 1
        children = node.get("child_spans") or []
        if isinstance(children, list):
            for ch in children:
                if isinstance(ch, Mapping):
                    visit(ch)

    for root in spans or []:
        if isinstance(root, Mapping):
            visit(root)

    collected.sort(key=lambda item: (item[0] == "", item[0], item[1]))

    steps: list[StepRecord] = []
    for idx, (_, _, sp) in enumerate(collected):
        span_id = str(sp.get("span_id") or f"span_{idx}")
        span_name = str(sp.get("span_name") or "span")
        service_name = str(sp.get("service_name") or "")
        steps.append(
            StepRecord(
                step_idx=idx,
                agent=span_name or service_name or "span",
                raw_agent=span_name,
                span_id=span_id,
                content=_render_span_content(sp),
                extra={
                    "span_name": span_name,
                    "service_name": service_name,
                    "parent_span_id": sp.get("parent_span_id"),
                    "status_code": sp.get("status_code", ""),
                },
            )
        )
    return steps


def _extract_trail_question(steps: Sequence[StepRecord]) -> str:
    q_re = re.compile(r'"question"\s*:\s*"((?:\\.|[^"\\])+)"')
    for s in steps[:10]:
        m = q_re.search(s.content)
        if m:
            try:
                return json.loads(f'"{m.group(1)}"')
            except json.JSONDecodeError:
                return m.group(1)
    return steps[0].content[:400] if steps else ""


def _build_trail_record(
    trace_obj: Mapping[str, Any],
    ann_obj: Mapping[str, Any] | None,
    *,
    subset: str,
    window: int,
    fallback_id: str = "",
) -> TraceRecord:
    trace_id = str(
        trace_obj.get("trace_id")
        or (ann_obj or {}).get("trace_id")
        or fallback_id
    )
    spans = trace_obj.get("spans") or []
    steps = flatten_trail_spans(spans)
    attach_context(steps, window=window)

    raw_errors = (ann_obj or {}).get("errors") or []
    norm_errors = []
    for err in raw_errors:
        if not isinstance(err, Mapping):
            continue
        loc = str(err.get("location") or "").strip()
        cat = normalize_trail_category(err.get("category") or "")
        norm_errors.append({
            "location": loc,
            "category": cat,
            "raw_category": str(err.get("category") or ""),
            "evidence": str(err.get("evidence") or ""),
            "description": str(err.get("description") or ""),
            "impact": str(err.get("impact") or ""),
        })

    scores_list = (ann_obj or {}).get("scores") or []
    scores = dict(scores_list[0]) if isinstance(scores_list, list) and scores_list and isinstance(scores_list[0], Mapping) else {}

    span_to_idx = {s.span_id: s.step_idx for s in steps}
    first_err = norm_errors[0] if norm_errors else None
    mistake_step = span_to_idx.get(first_err["location"]) if first_err else None
    mistake_agent = steps[mistake_step].agent if mistake_step is not None else ""
    failure_cat = first_err["category"] if first_err else ""

    return TraceRecord(
        trace_id=trace_id,
        dataset="trail",
        subset=subset,
        question=_extract_trail_question(steps),
        steps=steps,
        mistake_step=mistake_step,
        mistake_agent=mistake_agent,
        mistake_reason=first_err["description"] if first_err else "",
        failure_category=failure_cat,
        trail_errors=norm_errors,
        trail_scores=scores,
    )


_SPLIT_SPECS = (
    ("gaia", ("GAIA", "data/GAIA"), "processed_annotations_gaia"),
    ("swe_bench", ("SWE Bench", "SWE_Bench", "swe_bench", "data/SWE Bench", "data/swe_bench"), "processed_annotations_swe_bench"),
)


def load_trail_traces(
    data_path: str | Path,
    subset: str | None = None,
    *,
    window: int = 3,
    limit: int | None = None,
) -> list[TraceRecord]:
    """Load TRAIL traces from either GitHub directory layout or HF parquet/jsonl/json rows."""
    p = Path(data_path)
    if p.is_file():
        return _load_trail_file(p, subset=subset, window=window, limit=limit)

    if not p.is_dir():
        raise FileNotFoundError(f"TRAIL path not found: {data_path!r}")

    # Check if `benchmarking/` subdirectory exists inside a cloned repo root.
    bench_root = p / "benchmarking" if (p / "benchmarking").is_dir() else p

    want = (subset or "").strip().lower().replace("-", "_").replace(" ", "_")
    records: list[TraceRecord] = []

    found_split_dirs = False
    for split_key, trace_rel_candidates, ann_dir_name in _SPLIT_SPECS:
        if want and want not in ("all", split_key):
            continue
        trace_dir = next((bench_root / rel for rel in trace_rel_candidates if (bench_root / rel).is_dir()), None)
        if trace_dir is None:
            continue
        found_split_dirs = True
        ann_dir = bench_root / ann_dir_name

        for tf in sorted(trace_dir.glob("*.json")):
            if limit is not None and len(records) >= limit:
                return records
            trace_obj = loads_lenient_json(tf.read_text(encoding="utf-8"))
            ann_file = ann_dir / tf.name
            ann_obj = loads_lenient_json(ann_file.read_text(encoding="utf-8")) if ann_file.is_file() else None
            records.append(
                _build_trail_record(
                    trace_obj,
                    ann_obj,
                    subset=split_key,
                    window=window,
                    fallback_id=tf.stem,
                )
            )

    if found_split_dirs:
        return records

    # Fallback: directory containing parquet or jsonl files from HF snapshot_download.
    for ext in ("*.parquet", "*.jsonl", "*.json"):
        files = sorted(p.rglob(ext))
        if files:
            for f in files:
                chunk = _load_trail_file(
                    f,
                    subset=subset,
                    window=window,
                    limit=(limit - len(records)) if limit is not None else None,
                )
                records.extend(chunk)
                if limit is not None and len(records) >= limit:
                    return records[:limit]
            return records

    raise FileNotFoundError(f"No TRAIL splits or data files found under {data_path!r}")


def _load_trail_file(
    file_path: Path,
    *,
    subset: str | None = None,
    window: int = 3,
    limit: int | None = None,
) -> list[TraceRecord]:
    want = (subset or "").strip().lower().replace("-", "_").replace(" ", "_")
    rows: list[Mapping[str, Any]] = []
    if file_path.suffix == ".parquet":
        import pandas as pd

        df = pd.read_parquet(file_path)
        rows = df.to_dict("records")
    elif file_path.suffix == ".jsonl":
        rows = [loads_lenient_json(ln) for ln in file_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    else:
        obj = loads_lenient_json(file_path.read_text(encoding="utf-8"))
        rows = obj.get("rows", [obj]) if isinstance(obj, Mapping) else list(obj)

    out: list[TraceRecord] = []
    for idx, row in enumerate(rows):
        row_split = str(row.get("split") or row.get("subset") or "trail").lower().replace("-", "_").replace(" ", "_")
        if want and want not in ("all", row_split):
            continue
        raw_trace = row.get("trace") if "trace" in row else row
        raw_labels = row.get("labels") if "labels" in row else row.get("annotations")
        trace_obj = loads_lenient_json(raw_trace) if isinstance(raw_trace, (str, Mapping)) else {}
        ann_obj = loads_lenient_json(raw_labels) if isinstance(raw_labels, (str, Mapping)) and raw_labels else None
        tid = str(row.get("trace_id") or trace_obj.get("trace_id") or f"trail_{idx}")
        out.append(_build_trail_record(trace_obj, ann_obj, subset=row_split, window=window, fallback_id=tid))
        if limit is not None and len(out) >= limit:
            break
    return out


def load_trail(
    data_path: str | Path,
    subset: str | None = None,
    *,
    window: int = 3,
    limit: int | None = None,
) -> FailureTraceAdapter:
    """Load TRAIL as a `FailureTraceAdapter` (`src.traces` and `src.steps`)."""
    return FailureTraceAdapter(load_trail_traces(data_path, subset=subset, window=window, limit=limit))


# --- Pure-Python Official TRAIL Scorer ---------------------------------------------------


def _parse_trail_predictions(
    predictions: Mapping[str, Any] | Sequence[Mapping[str, Any]] | str | Path,
) -> dict[str, dict[str, Any]]:
    if isinstance(predictions, (str, Path)):
        p = Path(predictions)
        if p.is_dir():
            out: dict[str, dict[str, Any]] = {}
            for fp in sorted(p.glob("*.json")):
                try:
                    out[fp.stem] = loads_lenient_json(fp.read_text(encoding="utf-8"))
                except ValueError:
                    out[fp.stem] = {"errors": []}
            return out
        if p.is_file():
            text = p.read_text(encoding="utf-8").strip()
            if p.suffix == ".jsonl":
                items = [loads_lenient_json(ln) for ln in text.splitlines() if ln.strip()]
                return {str(it.get("trace_id") or it.get("trace")): it for it in items}
            if text.startswith("["):
                arr = json.loads(strip_trailing_commas(text))
                return {
                    str(it.get("trace_id") or it.get("trace")): dict(it)
                    for it in arr
                    if isinstance(it, Mapping) and (it.get("trace_id") or it.get("trace"))
                }
            obj = loads_lenient_json(text)
            if "errors" in obj and "trace_id" in obj:
                return {str(obj["trace_id"]): obj}
            return {str(k): (loads_lenient_json(v) if isinstance(v, str) else dict(v)) for k, v in obj.items()}
        obj = loads_lenient_json(str(predictions))
        return {str(k): dict(v) for k, v in obj.items()}

    if isinstance(predictions, Mapping):
        return {
            str(k[:-5] if str(k).endswith(".json") else k): (
                loads_lenient_json(v) if isinstance(v, str) else dict(v)
            )
            for k, v in predictions.items()
        }

    out = {}
    for row in predictions:
        tid = str(row.get("trace_id") or row.get("trace") or "")
        if tid.endswith(".json"):
            tid = tid[:-5]
        if tid:
            out[tid] = dict(row)
    return out


def score_trail(
    traces_or_adapter: Sequence[TraceRecord] | FailureTraceAdapter,
    predictions: Mapping[str, Any] | Sequence[Mapping[str, Any]] | str | Path,
    *,
    all_categories: Sequence[str] = TRAIL_CATEGORIES,
    n_boot: int = 2000,
    seed: int = 0,
) -> dict[str, Any]:
    """Compute official TRAIL metrics in pure Python:
    - `location_recall` (`location_accuracy`): mean per-trace `|gt_locs & pred_locs| / |gt_locs|`
    - `joint_location_category_recall` (`joint_accuracy`): mean per-trace `|gt_pairs & pred_pairs| / |gt_pairs|`
    - `weighted_f1`: support-weighted F1 over the 21 trace-level binary category indicators.
    """
    traces = traces_or_adapter.trace_records if isinstance(traces_or_adapter, FailureTraceAdapter) else list(traces_or_adapter)
    preds = _parse_trail_predictions(predictions)

    cat_index = {c: i for i, c in enumerate(all_categories)}
    n_cats = len(all_categories)

    loc_recalls: list[float] = []
    joint_recalls: list[float] = []
    y_true_rows: list[list[int]] = []
    y_pred_rows: list[list[int]] = []

    matched = 0
    for tr in traces:
        tid = tr.trace_id[:-5] if tr.trace_id.endswith(".json") else tr.trace_id
        pred = preds.get(tid) or preds.get(tr.trace_id) or {"errors": []}
        if tid in preds or tr.trace_id in preds:
            matched += 1

        gt_errors = tr.trail_errors or []
        pred_errors = pred.get("errors") or []

        gt_locs = [str(e.get("location") or "").strip() for e in gt_errors if str(e.get("location") or "").strip()]
        gt_cats = [normalize_trail_category(e.get("category"), all_categories) for e in gt_errors if e.get("category")]
        gt_pairs = {
            (str(e.get("location") or "").strip(), normalize_trail_category(e.get("category"), all_categories))
            for e in gt_errors
            if str(e.get("location") or "").strip() and e.get("category")
        }

        pred_locs = {
            str(e.get("location") or "").strip()
            for e in pred_errors
            if isinstance(e, Mapping) and str(e.get("location") or "").strip()
        }
        pred_cats = [
            normalize_trail_category(e.get("category"), all_categories)
            for e in pred_errors
            if isinstance(e, Mapping) and e.get("category")
        ]
        pred_pairs = {
            (str(e.get("location") or "").strip(), normalize_trail_category(e.get("category"), all_categories))
            for e in pred_errors
            if isinstance(e, Mapping) and str(e.get("location") or "").strip() and e.get("category")
        }

        gt_loc_set = set(gt_locs)
        loc_rec = (len(gt_loc_set & pred_locs) / len(gt_loc_set)) if gt_loc_set else 0.0
        joint_rec = (len(gt_pairs & pred_pairs) / len(gt_pairs)) if gt_pairs else 0.0
        loc_recalls.append(loc_rec)
        joint_recalls.append(joint_rec)

        yt = [0] * n_cats
        yp = [0] * n_cats
        for c in gt_cats:
            if c in cat_index:
                yt[cat_index[c]] = 1
        for c in pred_cats:
            if c in cat_index:
                yp[cat_index[c]] = 1
        y_true_rows.append(yt)
        y_pred_rows.append(yp)

    per_category: dict[str, dict[str, float | int]] = {}
    weighted_num = 0.0
    total_support = 0
    for c_idx, cat_name in enumerate(all_categories):
        tp_c = sum(1 for r in range(len(traces)) if y_true_rows[r][c_idx] == 1 and y_pred_rows[r][c_idx] == 1)
        fp_c = sum(1 for r in range(len(traces)) if y_true_rows[r][c_idx] == 0 and y_pred_rows[r][c_idx] == 1)
        fn_c = sum(1 for r in range(len(traces)) if y_true_rows[r][c_idx] == 1 and y_pred_rows[r][c_idx] == 0)
        sup_c = tp_c + fn_c
        prec = tp_c / (tp_c + fp_c) if (tp_c + fp_c) > 0 else 0.0
        rec = tp_c / sup_c if sup_c > 0 else 0.0
        f1 = (2.0 * prec * rec / (prec + rec)) if (prec + rec) > 0.0 else 0.0
        per_category[cat_name] = {
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "support": sup_c,
        }
        weighted_num += sup_c * f1
        total_support += sup_c

    weighted_f1 = (weighted_num / total_support) if total_support > 0 else 0.0
    loc_mean, loc_lo, loc_hi = bootstrap_ci(loc_recalls, n_boot=n_boot, seed=seed)
    jnt_mean, jnt_lo, jnt_hi = bootstrap_ci(joint_recalls, n_boot=n_boot, seed=seed)
    nonempty_idxs = [i for i, tr in enumerate(traces) if tr.trail_errors]
    loc_nonempty = (sum(loc_recalls[i] for i in nonempty_idxs) / len(nonempty_idxs)) if nonempty_idxs else 0.0
    jnt_nonempty = (sum(joint_recalls[i] for i in nonempty_idxs) / len(nonempty_idxs)) if nonempty_idxs else 0.0

    return {
        "n_traces": len(traces),
        "n_error_traces": len(nonempty_idxs),
        "n_predictions": matched,
        "location_recall": loc_mean,
        "location_accuracy": loc_mean,
        "location_recall_nonempty": loc_nonempty,
        "joint_location_category_recall": jnt_mean,
        "joint_accuracy": jnt_mean,
        "joint_recall_nonempty": jnt_nonempty,
        "weighted_f1": weighted_f1,
        "per_category": per_category,
        "ci95": {
            "location_recall": (loc_lo, loc_hi),
            "joint_location_category_recall": (jnt_lo, jnt_hi),
        },
    }
