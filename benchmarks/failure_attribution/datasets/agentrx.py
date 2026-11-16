"""AgentRx (`microsoft/AgentRx`) 1-indexed Trajectory IR / raw trace loader and official scorer."""

from __future__ import annotations

import json
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
)
from .whowhen import canonical_agent_name, lookup_prediction, normalize_agent_name, parse_whowhen_predictions

FAILURE_CASE_TO_CATEGORY: dict[int, str] = {
    1: "Instruction Adherence Failure",
    2: "Invention of New Information",
    3: "Invalid Invocation",
    4: "Misinterpretation of Tool Output",
    5: "Intent Plan Misalignment",
    6: "Underspecified User Intent",
    7: "Intent Not Supported",
    8: "Guardrails Triggered",
    9: "System Failure",
    10: "Inconclusive",
}

CATEGORY_TO_FAILURE_CASE: dict[str, int] = {v: k for k, v in FAILURE_CASE_TO_CATEGORY.items()}
AGENTRX_CATEGORIES: tuple[str, ...] = tuple(FAILURE_CASE_TO_CATEGORY[i] for i in range(1, 11))

_ENUM_TO_CASE: dict[str, int] = {
    "INSTRUCTION_ADHERENCE_FAILURE": 1,
    "INSTRUCTION_PLAN_ADHERENCE_FAILURE": 1,
    "INVENTION_OF_NEW_INFORMATION": 2,
    "INVALID_INVOCATION": 3,
    "MISINTERPRETATION_OF_TOOL_OUTPUT": 4,
    "INTENT_PLAN_MISALIGNMENT": 5,
    "UNDERSPECIFIED_USER_INTENT": 6,
    "INTENT_NOT_SUPPORTED": 7,
    "GUARDRAILS_TRIGGERED": 8,
    "SYSTEM_FAILURE": 9,
    "INCONCLUSIVE": 10,
}

FLASH_STEP_RE = re.compile(r"\bStep[-\s]*(\d+)\b", re.IGNORECASE)


def normalize_agentrx_category(category: Any) -> str:
    """Normalize an AgentRx category string, enum label, or case int (1..10) to canonical name."""
    if category is None or str(category).strip() == "":
        return ""
    if isinstance(category, int) or str(category).strip().isdigit():
        num = int(str(category).strip())
        return FAILURE_CASE_TO_CATEGORY.get(num, "Inconclusive")

    raw = str(category).strip()
    enum_part = raw.split(".")[-1].upper().replace(" ", "_").replace("-", "_").replace("/", "_")
    if enum_part in _ENUM_TO_CASE:
        return FAILURE_CASE_TO_CATEGORY[_ENUM_TO_CASE[enum_part]]

    low = raw.lower()
    if "instruction" in low and "adherence" in low:
        return "Instruction Adherence Failure"
    if "invention" in low or ("new" in low and "information" in low):
        return "Invention of New Information"
    if "invalid" in low and "invocation" in low:
        return "Invalid Invocation"
    if "misinterpretation" in low or "handoff" in low:
        return "Misinterpretation of Tool Output"
    if "intent" in low and ("plan" in low or "misalignment" in low):
        return "Intent Plan Misalignment"
    if "underspecified" in low or ("user" in low and "intent" in low and "not" not in low):
        return "Underspecified User Intent"
    if "not supported" in low or ("intent" in low and "not" in low and "supported" in low):
        return "Intent Not Supported"
    if "guardrail" in low or "policy violation" in low:
        return "Guardrails Triggered"
    if "system" in low and "failure" in low:
        return "System Failure"
    if "inconclusive" in low:
        return "Inconclusive"
    return raw


def extract_failure_case_number(value: Any) -> int:
    """Map an AgentRx category name, enum string, or integer to its 1..10 `FailureCase` id."""
    norm = normalize_agentrx_category(value)
    return CATEGORY_TO_FAILURE_CASE.get(norm, 10)


def parse_agentrx_ground_truth(gt_source: str | Path | Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """Parse AgentRx ground truth JSON and resolve `root_cause.failure_id` into `failures[*]`."""
    if isinstance(gt_source, (str, Path)):
        gt_data = json.loads(Path(gt_source).read_text(encoding="utf-8"))
    else:
        gt_data = list(gt_source)

    by_tid: dict[str, dict[str, Any]] = {}
    for entry in gt_data:
        tid = str(entry["trajectory_id"])
        failures = sorted(entry.get("failures") or [], key=lambda f: int(f.get("step_number", 0)))

        all_cats = {
            normalize_agentrx_category(f.get("failure_category"))
            for f in failures
            if f.get("failure_category")
        }
        earliest_cat = normalize_agentrx_category(failures[0].get("failure_category")) if failures else ""
        terminal_cat = normalize_agentrx_category(failures[-1].get("failure_category")) if failures else ""

        rc_obj = entry.get("root_cause") or {}
        rc_id = rc_obj.get("failure_id")
        rc_failure = next((f for f in failures if f.get("failure_id") == rc_id), None)
        if rc_failure is None and failures:
            rc_failure = failures[0]

        rc_step = int(rc_failure["step_number"]) if rc_failure and rc_failure.get("step_number") is not None else None
        rc_cat = normalize_agentrx_category(rc_failure.get("failure_category")) if rc_failure else ""
        rc_agent = str(rc_failure.get("failed_agent") or "") if rc_failure else ""
        rc_reason = (
            str(rc_obj.get("reason_for_root_cause") or "")
            or (str(rc_failure.get("step_reason") or "") if rc_failure else "")
            or str(entry.get("failure_summary") or "")
        )

        by_tid[tid] = {
            "failures": failures,
            "all_categories": all_cats,
            "earliest_category": earliest_cat,
            "terminal_category": terminal_cat,
            "root_cause_failure_id": rc_id,
            "root_cause_step": rc_step,
            "root_cause_category": rc_cat,
            "root_cause_agent": rc_agent,
            "root_cause_reason": rc_reason,
            "failure_summary": str(entry.get("failure_summary") or ""),
        }
    return by_tid


# --- Trajectory IR & Raw Trace Converters (1-indexed steps) ------------------------------


def _steps_from_ir(ir_obj: Mapping[str, Any]) -> list[StepRecord]:
    steps: list[StepRecord] = []
    for i, s in enumerate(ir_obj.get("steps") or []):
        idx = int(s.get("index", i + 1))
        subs = s.get("substeps") or []
        if not subs:
            raw_role = str(s.get("role") or "unknown")
            content = str(s.get("content") or "")
        elif len(subs) == 1:
            raw_role = str(subs[0].get("role") or "unknown")
            content = str(subs[0].get("content") or "")
        else:
            raw_role = str(subs[0].get("role") or "step")
            content = "\n".join(
                f"[{sub.get('sub_index', j + 1)}] {sub.get('role', 'unknown')}: {sub.get('content', '')}"
                for j, sub in enumerate(subs)
            )
        steps.append(
            StepRecord(
                step_idx=idx,
                agent=canonical_agent_name(raw_role) or raw_role,
                raw_agent=raw_role,
                content=content,
            )
        )
    return steps


def _steps_from_tau_messages(messages: Sequence[Mapping[str, Any]]) -> list[StepRecord]:
    steps: list[StepRecord] = []
    for i, m in enumerate(messages):
        role = str(m.get("role") or "unknown").strip() or "unknown"
        raw_content = m.get("content") if m.get("content") is not None else m.get("message", "")
        parts: list[str] = []
        if raw_content:
            parts.append(str(raw_content).strip())
        tool_calls = m.get("tool_calls") or []
        if tool_calls:
            parts.append(json.dumps(tool_calls, ensure_ascii=False, sort_keys=True))
        if m.get("function"):
            parts.append(f"[function]\n{m['function']}")
        if m.get("response"):
            parts.append(f"[response]\n{m['response']}")
        steps.append(
            StepRecord(
                step_idx=i + 1,
                agent=canonical_agent_name(role) or role,
                raw_agent=role,
                content="\n\n".join(p for p in parts if p),
            )
        )
    return steps


def _steps_from_flash_events(events: Sequence[Mapping[str, Any]]) -> list[StepRecord]:
    grouped: dict[int, list[tuple[str, str]]] = {1: []}
    cur = 1
    for e in events:
        if e.get("type") == "LLMCallEvent":
            continue
        role = str(e.get("source") or e.get("role") or "unknown")
        content = str(e.get("message") if e.get("message") is not None else e.get("content") or "")
        if "(Step-" in role or content.lstrip().startswith("Step-"):
            m = FLASH_STEP_RE.search(role) or FLASH_STEP_RE.search(content.lstrip())
            if m:
                cur = int(m.group(1))
                grouped.setdefault(cur, [])
        grouped[cur].append((role, content))

    steps: list[StepRecord] = []
    for idx in sorted(grouped):
        subs = grouped[idx]
        if not subs:
            continue
        first_role = subs[0][0]
        body = (
            subs[0][1]
            if len(subs) == 1
            else "\n".join(f"[{j + 1}] {r}: {c}" for j, (r, c) in enumerate(subs))
        )
        steps.append(
            StepRecord(
                step_idx=idx,
                agent=canonical_agent_name(first_role) or first_role,
                raw_agent=first_role,
                content=body,
            )
        )
    return steps


def _attach_gt(rec: TraceRecord, gt_map: Mapping[str, dict[str, Any]]) -> TraceRecord:
    info = gt_map.get(rec.trace_id)
    if not info:
        return rec
    rec.mistake_step = info["root_cause_step"]
    rec.mistake_agent = info["root_cause_agent"] or rec.speaker_at(info["root_cause_step"])
    rec.mistake_reason = info["root_cause_reason"]
    rec.failure_category = info["root_cause_category"]
    rec.extra.update({
        "all_categories": sorted(info["all_categories"]),
        "earliest_category": info["earliest_category"],
        "terminal_category": info["terminal_category"],
        "failures": info["failures"],
        "root_cause_failure_id": info["root_cause_failure_id"],
    })
    return rec


def _normalize_single_item(
    obj: Any,
    *,
    default_id: str,
    domain: str,
    window: int,
    gt_map: Mapping[str, dict[str, Any]],
) -> TraceRecord:
    if isinstance(obj, Mapping) and isinstance(obj.get("steps"), list):
        tid = str(obj.get("trajectory_id") or default_id)
        instr = str(obj.get("instruction") or "")
        steps = _steps_from_ir(obj)
        if not instr and steps:
            instr = steps[0].content[:1000]
        attach_context(steps, window=window)
        rec = TraceRecord(trace_id=tid, dataset="agentrx", subset=domain, question=instr, steps=steps)
        if "ground_truth" in obj and isinstance(obj["ground_truth"], Mapping):
            gt_inline = parse_agentrx_ground_truth([{"trajectory_id": tid, **obj["ground_truth"]}])
            rec = _attach_gt(rec, gt_inline)
        elif "failures" in obj and "root_cause" in obj:
            gt_inline = parse_agentrx_ground_truth([{**obj, "trajectory_id": tid}])
            rec = _attach_gt(rec, gt_inline)
        return _attach_gt(rec, gt_map)

    if isinstance(obj, Mapping):
        tid = str(obj.get("trajectory_id") or obj.get("task_id") or obj.get("traceId") or default_id)
        events = (
            obj.get("traj")
            or obj.get("events")
            or obj.get("messages")
            or obj.get("trajectory")
            or obj.get("history")
            or []
        )
        if domain == "flash" or (events and isinstance(events[0], Mapping) and "source" in events[0]):
            steps = _steps_from_flash_events(events)
        else:
            steps = _steps_from_tau_messages(events)
        instr = str(
            obj.get("instruction")
            or (obj.get("info", {}) or {}).get("task", {}).get("instruction", "")
            or (steps[0].content[:1000] if steps else "")
        )
        attach_context(steps, window=window)
        rec = TraceRecord(trace_id=tid, dataset="agentrx", subset=domain, question=instr, steps=steps)
        if "failures" in obj and "root_cause" in obj:
            gt_inline = parse_agentrx_ground_truth([{**obj, "trajectory_id": tid}])
            rec = _attach_gt(rec, gt_inline)
        return _attach_gt(rec, gt_map)

    if isinstance(obj, list):
        # Raw Magentic-One / flat chat message list `[{"role": ..., "content": ...}, ...]`
        steps = _steps_from_tau_messages(obj)
        instr = next(
            (str(e.get("content") or "").strip() for e in obj if str(e.get("role") or "").lower() in ("human", "user")),
            steps[0].content[:1000] if steps else "",
        )
        attach_context(steps, window=window)
        rec = TraceRecord(trace_id=default_id, dataset="agentrx", subset=domain or "magentic", question=instr, steps=steps)
        return _attach_gt(rec, gt_map)

    raise ValueError(f"Unsupported AgentRx trajectory object type: {type(obj).__name__}")


def load_agentrx_traces(
    data_path: str | Path,
    subset: str | None = "auto",
    *,
    ground_truth_path: str | Path | None = None,
    window: int = 3,
    limit: int | None = None,
) -> list[TraceRecord]:
    """Load AgentRx 1-indexed trajectories + root-cause ground truth."""
    p = Path(data_path)
    domain = (subset or "auto").strip().lower()

    gt_map: dict[str, dict[str, Any]] = {}
    if ground_truth_path is not None:
        gt_map.update(parse_agentrx_ground_truth(ground_truth_path))

    # Auto-discover AgentRx repo layout (`AgentRx/` or `AgentRx/data/`).
    data_root = p / "data" if (p / "data").is_dir() else p
    gt_dir = data_root / "ground_truth"
    if gt_dir.is_dir() and not gt_map:
        if domain in ("auto", "all", "tau", "tau_retail") and (gt_dir / "tau_ground_truth.json").is_file():
            gt_map.update(parse_agentrx_ground_truth(gt_dir / "tau_ground_truth.json"))
        if domain in ("auto", "all", "magentic", "magentic_one") and (gt_dir / "magentic_one_ground_truth.json").is_file():
            gt_map.update(parse_agentrx_ground_truth(gt_dir / "magentic_one_ground_truth.json"))

    records: list[TraceRecord] = []

    if p.is_file():
        if p.suffix == ".jsonl":
            for idx, ln in enumerate(p.read_text(encoding="utf-8").splitlines()):
                if not ln.strip():
                    continue
                records.append(
                    _normalize_single_item(
                        json.loads(ln),
                        default_id=f"{p.stem}_{idx + 1}",
                        domain=domain,
                        window=window,
                        gt_map=gt_map,
                    )
                )
                if limit is not None and len(records) >= limit:
                    break
            return records

        obj = json.loads(p.read_text(encoding="utf-8-sig"))
        if isinstance(obj, list) and obj and isinstance(obj[0], Mapping) and (
            any(k in obj[0] for k in ("steps", "traj", "events", "messages", "trajectory"))
        ):
            for idx, item in enumerate(obj):
                tid = str(item.get("trajectory_id") or item.get("task_id") or f"{p.stem}_{idx + 1}")
                if gt_map and tid not in gt_map:
                    continue
                records.append(
                    _normalize_single_item(item, default_id=tid, domain=domain, window=window, gt_map=gt_map)
                )
                if limit is not None and len(records) >= limit:
                    break
            return records

        records.append(_normalize_single_item(obj, default_id=p.stem, domain=domain, window=window, gt_map=gt_map))
        return records[:limit] if limit is not None else records

    if p.is_dir():
        if domain in ("auto", "all", "tau", "tau_retail"):
            tau_file = data_root / "tau_retail" / "tau_dataset_failed.json"
            if tau_file.is_file():
                tau_recs = load_agentrx_traces(
                    tau_file,
                    subset="tau",
                    ground_truth_path=gt_dir / "tau_ground_truth.json" if (gt_dir / "tau_ground_truth.json").is_file() else ground_truth_path,
                    window=window,
                    limit=limit,
                )
                records.extend(tau_recs)
                if limit is not None and len(records) >= limit:
                    return records[:limit]

        if domain in ("auto", "all", "magentic", "magentic_one"):
            mag_dir = data_root / "magentic_dataset"
            if mag_dir.is_dir():
                mag_gt = (
                    parse_agentrx_ground_truth(gt_dir / "magentic_one_ground_truth.json")
                    if (gt_dir / "magentic_one_ground_truth.json").is_file()
                    else gt_map
                )
                for fp in sorted(mag_dir.glob("*.json")):
                    if fp.name == "steps_by_id.json":
                        continue
                    if mag_gt and fp.stem not in mag_gt:
                        continue
                    obj = json.loads(fp.read_text(encoding="utf-8-sig"))
                    records.append(
                        _normalize_single_item(obj, default_id=fp.stem, domain="magentic", window=window, gt_map=mag_gt)
                    )
                    if limit is not None and len(records) >= limit:
                        return records[:limit]

        if records:
            return records

        # Generic directory of IR / raw JSON files.
        for fp in sorted(p.glob("*.json")):
            obj = json.loads(fp.read_text(encoding="utf-8-sig"))
            records.append(_normalize_single_item(obj, default_id=fp.stem, domain=domain, window=window, gt_map=gt_map))
            if limit is not None and len(records) >= limit:
                return records[:limit]
        return records

    raise FileNotFoundError(f"AgentRx data path not found: {data_path!r}")


def load_agentrx(
    data_path: str | Path,
    subset: str | None = "auto",
    *,
    ground_truth_path: str | Path | None = None,
    window: int = 3,
    limit: int | None = None,
) -> FailureTraceAdapter:
    """Load AgentRx as a `FailureTraceAdapter` (`src.traces` and `src.steps`, 1-indexed `step_idx`)."""
    return FailureTraceAdapter(
        load_agentrx_traces(
            data_path,
            subset=subset,
            ground_truth_path=ground_truth_path,
            window=window,
            limit=limit,
        )
    )


# --- Official AgentRx Scorer -------------------------------------------------------------


def score_agentrx(
    traces_or_adapter: Sequence[TraceRecord] | FailureTraceAdapter,
    predictions: Mapping[str, Mapping[str, Any]] | str | Path,
    *,
    n_boot: int = 2000,
    seed: int = 0,
) -> dict[str, Any]:
    """Score AgentRx 1-indexed root-cause localization & 10-category classification:
    - `step_acc`, `step_pm1`..`step_pm5`
    - `step_mae` (mean `|pred_step - gt_step|`)
    - `norm_step_dist` (mean `|pred_step - gt_step| / max(n_steps, 1)`)
    - `category_acc` (root-cause category match), `any_category_acc`, `earliest_category_acc`,
      `terminal_category_acc`, `agent_acc`, and `both_exact`.
    """
    traces = traces_or_adapter.trace_records if isinstance(traces_or_adapter, FailureTraceAdapter) else list(traces_or_adapter)
    preds = (
        parse_whowhen_predictions(predictions)
        if isinstance(predictions, (str, Path))
        else dict(predictions)
    )

    step_vec: list[float] = []
    pm_vecs: dict[int, list[float]] = {k: [] for k in range(1, 6)}
    mae_vec: list[float] = []
    norm_dist_vec: list[float] = []
    cat_vec: list[float] = []
    any_cat_vec: list[float] = []
    earliest_cat_vec: list[float] = []
    terminal_cat_vec: list[float] = []
    agent_vec: list[float] = []
    both_vec: list[float] = []

    matched = 0
    for tr in traces:
        pred = lookup_prediction(preds, tr.trace_id)
        gt_step = tr.mistake_step
        gt_cat = normalize_agentrx_category(tr.failure_category)
        gt_agent = normalize_agent_name(tr.mistake_agent)
        all_cats = set(tr.extra.get("all_categories") or ([gt_cat] if gt_cat else []))
        earliest_cat = normalize_agentrx_category(tr.extra.get("earliest_category") or gt_cat)
        terminal_cat = normalize_agentrx_category(tr.extra.get("terminal_category") or gt_cat)

        if not pred:
            step_vec.append(0.0)
            for k in range(1, 6):
                pm_vecs[k].append(0.0)
            cat_vec.append(0.0)
            any_cat_vec.append(0.0)
            earliest_cat_vec.append(0.0)
            terminal_cat_vec.append(0.0)
            agent_vec.append(0.0)
            both_vec.append(0.0)
            continue

        matched += 1
        raw_step = (
            pred.get("predicted_step")
            if pred.get("predicted_step") is not None
            else pred.get("step_mean", pred.get("step_median"))
        )
        if isinstance(raw_step, (int, float)):
            pred_step: int | None = int(round(raw_step))
        elif raw_step is not None and str(raw_step).strip().lstrip("-").isdigit():
            pred_step = int(str(raw_step).strip())
        else:
            pred_step = None

        raw_cat = (
            pred.get("failure_category")
            or pred.get("category")
            or pred.get("most_common_failure")
            or ""
        )
        pred_cat = normalize_agentrx_category(raw_cat)
        pred_agent = normalize_agent_name(pred.get("predicted_agent") or "")

        step_ok = bool(pred_step is not None and gt_step is not None and pred_step == gt_step)
        step_vec.append(1.0 if step_ok else 0.0)

        for k in range(1, 6):
            pm_ok = bool(pred_step is not None and gt_step is not None and abs(pred_step - gt_step) <= k)
            pm_vecs[k].append(1.0 if pm_ok else 0.0)

        if pred_step is not None and gt_step is not None:
            dist = float(abs(pred_step - gt_step))
            mae_vec.append(dist)
            norm_dist_vec.append(dist / max(tr.n_steps, 1))

        cat_ok = bool(pred_cat and gt_cat and pred_cat == gt_cat)
        cat_vec.append(1.0 if cat_ok else 0.0)
        any_cat_vec.append(1.0 if (pred_cat and pred_cat in all_cats) else 0.0)
        earliest_cat_vec.append(1.0 if (pred_cat and earliest_cat and pred_cat == earliest_cat) else 0.0)
        terminal_cat_vec.append(1.0 if (pred_cat and terminal_cat and pred_cat == terminal_cat) else 0.0)

        agent_ok = bool(pred_agent and gt_agent and pred_agent == gt_agent)
        agent_vec.append(1.0 if agent_ok else 0.0)
        both_vec.append(1.0 if (step_ok and (cat_ok if gt_cat else agent_ok)) else 0.0)

    step_mean, step_lo, step_hi = bootstrap_ci(step_vec, n_boot=n_boot, seed=seed)
    cat_mean, cat_lo, cat_hi = bootstrap_ci(cat_vec, n_boot=n_boot, seed=seed)

    return {
        "n_traces": len(traces),
        "n_predictions": matched,
        "step_acc": step_mean,
        **{f"step_pm{k}": (sum(pm_vecs[k]) / len(pm_vecs[k]) if pm_vecs[k] else 0.0) for k in range(1, 6)},
        "step_mae": (sum(mae_vec) / len(mae_vec)) if mae_vec else None,
        "norm_step_dist": (sum(norm_dist_vec) / len(norm_dist_vec)) if norm_dist_vec else None,
        "category_acc": cat_mean,
        "any_category_acc": (sum(any_cat_vec) / len(any_cat_vec)) if any_cat_vec else 0.0,
        "earliest_category_acc": (sum(earliest_cat_vec) / len(earliest_cat_vec)) if earliest_cat_vec else 0.0,
        "terminal_category_acc": (sum(terminal_cat_vec) / len(terminal_cat_vec)) if terminal_cat_vec else 0.0,
        "agent_acc": (sum(agent_vec) / len(agent_vec)) if agent_vec else 0.0,
        "both_exact": (sum(both_vec) / len(both_vec)) if both_vec else 0.0,
        "ci95": {
            "step_acc": (step_lo, step_hi),
            "category_acc": (cat_lo, cat_hi),
        },
    }
