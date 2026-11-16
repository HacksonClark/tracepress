"""TracePress failure-attribution pipelines: `all_at_once`, `map_reduce`, `fold`, `staged`, and `trail_pipeline`."""

from __future__ import annotations

import copy
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import dspy
import tracepress as tp
from tracepress.adapters.base import Adapter
from tracepress.plan import Table
from tracepress.rows import Row

from .datasets.agentrx import AGENTRX_CATEGORIES, normalize_agentrx_category
from .datasets.common import FailureTraceAdapter, StepRecord, TraceRecord, attach_context
from .datasets.longrca import canonical_role, normalize_role
from .datasets.trail import TRAIL_CATEGORIES, normalize_trail_category
from .datasets.whowhen import canonical_agent_name, normalize_agent_name

# --- Instructions ------------------------------------------------------------------------

STORY_INSTRUCTIONS = (
    "Reconstruct the trajectory's execution arc from its steps:\n"
    "1. `plan`: What approach, roles, or delegation plan the team adopted to solve `question`.\n"
    "2. `final_answer`: What final output or termination state the system produced.\n"
    "3. `visible_failure`: Specifically how the run failed (wrong answer, unhandled exception, "
    "violated constraint, infinite loop, or premature stop), without guessing the root cause yet."
)

SCREEN_INSTRUCTIONS = (
    "Inspect one step of a failed multi-agent run in context (`prev_context` and `upstream_handoff`).\n"
    "Set `flagged=true` if this step introduces, propagates, or reveals a concrete mistake: "
    "a flawed plan/delegation, wrong assumption, hallucinated value, misread tool/search output, "
    "buggy code, skipped constraint, or execution failure. "
    "In `suspect_note`, concisely state the concrete evidence and whether it may originate from "
    "`upstream_handoff`."
)

TRACE_BACK_INSTRUCTIONS = (
    "Determine whether this flagged step is the earliest decisive root cause or a downstream symptom.\n"
    "- Check `upstream_handoff`, `prev_context`, and `flagged_digest` (all flagged steps in the trace).\n"
    "- If this step is an executor (`Computer_terminal`, `tool`, etc.) failing on code or arguments "
    "written earlier, or an agent following a flawed instruction from an upstream planner/orchestrator, "
    "trace responsibility back to that upstream step (`origin_step_idx`) and `responsible_agent`.\n"
    "- Set `is_root_cause=true` only if the decisive error originated here (or at `origin_step_idx` "
    "with no earlier cause)."
)

DECIDE_INSTRUCTIONS = (
    "Identify the single earliest decisive root-cause mistake that made the trajectory fail.\n"
    "- Prefer the earliest unrecovered error whose correction would have prevented the failure, "
    "not late symptoms, terminal crashes, or final answer formatting turns.\n"
    "- When a `Computer_terminal` or tool step fails because an upstream coder/expert wrote bad code "
    "or arguments, attribute `mistake_agent` and `mistake_step` to the agent/step that introduced the defect "
    "(unless the benchmark convention blames an explicit handoff step).\n"
    "- Pick `mistake_agent` from `known_agents` whenever applicable, output `mistake_step` as the exact "
    "`step_idx` integer, and explain the causal chain in `reason`."
)

ALL_AT_ONCE_INSTRUCTIONS = (
    "Analyze the complete failed trajectory and attribute the failure to the single responsible agent "
    "(`mistake_agent`, chosen from `known_agents`) and the earliest decisive `step_idx` (`mistake_step`) "
    "whose error caused the task to fail. Explain why in `reason`."
)

MAP_STEP_INSTRUCTIONS = (
    "Write a concise factual digest (`step_summary`) of what this agent did or observed at `step_idx`, "
    "and set `is_suspect=true` if this step contains an error, bad assumption, failed command, or "
    "constraint violation."
)

FOLD_INSTRUCTIONS = (
    "Steps of a failed multi-agent trajectory arrive one by one in chronological order.\n"
    "- Update `running_summary` with key actions, tool results, and state changes.\n"
    "- When you encounter the earliest decisive mistake that derails the task, record its `step_idx` "
    "in `mistake_step`, its responsible agent in `mistake_agent`, and why in `reason`.\n"
    "- Once a genuine root-cause mistake is recorded (`mistake_step >= 0`), keep that root cause "
    "through downstream cascading symptoms; only replace it if an earlier mistake was fully recovered "
    "and a new independent mistake later caused the final failure."
)

TRAIL_SCREEN_INSTRUCTIONS = (
    "Inspect one OpenTelemetry span (`span_id`, `agent`, `content`, `prev_context`) from an agent trace.\n"
    "Set `has_error=true` if this span exhibits any error from the 21 TRAIL subcategories:\n"
    + ", ".join(TRAIL_CATEGORIES)
    + ".\n"
    "If `has_error=true`, name the best-matching TRAIL subcategory in `candidate_category`, quote "
    "`candidate_evidence`, and describe the issue in `candidate_description`."
)

TRAIL_REDUCE_INSTRUCTIONS = (
    "Consolidate span-level findings into the complete list of TRAIL `errors` for this trace.\n"
    "Each item in `errors` MUST be a dict with keys:\n"
    "  `location` (exact `span_id`), `category` (one of the 21 TRAIL leaf categories: "
    + ", ".join(TRAIL_CATEGORIES)
    + "), `evidence`, `description`, and `impact` ('LOW'|'MEDIUM'|'HIGH').\n"
    "Rules: For 'Resource Abuse', report only the LAST span instance; for all other categories, "
    "report the FIRST span instance where that error occurs. Return `errors=[]` if no errors occurred."
)


# --- Helpers: Context Window, Compiled State, LM Copying, Validation ---------------------


def _with_window(src: Adapter, window: int) -> Adapter:
    if not isinstance(src, FailureTraceAdapter):
        return src
    rebuilt: list[TraceRecord] = []
    for tr in src.trace_records:
        copied_steps = [
            StepRecord(
                step_idx=s.step_idx,
                agent=s.agent,
                content=s.content,
                raw_agent=s.raw_agent,
                span_id=s.span_id,
                extra=dict(s.extra),
            )
            for s in tr.steps
        ]
        attach_context(copied_steps, window=window)
        rebuilt.append(copy.replace(tr, steps=copied_steps))
    return FailureTraceAdapter(rebuilt)


def load_compiled_state(compiled_state: str | Path | Mapping[str, Any] | None) -> dict[str, Any]:
    if compiled_state is None:
        return {}
    if isinstance(compiled_state, Mapping):
        return dict(compiled_state)
    return json.loads(Path(compiled_state).read_text(encoding="utf-8"))


def _stage_instructions_and_module(
    state: Mapping[str, Any],
    stage_name: str,
    default_instructions: str,
    base_module: type[dspy.Module] = dspy.Predict,
) -> tuple[str, Any]:
    """Extract compiled instructions + few-shot demos for `stage_name` if present in `state`."""
    raw = state.get(stage_name)
    if not isinstance(raw, Mapping):
        return default_instructions, base_module

    inner = raw.get("predict") if isinstance(raw.get("predict"), Mapping) else raw
    sig_info = inner.get("signature") if isinstance(inner.get("signature"), Mapping) else {}
    instructions = str(sig_info.get("instructions") or raw.get("instructions") or default_instructions)
    demos = list(inner.get("demos") or raw.get("demos") or [])

    if not demos:
        return instructions, base_module

    def factory(sig):
        mod = base_module(sig)
        target = getattr(mod, "predict", mod)
        target.demos = [dspy.Example(**d) if isinstance(d, Mapping) else d for d in demos]
        return mod

    return instructions, factory


def copy_lm_for_vote(lm_spec_or_obj: Any, rollout_id: int, temperature: float = 0.7) -> Any:
    """Resolve `lm_spec_or_obj` via TracePress and return `.copy(rollout_id=v, temperature=...)` if supported."""
    base_lm = tp.lm.resolve_lm(lm_spec_or_obj)
    if rollout_id == 0 and temperature is None:
        return base_lm
    if hasattr(base_lm, "copy"):
        try:
            copied = base_lm.copy(rollout_id=rollout_id, temperature=temperature)
            if hasattr(base_lm, "_tp_label") and not hasattr(copied, "_tp_label"):
                copied._tp_label = base_lm._tp_label
            return copied
        except Exception:
            return base_lm
    return base_lm


def _norm_for_dataset(name: str, dataset: str) -> str:
    if dataset == "longrca":
        return normalize_role(name)
    return normalize_agent_name(name)


def _canon_for_dataset(name: str, dataset: str) -> str:
    if dataset == "longrca":
        return canonical_role(name)
    return canonical_agent_name(name)


def _parse_int_step(raw: Any) -> int | None:
    if isinstance(raw, bool):
        return None
    if isinstance(raw, int):
        return raw
    if isinstance(raw, float):
        return int(round(raw))
    text = str(raw or "").strip()
    if not text:
        return None
    try:
        return int(text)
    except ValueError:
        import re

        m = re.search(r"-?\d+", text)
        return int(m.group(0)) if m else None


def _make_validator(src: Adapter, *, with_category: bool = False):
    by_id = src.by_id if isinstance(src, FailureTraceAdapter) else {}

    def validate(row: Row) -> dict[str, Any]:
        tr = by_id.get(row.trace)
        dataset = str(tr.dataset if tr is not None else row.get("dataset", "whowhen"))
        known_agents: list[str] = list(
            tr.known_agents if tr is not None else (row.get("known_agents", None) or [])
        )

        # 1. Clamp / snap `mistake_step` onto valid step indices for this trace.
        parsed_step = _parse_int_step(row.get("mistake_step", None))
        if tr is not None and tr.steps:
            valid_indices = [s.step_idx for s in tr.steps]
            lo, hi = min(valid_indices), max(valid_indices)
            if parsed_step is None or parsed_step < 0:
                valid_step = valid_indices[0]
            elif parsed_step in tr.step_by_idx:
                valid_step = parsed_step
            else:
                clamped = max(lo, min(hi, parsed_step))
                valid_step = min(valid_indices, key=lambda idx: abs(idx - clamped))
        else:
            n_steps = int(row.get("n_steps", 0) or 0)
            base_idx = 1 if dataset == "agentrx" else 0
            max_idx = max(base_idx, n_steps - (0 if base_idx == 1 else 1))
            if parsed_step is None or parsed_step < base_idx:
                valid_step = base_idx
            elif n_steps > 0:
                valid_step = max(base_idx, min(max_idx, parsed_step))
            else:
                valid_step = parsed_step

        # 2. Resolve `mistake_agent`: keep valid trace-known agents (e.g. upstream orchestrator/coder),
        #    falling back to speaker of `valid_step` when blank or unknown.
        raw_agent = _canon_for_dataset(str(row.get("mistake_agent", "") or ""), dataset)
        norm_agent = _norm_for_dataset(raw_agent, dataset)
        known_by_norm = {_norm_for_dataset(a, dataset): a for a in known_agents if a}

        placeholder_norms = {"", "unknown", "unknown agent", "none", "n/a", "null"}
        resolved_agent = ""
        if norm_agent not in placeholder_norms:
            if norm_agent in known_by_norm:
                resolved_agent = known_by_norm[norm_agent]
            else:
                # Unique substring match against known trace agents (e.g. "websurfer" -> "WebSurfer").
                matches = [
                    canon
                    for kn, canon in known_by_norm.items()
                    if kn and (norm_agent in kn or kn in norm_agent)
                ]
                if len(matches) == 1:
                    resolved_agent = matches[0]
                elif not known_by_norm:
                    resolved_agent = raw_agent

        if not resolved_agent:
            speaker = tr.speaker_at(valid_step) if tr is not None else ""
            resolved_agent = speaker or raw_agent or (known_agents[0] if known_agents else "Unknown")

        out: dict[str, Any] = {
            "mistake_step": int(valid_step),
            "mistake_agent": str(resolved_agent),
            "reason": str(row.get("reason", "") or "").strip(),
        }
        if with_category:
            out["failure_category"] = normalize_agentrx_category(row.get("failure_category", ""))
        return out

    return validate


def _validator_columns(*, with_category: bool = False) -> dict[str, Any]:
    cols: dict[str, Any] = {"mistake_step": int, "mistake_agent": str, "reason": str}
    if with_category:
        cols["failure_category"] = str
    return cols


# --- Pipelines 1-3: `all_at_once`, `map_reduce`, `fold` ----------------------------------


def all_at_once(
    src: Adapter,
    *,
    with_gt: bool = True,
    with_category: bool = False,
    lm: Any = None,
    module: type[dspy.Module] = dspy.ChainOfThought,
    chunk_chars: int | None = None,
    on_error: str = "skip",
) -> Table:
    """Single-reduce baseline over raw trajectory steps."""
    gt_in = "ground_truth, " if with_gt else ""
    cat_out = ", failure_category: str" if with_category else ""
    instructions = ALL_AT_ONCE_INSTRUCTIONS
    if with_category:
        instructions += "\nAlso classify `failure_category` into one of: " + ", ".join(AGENTRX_CATEGORIES) + "."

    steps = src.table("steps").join(src.table("traces"))
    reduced = steps.group_by("trace").lm_reduce(
        f"question, {gt_in}known_agents, step_idx, agent, content -> mistake_step: int, mistake_agent: str, reason: str{cat_out}",
        instructions=instructions,
        lm=lm,
        module=module,
        chunk_chars=chunk_chars,
        on_error=on_error,
    )
    return reduced.map(
        _make_validator(src, with_category=with_category),
        columns=_validator_columns(with_category=with_category),
    )


def map_reduce(
    src: Adapter,
    *,
    with_gt: bool = True,
    with_category: bool = False,
    lm: Any = None,
    screen_lm: Any = None,
    module: type[dspy.Module] = dspy.ChainOfThought,
    chunk_chars: int | None = None,
    on_error: str = "skip",
) -> Table:
    """Two-stage `lm_map` per-step digest + `lm_reduce` root-cause judge."""
    gt_in = "ground_truth, " if with_gt else ""
    cat_out = ", failure_category: str" if with_category else ""
    decide_inst = DECIDE_INSTRUCTIONS
    if with_category:
        decide_inst += "\nAlso classify `failure_category` into one of: " + ", ".join(AGENTRX_CATEGORIES) + "."

    steps = src.table("steps").join(src.table("traces"))
    mapped = steps.lm_map(
        f"question, {gt_in}step_idx, agent, prev_context, content -> step_summary: str, is_suspect: bool",
        instructions=MAP_STEP_INSTRUCTIONS,
        lm=screen_lm if screen_lm is not None else lm,
        on_error=on_error,
    )
    reduced = mapped.group_by("trace").lm_reduce(
        f"question, {gt_in}known_agents, step_idx, agent, step_summary, is_suspect -> mistake_step: int, mistake_agent: str, reason: str{cat_out}",
        instructions=decide_inst,
        lm=lm,
        module=module,
        chunk_chars=chunk_chars,
        on_error=on_error,
    )
    return reduced.map(
        _make_validator(src, with_category=with_category),
        columns=_validator_columns(with_category=with_category),
    )


def fold(
    src: Adapter,
    *,
    with_gt: bool = True,
    with_category: bool = False,
    lm: Any = None,
    module: type[dspy.Module] = dspy.Predict,
    on_error: str = "skip",
) -> Table:
    """Sequential `lm_fold` carrying root-cause state forward one step at a time."""
    gt_in = "ground_truth, " if with_gt else ""
    cat_state = ", failure_category: str" if with_category else ""
    cat_in = ", failure_category" if with_category else ""
    instructions = FOLD_INSTRUCTIONS
    if with_category:
        instructions += "\nMaintain `failure_category` from: " + ", ".join(AGENTRX_CATEGORIES) + "."

    init_state: dict[str, Any] = {
        "running_summary": "",
        "mistake_step": -1,
        "mistake_agent": "",
        "reason": "",
    }
    if with_category:
        init_state["failure_category"] = ""

    steps = src.table("steps").join(src.table("traces"))
    folded = steps.group_by("trace").lm_fold(
        f"question, {gt_in}known_agents, running_summary, mistake_step, mistake_agent, reason{cat_in}, "
        f"step_idx, agent, content -> running_summary: str, mistake_step: int, mistake_agent: str, reason: str{cat_state}",
        init=init_state,
        instructions=instructions,
        lm=lm,
        module=module,
        on_error=on_error,
    )
    return folded.map(
        _make_validator(src, with_category=with_category),
        columns=_validator_columns(with_category=with_category),
    )


# --- Pipeline 4: 4-Stage Causal Attribution + Majority Vote (`staged`) -------------------


def _summarize_flagged_steps(rows: list[Row]) -> dict[str, Any]:
    """Build per-trace `flagged_digest` and safe fallback step indices when 0 steps are flagged."""
    flagged = [r for r in rows if bool(r.get("flagged", False))]
    if flagged:
        lines = [
            f"[step {r.step_idx}] {r.agent}: {str(r.get('suspect_note') or '').strip()}"
            for r in flagged
        ]
        return {
            "flagged_digest": "\n".join(lines),
            "n_flagged": len(flagged),
            "fallback_indices": [],
        }

    # Safe fallback when 0 steps were flagged: pick up to 3 anchor steps (first non-user, midpoint, last)
    # so the trace is never dropped before Stage 3 / Stage 4.
    indices = [r.step_idx for r in rows]
    non_user = [r.step_idx for r in rows if str(r.agent).lower() not in ("human", "user")]
    picks: list[int] = []
    if non_user:
        picks.append(non_user[0])
    elif indices:
        picks.append(indices[0])
    if len(indices) >= 3:
        mid = indices[len(indices) // 2]
        if mid not in picks:
            picks.append(mid)
    if len(indices) >= 2 and indices[-1] not in picks:
        picks.append(indices[-1])

    return {
        "flagged_digest": "No step was explicitly flagged in Stage 2; evaluating fallback anchor steps.",
        "n_flagged": 0,
        "fallback_indices": picks,
    }


def _single_staged_branch(
    steps_with_story: Table,
    src: Adapter,
    *,
    with_gt: bool,
    with_category: bool,
    branch_lm: Any,
    branch_screen_lm: Any,
    state: Mapping[str, Any],
    chunk_chars: int | None,
    on_error: str,
) -> Table:
    gt_in = "ground_truth, " if with_gt else ""
    cat_out = ", failure_category: str" if with_category else ""

    screen_inst, screen_mod = _stage_instructions_and_module(state, "screen", SCREEN_INSTRUCTIONS, dspy.Predict)
    tb_inst, tb_mod = _stage_instructions_and_module(state, "trace_back", TRACE_BACK_INSTRUCTIONS, dspy.Predict)
    decide_inst, decide_mod = _stage_instructions_and_module(state, "decide", DECIDE_INSTRUCTIONS, dspy.ChainOfThought)
    if with_category:
        decide_inst += "\nAlso classify `failure_category` into one of: " + ", ".join(AGENTRX_CATEGORIES) + "."

    # Stage 2: Screen every step in parallel with story + sliding window + upstream handoff.
    screened = steps_with_story.lm_map(
        f"question, {gt_in}plan, final_answer, visible_failure, "
        "step_idx, agent, prev_context, upstream_handoff, content -> flagged: bool, suspect_note: str",
        instructions=screen_inst,
        lm=branch_screen_lm,
        module=screen_mod,
        on_error=on_error,
    )

    # Attach all flagged notes per trace + safe 0-flagged fallback.
    flagged_info = screened.group_by("trace").reduce(
        _summarize_flagged_steps,
        columns={"flagged_digest": str, "n_flagged": int, "fallback_indices": list},
    )
    candidates = screened.join(flagged_info).filter(
        lambda r: bool(r.flagged) or (r.n_flagged == 0 and r.step_idx in (r.fallback_indices or ()))
    )

    # Stage 3: Trace-back each candidate step to distinguish symptoms from upstream root causes.
    traced = candidates.lm_map(
        "question, plan, visible_failure, flagged_digest, "
        "step_idx, agent, prev_context, upstream_handoff, content, suspect_note "
        "-> is_root_cause: bool, origin_step_idx: int, responsible_agent: str, causal_note: str",
        instructions=tb_inst,
        lm=branch_lm,
        module=tb_mod,
        on_error=on_error,
    )

    # Stage 4: Decide root-cause `(mistake_step, mistake_agent, reason)` + Python validator.
    decided = traced.group_by("trace").lm_reduce(
        f"question, {gt_in}known_agents, plan, final_answer, visible_failure, "
        "step_idx, agent, suspect_note, is_root_cause, origin_step_idx, responsible_agent, causal_note "
        f"-> mistake_step: int, mistake_agent: str, reason: str{cat_out}",
        instructions=decide_inst,
        lm=branch_lm,
        module=decide_mod,
        chunk_chars=chunk_chars,
        on_error=on_error,
    )
    return decided.map(
        _make_validator(src, with_category=with_category),
        columns=_validator_columns(with_category=with_category),
    )


def _aggregate_votes(vote_k: int, src: Adapter, *, with_category: bool = False):
    by_id = src.by_id if isinstance(src, FailureTraceAdapter) else {}

    def vote_fn(row: Row) -> dict[str, Any]:
        tr = by_id.get(row.trace)
        dataset = str(tr.dataset if tr is not None else row.get("dataset", "whowhen"))

        ballots: list[tuple[int, str, str, str, str]] = []
        for v in range(vote_k):
            s = row.get(f"mistake_step_v{v}", None)
            a = row.get(f"mistake_agent_v{v}", None)
            r = str(row.get(f"reason_v{v}", "") or "")
            c = str(row.get(f"failure_category_v{v}", "") or "")
            if s is not None and a is not None:
                ballots.append((int(s), _norm_for_dataset(str(a), dataset), str(a), r, c))

        if not ballots:
            fallback_step = tr.steps[0].step_idx if (tr is not None and tr.steps) else (1 if dataset == "agentrx" else 0)
            fallback_agent = tr.speaker_at(fallback_step) if tr is not None else "Unknown"
            out: dict[str, Any] = {
                "mistake_step": fallback_step,
                "mistake_agent": fallback_agent or "Unknown",
                "reason": "",
                "vote_agreement": 0.0,
            }
            if with_category:
                out["failure_category"] = "Inconclusive"
            return out

        pair_counts = Counter((s, na) for s, na, *_ in ballots)
        step_counts = Counter(s for s, *_ in ballots)

        # Majority vote on `(mistake_step, normalized_agent)`, breaking ties by step majority then earliest `mistake_step`.
        best_pair = min(
            pair_counts,
            key=lambda pair: (-pair_counts[pair], -step_counts[pair[0]], pair[0], pair[1]),
        )
        winning_ballots = [b for b in ballots if (b[0], b[1]) == best_pair]
        chosen = winning_ballots[0]

        out = {
            "mistake_step": chosen[0],
            "mistake_agent": chosen[2],
            "reason": chosen[3],
            "vote_agreement": pair_counts[best_pair] / len(ballots),
        }
        if with_category:
            cats = [b[4] for b in winning_ballots if b[4]] or [b[4] for b in ballots if b[4]]
            out["failure_category"] = Counter(cats).most_common(1)[0][0] if cats else "Inconclusive"
        return out

    return vote_fn


def staged(
    src: Adapter,
    *,
    with_gt: bool = True,
    with_category: bool = False,
    window: int = 3,
    vote_k: int = 1,
    lm: Any = None,
    screen_lm: Any = None,
    compiled_state: str | Path | Mapping[str, Any] | None = None,
    vote_temperature: float = 0.7,
    chunk_chars: int | None = None,
    on_error: str = "skip",
) -> Table:
    """4-stage TracePress root-cause attribution pipeline (`story` -> `screen` -> `trace_back` -> `decide`),
    with optional `vote_k` self-consistency majority voting over Stages 2-4.
    """
    if vote_k < 1:
        raise ValueError(f"vote_k must be >= 1, got {vote_k}")

    if window != 3 and isinstance(src, FailureTraceAdapter):
        src = _with_window(src, window)

    state = load_compiled_state(compiled_state)
    gt_in = "ground_truth, " if with_gt else ""

    # Stage 1: Reconstruct trace-level story (`plan`, `final_answer`, `visible_failure`) once.
    story_inst, story_mod = _stage_instructions_and_module(state, "story", STORY_INSTRUCTIONS, dspy.Predict)
    base_steps = src.table("steps").join(src.table("traces"))
    story = base_steps.group_by("trace").lm_reduce(
        f"question, {gt_in}step_idx, agent, content -> plan: str, final_answer: str, visible_failure: str",
        instructions=story_inst,
        lm=lm,
        module=story_mod,
        chunk_chars=chunk_chars,
        on_error=on_error,
    )
    steps_with_story = base_steps.join(story)

    if vote_k == 1:
        return _single_staged_branch(
            steps_with_story,
            src,
            with_gt=with_gt,
            with_category=with_category,
            branch_lm=lm,
            branch_screen_lm=screen_lm if screen_lm is not None else lm,
            state=state,
            chunk_chars=chunk_chars,
            on_error=on_error,
        )

    # Stage 5 (`vote_k > 1`): Run Stages 2-4 `vote_k` times sharing Stage 1 `story`, with distinct rollout copies.
    joined_votes = src.table("traces").join(story)
    for v in range(vote_k):
        lm_v = copy_lm_for_vote(lm, rollout_id=v, temperature=vote_temperature)
        screen_lm_v = (
            copy_lm_for_vote(screen_lm, rollout_id=v, temperature=vote_temperature)
            if screen_lm is not None
            else lm_v
        )
        branch = _single_staged_branch(
            steps_with_story,
            src,
            with_gt=with_gt,
            with_category=with_category,
            branch_lm=lm_v,
            branch_screen_lm=screen_lm_v,
            state=state,
            chunk_chars=chunk_chars,
            on_error=on_error,
        )
        rename_cols: dict[str, Any] = {
            f"mistake_step_v{v}": int,
            f"mistake_agent_v{v}": str,
            f"reason_v{v}": str,
        }
        if with_category:
            rename_cols[f"failure_category_v{v}"] = str

        def _rename(r: Row, v_idx: int = v) -> dict[str, Any]:
            d: dict[str, Any] = {
                f"mistake_step_v{v_idx}": r.mistake_step,
                f"mistake_agent_v{v_idx}": r.mistake_agent,
                f"reason_v{v_idx}": r.reason,
            }
            if with_category:
                d[f"failure_category_v{v_idx}"] = r.failure_category
            return d

        branch_renamed = branch.map(_rename, columns=rename_cols).select(*rename_cols.keys())
        joined_votes = joined_votes.join(branch_renamed)

    out_cols = {**_validator_columns(with_category=with_category), "vote_agreement": float}
    return joined_votes.map(
        _aggregate_votes(vote_k, src, with_category=with_category),
        columns=out_cols,
    )


# --- Pipeline 5: Multi-Label Span Localization & Taxonomy Classification (`trail_pipeline`)


def _summarize_trail_candidates(rows: list[Row]) -> dict[str, Any]:
    flagged = [r for r in rows if bool(r.get("has_error", False))]
    return {
        "n_flagged_spans": len(flagged),
        "fallback_span_idx": rows[0].step_idx if rows else 0,
    }


def _make_trail_sanitizer(src: Adapter):
    by_id = src.by_id if isinstance(src, FailureTraceAdapter) else {}

    def sanitize(row: Row) -> dict[str, Any]:
        tr = by_id.get(row.trace)
        valid_spans = {s.span_id for s in tr.steps if s.span_id} if tr is not None else set()
        raw_errors = row.get("errors", None) or []

        cleaned: list[dict[str, Any]] = []
        seen_pairs: set[tuple[str, str]] = set()
        for item in raw_errors:
            if hasattr(item, "model_dump"):
                item = item.model_dump()
            if not isinstance(item, Mapping):
                continue
            loc = str(item.get("location") or item.get("span_id") or "").strip()
            if valid_spans and loc not in valid_spans:
                continue
            cat = normalize_trail_category(item.get("category") or "")
            if cat not in TRAIL_CATEGORIES:
                continue
            pair = (loc, cat)
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            cleaned.append({
                "location": loc,
                "category": cat,
                "evidence": str(item.get("evidence") or ""),
                "description": str(item.get("description") or ""),
                "impact": str(item.get("impact") or "MEDIUM").upper(),
            })
        return {"errors": cleaned}

    return sanitize


def trail_pipeline(
    src: Adapter,
    *,
    lm: Any = None,
    screen_lm: Any = None,
    module: type[dspy.Module] = dspy.ChainOfThought,
    chunk_chars: int | None = None,
    on_error: str = "skip",
) -> Table:
    """Per-span `lm_map` screening + per-trace `lm_reduce` emitting TRAIL `errors: list[dict]`."""
    spans = src.table("steps").join(src.table("traces"))
    screened = spans.lm_map(
        "question, span_id, step_idx, agent, prev_context, content "
        "-> has_error: bool, candidate_category: str, candidate_evidence: str, candidate_description: str",
        instructions=TRAIL_SCREEN_INSTRUCTIONS,
        lm=screen_lm if screen_lm is not None else lm,
        on_error=on_error,
    )
    counts = screened.group_by("trace").reduce(
        _summarize_trail_candidates,
        columns={"n_flagged_spans": int, "fallback_span_idx": int},
    )
    filtered = screened.join(counts).filter(
        lambda r: bool(r.has_error) or (r.n_flagged_spans == 0 and r.step_idx == r.fallback_span_idx)
    )
    reduced = filtered.group_by("trace").lm_reduce(
        "question, span_id, step_idx, agent, has_error, candidate_category, candidate_evidence, candidate_description "
        "-> errors: list[dict]",
        instructions=TRAIL_REDUCE_INSTRUCTIONS,
        lm=lm,
        module=module,
        chunk_chars=chunk_chars,
        on_error=on_error,
    )
    return reduced.map(_make_trail_sanitizer(src), columns={"errors": list})
