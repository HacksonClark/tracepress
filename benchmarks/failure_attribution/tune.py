"""DSPy module wrapper and MIPROv2 / GEPA prompt compilation for `pipelines.staged`."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import dspy

from .datasets.common import TraceRecord
from .datasets.longrca import normalize_role
from .datasets.whowhen import normalize_agent_name
from .pipelines import (
    DECIDE_INSTRUCTIONS,
    SCREEN_INSTRUCTIONS,
    STORY_INSTRUCTIONS,
    TRACE_BACK_INSTRUCTIONS,
)


def _build_story_sig(with_gt: bool) -> type[dspy.Signature]:
    fields: dict[str, Any] = {"question": (str, dspy.InputField())}
    if with_gt:
        fields["ground_truth"] = (str, dspy.InputField())
    fields["steps"] = (list[dict], dspy.InputField(desc="one record per step: step_idx, agent, content"))
    fields["plan"] = (str, dspy.OutputField())
    fields["final_answer"] = (str, dspy.OutputField())
    fields["visible_failure"] = (str, dspy.OutputField())
    return dspy.Signature(fields, STORY_INSTRUCTIONS)


def _build_screen_sig(with_gt: bool) -> type[dspy.Signature]:
    fields: dict[str, Any] = {"question": (str, dspy.InputField())}
    if with_gt:
        fields["ground_truth"] = (str, dspy.InputField())
    fields.update({
        "plan": (str, dspy.InputField()),
        "final_answer": (str, dspy.InputField()),
        "visible_failure": (str, dspy.InputField()),
        "step_idx": (int, dspy.InputField()),
        "agent": (str, dspy.InputField()),
        "prev_context": (str, dspy.InputField()),
        "upstream_handoff": (str, dspy.InputField()),
        "content": (str, dspy.InputField()),
        "flagged": (bool, dspy.OutputField()),
        "suspect_note": (str, dspy.OutputField()),
    })
    return dspy.Signature(fields, SCREEN_INSTRUCTIONS)


def _build_trace_back_sig() -> type[dspy.Signature]:
    fields: dict[str, Any] = {
        "question": (str, dspy.InputField()),
        "plan": (str, dspy.InputField()),
        "visible_failure": (str, dspy.InputField()),
        "flagged_digest": (str, dspy.InputField()),
        "step_idx": (int, dspy.InputField()),
        "agent": (str, dspy.InputField()),
        "prev_context": (str, dspy.InputField()),
        "upstream_handoff": (str, dspy.InputField()),
        "content": (str, dspy.InputField()),
        "suspect_note": (str, dspy.InputField()),
        "is_root_cause": (bool, dspy.OutputField()),
        "origin_step_idx": (int, dspy.OutputField()),
        "responsible_agent": (str, dspy.OutputField()),
        "causal_note": (str, dspy.OutputField()),
    }
    return dspy.Signature(fields, TRACE_BACK_INSTRUCTIONS)


def _build_decide_sig(with_gt: bool) -> type[dspy.Signature]:
    fields: dict[str, Any] = {"question": (str, dspy.InputField())}
    if with_gt:
        fields["ground_truth"] = (str, dspy.InputField())
    fields.update({
        "known_agents": (list, dspy.InputField()),
        "plan": (str, dspy.InputField()),
        "final_answer": (str, dspy.InputField()),
        "visible_failure": (str, dspy.InputField()),
        "steps": (
            list[dict],
            dspy.InputField(
                desc="traced candidate steps with step_idx, agent, suspect_note, is_root_cause, origin_step_idx, responsible_agent, causal_note"
            ),
        ),
        "mistake_step": (int, dspy.OutputField()),
        "mistake_agent": (str, dspy.OutputField()),
        "reason": (str, dspy.OutputField()),
    })
    return dspy.Signature(fields, DECIDE_INSTRUCTIONS)


class StagedAttributionModule(dspy.Module):
    """Standalone DSPy `Module` mirroring `pipelines.staged`'s 4 predictors (`story`, `screen`, `trace_back`, `decide`)
    so DSPy optimizers (`MIPROv2`, `GEPA`, `BootstrapFewShot`) can tune instructions & demos and export JSON state.
    """

    def __init__(self, *, with_gt: bool = True):
        super().__init__()
        self.with_gt = with_gt
        self.story = dspy.Predict(_build_story_sig(with_gt))
        self.screen = dspy.Predict(_build_screen_sig(with_gt))
        self.trace_back = dspy.Predict(_build_trace_back_sig())
        self.decide = dspy.ChainOfThought(_build_decide_sig(with_gt))

    def forward(
        self,
        question: str,
        steps: list[dict[str, Any]],
        known_agents: list[str] | None = None,
        ground_truth: str = "",
        **_: Any,
    ) -> dspy.Prediction:
        known = list(known_agents or [])
        if not known:
            seen: dict[str, str] = {}
            for s in steps:
                a = str(s.get("agent") or "").strip()
                if a and a.lower() not in seen:
                    seen[a.lower()] = a
            known = list(seen.values())

        raw_steps = [
            {"step_idx": int(s["step_idx"]), "agent": str(s["agent"]), "content": str(s["content"])}
            for s in steps
        ]
        gt_kw = {"ground_truth": ground_truth} if self.with_gt else {}

        story_pred = self.story(question=question, steps=raw_steps, **gt_kw)
        plan = str(story_pred.plan)
        final_answer = str(story_pred.final_answer)
        visible_failure = str(story_pred.visible_failure)

        screened_rows: list[dict[str, Any]] = []
        for s in steps:
            sp = self.screen(
                question=question,
                plan=plan,
                final_answer=final_answer,
                visible_failure=visible_failure,
                step_idx=int(s["step_idx"]),
                agent=str(s["agent"]),
                prev_context=str(s.get("prev_context") or "(none)"),
                upstream_handoff=str(s.get("upstream_handoff") or "(none)"),
                content=str(s["content"]),
                **gt_kw,
            )
            screened_rows.append({**s, "flagged": bool(sp.flagged), "suspect_note": str(sp.suspect_note)})

        flagged = [r for r in screened_rows if r["flagged"]]
        if flagged:
            digest = "\n".join(f"[step {r['step_idx']}] {r['agent']}: {r['suspect_note']}" for r in flagged)
            candidates = flagged
        else:
            digest = "No step was explicitly flagged in Stage 2; evaluating fallback anchor steps."
            candidates = [screened_rows[0]] if screened_rows else []
            if len(screened_rows) >= 2 and screened_rows[-1] is not candidates[0]:
                candidates.append(screened_rows[-1])

        traced_candidates: list[dict[str, Any]] = []
        for c in candidates:
            tb = self.trace_back(
                question=question,
                plan=plan,
                visible_failure=visible_failure,
                flagged_digest=digest,
                step_idx=int(c["step_idx"]),
                agent=str(c["agent"]),
                prev_context=str(c.get("prev_context") or "(none)"),
                upstream_handoff=str(c.get("upstream_handoff") or "(none)"),
                content=str(c["content"]),
                suspect_note=str(c["suspect_note"]),
            )
            traced_candidates.append({
                "step_idx": int(c["step_idx"]),
                "agent": str(c["agent"]),
                "suspect_note": str(c["suspect_note"]),
                "is_root_cause": bool(tb.is_root_cause),
                "origin_step_idx": int(tb.origin_step_idx),
                "responsible_agent": str(tb.responsible_agent),
                "causal_note": str(tb.causal_note),
            })

        decided = self.decide(
            question=question,
            known_agents=known,
            plan=plan,
            final_answer=final_answer,
            visible_failure=visible_failure,
            steps=traced_candidates,
            **gt_kw,
        )
        return dspy.Prediction(
            mistake_step=int(decided.mistake_step),
            mistake_agent=str(decided.mistake_agent),
            reason=str(decided.reason),
            plan=plan,
            visible_failure=visible_failure,
        )


def traces_to_dspy_examples(traces: Sequence[TraceRecord], *, with_gt: bool = True) -> list[dspy.Example]:
    """Convert canonical `TraceRecord`s into `dspy.Example`s for `StagedAttributionModule`."""
    examples: list[dspy.Example] = []
    input_keys = ["question", "steps", "known_agents"] + (["ground_truth"] if with_gt else [])
    for tr in traces:
        step_dicts = [
            {
                "step_idx": s.step_idx,
                "agent": s.agent,
                "content": s.content,
                "prev_context": s.prev_context,
                "upstream_handoff": s.upstream_handoff,
            }
            for s in tr.steps
        ]
        ex = dspy.Example(
            trace_id=tr.trace_id,
            dataset=tr.dataset,
            question=tr.question,
            ground_truth=tr.ground_truth,
            known_agents=tr.known_agents,
            steps=step_dicts,
            mistake_step=tr.mistake_step,
            mistake_agent=tr.mistake_agent,
            mistake_reason=tr.mistake_reason,
        ).with_inputs(*input_keys)
        examples.append(ex)
    return examples


def attribution_metric(
    example: dspy.Example | Any,
    pred: dspy.Prediction | Any,
    trace: Any = None,
    pred_name: str | None = None,
    pred_trace: Any = None,
) -> float | dspy.Prediction:
    """`0.5 * step_exact + 0.5 * agent_exact` (compatible with both MIPROv2 and GEPA)."""
    dataset = str(getattr(example, "dataset", "whowhen") or "whowhen")
    norm_fn = normalize_role if dataset == "longrca" else normalize_agent_name

    gt_step = getattr(example, "mistake_step", None)
    gt_agent = norm_fn(getattr(example, "mistake_agent", ""))

    try:
        pred_step: int | None = int(getattr(pred, "mistake_step", None))
    except (TypeError, ValueError):
        pred_step = None
    pred_agent = norm_fn(getattr(pred, "mistake_agent", ""))

    step_ok = 1.0 if (gt_step is not None and pred_step == int(gt_step)) else 0.0
    agent_ok = 1.0 if (gt_agent and pred_agent == gt_agent) else 0.0
    score = 0.5 * step_ok + 0.5 * agent_ok

    # GEPA passes `pred_name` expecting either float or `dspy.Prediction(score=..., feedback=...)`.
    if pred_name is not None:
        fb = (
            f"Score={score:.2f} (step_exact={int(step_ok)}, agent_exact={int(agent_ok)}). "
            f"Expected agent={getattr(example, 'mistake_agent', '')!r} at step={gt_step}; "
            f"got agent={getattr(pred, 'mistake_agent', '')!r} at step={pred_step}. "
            f"Gold rationale: {getattr(example, 'mistake_reason', '')}"
        )
        return dspy.Prediction(score=score, feedback=fb)
    return score


def export_compiled_state(module: StagedAttributionModule, out_path: str | Path) -> Path:
    """Serialize `StagedAttributionModule` state as JSON loadable by `pipelines.staged(compiled_state=...)`."""
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "story": module.story.dump_state(),
        "screen": module.screen.dump_state(),
        "trace_back": module.trace_back.dump_state(),
        "decide": module.decide.dump_state(),
    }
    path.write_text(json.dumps(state, indent=2, default=str) + "\n", encoding="utf-8")
    return path


def compile_staged(
    trainset: Sequence[TraceRecord | dspy.Example],
    *,
    valset: Sequence[TraceRecord | dspy.Example] | None = None,
    optimizer: str = "miprov2",
    with_gt: bool = True,
    lm: Any = None,
    reflection_lm: Any = None,
    auto: str = "light",
    out_path: str | Path | None = None,
    num_threads: int = 8,
    **optimizer_kwargs: Any,
) -> tuple[StagedAttributionModule, dict[str, Any]]:
    """Compile `StagedAttributionModule` with `MIPROv2` or `GEPA` and optionally write JSON state."""
    import tracepress as tp

    resolved_lm = tp.lm.resolve_lm(lm)
    train_examples = (
        traces_to_dspy_examples(trainset, with_gt=with_gt)  # type: ignore[arg-type]
        if trainset and isinstance(trainset[0], TraceRecord)
        else list(trainset)
    )
    val_examples = (
        traces_to_dspy_examples(valset, with_gt=with_gt)  # type: ignore[arg-type]
        if valset and isinstance(valset[0], TraceRecord)
        else (list(valset) if valset is not None else None)
    )

    student = StagedAttributionModule(with_gt=with_gt)
    opt_key = optimizer.strip().lower()

    with dspy.context(lm=resolved_lm):
        if opt_key == "miprov2":
            teleprompter = dspy.MIPROv2(
                metric=attribution_metric,
                auto=auto,
                num_threads=num_threads,
                **optimizer_kwargs,
            )
            compiled = teleprompter.compile(student, trainset=train_examples, valset=val_examples)
        elif opt_key == "gepa":
            refl = tp.lm.resolve_lm(reflection_lm) if reflection_lm is not None else resolved_lm
            teleprompter = dspy.GEPA(
                metric=attribution_metric,
                auto=auto,
                reflection_lm=refl,
                num_threads=num_threads,
                **optimizer_kwargs,
            )
            compiled = teleprompter.compile(student, trainset=train_examples, valset=val_examples)
        else:
            raise ValueError(f"Unsupported optimizer {optimizer!r}; choose 'miprov2' or 'gepa'")

    state = {
        "story": compiled.story.dump_state(),
        "screen": compiled.screen.dump_state(),
        "trace_back": compiled.trace_back.dump_state(),
        "decide": compiled.decide.dump_state(),
    }
    if out_path is not None:
        export_compiled_state(compiled, out_path)
    return compiled, state
