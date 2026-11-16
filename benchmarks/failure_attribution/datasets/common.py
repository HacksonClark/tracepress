"""Canonical trace/step records, context builders, bootstrap CIs, and TracePress Adapter."""

from __future__ import annotations

import math
import random
import re
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field
from typing import Any

import tracepress as tp
from tracepress.rows import Closed, Event, Row, RowAdded
from tracepress.schema import Schema

EXECUTOR_ROLES = frozenset({
    "computer_terminal",
    "computerterminal",
    "code_executor",
    "executor",
    "terminal",
    "tool",
    "bash",
    "system",
})

HANDOFF_ARROW_RE = re.compile(r"^\s*(.+?)\s*\(\s*(?:->|→)\s*([^()]+?)\s*\)\s*$")
ORCH_TARGET_RE = re.compile(r"Orchestration target:\s*([A-Za-z0-9_.-]+)", re.IGNORECASE)
NEXT_SPEAKER_RE = re.compile(r'"next_speaker"\s*:\s*\{[^{}]*"answer"\s*:\s*"([^"]+)"', re.DOTALL)


def clip_text(text: str, limit: int) -> str:
    """Clip text to at most `limit` chars while preserving head and tail context."""
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    head = limit * 2 // 3
    tail = max(limit - head - 24, 40)
    omitted = len(text) - head - tail
    return f"{text[:head]} ... [{omitted} chars] ... {text[-tail:]}"


def _norm_key(name: str) -> str:
    base = re.sub(r"\s*\(.*\)\s*$", "", (name or "").strip())
    return re.sub(r"[\s_-]+", "_", base.lower()).strip("_")


def is_executor_agent(name: str) -> bool:
    key = _norm_key(name)
    return key in EXECUTOR_ROLES


@dataclass
class StepRecord:
    """One normalized step or span in a failure-attribution trajectory."""

    step_idx: int
    agent: str
    content: str
    raw_agent: str = ""
    span_id: str = ""
    prev_context: str = ""
    upstream_handoff: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.raw_agent:
            self.raw_agent = self.agent


@dataclass
class TraceRecord:
    """One trajectory with optional ground-truth failure attribution metadata."""

    trace_id: str
    dataset: str
    subset: str = ""
    question: str = ""
    ground_truth: str = ""
    steps: list[StepRecord] = field(default_factory=list)
    mistake_step: int | None = None
    mistake_agent: str = ""
    mistake_reason: str = ""
    failure_category: str = ""
    trail_errors: list[dict[str, Any]] = field(default_factory=list)
    trail_scores: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def n_steps(self) -> int:
        return len(self.steps)

    @property
    def known_agents(self) -> list[str]:
        seen: dict[str, str] = {}
        for s in self.steps:
            a = (s.agent or "").strip()
            if a and a.lower() not in seen:
                seen[a.lower()] = a
        return list(seen.values())

    @property
    def step_by_idx(self) -> dict[int, StepRecord]:
        return {s.step_idx: s for s in self.steps}

    def speaker_at(self, step_idx: int | None) -> str:
        if step_idx is None:
            return ""
        rec = self.step_by_idx.get(step_idx)
        return rec.agent if rec is not None else ""


def _detect_handoff_target(step: StepRecord) -> tuple[str, str] | None:
    """Return `(sender, target)` if `step` is an explicit handoff/delegation turn."""
    m = HANDOFF_ARROW_RE.match(step.raw_agent or "")
    if m:
        return m.group(1).strip(), m.group(2).strip()
    m_orch = ORCH_TARGET_RE.search(step.content or "")
    if m_orch:
        return step.agent, m_orch.group(1).strip()
    m_next = NEXT_SPEAKER_RE.search(step.content or "")
    if m_next:
        return step.agent, m_next.group(1).strip()
    if (step.content or "").strip().lower().startswith("next speaker"):
        parts = step.content.replace(":", " ").replace("[", " ").replace("]", " ").replace('"', " ").split()
        if len(parts) >= 3:
            return step.agent, parts[2].strip()
    return None


def attach_context(
    steps: list[StepRecord],
    window: int = 3,
    *,
    max_chars_per_step: int = 360,
    handoff_chars: int = 560,
) -> list[StepRecord]:
    """Populate `prev_context` and `upstream_handoff` on each `StepRecord` in `steps`.

    - `prev_context`: sliding window of up to `window` immediately preceding steps.
    - `upstream_handoff`: captures either:
      1. The most recent explicit `Sender (-> Target)` / orchestrator delegation directed at
         the current speaker (or emitted on this turn itself); or
      2. For executor steps (`Computer_terminal`, `tool`, etc.), the immediately preceding
         non-executor planner/coder/expert step whose code or tool invocation is being run.
    """
    latest_handoff_for: dict[str, str] = {}
    last_any_handoff: str = ""
    last_non_executor: StepRecord | None = None

    for i, cur in enumerate(steps):
        if window > 0 and i > 0:
            start = max(0, i - window)
            ctx_lines = [
                f"[step {s.step_idx}] {s.raw_agent or s.agent}: {clip_text(s.content, max_chars_per_step)}"
                for s in steps[start:i]
            ]
            cur.prev_context = "\n".join(ctx_lines)
        else:
            cur.prev_context = "(none — first step)"

        self_handoff = _detect_handoff_target(cur)
        cur_key = _norm_key(cur.agent)

        handoff_note = ""
        if self_handoff is not None:
            sender, target = self_handoff
            handoff_note = (
                f"[step {cur.step_idx}] {sender} hands off to {target}: "
                f"{clip_text(cur.content, handoff_chars)}"
            )
            latest_handoff_for[_norm_key(target)] = handoff_note
            last_any_handoff = handoff_note
        elif is_executor_agent(cur.agent) and last_non_executor is not None:
            handoff_note = (
                f"Executing code/tool from [step {last_non_executor.step_idx}] "
                f"{last_non_executor.agent}: {clip_text(last_non_executor.content, handoff_chars)}"
            )
        elif cur_key in latest_handoff_for:
            handoff_note = latest_handoff_for[cur_key]
        elif i > 0 and _detect_handoff_target(steps[i - 1]) is not None:
            handoff_note = last_any_handoff

        cur.upstream_handoff = handoff_note or "(none)"

        if not is_executor_agent(cur.agent) and _norm_key(cur.agent) not in ("human", "user"):
            last_non_executor = cur

    return steps


class FailureTraceAdapter(tp.Adapter):
    """TracePress Adapter exposing `src.traces` and `src.steps` with explicit schemas.

    Trace-level columns on `traces` carry `grain=1`, so `src.steps.join(src.traces).group_by("trace")`
    automatically treats `question`, `ground_truth`, `n_steps`, and `known_agents` as scalar
    group constants in `lm_reduce` and `lm_fold` while keeping output keys at `("trace",)`.
    """

    levels = ("trace", "step")
    bounded = True

    def __init__(self, traces: Sequence[TraceRecord]):
        self.trace_records: list[TraceRecord] = list(traces)
        self.by_id: dict[str, TraceRecord] = {t.trace_id: t for t in self.trace_records}

    def __len__(self) -> int:
        return len(self.trace_records)

    def tables(self) -> dict[str, Schema]:
        trace_cols = {
            "question": str,
            "ground_truth": str,
            "dataset": str,
            "subset": str,
            "n_steps": int,
            "known_agents": list,
        }
        step_cols = {
            "step_idx": int,
            "agent": str,
            "raw_agent": str,
            "content": str,
            "prev_context": str,
            "upstream_handoff": str,
            "span_id": str,
        }
        return {
            "traces": Schema(
                levels=("trace",),
                columns=trace_cols,
                grain={k: 1 for k in trace_cols},
            ),
            "steps": Schema(
                levels=("trace", "step"),
                columns=step_cols,
                record=("step_idx", "agent", "raw_agent", "content", "span_id"),
            ),
        }

    async def events(self, table: str) -> AsyncIterator[Event]:
        if table == "traces":
            for tr in self.trace_records:
                yield RowAdded(
                    Row(
                        (("trace", tr.trace_id),),
                        {
                            "question": tr.question,
                            "ground_truth": tr.ground_truth,
                            "dataset": tr.dataset,
                            "subset": tr.subset,
                            "n_steps": tr.n_steps,
                            "known_agents": tr.known_agents,
                        },
                    )
                )
            return

        if table == "steps":
            for tr in self.trace_records:
                tkey = (("trace", tr.trace_id),)
                for st in tr.steps:
                    yield RowAdded(
                        Row(
                            (*tkey, ("step", st.step_idx)),
                            {
                                "step_idx": st.step_idx,
                                "agent": st.agent,
                                "raw_agent": st.raw_agent,
                                "content": st.content,
                                "prev_context": st.prev_context,
                                "upstream_handoff": st.upstream_handoff,
                                "span_id": st.span_id,
                            },
                        )
                    )
                yield Closed(tkey)
            return

        raise KeyError(f"Unknown table {table!r}")


# --- Bootstrap Confidence Intervals ------------------------------------------------------


def _percentile(sorted_vals: list[float], q: float) -> float:
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    pos = q * (len(sorted_vals) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return sorted_vals[lo]
    frac = pos - lo
    return sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac


def bootstrap_ci(
    values: Sequence[float],
    *,
    n_boot: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """Return `(mean, ci_low, ci_high)` via non-parametric percentile bootstrap."""
    n = len(values)
    if n == 0:
        return 0.0, 0.0, 0.0
    obs_mean = sum(values) / n
    if n_boot <= 0 or n == 1:
        return obs_mean, obs_mean, obs_mean

    rng = random.Random(seed)
    vals = list(values)
    means = []
    for _ in range(n_boot):
        total = 0.0
        for _ in range(n):
            total += vals[rng.randrange(n)]
        means.append(total / n)
    means.sort()
    return obs_mean, _percentile(means, alpha / 2.0), _percentile(means, 1.0 - alpha / 2.0)


def paired_bootstrap_diff(
    values_a: Sequence[float],
    values_b: Sequence[float],
    *,
    n_boot: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """Return `(mean_diff, ci_low, ci_high)` for paired per-item differences `a_i - b_i`."""
    if len(values_a) != len(values_b):
        raise ValueError(f"Paired arrays must have equal length, got {len(values_a)} vs {len(values_b)}")
    diffs = [float(a) - float(b) for a, b in zip(values_a, values_b)]
    return bootstrap_ci(diffs, n_boot=n_boot, seed=seed, alpha=alpha)
