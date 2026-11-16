"""Hermetic fixtures and fake LMs for the failure-attribution benchmark test suite."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

WORKSPACE_ROOT = str(Path(__file__).resolve().parents[3])
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

import dspy
import litellm
import pytest
import tracepress as tp
from tracepress import lm as lm_config

FIELD_RE = re.compile(r"\[\[ ## (\w+) ## \]\]\n(.*?)(?=\n\n(?:\[\[ ## |Respond with)|\Z)", re.S)


class FuncLM(dspy.BaseLM):
    """Deterministic fake LM for `dspy.ChatAdapter()` that computes outputs from `(fields, rollout_id, prompt)`.

    Tracks calls and rollout IDs even across `.copy(rollout_id=v, temperature=0.7)` clones via a shared log list.
    """

    def __init__(self, fn):
        super().__init__(model="func-fa")
        self.fn = fn
        self.shared_calls: list[dict[str, Any]] = []

    def copy(self, **kwargs):
        clone = FuncLM(self.fn)
        clone.kwargs = {**self.kwargs, **kwargs}
        clone.shared_calls = self.shared_calls
        return clone

    @property
    def calls(self) -> int:
        return len(self.shared_calls)

    def forward(self, prompt=None, messages=None, **kwargs):
        user_msg = messages[-1]["content"] if messages else (prompt or "")
        sys_msg = messages[0]["content"] if messages and len(messages) > 1 else ""
        fields = dict(FIELD_RE.findall(user_msg))
        merged_kw = {**self.kwargs, **kwargs}
        rollout_id = merged_kw.get("rollout_id")
        self.shared_calls.append({
            "fields": fields,
            "rollout_id": rollout_id,
            "temperature": merged_kw.get("temperature"),
            "user_msg": user_msg,
            "sys_msg": sys_msg,
        })
        out = self.fn(fields, rollout_id=rollout_id, user_msg=user_msg, sys_msg=sys_msg)
        if isinstance(out, str):
            content = out
        else:
            content = "".join(f"[[ ## {k} ## ]]\n{v}\n\n" for k, v in out.items()) + "[[ ## completed ## ]]"
        return litellm.ModelResponse(
            choices=[{"message": {"role": "assistant", "content": content}}],
            model="func-fa",
            usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        )

    async def aforward(self, prompt=None, messages=None, **kwargs):
        return self.forward(prompt=prompt, messages=messages, **kwargs)


@pytest.fixture(autouse=True)
def isolated_tp_config(monkeypatch):
    """Disable telemetry, use `ChatAdapter` for `FuncLM`, and isolate DSPy/TracePress state."""
    prev_tel = tp.configure().telemetry
    prev_adp = tp.configure().adapter
    prev_lm = tp.configure().lm
    monkeypatch.setattr(lm_config, "_lm_cache", {})
    tp.configure(telemetry=False, adapter=dspy.ChatAdapter(), lm=None)
    yield
    tp.configure(telemetry=prev_tel, adapter=prev_adp, lm=prev_lm)


@pytest.fixture
def whowhen_dir(tmp_path: Path) -> Path:
    """Synthetic Who&When root covering `Hand-Crafted` (`20.json` handoff mismatch + `(thought)`)
    and `Algorithm-Generated` (`91.json` `Blu-Ray_Expert` + `Computer_terminal` execution error).
    """
    hand = tmp_path / "Who&When" / "Hand-Crafted"
    algo = tmp_path / "Who&When" / "Algorithm-Generated"
    hand.mkdir(parents=True)
    algo.mkdir(parents=True)

    # Hand-Crafted 20.json: speaker at step 3 is `Orchestrator (-> WebSurfer)` while `mistake_agent` is `WebSurfer`!
    (hand / "20.json").write_text(
        json.dumps({
            "question": "Difference in seconds between two FRB X-ray diagrams?",
            "ground_truth": "0.2",
            "mistake_agent": "WebSurfer",
            "mistake_step": "3",
            "mistake_reason": "WebSurfer failed to extract the PDF diagram time span.",
            "history": [
                {"role": "human", "content": "Find difference in seconds."},
                {"role": "Orchestrator (thought)", "content": "Initial plan: ask WebSurfer."},
                {"role": "Orchestrator (thought)", "content": '{"next_speaker": {"answer": "WebSurfer"}}'},
                {"role": "Orchestrator (-> WebSurfer)", "content": "Search arxiv 2103.07786 PDF diagram."},
                {"role": "WebSurfer", "content": "Opened abstract only, skipped PDF."},
            ],
        }),
        encoding="utf-8",
    )
    # Hand-Crafted 1.json: `Orchestrator (thought)` speaker normalized to `Orchestrator`.
    (hand / "1.json").write_text(
        json.dumps({
            "question": "Compute 6 * 7.",
            "ground_truth": "42",
            "mistake_agent": "Orchestrator",
            "mistake_step": "1",
            "mistake_reason": "Orchestrator gave wrong formula.",
            "history": [
                {"role": "human", "content": "Compute 6 * 7."},
                {"role": "Orchestrator (thought)", "content": "Plan: compute 6 + 7 = 13."},
                {"role": "Assistant", "content": "FINAL ANSWER: 13"},
            ],
        }),
        encoding="utf-8",
    )

    # Algorithm-Generated 91.json: hyphenated `Blu-Ray_Expert` + `Computer_terminal`.
    (algo / "91.json").write_text(
        json.dumps({
            "question": "Oldest Blu-Ray title in spreadsheet?",
            "ground_truth": "Time-Parking 2: Parallel Universe",
            "mistake_agent": "Blu-Ray_Expert",
            "mistake_step": "2",
            "mistake_reason": "Dropped all NaN Platform rows.",
            "history": [
                {"name": "Data_Analysis_Expert", "role": "assistant", "content": "Let us inspect spreadsheet."},
                {"name": "Computer_terminal", "role": "user", "content": "exitcode: 1 KeyError: Platform"},
                {"name": "Blu-Ray_Expert", "role": "user", "content": "df = df.dropna(subset=['Platform'])"},
                {"name": "Computer_terminal", "role": "user", "content": "No Blu-Ray entries found."},
            ],
        }),
        encoding="utf-8",
    )
    # Algorithm-Generated 14.json: `mistake_step=1` is `Computer_terminal`, `mistake_agent` is `Culinary_Awards_Expert`!
    (algo / "14.json").write_text(
        json.dumps({
            "question": "James Beard winner 2019?",
            "ground_truth": "Kwame Onwuachi",
            "mistake_agent": "Culinary_Awards_Expert",
            "mistake_step": "1",
            "mistake_reason": "Wrote buggy scraper executed in terminal.",
            "history": [
                {"name": "Culinary_Awards_Expert", "role": "assistant", "content": "```python\nraise ValueError('bad')\n```"},
                {"name": "Computer_terminal", "role": "user", "content": "exitcode: 1 ValueError: bad"},
                {"name": "Verification_Expert", "role": "assistant", "content": "TERMINATE"},
            ],
        }),
        encoding="utf-8",
    )
    return tmp_path / "Who&When"


@pytest.fixture
def trail_dir(tmp_path: Path) -> Path:
    """Synthetic TRAIL benchmark directory with nested `child_spans`, trailing comma JSON, and category typos."""
    bench = tmp_path / "trail-benchmark" / "benchmarking"
    gaia_traces = bench / "data" / "GAIA"
    gaia_anns = bench / "processed_annotations_gaia"
    gaia_traces.mkdir(parents=True)
    gaia_anns.mkdir(parents=True)

    tid = "a96c6811716c0473b86a23321db79c34"
    trace_payload = {
        "trace_id": tid,
        "spans": [
            {
                "timestamp": "2025-03-19T16:46:37.000000Z",
                "span_id": "root_span_01",
                "parent_span_id": None,
                "span_name": "main",
                "service_name": "gaia-app",
                "status_code": "Unset",
                "span_attributes": {"llm.model": "o3-mini"},
                "logs": [{"severity_text": "INFO", "body": {"question": "What percentage of penguins?"}}],
                "child_spans": [
                    {
                        "timestamp": "2025-03-19T16:46:38.000000Z",
                        "span_id": "bb1b825898c2697c",
                        "parent_span_id": "root_span_01",
                        "span_name": "plan_step",
                        "service_name": "gaia-app",
                        "status_code": "Ok",
                        "span_attributes": {"gen_ai.completion": "Plan missing end tag, comma, [test]"},
                        "logs": [],
                        "child_spans": [],
                    },
                    {
                        "timestamp": "2025-03-19T16:46:39.000000Z",
                        "span_id": "c46c0dbcedd707cc",
                        "parent_span_id": "root_span_01",
                        "span_name": "solve_step",
                        "service_name": "gaia-app",
                        "status_code": "Error",
                        "status_message": "Simulated data used",
                        "span_attributes": {},
                        "logs": [{"severity_text": "ERROR", "body": "Assumed 20,000,000"}],
                        "child_spans": [],
                    },
                ],
            }
        ],
    }
    (gaia_traces / f"{tid}.json").write_text(json.dumps(trace_payload), encoding="utf-8")

    # Intentional trailing comma before `]` AND human category typos (`Instruction non complience`, `Goal deviation`)!
    raw_ann_with_trailing_comma = """{
        "trace_id": "a96c6811716c0473b86a23321db79c34",
        "errors": [
            {
                "category": "Instruction non complience",
                "location": "bb1b825898c2697c",
                "evidence": "Missing <end_plan>",
                "description": "Did not close plan tag",
                "impact": "LOW"
            },
            {
                "category": "Goal deviation",
                "location": "c46c0dbcedd707cc",
                "evidence": "Simulated data",
                "description": "Used fake population",
                "impact": "HIGH"
            },
        ],
        "scores": [{"overall": 2.25}]
    }"""
    (gaia_anns / f"{tid}.json").write_text(raw_ann_with_trailing_comma, encoding="utf-8")
    return tmp_path / "trail-benchmark"
