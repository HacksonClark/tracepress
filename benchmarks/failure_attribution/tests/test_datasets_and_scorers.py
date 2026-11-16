"""Unit tests for all 4 dataset loaders, schema invariants, edge cases, and official scorers."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import tracepress as tp

from benchmarks.failure_attribution.datasets import (
    FailureTraceAdapter,
    StepRecord,
    TRAIL_CATEGORIES,
    TraceRecord,
    attach_context,
    load_agentrx,
    load_longrca,
    load_trail,
    load_whowhen,
    normalize_role,
    paired_diff_whowhen,
    parse_whowhen_predictions,
    score_agentrx,
    score_longrca,
    score_trail,
    score_whowhen,
    trivial_baselines,
    write_whowhen_predictions,
)
from benchmarks.failure_attribution.datasets.agentrx import (
    extract_failure_case_number,
    normalize_agentrx_category,
)
from benchmarks.failure_attribution.datasets.trail import (
    loads_lenient_json,
    normalize_trail_category,
)


# --- 1. FailureTraceAdapter & TracePress 0.1.3+ Schema Invariants ------------------------


def test_adapter_schema_join_constants_and_key_shadow_guard(whowhen_dir: Path, monkeypatch):
    src = load_whowhen(whowhen_dir, subset="algo")
    assert len(src) == 2
    assert set(src.tables()) == {"traces", "steps"}

    # Naming a column `step` on a `("trace", "step")` table must fail at build time.
    with pytest.raises(ValueError, match="shadow"):
        src.steps.map(lambda r: {"step": 0}, columns={"step": int})

    # Joining `src.traces` onto `src.steps` gives `question` & `known_agents` grain=1,
    # so `.group_by("trace")` keeps output levels `("trace",)` and lifts `question` as a scalar!
    joined = src.steps.join(src.traces)
    assert joined.schema.grain["question"] == 1
    assert joined.schema.grain["known_agents"] == 1

    grouped = joined.group_by("trace").reduce(
        lambda rs: {"count": len(rs)},
        columns={"count": int},
    )
    assert grouped.schema.levels == ("trace",)
    rows = grouped.collect()
    assert len(rows) == 2
    assert all("question" in r.values and r.count >= 3 for r in rows)


def test_attach_context_detects_arrow_handoff_and_terminal_predecessor():
    steps = [
        StepRecord(0, "human", "Solve task", raw_agent="human"),
        StepRecord(1, "Orchestrator", "Ask WebSurfer to open PDF", raw_agent="Orchestrator (-> WebSurfer)"),
        StepRecord(2, "WebSurfer", "Clicked abstract only", raw_agent="WebSurfer"),
        StepRecord(3, "Data_Expert", "```python\n1/0\n```", raw_agent="Data_Expert"),
        StepRecord(4, "Computer_terminal", "ZeroDivisionError", raw_agent="Computer_terminal"),
    ]
    attach_context(steps, window=2)

    assert steps[0].prev_context == "(none — first step)"
    assert "Orchestrator hands off to WebSurfer" in steps[1].upstream_handoff
    assert "Orchestrator hands off to WebSurfer" in steps[2].upstream_handoff
    assert "Executing code/tool from [step 3] Data_Expert" in steps[4].upstream_handoff
    assert "[step 2]" in steps[4].prev_context and "[step 3]" in steps[4].prev_context
    assert "[step 0]" not in steps[4].prev_context  # window=2


# --- 2. Who&When Loader, Parser, Scorer, Edge Cases & Trivial Baselines ------------------


def test_whowhen_hyphenated_agent_and_paper_substring_inflation(whowhen_dir: Path, tmp_path: Path):
    algo_src = load_whowhen(whowhen_dir, subset="algo")
    t91 = algo_src.by_id["91.json"]
    assert t91.mistake_agent == "Blu-Ray_Expert"
    assert t91.steps[2].agent == "Blu-Ray_Expert"

    # Roundtrip through upstream text block writer & parser preserves hyphen in `Blu-Ray_Expert`.
    pred_file = tmp_path / "preds.txt"
    write_whowhen_predictions(
        [
            {"trace_id": "91.json", "predicted_agent": "Blu-Ray_Expert", "predicted_step": 2, "reason": "ok"},
            # Substring trap: `Culinary_Awards_Expert_10` & step `10` contain `Culinary_Awards_Expert` & `1`!
            {"trace_id": "14.json", "predicted_agent": "Culinary_Awards_Expert_10", "predicted_step": 10, "reason": "sub"},
        ],
        pred_file,
    )
    parsed = parse_whowhen_predictions(pred_file)
    assert parsed["91.json"]["predicted_agent"] == "Blu-Ray_Expert"
    assert parsed["91.json"]["predicted_step"] == 2

    res = score_whowhen(algo_src, pred_file, n_boot=200, seed=0)
    # Exact: only 91.json is right (1/2 = 0.5). Paper substring: both match (2/2 = 1.0)!
    assert res["agent"] == pytest.approx(0.5)
    assert res["step"] == pytest.approx(0.5)
    assert res["both_exact"] == pytest.approx(0.5)
    assert res["paper_agent"] == pytest.approx(1.0)
    assert res["paper_step"] == pytest.approx(1.0)


def test_whowhen_handcrafted_parenthetical_and_20json_handoff_mismatch(whowhen_dir: Path):
    hand_src = load_whowhen(whowhen_dir, subset="hand")
    t1 = hand_src.by_id["1.json"]
    t20 = hand_src.by_id["20.json"]
    assert t1.steps[1].raw_agent == "Orchestrator (thought)"
    assert t1.steps[1].agent == "Orchestrator"
    assert t20.steps[3].raw_agent == "Orchestrator (-> WebSurfer)"
    assert t20.steps[3].agent == "Orchestrator"
    assert t20.mistake_agent == "WebSurfer" and t20.mistake_step == 3

    # Predicting `(WebSurfer, step 3)` on 20.json gets `both_exact=1.0`, while `agent_from_step` is 0
    # because step 3's speaker is `Orchestrator`, not `WebSurfer`!
    preds_a = {
        "1.json": {"predicted_agent": "Orchestrator (thought)", "predicted_step": 1},
        "20.json": {"predicted_agent": "WebSurfer", "predicted_step": 3},
    }
    preds_b = {
        "1.json": {"predicted_agent": "Assistant", "predicted_step": 2},
        "20.json": {"predicted_agent": "Orchestrator", "predicted_step": 3},
    }
    res_a = score_whowhen(hand_src, preds_a, n_boot=200, seed=0)
    assert res_a["agent"] == pytest.approx(1.0)
    assert res_a["step"] == pytest.approx(1.0)
    assert res_a["both_exact"] == pytest.approx(1.0)
    assert res_a["agent_from_step"] == pytest.approx(0.5)

    diffs = paired_diff_whowhen(hand_src, preds_a, preds_b, n_boot=200, seed=0)
    assert diffs["both_exact"]["diff"] == pytest.approx(1.0)

    triv = trivial_baselines(hand_src, n_boot=100, seed=0)
    assert set(triv) == {
        "always WebSurfer",
        "always Verification_Expert",
        "always step 8",
        "always step 1",
        "uniform_step(seed=0)",
    }
    assert triv["always WebSurfer"]["agent"] == pytest.approx(0.5)
    assert triv["always step 1"]["step"] == pytest.approx(0.5)


# --- 3. TRAIL Loader, Trailing Comma Fix, 21-Category Typos, and Official Scorer ---------


def test_trail_loader_typos_trailing_comma_and_scorer(trail_dir: Path, tmp_path: Path):
    # Verify all documented human annotation typos map to canonical 21 TRAIL categories.
    assert len(TRAIL_CATEGORIES) == 21
    assert normalize_trail_category("Instruction non complience") == "Instruction Non-compliance"
    assert normalize_trail_category("Goal deviation") == "Goal Deviation"
    assert normalize_trail_category("Context Handling Failure") == "Context Handling Failures"
    assert normalize_trail_category("Task Orchestration Errors") == "Task Orchestration"
    assert normalize_trail_category("Formatting Error") == "Formatting Errors"
    assert normalize_trail_category("Tool Selection") == "Tool Selection Errors"
    assert normalize_trail_category(" Incorrect Problem Identification") == "Incorrect Problem Identification"

    # Verify trailing comma stripper does NOT corrupt `, ]` inside string literals.
    obj = loads_lenient_json('{"note": "a, ] b", "items": [1, 2, ], }')
    assert obj == {"note": "a, ] b", "items": [1, 2]}

    src = load_trail(trail_dir, subset="gaia")
    assert len(src) == 1
    tr = src.trace_records[0]
    assert [s.span_id for s in tr.steps] == ["root_span_01", "bb1b825898c2697c", "c46c0dbcedd707cc"]
    assert "penguins" in tr.question
    assert [(e["location"], e["category"]) for e in tr.trail_errors] == [
        ("bb1b825898c2697c", "Instruction Non-compliance"),
        ("c46c0dbcedd707cc", "Goal Deviation"),
    ]

    # Score partial prediction (1 of 2 errors matched -> recall=0.5, weighted_f1=0.5).
    partial_preds = {
        tr.trace_id: {
            "errors": [{"location": "bb1b825898c2697c", "category": "Instruction Non-Compliance"}]
        }
    }
    scored = score_trail(src, partial_preds, n_boot=100, seed=0)
    assert scored["location_recall"] == pytest.approx(0.5)
    assert scored["joint_location_category_recall"] == pytest.approx(0.5)
    assert scored["weighted_f1"] == pytest.approx(0.5)

    # Also verify JSON array prediction files (`[{"trace_id": ..., "errors": [...]}, ...]`).
    arr_pred_file = tmp_path / "trail_preds_array.json"
    arr_pred_file.write_text(
        json.dumps([{"trace_id": tr.trace_id, "errors": tr.trail_errors}]),
        encoding="utf-8",
    )
    assert score_trail(src, arr_pred_file, n_boot=50)["weighted_f1"] == pytest.approx(1.0)

    # Also verify HF JSONL format (`trace` and `labels` as JSON strings).
    hf_jsonl = tmp_path / "trail_hf.jsonl"
    hf_jsonl.write_text(
        json.dumps({
            "trace_id": "hf_01",
            "split": "swe_bench",
            "trace": json.dumps({"trace_id": "hf_01", "spans": [{"span_id": "sp1", "span_name": "tool"}]}),
            "labels": '{"errors": [{"location": "sp1", "category": "Timeout Issue",}], }',
        })
        + "\n",
        encoding="utf-8",
    )
    hf_src = load_trail(hf_jsonl, subset="swe_bench")
    assert len(hf_src) == 1
    assert hf_src.trace_records[0].trail_errors[0]["category"] == "Timeout Issues"


# --- 4. LongRCA Loader, `normalize_role`, Independent Role Acc, & Source-Weighted MAE ----


def test_longrca_normalize_role_and_source_weighted_mae(tmp_path: Path):
    # `normalize_role` strips `(-> Target)` / `(→ Target)` while keeping non-arrow parentheticals.
    assert normalize_role("DiagnostAgent (-> ActionAgent)") == "diagnostagent"
    assert normalize_role("Orchestrator (→ WebSurfer)") == "orchestrator"
    assert normalize_role("Orchestrator (thought)") == "orchestrator (thought)"

    rows = [
        {
            "question_ID": "travelplanner__001",
            "history": [
                {"step": 0, "role": "user", "name": "Computer_terminal", "content": "Plan trip to Seattle."},
                {"step": 1, "role": "assistant", "name": "verifier_agent", "content": "Bad night split"},
                {"step": 2, "role": "user", "name": "Computer_terminal", "content": "Done"},
            ],
            "mistake_agent": "verifier_agent",
            "mistake_step": 1,
            "mistake_reason": "Wrong night split",
        },
        {
            "question_ID": "terminal_bench_2__005",
            "history": [
                {"step": 0, "role": "user", "name": "human", "content": "Fix CLI tool."},
                {"step": 1, "role": "assistant", "name": "DiagnostAgent (-> ActionAgent)", "content": "Bad spec"},
                {"step": 2, "role": "user", "name": "Computer_terminal", "content": "exit 1"},
            ],
            "mistake_agent": "DiagnostAgent",
            "mistake_step": 1,
            "mistake_reason": "Bad handoff",
        },
    ]
    df = pd.DataFrame(rows)
    src = load_longrca(df, subset="mini")
    assert len(src) == 2

    # TravelPlanner err = |1 - 1| = 0; Terminal-Bench err = |11 - 1| = 10.
    # Also on terminal_bench_2__005, step 1 speaker is `DiagnostAgent`, but we predict wrong agent `ActionAgent`
    # AND on travelplanner__001 we predict `verifier_agent (-> writer)` (arrow stripped -> matches!).
    preds = {
        "travelplanner__001": {"predicted_agent": "verifier_agent (-> writer)", "predicted_step": 1},
        "terminal_bench_2__005": {"predicted_agent": "ActionAgent", "predicted_step": 11},
    }
    res = score_longrca(src, preds, n_boot=100, seed=0)
    assert res["agent_acc"] == pytest.approx(0.5)  # strictly explicit predicted_agent, NOT step speaker!
    assert res["step_acc"] == pytest.approx(0.5)
    assert res["step_pm5"] == pytest.approx(0.5)
    assert res["both_exact"] == pytest.approx(0.5)
    assert res["micro_mae"] == pytest.approx(5.0)
    # Source-weighted MAE: weights are travelplanner=685, terminal_bench_2=42 -> (685*0 + 42*10)/(685+42)
    expected_weighted = (685.0 * 0.0 + 42.0 * 10.0) / (685.0 + 42.0)
    assert res["root_mae"] == pytest.approx(expected_weighted)


# --- 5. AgentRx 1-Indexed Loader, Root-Cause Lookup, & 10-Category Scorer ----------------


def test_agentrx_one_indexed_root_cause_lookup_and_metrics(tmp_path: Path):
    assert normalize_agentrx_category("FailureCase.INVENTION_OF_NEW_INFORMATION") == "Invention of New Information"
    assert normalize_agentrx_category("Instruction/Plan Adherence Failure") == "Instruction Adherence Failure"
    assert extract_failure_case_number("Misinterpretation of Tool Output") == 4

    ir_file = tmp_path / "tau_ir.json"
    gt_file = tmp_path / "tau_gt.json"
    ir_file.write_text(
        json.dumps([
            {
                "trajectory_id": "2",
                "instruction": "Count available t-shirts.",
                "steps": [
                    {"index": 1, "substeps": [{"sub_index": 1, "role": "user", "content": "How many t-shirts?"}]},
                    {"index": 2, "substeps": [{"sub_index": 1, "role": "assistant", "content": "Skipped auth"}]},
                    {"index": 3, "substeps": [{"sub_index": 1, "role": "tool", "content": '{"items": [1,2,3]}'}]},
                    {"index": 4, "substeps": [{"sub_index": 1, "role": "assistant", "content": "There are 2."}]},
                ],
            }
        ]),
        encoding="utf-8",
    )
    # Notice `failures` has failure_id=1 at step 2 AND failure_id=2 at step 4, with `root_cause.failure_id = 2`!
    gt_file.write_text(
        json.dumps([
            {
                "trajectory_id": 2,
                "failures": [
                    {
                        "failure_id": 1,
                        "step_number": 2,
                        "failure_category": "Instruction Adherence Failure",
                        "failed_agent": "Assistant",
                    },
                    {
                        "failure_id": 2,
                        "step_number": 4,
                        "failure_category": "Misinterpretation of Tool Output",
                        "failed_agent": "Assistant",
                    },
                ],
                "root_cause": {"failure_id": 2, "reason_for_root_cause": "Wrong t-shirt count"},
            }
        ]),
        encoding="utf-8",
    )

    src = load_agentrx(ir_file, ground_truth_path=gt_file)
    assert len(src) == 1
    tr = src.trace_records[0]
    assert [s.step_idx for s in tr.steps] == [1, 2, 3, 4]
    assert tr.mistake_step == 4
    assert tr.failure_category == "Misinterpretation of Tool Output"

    # Predict step 3 (off by 1 from step 4; n_steps=4 -> norm_step_dist = 1/4 = 0.25) and root category.
    preds = {"2": {"predicted_agent": "Assistant", "predicted_step": 3, "failure_category": "4"}}
    res = score_agentrx(src, preds, n_boot=100, seed=0)
    assert res["step_acc"] == pytest.approx(0.0)
    assert res["step_pm1"] == pytest.approx(1.0)
    assert res["step_mae"] == pytest.approx(1.0)
    assert res["norm_step_dist"] == pytest.approx(0.25)
    assert res["category_acc"] == pytest.approx(1.0)
    assert res["any_category_acc"] == pytest.approx(1.0)
    assert res["earliest_category_acc"] == pytest.approx(0.0)
    assert res["terminal_category_acc"] == pytest.approx(1.0)


def test_agentrx_raw_magentic_and_flash_formats_and_longrca_json_dir(tmp_path: Path):
    # 1. Raw Magentic flat message array + Flash `(Step-N)` event grouping
    mag_file = tmp_path / "mag_01.json"
    mag_file.write_text(
        json.dumps([
            {"role": "human", "content": "Find board member."},
            {"role": "Orchestrator (-> WebSurfer)", "content": "Search board."},
            {"role": "WebSurfer", "content": "Results."},
        ]),
        encoding="utf-8",
    )
    mag_src = load_agentrx(mag_file, subset="magentic")
    assert [s.step_idx for s in mag_src.trace_records[0].steps] == [1, 2, 3]
    assert mag_src.trace_records[0].steps[1].agent == "Orchestrator"

    flash_file = tmp_path / "flash_01.json"
    flash_file.write_text(
        json.dumps({
            "trajectory_id": "f1",
            "events": [
                {"type": "LLMCallEvent", "source": "ignored", "message": "skip me"},
                {"source": "Orchestrator (thought)", "message": "Initial plan: check DB"},
                {"source": "Executor (Step-2)", "message": "Step-2 query failed"},
            ],
        }),
        encoding="utf-8",
    )
    flash_src = load_agentrx(flash_file, subset="flash")
    assert [s.step_idx for s in flash_src.trace_records[0].steps] == [1, 2]

    # 2. LongRCA nested directory layout `data/mini/<bench>/<qid>.json`
    lrca_dir = tmp_path / "longrca_repo" / "data" / "mini" / "vitabench"
    lrca_dir.mkdir(parents=True)
    (lrca_dir / "vitabench__005.json").write_text(
        json.dumps({
            "question_ID": "vitabench__005",
            "history": [
                {"step": 0, "role": "assistant", "name": "Action_Expert", "content": "Hello, how can I help?"},
                {"step": 1, "role": "user", "name": "Computer_terminal", "content": "Order dinner."},
            ],
            "mistake_agent": "Action_Expert",
            "mistake_step": 0,
        }),
        encoding="utf-8",
    )
    lrca_src = load_longrca(tmp_path / "longrca_repo", subset="mini")
    assert len(lrca_src) == 1
    assert lrca_src.trace_records[0].trace_id == "vitabench__005"


def test_download_safe_extract_tarball_blocks_path_traversal(tmp_path: Path):
    import io
    import tarfile
    from benchmarks.failure_attribution.datasets.download import _safe_extract_tarball

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        good = tarfile.TarInfo("repo-main/sub/ok.json")
        payload = b'{"ok": true}'
        good.size = len(payload)
        tf.addfile(good, io.BytesIO(payload))

        evil = tarfile.TarInfo("repo-main/../../escape.txt")
        evil_payload = b"nope"
        evil.size = len(evil_payload)
        tf.addfile(evil, io.BytesIO(evil_payload))

    dest = tmp_path / "extracted"
    _safe_extract_tarball(buf.getvalue(), dest)
    assert (dest / "sub" / "ok.json").is_file()
    assert not (tmp_path / "escape.txt").exists()

