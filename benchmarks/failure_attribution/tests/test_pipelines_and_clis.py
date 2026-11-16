"""Integration tests for all TracePress failure-attribution pipelines, DSPy tuning, shim, and CLIs."""

from __future__ import annotations

import json
from pathlib import Path

import dspy
import pytest

from benchmarks.failure_attribution import (
    StagedAttributionModule,
    all_at_once,
    attribution_metric,
    estimate_cost,
    fold,
    load_trail,
    load_whowhen,
    make_fake_preds,
    map_reduce,
    run_baselines,
    score_trail,
    score_whowhen,
    staged,
    trail_pipeline,
    whowhen_tp,
)
from benchmarks.failure_attribution.run_baselines import TracePressOpenAIShim
from benchmarks.failure_attribution.tests.conftest import FuncLM
from benchmarks.failure_attribution.tune import export_compiled_state, traces_to_dspy_examples


def _staged_fake_lm(*, zero_flags_on: str | None = None, vote_behavior: bool = False) -> FuncLM:
    """Deterministic `FuncLM` implementing responses for `story`, `screen`, `trace_back`, and `decide`."""

    def respond(fields: dict[str, str], rollout_id=None, **_):
        # Stage 1: story (`steps` input, no `known_agents` and no `is_suspect`)
        if "steps" in fields and "known_agents" not in fields and "has_error" not in user_keys(fields):
            return {
                "plan": "Inspect spreadsheet and filter Blu-Ray rows.",
                "final_answer": "No Blu-Ray entries found.",
                "visible_failure": "Wrongly dropped valid rows or crashed.",
            }
        # Stage 2: screen (`upstream_handoff` in fields and `flagged_digest` not in fields)
        if "upstream_handoff" in fields and "flagged_digest" not in fields:
            q = fields.get("question", "")
            if zero_flags_on and zero_flags_on in q:
                return {"flagged": "False", "suspect_note": "clean"}
            idx = int(fields["step_idx"].strip())
            is_bad = idx in (1, 2, 3)
            return {"flagged": str(is_bad), "suspect_note": f"suspect at step {idx}"}
        # Stage 3: trace_back (`flagged_digest` in fields)
        if "flagged_digest" in fields:
            idx = int(fields["step_idx"].strip())
            return {
                "is_root_cause": str(idx in (1, 2)),
                "origin_step_idx": str(idx),
                "responsible_agent": fields.get("agent", "Unknown"),
                "causal_note": f"traced step {idx}",
            }
        # Stage 4: decide (`known_agents` in fields)
        if "known_agents" in fields:
            q = fields.get("question", "")
            if vote_behavior:
                # Rollouts 0 and 2 agree on (2, Blu-Ray_Expert); rollout 1 disagrees with (99, Unknown)
                if rollout_id == 1:
                    return {
                        "reasoning": "outlier vote",
                        "mistake_step": "99",
                        "mistake_agent": "Unknown",
                        "reason": "outlier",
                    }
                return {
                    "reasoning": "consensus vote",
                    "mistake_step": "2",
                    "mistake_agent": "Blu-Ray_Expert",
                    "reason": "Dropped NaN Platform rows.",
                }
            if "James Beard" in q:
                # Explicitly blame upstream coder `Culinary_Awards_Expert` at step 1 (whose speaker is `Computer_terminal`)!
                return {
                    "reasoning": "Upstream coder wrote bad script run at step 1.",
                    "mistake_step": "1",
                    "mistake_agent": "Culinary_Awards_Expert",
                    "reason": "Buggy python block.",
                }
            # Test validator clamping & blank-agent speaker fallback on 91.json when rollout_id is None:
            return {
                "reasoning": "Step 2 dropped NaN rows.",
                "mistake_step": "2",
                "mistake_agent": "",  # blank -> validator resolves speaker at step 2 (`Blu-Ray_Expert`)!
                "reason": "Dropped NaN Platform rows.",
            }
        raise AssertionError(f"Unhandled signature fields: {sorted(fields)}")

    def user_keys(f: dict[str, str]) -> set[str]:
        return set(f)

    return FuncLM(respond)


def test_all_at_once_map_reduce_and_fold_pipelines(whowhen_dir: Path):
    src = load_whowhen(whowhen_dir, subset="algo")

    # 1. `all_at_once` (also test `with_gt=False` omits `ground_truth` from prompt)
    aao_lm = FuncLM(
        lambda fields, **_: {
            "reasoning": " analyzed ",
            "mistake_step": "999",  # out-of-bounds -> validator clamps to max step (3 or 2)!
            "mistake_agent": "Blu-Ray_Expert" if "Blu-Ray" in fields.get("question", "") else "Culinary_Awards_Expert",
            "reason": "found it",
        }
    )
    rows_no_gt = all_at_once(src, with_gt=False, lm=aao_lm, on_error="raise").collect()
    assert len(rows_no_gt) == 2
    assert all("ground_truth" not in c["fields"] for c in aao_lm.shared_calls)
    by_trace = {r.trace: r for r in rows_no_gt}
    assert by_trace["91.json"].mistake_step == 3  # clamped from 999 to len(steps)-1 == 3!
    assert by_trace["91.json"].mistake_agent == "Blu-Ray_Expert"

    # 2. `map_reduce` with separate `screen_lm` and main `lm`
    screen_lm = FuncLM(lambda fields, **_: {"step_summary": f"s{fields['step_idx']}", "is_suspect": "True"})
    judge_lm = FuncLM(
        lambda fields, **_: {
            "reasoning": "r",
            "mistake_step": "2" if "Blu-Ray" in fields["question"] else "1",
            "mistake_agent": "Blu-Ray_Expert" if "Blu-Ray" in fields["question"] else "Culinary_Awards_Expert",
            "reason": "mr",
        }
    )
    mr_rows = map_reduce(src, lm=judge_lm, screen_lm=screen_lm, on_error="raise").collect()
    assert len(mr_rows) == 2
    assert screen_lm.calls == sum(t.n_steps for t in src.trace_records)
    assert judge_lm.calls == 2
    assert score_whowhen(src, {r.trace: {"predicted_agent": r.mistake_agent, "predicted_step": r.mistake_step} for r in mr_rows}, n_boot=50)["both_exact"] == pytest.approx(1.0)

    # 3. `fold` sequential state updates
    fold_lm = FuncLM(
        lambda fields, **_: {
            "running_summary": "updated",
            "mistake_step": "2" if "Blu-Ray" in fields["question"] else "1",
            "mistake_agent": "Blu-Ray_Expert" if "Blu-Ray" in fields["question"] else "Culinary_Awards_Expert",
            "reason": "folded",
        }
    )
    fold_rows = fold(src, lm=fold_lm, on_error="raise").collect()
    assert len(fold_rows) == 2
    assert fold_lm.calls == sum(t.n_steps for t in src.trace_records)


def test_staged_zero_flag_fallback_upstream_blame_and_voting(whowhen_dir: Path):
    src = load_whowhen(whowhen_dir, subset="algo")

    # Test 0-flagged-steps safe fallback on `James Beard` (`14.json`) + speaker fallback on `91.json`
    lm = _staged_fake_lm(zero_flags_on="James Beard", vote_behavior=False)
    rows = staged(src, window=2, vote_k=1, lm=lm, on_error="raise").collect()
    assert len(rows) == 2
    by_id = {r.trace: r for r in rows}

    # `14.json` had 0 flagged steps yet was NOT dropped, AND preserved explicit upstream `Culinary_Awards_Expert` at step 1!
    assert by_id["14.json"].mistake_step == 1
    assert by_id["14.json"].mistake_agent == "Culinary_Awards_Expert"
    # `91.json` had blank `mistake_agent` -> validator resolved speaker at step 2 (`Blu-Ray_Expert`)!
    assert by_id["91.json"].mistake_step == 2
    assert by_id["91.json"].mistake_agent == "Blu-Ray_Expert"

    # Now test `vote_k=3`: Stage 1 `story` runs ONCE per trace, Stages 2-4 run 3x with rollout_ids {0, 1, 2}!
    v_lm = _staged_fake_lm(vote_behavior=True)
    v_rows = staged(src, vote_k=3, lm=v_lm, on_error="raise").collect()
    assert len(v_rows) == 2
    rollouts_seen = {c["rollout_id"] for c in v_lm.shared_calls if c["rollout_id"] is not None}
    assert rollouts_seen == {0, 1, 2}
    story_calls = [c for c in v_lm.shared_calls if "steps" in c["fields"] and "known_agents" not in c["fields"]]
    assert len(story_calls) == 2  # shared upstream Stage 1 ran once per trace!
    v91 = {r.trace: r for r in v_rows}["91.json"]
    assert v91.mistake_step == 2
    assert v91.mistake_agent == "Blu-Ray_Expert"
    assert v91.vote_agreement == pytest.approx(2.0 / 3.0)


def test_staged_dspy_module_metric_and_compiled_state_roundtrip(whowhen_dir: Path, tmp_path: Path):
    src = load_whowhen(whowhen_dir, subset="algo")
    examples = traces_to_dspy_examples(src.trace_records)
    assert len(examples) == 2

    lm = _staged_fake_lm()
    mod = StagedAttributionModule(with_gt=True)
    mod.decide.predict.signature = mod.decide.predict.signature.with_instructions("Custom compiled decide rubric.")
    state_path = export_compiled_state(mod, tmp_path / "compiled_staged.json")
    assert state_path.is_file()

    with dspy.context(lm=lm, adapter=dspy.ChatAdapter()):
        pred = mod(**examples[0].inputs())
    assert pred.mistake_step == examples[0].mistake_step
    assert pred.mistake_agent == examples[0].mistake_agent

    # Check both float (MIPROv2) and Prediction+feedback (GEPA) modes of `attribution_metric`.
    exact_pred = dspy.Prediction(mistake_step=examples[0].mistake_step, mistake_agent=examples[0].mistake_agent)
    half_pred = dspy.Prediction(mistake_step=99, mistake_agent=examples[0].mistake_agent)
    assert attribution_metric(examples[0], exact_pred) == pytest.approx(1.0)
    assert attribution_metric(examples[0], half_pred) == pytest.approx(0.5)
    gepa_res = attribution_metric(examples[0], half_pred, pred_name="decide")
    assert gepa_res.score == pytest.approx(0.5) and "Expected agent=" in gepa_res.feedback

    # Load `compiled_state` back into `pipelines.staged` and verify custom instruction reaches the LM prompt.
    rows = staged(src, lm=lm, compiled_state=state_path, on_error="raise").collect()
    assert len(rows) == 2
    assert any("Custom compiled decide rubric." in c["sys_msg"] for c in lm.shared_calls)


def test_trail_pipeline_end_to_end(trail_dir: Path):
    src = load_trail(trail_dir, subset="gaia")

    def trail_fn(fields: dict[str, str], **_):
        if "span_id" in fields and "candidate_category" not in fields:
            sp = fields["span_id"].strip()
            has_err = sp == "bb1b825898c2697c"
            return {
                "has_error": str(has_err),
                "candidate_category": "Instruction non complience" if has_err else "",
                "candidate_evidence": "missing <end_plan>",
                "candidate_description": "tag missing",
            }
        return {
            "reasoning": "consolidating spans",
            "errors": json.dumps([
                {
                    "location": "bb1b825898c2697c",
                    "category": "Instruction non complience",  # sanitizer normalizes typo!
                    "evidence": "missing <end_plan>",
                    "description": "tag missing",
                    "impact": "low",
                },
                {
                    "location": "nonexistent_span_id",  # sanitizer drops invalid span_id!
                    "category": "Goal Deviation",
                },
            ]),
        }

    lm = FuncLM(trail_fn)
    rows = trail_pipeline(src, lm=lm, on_error="raise").collect()
    assert len(rows) == 1
    assert rows[0].errors == [
        {
            "location": "bb1b825898c2697c",
            "category": "Instruction Non-compliance",
            "evidence": "missing <end_plan>",
            "description": "tag missing",
            "impact": "LOW",
        }
    ]
    scored = score_trail(src, {rows[0].trace: {"errors": rows[0].errors}}, n_boot=50)
    assert scored["location_recall"] == pytest.approx(0.5)


def test_openai_shim_and_cli_entrypoints(whowhen_dir: Path, tmp_path: Path):
    # 1. `TracePressOpenAIShim` + `run_baselines.run_baseline` (`all_at_once`, `step_by_step`, `binary_search`)
    shim_lm = FuncLM(
        lambda _fields, user_msg="", **_: (
            "1. Yes.\n2. Reason: bad code"
            if "Respond ONLY in the format: 1. Yes/No" in user_msg
            else (
                "lower half"
                if "upper half" in user_msg
                else "Agent Name: Blu-Ray_Expert\nStep Number: 2\nReason for Mistake: dropped NaNs"
            )
        )
    )
    shim = TracePressOpenAIShim(shim_lm)
    for method in ("all_at_once", "step_by_step", "binary_search"):
        out_log = tmp_path / f"base_{method}.txt"
        run_baselines.run_baseline(
            method=method,
            data_dir=whowhen_dir,
            subset="algo",
            out_path=out_log,
            client=shim,
            model="shim",
            limit=1,
            no_gt=True,
        )
        parsed = whowhen_tp.score_whowhen(load_whowhen(whowhen_dir, subset="algo", limit=1), out_log, n_boot=50)
        assert parsed["n_predictions"] == 1

    # 2. `make_fake_preds` + `whowhen_tp score` (with paired diff + `--trivial`) + `estimate_cost`
    oracle_txt = tmp_path / "oracle.txt"
    rand_txt = tmp_path / "rand.txt"
    score_json = tmp_path / "scores.json"
    cost_json = tmp_path / "costs.json"

    assert make_fake_preds.main(["--dataset", "whowhen", "--data", str(whowhen_dir), "--subset", "algo", "--mode", "oracle", "--out", str(oracle_txt)]) == 0
    assert make_fake_preds.main(["--dataset", "whowhen", "--data", str(whowhen_dir), "--subset", "algo", "--mode", "random", "--out", str(rand_txt)]) == 0
    assert whowhen_tp.main([
        "score",
        "--dataset", "whowhen",
        "--data", str(whowhen_dir),
        "--subset", "algo",
        "--trivial",
        "--n-boot", "100",
        "--json-out", str(score_json),
        str(oracle_txt),
        str(rand_txt),
    ]) == 0
    metrics = json.loads(score_json.read_text(encoding="utf-8"))
    assert metrics[str(oracle_txt)]["both_exact"] == pytest.approx(1.0)
    assert "paired_diffs" in metrics and "trivial_baselines" in metrics

    assert estimate_cost.main(["--whowhen-dir", str(whowhen_dir), "--json-out", str(cost_json)]) == 0
    costs = json.loads(cost_json.read_text(encoding="utf-8"))
    assert costs["whowhen_algo"]["methods"]["tp staged (k=1)"]["max_seq_depth"] == 4


def test_vote_tie_breaking_and_whowhen_tp_run_subcommand(whowhen_dir: Path, tmp_path: Path):
    import tracepress as tp

    src = load_whowhen(whowhen_dir, subset="hand", limit=1)  # `1.json` (steps 0, 1, 2)

    # 3 votes with 3 distinct `(step, agent)` pairs, where step 1 appears TWICE (with two agents) vs step 2 ONCE:
    # tie-breaker picks step majority (step 1), and earliest among tied pairs!
    def tie_fn(fields: dict[str, str], rollout_id=None, **_):
        if "steps" in fields and "known_agents" not in fields:
            return {"plan": "p", "final_answer": "a", "visible_failure": "f"}
        if "upstream_handoff" in fields and "flagged_digest" not in fields:
            return {"flagged": "True", "suspect_note": "n"}
        if "flagged_digest" in fields:
            return {"is_root_cause": "True", "origin_step_idx": "1", "responsible_agent": "Orchestrator", "causal_note": "c"}
        if rollout_id == 0:
            return {"reasoning": "r0", "mistake_step": "2", "mistake_agent": "Assistant", "reason": "r0"}
        if rollout_id == 1:
            return {"reasoning": "r1", "mistake_step": "1", "mistake_agent": "human", "reason": "r1"}
        return {"reasoning": "r2", "mistake_step": "1", "mistake_agent": "Orchestrator", "reason": "r2"}

    tie_lm = FuncLM(tie_fn)
    rows = staged(src, vote_k=3, lm=tie_lm, on_error="raise").collect()
    assert rows[0].mistake_step == 1  # step-majority tie-break beat step 2!

    # Also test `whowhen_tp.py run` end-to-end when LM is pre-configured on `tp.configure(lm=...)`.
    tp.configure(lm=_staged_fake_lm())
    out_run = tmp_path / "tp_run_preds.txt"
    rc = whowhen_tp.main([
        "run",
        "--dataset", "whowhen",
        "--data", str(whowhen_dir),
        "--subset", "algo",
        "--method", "staged",
        "--out", str(out_run),
        "--n-boot", "50",
    ])
    assert rc == 0 and out_run.is_file()
    assert score_whowhen(load_whowhen(whowhen_dir, subset="algo"), out_run, n_boot=50)["both_exact"] == pytest.approx(1.0)

