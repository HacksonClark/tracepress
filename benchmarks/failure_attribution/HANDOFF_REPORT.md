# Failure-Attribution Benchmark & Research Suite — Technical Handoff Report

## 1. Architecture & TracePress 0.1.3+ (`0d5f405`) Verification

All components live strictly under `benchmarks/failure_attribution/` and build directly on TracePress `0.1.3` (`0d5f405`):

1. **Key-Shadowing Guard (`step_idx` vs `step`)**:
   - In TracePress `plan.py`, `Table.map` rejects any output column whose name matches a key level (`ValueError: map: column(s) ['step'] would shadow key level(s) of the same name`), and `lm_map` / `lm_reduce` treat an input named `step` on a `("trace", "step")` table as a whole-row record dict.
   - Every dataset loader therefore stores the integer step index in `step_idx` (`0`-based on Who&When, LongRCA, and TRAIL; `1`-based on AgentRx) and span identifiers in `span_id`, while prediction outputs emit `mistake_step: int` and `mistake_agent: str`.
2. **Typed `FailureTraceAdapter(tp.Adapter)` + `Table.join` Constant Lifting**:
   - `FailureTraceAdapter` exposes two closed-schema tables:
     - `src.traces` keyed by `("trace",)` with `grain={"question": 1, "ground_truth": 1, "dataset": 1, "subset": 1, "n_steps": 1, "known_agents": 1}`.
     - `src.steps` keyed by `("trace", "step")` with columns `step_idx`, `agent`, `raw_agent`, `content`, `prev_context`, `upstream_handoff`, and `span_id`.
   - Calling `src.steps.join(src.traces).group_by("trace")` automatically lifts `question`, `ground_truth`, and `known_agents` as scalar group constants (`self._constants`) while keeping output table keys at `("trace",)`. Consequently, trace-level intermediate tables (such as Stage 1 `story`) join directly back onto `steps` (`steps.join(story)`) with full build-time column checking.
   - Every `.map(...)` and `.reduce(...)` in `pipelines.py` passes explicit `columns={...}` so schemas never degrade to `open=True` (which would otherwise clear `grain` metadata across joins).
3. **4-Stage Causal Attribution (`pipelines.staged`) & Majority Vote (`vote_k`)**:
   - **Stage 1 (`story`)**: `lm_reduce` over raw steps reconstructs `(plan, final_answer, visible_failure)` once per trace.
   - **Stage 2 (`screen`)**: `lm_map` screens every step in parallel with `(story, prev_context, upstream_handoff)` into `(flagged: bool, suspect_note)` (routing to `screen_lm` if supplied, e.g. a fast classifier or cheaper model).
   - **Stage 3 (`trace_back`)**: A per-trace Python `reduce` builds `flagged_digest` across all flagged steps and attaches safe fallback anchor steps whenever `n_flagged == 0` so zero traces are ever dropped. `lm_map` then tests whether each candidate step is an upstream root cause or a symptom of an earlier coder/orchestrator handoff (`is_root_cause, origin_step_idx, responsible_agent, causal_note`).
   - **Stage 4 (`decide` + validator)**: `lm_reduce` (`dspy.ChainOfThought`) selects `(mistake_step, mistake_agent, reason)`, followed by a deterministic Python validator that clamps out-of-range steps onto valid `step_idx` values and resolves `mistake_agent` (preserving explicitly blamed trace-known upstream agents such as `Culinary_Awards_Expert` or `WebSurfer` even when the step speaker is `Computer_terminal` or `Orchestrator (-> WebSurfer)`, while falling back to the step speaker when blank/unknown).
   - **Stage 5 (`vote_k > 1`)**: Shares the single Stage 1 `story` DAG node across `vote_k` branches of Stages 2–4, cloning the LM with `lm.copy(rollout_id=v, temperature=0.7)` for cache-distinct rollout diversity, and aggregates via majority vote on `(mistake_step, normalized_agent)` with tie-breaking by step majority then earliest `mistake_step`.

---

## 2. Dataset & Scorer Edge Cases Verified Against Real Data

All four loaders and scorers were verified against the complete real datasets in `/tmp/fa-research`:

1. **Who&When (`Hand-Crafted` $N=58$, `Algorithm-Generated` $N=126$)**:
   - **Hyphenated agent bug in upstream `evaluate.py`**: Upstream regex `r"Agent Name:\s*([\w_]+)"` truncates `Blu-Ray_Expert` (`Algorithm-Generated/91.json`) to `Blu`. `whowhen.py` parses the full agent line and preserves hyphens.
   - **Magentic-One parenthetical suffixes**: `Hand-Crafted` steps carry `role` values like `Orchestrator (thought)` and `Orchestrator (-> WebSurfer)`; `canonical_agent_name` strips `\s*\(.*\)\s*$`.
   - **Speaker $\neq$ `mistake_agent` mismatches**: Verified on all 6 real mismatch traces (`Hand-Crafted` `20.json`, `22.json`, `49.json`; `Algorithm-Generated` `14.json`, `15.json`, `59.json`). Because of these 6 traces, oracle predictions score `100.00%` on `agent`, `step`, and `both_exact`, while `agent_from_step` (speaker of `mistake_step`) is `94.83%` (`55/58`) on `Hand-Crafted` and `97.62%` (`123/126`) on `Algorithm-Generated`.
   - **Pre-PR-#15 substring inflation**: Reported alongside exact match via `paper_agent` and `paper_step` for comparison with published pre-fix tables.

2. **TRAIL ($N=148$: 117 GAIA, 31 SWE Bench)**:
   - **Malformed annotation JSON**: `processed_annotations_gaia/a96c6811716c0473b86a23321db79c34.json` contains an illegal trailing comma at line 38 (`}, ]`). `strip_trailing_commas` strips trailing commas outside string literals without corrupting valid string contents.
   - **Human annotation category typos**: Across all 148 real files, 31 distinct raw category strings appear (including `"Instruction non complience"`, `"Goal deviation"`, `"Context Handling Failure"`, `"Formatting Error"`, `"Tool Selection"`, `"Task Orchestration Error"`, `"Task Orchestration Errors"`, and `" Incorrect Problem Identification"`). Note that upstream `calculate_scores.py`'s substring fallback (`category_no_spaces in std_cat`) actually fails on `"Task Orchestration Errors"` because it is *longer* than `"Task Orchestration"`; `normalize_trail_category` maps all 31 raw strings cleanly onto the 21 official categories, yielding `weighted_f1 = 100.00%` for the oracle.
   - **4 error-free traces ceiling**: 4 of the 148 TRAIL traces have `errors: []`. Because official `calculate_scores.py` defines per-trace location accuracy as `0` when `gt_locations` is empty, the official `location_recall` ceiling over all 148 traces is `144 / 148 = 97.30%` (`location_recall_nonempty = 100.00%`). Both metrics are returned by `score_trail`.

3. **LongRCA-Mini ($N=200$, 40 per benchmark across 5 benchmarks)**:
   - **Role normalization**: Implements verbatim `normalize_role` (`re.sub(r"\s*\((?:->|→)\s*[^()]+\)\s*$", "", text)` + NFKC casefold), resolving handoff labels like `DiagnostAgent (-> ActionAgent)` (`swe_bench_pro__057`, `terminal_bench_2__005`) while preserving non-arrow parentheticals.
   - **Independent role scoring & source-weighted MAE**: Per `longrca-bench.github.io/scripts/export_paper_results.py`, `agent_acc` is scored strictly from `predicted_agent` (never derived from `predicted_step`'s speaker). `root_mae` weights per-benchmark MAEs by the full-benchmark slice sizes (`swe_bench_pro: 128`, `terminal_bench_2: 42`, `travelplanner: 685`, `vitabench: 108`, `webarena_verified: 177`). On `LongRCA-Mini`:
     - Oracle: `agent_acc = 100.00%`, `step_acc = 100.00%`, `root_mae = 0.00`.
     - Uniform random (`seed=0`): `agent_acc = 24.00%`, `step_acc = 1.50%`, `step_pm5 = 13.00%`, `root_mae = 49.01` (`micro_mae = 54.52`).

4. **AgentRx ($N=73$ matched trajectories across `tau` and `magentic`)**:
   - Converts raw `tau_dataset_failed.json` (29 labeled) and `magentic_dataset/*.json` (44 labeled), as well as pre-normalized Trajectory IR and `flash` logs, into 1-indexed `StepRecord`s (`step_idx >= 1`).
   - Resolves `root_cause.failure_id` into `failures[*]` and normalizes all 10 `FailureCase` categories. Oracle achieves `100.00%` `step_acc`, `100.00%` `category_acc`, and `0.00` `step_mae` / `norm_step_dist` across all 73 trajectories.

---

## 3. Empirical Cost & Scalability Insights

Tokenized with `tiktoken` (`o200k_base`) across all real traces:

- **Quadratic blowup of `step_by_step` on long horizons**: On `LongRCA-Mini` ($N=200$, mean 161.6 steps, max 661 steps), full `step_by_step` consumes **1,645.6 Mtok** ($166.01 at $0.10/Mtok, or **>$4,100** at $2.50/Mtok) with sequential depth **661**; even oracle-early-stopped `step_by_step` consumes **265.1 Mtok** with max sequential depth **518**.
- **Bounded-depth parallel scaling in TracePress**:
  - `tp map_reduce` compresses prompt volume by **41x** vs full `step_by_step` (`39.8 Mtok`) at sequential depth **2**.
  - `tp staged (k=1)` performs full story reconstruction, context-aware screening with upstream handoffs, causal trace-back on flagged candidates, and Chain-of-Thought root-cause adjudication in `100.6 Mtok` ($11.25 on GPT-6-Luna) with mean sequential depth **5.2** (max **8** even on 393k-token traces).
  - Mixing a fast classifier (`--screen-lm typesafe:jev-latest` or a lightweight flash model) for Stage 2 screening with a frontier reasoning model for Stages 1, 3, and 4 cuts `staged` cost by an additional 60–75% because ~78% of `staged` calls occur in Stage 2.

---

## 4. Recommended Ablation & Optimization Matrix

1. **Method ladder** (`--no-gt` and default `with_gt` reported separately — never mix!):
   - Baselines via `run_baselines.py`: `all_at_once`, `binary_search`, `step_by_step`.
   - TracePress pipelines via `whowhen_tp.py run`: `all_at_once` -> `map_reduce` -> `fold` -> `staged (vote_k=1)` -> `staged (vote_k=3)`.
2. **Causal component ablations on `staged`**:
   - Window size `--window 0` vs `--window 3` vs `--window 5`.
   - Heterogeneous `--screen-lm` vs homogeneous `--lm`.
   - `MIPROv2` vs `GEPA` prompt compilation via `tune.compile_staged` (fitted on `Who&When Algorithm-Generated` or a stratified dev split, evaluated zero-shot on `Hand-Crafted` and `LongRCA-Mini`).
3. **Statistical reporting**:
   - Always pass multiple prediction files to `whowhen_tp.py score ... pred_a.txt pred_b.txt` to report paired bootstrap 95% CIs (`n_boot=2000, seed=0`) alongside exact `agent`, `step`, `both_exact`, and `step_pm5`.
