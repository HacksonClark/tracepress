# TracePress Failure-Attribution Benchmark & Research Suite

A unified benchmark harness, official scorer suite, and TracePress multi-stage attribution pipeline collection covering four public agent failure-attribution benchmarks:

1. **Who&When** (`ag2ai/Agents_Failure_Attribution`, ICML 2025) — `Hand-Crafted` ($N=58$) and `Algorithm-Generated` ($N=126$), 0-indexed root-cause `(agent, step)` localization.
2. **LongRCA-Bench** (`CLoud5-real/longrca-bench`, arXiv:2608.15242) — Long-horizon trajectories (`LongRCA-Mini` $N=200$, `LongRCA-Full` $N=1{,}140$; up to 661 steps / 393k tokens per trace) across 5 benchmarks (`swe_bench_pro`, `terminal_bench_2`, `travelplanner`, `vitabench`, `webarena_verified`).
3. **AgentRx** (`microsoft/AgentRx`, arXiv:2602.02475) — 1-indexed Trajectory IR & raw traces (`tau`, `magentic`, `flash`) with 10-category `FailureCase` root-cause classification.
4. **TRAIL** (`PatronusAI/trail-benchmark`) — OpenTelemetry span trees (`GAIA` + `SWE Bench`, $N=148$) with multi-label `span_id` localization and 21-category error classification.

---

## Directory Layout

- `datasets/`
  - `common.py` — Canonical `TraceRecord` / `StepRecord`, sliding-window + handoff builder `attach_context`, bootstrap 95% CIs, and `FailureTraceAdapter(tp.Adapter)` (`src.traces` & `src.steps`).
  - `whowhen.py` — Who&When loader, upstream-compatible `Prediction for X.json:` writer/parser, exact + paper-substring scorer, paired bootstrap diffs, and `trivial_baselines`.
  - `longrca.py` — LongRCA parquet/JSON/DataFrame loader, official `normalize_role`, independent `agent_acc`, `step_acc`, `step_pm5`, and source-weighted `root_mae`.
  - `agentrx.py` — AgentRx 1-indexed Trajectory IR / raw domain loader, `root_cause.failure_id` resolver, 10-category `FailureCase` normalizer, and official scorer.
  - `trail.py` — TRAIL recursive `spans`/`child_spans` flattener, lenient trailing-comma JSON parser, 21-category typo normalizer, and pure-Python official scorer.
  - `download.py` — Pure-Python downloader fetching datasets into `/tmp/fa-datasets` (never inside the git repository).
- `pipelines.py` — TracePress pipelines: `all_at_once`, `map_reduce`, `fold`, `staged` (4-stage causal attribution + `vote_k` majority vote), and `trail_pipeline`.
- `tune.py` — `StagedAttributionModule(dspy.Module)`, `attribution_metric`, and `compile_staged` (`MIPROv2` / `GEPA`) exporting JSON state for `pipelines.staged(..., compiled_state=...)`.
- `whowhen_tp.py` — Unified `run` and `score` CLI across all 4 benchmarks.
- `run_baselines.py` — Runs upstream `Automated_FA/Lib/utils.py` (`all_at_once`, `step_by_step`, `binary_search`) via OpenAI-compatible HTTP endpoints OR `TracePressOpenAIShim` (`--tp-lm`).
- `estimate_cost.py` — Empirical LM call, prompt Mtok, USD cost, and sequential-depth estimator.
- `make_fake_preds.py` — Generates `oracle`, `random`, and constant baseline prediction files.
- `tests/` — Hermetic unit & integration test suite (`uv run pytest benchmarks/failure_attribution/tests`).

---

## Quickstart Commands

### 1. Download Public Datasets (Outside Repo)

```bash
uv run python benchmarks/failure_attribution/datasets/download.py \
  --dataset all --dest /tmp/fa-datasets
```

*(Reading `.parquet` files such as `longrca-mini.parquet` uses `pyarrow`; invoke with `uv run --with pyarrow ...` if `pyarrow` is not installed in the base venv.)*

### 2. Run TracePress Pipelines (`whowhen_tp.py run`)

With an API gateway (`OPENROUTER_API_KEY`, `PORTKEY_API_KEY`, or `TYPESAFE_API_KEY` for classification screening):

```bash
# 4-stage causal attribution on Who&When Algorithm-Generated
OPENROUTER_API_KEY=... uv run python benchmarks/failure_attribution/whowhen_tp.py run \
  --dataset whowhen \
  --data /tmp/fa-datasets/Agents_Failure_Attribution/Who\&When \
  --subset algo \
  --method staged \
  --lm openrouter:openai/gpt-6-luna \
  --vote-k 3 \
  --out /tmp/fa-outputs/whowhen_algo_staged_v3.txt

# Withhold ground-truth final answers (--no-gt) on LongRCA-Mini
OPENROUTER_API_KEY=... uv run --with pyarrow python benchmarks/failure_attribution/whowhen_tp.py run \
  --dataset longrca \
  --data /tmp/fa-datasets/longrca/longrca-mini.parquet \
  --subset mini \
  --method staged \
  --no-gt \
  --lm openrouter:openai/gpt-6-luna \
  --out /tmp/fa-outputs/longrca_mini_staged.json

# Multi-label span localization & 21-category classification on TRAIL
OPENROUTER_API_KEY=... uv run python benchmarks/failure_attribution/whowhen_tp.py run \
  --dataset trail \
  --data /tmp/fa-datasets/trail-benchmark \
  --subset gaia \
  --method trail_pipeline \
  --lm openrouter:openai/gpt-6-luna \
  --out /tmp/fa-outputs/trail_gaia_preds.json
```

With subscription / local / custom LMs (`claude-code:sonnet`, `codex`, `cursor`, `ollama_chat/...`, or `TRACEPRESS_LM`):

```bash
uv run python benchmarks/failure_attribution/whowhen_tp.py run \
  --dataset whowhen \
  --data /tmp/fa-datasets/Agents_Failure_Attribution/Who\&When \
  --subset hand \
  --method staged \
  --lm claude-code:sonnet \
  --limit 10 \
  --out /tmp/fa-outputs/whowhen_hand_sonnet.txt
```

### 3. Run Upstream Baselines (`run_baselines.py`)

```bash
# Via TracePressOpenAIShim (works with subscription LMs and any TracePress LM spec!)
uv run python benchmarks/failure_attribution/run_baselines.py \
  --data /tmp/fa-datasets/Agents_Failure_Attribution/Who\&When \
  --subset algo \
  --method binary_search \
  --tp-lm openrouter:openai/gpt-6-luna \
  --limit 20 \
  --out /tmp/fa-outputs/algo_binsearch.txt

# Or directly via an OpenAI-compatible endpoint
OPENROUTER_API_KEY=... uv run python benchmarks/failure_attribution/run_baselines.py \
  --data /tmp/fa-datasets/Agents_Failure_Attribution/Who\&When \
  --subset hand \
  --method step_by_step \
  --base-url https://openrouter.ai/api/v1 \
  --api-key-env OPENROUTER_API_KEY \
  --model openai/gpt-6-luna \
  --out /tmp/fa-outputs/hand_sbs.txt
```

### 4. Score Predictions, Trivial Baselines & Paired Bootstrap Diffs

```bash
uv run python benchmarks/failure_attribution/whowhen_tp.py score \
  --dataset whowhen \
  --data /tmp/fa-datasets/Agents_Failure_Attribution/Who\&When \
  --subset hand \
  --trivial \
  /tmp/fa-outputs/whowhen_hand_sonnet.txt \
  /tmp/fa-outputs/hand_sbs.txt
```

### 5. Compile Prompts with DSPy (`MIPROv2` / `GEPA`)

```python
import tracepress as tp
from benchmarks.failure_attribution import compile_staged, load_whowhen, staged

tp.configure(lm="openrouter:openai/gpt-6-luna", telemetry=False)
train = load_whowhen("/tmp/fa-datasets/Agents_Failure_Attribution/Who&When", subset="algo", limit=30)
_, _ = compile_staged(train.trace_records, optimizer="miprov2", out_path="/tmp/fa-outputs/staged_mipro.json")

val = load_whowhen("/tmp/fa-datasets/Agents_Failure_Attribution/Who&When", subset="hand")
rows = staged(val, compiled_state="/tmp/fa-outputs/staged_mipro.json", vote_k=3).collect()
```

---

## Dataset License & Gating Audit

| Dataset | Repo / Source | Scale | Step Indexing | License | Gated? | Notes |
|---|---|---:|---|---|---|---|
| **Who&When** | `ag2ai/Agents_Failure_Attribution` | 58 Hand + 126 Algo | 0-indexed | MIT | No | Fixes upstream `[\w_]+` hyphen truncation (`Blu-Ray_Expert` in `91.json`) & pre-PR-#15 substring inflation. |
| **TRAIL** | `PatronusAI/trail-benchmark` / HF `patronus-ai/trail-benchmark` | 148 (117 GAIA, 31 SWE) | OpenTelemetry `span_id` | MIT (GitHub) / CC-BY-4.0 (HF) | No | Fixes trailing comma in `a96c6811716c0473b86a23321db79c34.json` & human category typos (`Instruction non complience`, `Task Orchestration Errors`). |
| **LongRCA-Bench** | `CLoud5-real/longrca-bench` | 200 Mini / 1,140 Full | 0-indexed | Public HF (upstream benchmark terms) | No | Independent `normalize_role` role accuracy + source-weighted `root_mae` across 5 benchmarks. |
| **AgentRx** | `microsoft/AgentRx` | 73+ labeled (`tau`, `magentic`, `flash`) | **1-indexed** | MIT | No | Resolves `root_cause.failure_id` into `failures[*]`; 10-category `FailureCase` taxonomy. |
| **Who&When Pro** | `Leoxx/whowhen_pro` | 12,326 traces | 0/1-indexed by modality | CC-BY-4.0 (HF) | Partially / Large multimodal | Warm-start injected errors across text, image, GUI, and video. |
| **TraceElephant** | `TraceElephant/TraceElephant` | 220 annotated | Framework-specific | CC-BY-4.0 | Partially (HF data separate from code repo) | GitHub repo ships execution harnesses (`Captain-Agent`, `SWE-Agent`) + `trace_locate` evaluators. |

---

## Baseline Code Audit

- **Who&When (`Automated_FA/Lib/utils.py`)**: Ships `all_at_once`, `step_by_step` (early-stopping on first `"1. yes"`), and recursive `binary_search`, coupled to `AzureOpenAI` and stdout redirection. Wrapped cleanly in `run_baselines.py` via `TracePressOpenAIShim` + temp-dir `--limit` / `--no-gt` support.
- **AgentDebugX (`AgentDebugX/AgentDebugX`, MIT)**: Ships `HeuristicAttributor`, `AllAtOnceAttributor`, `BinarySearchAttributor`, `DeepDebug`, and MoA/MoE ensembles over its own `AgentTrajectory` schema, plus `CorrectedAction` substitution reruns.
- **AgentRx (`microsoft/AgentRx`, MIT)**: 5-stage domain-invariant synthesis pipeline (`IR -> Static Invariants -> Dynamic Step-by-Step Invariants -> Checker -> LLM Judge`), defaulting to Azure AD / MSR internal `TRAPI` endpoints; requires $O(T)$ dynamic invariant synthesis calls per trace.
- **RCTA / LongRCA (`longrca-bench.github.io`)**: Public repository provides leaderboard validation (`validate_data.py`), official `normalize_role`, and TSV export (`export_paper_results.py`), but **does not release the RCTA inference code** (only reference scores on `DeepSeek-V4-Flash`). Our `pipelines.staged` implements open-source 4-stage causal root-cause attribution (`story -> screen -> trace_back -> decide`) directly in TracePress with constant sequential depth.

---

## Empirical Cost & Sequential Depth (`o200k_base`, `$0.10 / $0.50` per Mtok)

### Who&When Hand-Crafted ($N=58$, 2,993 steps, mean/max steps = 51.6 / 130)

| Method | LM Calls | Calls/Trace | Prompt Mtok | Output Mtok | Est. USD | Mean Seq Depth | Max Seq Depth |
|---|---:|---:|---:|---:|---:|---:|---:|
| `all_at_once` | 80 | 1.4 | 0.997 | 0.014 | $0.107 | 1.2 | 3 |
| `step_by_step (early_stop)` | 937 | 16.2 | 6.667 | 0.084 | $0.709 | 16.2 | 83 |
| `step_by_step (full)` | 2,993 | 51.6 | 42.296 | 0.269 | $4.364 | 51.6 | 130 |
| `binary_search` | 302 | 5.2 | 1.983 | 0.006 | $0.201 | 5.2 | 7 |
| `tp map_reduce` | 3,051 | 52.6 | 2.803 | 0.191 | $0.376 | 2.0 | 2 |
| `tp fold` | 2,993 | 51.6 | 2.325 | 0.419 | $0.442 | 51.6 | 130 |
| `tp staged (k=1)` | 3,874 | 66.8 | 6.103 | 0.228 | $0.724 | 4.2 | 6 |

### Who&When Algorithm-Generated ($N=126$, 1,099 steps, mean/max steps = 8.7 / 10)

| Method | LM Calls | Calls/Trace | Prompt Mtok | Output Mtok | Est. USD | Mean Seq Depth | Max Seq Depth |
|---|---:|---:|---:|---:|---:|---:|---:|
| `all_at_once` | 128 | 1.0 | 0.403 | 0.023 | $0.052 | 1.0 | 2 |
| `step_by_step (early_stop)` | 504 | 4.0 | 0.832 | 0.045 | $0.106 | 4.0 | 10 |
| `step_by_step (full)` | 1,099 | 8.7 | 2.386 | 0.099 | $0.288 | 8.7 | 10 |
| `binary_search` | 415 | 3.3 | 0.845 | 0.008 | $0.089 | 3.3 | 4 |
| `tp map_reduce` | 1,225 | 9.7 | 1.062 | 0.091 | $0.152 | 2.0 | 2 |
| `tp fold` | 1,099 | 8.7 | 0.889 | 0.154 | $0.166 | 8.7 | 10 |
| `tp staged (k=1)` | 1,600 | 12.7 | 2.021 | 0.126 | $0.265 | 4.0 | 5 |

### LongRCA-Mini ($N=200$, 32,317 steps, mean/median/max steps = 161.6 / 123 / 661, max trace = 393,570 tok)

| Method | LM Calls | Calls/Trace | Prompt Mtok | Output Mtok | Est. USD | Mean Seq Depth | Max Seq Depth |
|---|---:|---:|---:|---:|---:|---:|---:|
| `all_at_once` | 820 | 4.1 | 12.193 | 0.148 | $1.293 | 2.2 | 5 |
| `step_by_step (early_stop)` | 11,561 | 57.8 | 265.103 | 1.040 | $27.031 | 57.8 | 518 |
| `step_by_step (full)` | 32,317 | 161.6 | 1,645.556 | 2.909 | $166.010 | 161.6 | 661 |
| `binary_search` | 1,375 | 6.9 | 25.311 | 0.028 | $2.545 | 6.9 | 9 |
| `tp map_reduce` | 32,517 | 162.6 | 39.841 | 1.979 | $4.974 | 2.0 | 2 |
| `tp fold` | 32,317 | 161.6 | 32.602 | 4.524 | $5.522 | 161.6 | 661 |
| `tp staged (k=1)` | 41,420 | 207.1 | 100.648 | 2.366 | $11.248 | 5.2 | 8 |

---

## Trivial Baselines & Oracle Ceiling Verification

### Who&When Zero-LM Baselines

| Subset | Baseline | Agent Exact | Step Exact | Both Exact | Step $\pm 1$ | Step $\pm 5$ |
|---|---|---:|---:|---:|---:|---:|
| **Hand-Crafted** ($N=58$) | `always WebSurfer` | **56.90%** | 0.00% | 0.00% | 0.00% | 0.00% |
| **Hand-Crafted** ($N=58$) | `always Verification_Expert` | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| **Hand-Crafted** ($N=58$) | `always step 8` | 46.55% | **15.52%** | **15.52%** | 20.69% | **60.34%** |
| **Hand-Crafted** ($N=58$) | `always step 1` | 31.03% | 1.72% | 1.72% | 1.72% | 27.59% |
| **Hand-Crafted** ($N=58$) | `uniform_step(seed=0)` | 41.38% | 1.72% | 1.72% | 12.07% | 37.93% |
| **Algorithm-Generated** ($N=126$) | `always WebSurfer` | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| **Algorithm-Generated** ($N=126$) | `always Verification_Expert` | 14.29% | 0.00% | 0.00% | 0.00% | 0.00% |
| **Algorithm-Generated** ($N=126$) | `always step 8` | 19.05% | 9.52% | 9.52% | 13.49% | 48.41% |
| **Algorithm-Generated** ($N=126$) | `always step 1` | **42.86%** | **26.98%** | **26.19%** | **51.59%** | 86.51% |
| **Algorithm-Generated** ($N=126$) | `uniform_step(seed=0)` | 31.75% | 13.49% | 12.70% | 38.10% | **89.68%** |

---

## Running Tests

```bash
uv run pytest benchmarks/failure_attribution/tests -v
```
