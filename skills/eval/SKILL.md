---
name: eval
description: Quickly run prompt, code, or agent evaluations using the Claude Code, Codex, or Cursor subscription you already pay for — zero API keys needed. Invoke with /eval (or $eval in Codex) to run a JSONL/CSV suite, synthesize seed eval cases, compare A/B variants, or grade outputs with an LLM judge.
disable-model-invocation: true
argument-hint: "[target-or-question] [--cases suite.jsonl] [--compare prompt_b.txt]"
---

# Subscription Eval (`/eval` / `$eval`)

Run reproducible evaluations using **the AI coding subscription you are already coding from** (Claude Code, OpenAI Codex, or Cursor) — zero external API keys, zero pip installs (pure Python 3.10+ stdlib), SQLite-cached so reruns only evaluate what changed, with bootstrap 95% confidence intervals and paired A/B comparisons.

Parse any arguments after `/eval` or `$eval` in the user's message (`$ARGUMENTS`). Locate this skill's scripts relative to this `SKILL.md` file (`scripts/sub_eval.py`).

## Step 1 — Check execution mode (`doctor`)

Run the zero-dependency doctor check:

```bash
python3 <skill_dir>/scripts/sub_eval.py doctor --json
```

Pick the execution path:
1. **CLI Subscription Mode (`recommended_mode == "cli"`)**: `sub_eval.py run` invokes your logged-in `claude -p`, `codex exec`, or Cursor `agent -p --mode=ask` concurrently (with tools disabled and environment sanitized so nested calls never collide with your active session or use API-key billing).
2. **In-Session Subagent Mode (`recommended_mode == "subagent"`)**: When inside a sandbox without local CLI binaries (or whenever native subagents are preferred), run `sub_eval.py prepare` to shard prompts/rubrics into JSON batches, spawn parallel `Task` / `Agent` subagents from your current coding session to answer/grade each shard, and run `sub_eval.py ingest` to cache, score, and render the report!

## Step 2 — Match what the user asked for

### Workflow A: Run an existing eval suite
When the user points `/eval` at a `.jsonl`, `.json`, or `.csv` file (optionally with `--prompt` or `--cmd`):

```bash
# Prompt / model eval:
python3 <skill_dir>/scripts/sub_eval.py run evals/cases.jsonl \
  --prompt prompts/system.txt \
  --grader auto \
  --out .eval-cache/latest_report.md

# Local code / CLI under test (--cmd) graded by your subscription as an LLM judge:
python3 <skill_dir>/scripts/sub_eval.py run evals/cases.jsonl \
  --cmd "python3 my_tool.py" \
  --grader llm_judge \
  --rubric "Output must answer the question accurately without hallucinated flags."
```

### Workflow B: Quick ad-hoc `/eval` (no dataset exists yet)
When the user types something like `/eval "does our date parser handle timezones and leap years?"` or `/eval src/prompts/triage.md`:
1. Read the target prompt, function, or module to understand inputs, edge cases, and failure modes.
2. Author a focused 8–20 case dataset at `evals/<slug>.jsonl` covering happy paths, tricky edge cases, and adversarial inputs (each line: `{"id": "...", "input": "...", "expected": "...", "rubric": "..."}` — see `references/suite-format.md`).
3. Run `sub_eval.py run` (or `prepare` -> subagents -> `ingest`), inspect misses, and show the Markdown summary table plus concrete failure examples.

### Workflow C: A/B comparison with paired 95% confidence intervals
When comparing two prompts or two saved runs:

```bash
# Run both prompts on the same suite (cached calls are reused automatically):
python3 <skill_dir>/scripts/sub_eval.py run evals/cases.jsonl \
  --prompt prompts/v1.txt \
  --compare prompts/v2.txt

# Or compare any two saved run JSONs in .eval-cache/runs/:
python3 <skill_dir>/scripts/sub_eval.py compare .eval-cache/runs/run_a.json .eval-cache/runs/run_b.json
```

### Workflow D: In-session subagent execution (`prepare` -> subagents -> `ingest`)
When no CLI binary is installed on `PATH`:
1. Prepare shards (if `--cmd` is provided, `prepare` executes `--cmd` locally first and emits judge prompts!):
   ```bash
   python3 <skill_dir>/scripts/sub_eval.py prepare evals/cases.jsonl \
     --prompt prompts/v1.txt \
     --work-dir .eval-cache/batch
   ```
2. For each `.eval-cache/batch/shard_XX.json` in parallel, spawn a subagent instructed to read the shard's `items` and write `.eval-cache/batch/shard_XX.out.json` with `[{"id": "...", "output": "...", "judge": {"pass": true, "score": 1.0, "reason": "..."}}]`.
3. Ingest, score, cache, and print the report:
   ```bash
   python3 <skill_dir>/scripts/sub_eval.py ingest .eval-cache/batch --out .eval-cache/latest_report.md
   ```

## Step 3 — Reporting rules

- Always include `.eval-cache/` in `.gitignore` if creating evals in a repo.
- Report **Pass Rate**, **Mean Score**, **Bootstrap 95% CI**, **Cache Hits**, and (for comparisons) **Paired Delta + 95% CI** alongside the specific case IDs that regressed or improved.
- For dataset schemas, graders (`exact`, `contains`, `regex`, `json_match`, `llm_judge`), and standalone repo export (`scripts/export_standalone.sh`), read `references/suite-format.md`.
