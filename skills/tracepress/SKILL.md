---
name: tracepress
description: Analyze, summarize, audit, and monitor agent traces (Harbor, ATIF, Claude Code, Codex, Cursor, OpenTelemetry) using your existing Claude Code, Codex, or Cursor subscription — no separate LLM API key required. Invoke as /tracepress or when asked to inspect agent trajectories, failure modes, or coding sessions.
argument-hint: "[traces-path-or-source] [question]"
---

# TracePress (Subscription-Powered Trace Analysis)

TracePress runs DSPy-powered map/reduce and streaming folds over agent traces in Python — and uses the **Claude Code, OpenAI Codex, or Cursor subscription you are already logged into**, with no separate OpenRouter or provider API key required.

Parse any arguments passed after `/tracepress` or `$tracepress` in the user's message (`$ARGUMENTS`) as the target trace source (`harbor`, `atif`, `claude_code`, `codex`, `cursor`, `otel`, or a file/directory path) and the question to answer.

## Step 1 — Pick the zero-key LM backend

Run the built-in subscription check first:

```bash
uv run tracepress doctor
# or: python3 -m tracepress doctor --json
```

Choose the execution path based on what is available:

1. **Existing AI subscription on this machine (default — $0 extra API spend):**
   ```python
   import tracepress as tp

   tp.configure(lm="subscription")          # auto-picks Claude Code, Codex, or Cursor
   # Or pick explicitly:
   # tp.configure(lm="claude-code:sonnet")  # uses ~/.claude OAuth or `claude -p`
   # tp.configure(lm="codex:gpt-5.4")       # uses ~/.codex/auth.json or `codex exec`
   # tp.configure(lm="cursor:auto")         # uses Cursor `agent -p --mode=ask`
   ```
   - `claude-code` automatically uses fast direct OAuth HTTP when `~/.claude/.credentials.json` is present, and falls back to headless `claude -p` (e.g. on macOS Keychain logins), stripping `CLAUDECODE` and `ANTHROPIC_API_KEY` so nested calls inside Claude Code work cleanly against your subscription.
   - `codex` uses `~/.codex/auth.json` OAuth HTTP when present, and falls back to `codex exec --sandbox read-only --ephemeral`.
   - `cursor` runs `agent -p --mode=ask --trust` inside an isolated tool-denied temporary workspace so prompts never read or modify repo files.
   - Every call goes through DSPy's persistent disk cache, so re-running or iterating on downstream reduces reuses all cached step calls for free.

2. **Explicit API gateway (when the user sets an API key or runs thousands of rows):**
   - `OPENROUTER_API_KEY` -> `tp.configure(lm="openrouter:openai/gpt-6-luna")`
   - `TYPESAFE_API_KEY` -> `lm="typesafe:jev-latest"` on classification-only stages (`Noul`, `Choice`, `Score`)
   - `PORTKEY_API_KEY` -> `tp.configure(lm="portkey:<slug>/<model>")`

3. **Keyless cloud sandbox fallback (no local CLI binary and no API key):**
   When running inside a remote sandbox where `tracepress doctor` reports `recommended: null` and no API keys exist in the environment:
   - Still use TracePress's free Python engine (`tp.read.*`, `.filter()`, `.map()`, `.join()`, `.group_by().reduce()`) to load, normalize, and slice traces at zero cost.
   - Export compact step/trace batches to `/tmp/tp_batches/*.json`, fan out parallel native subagents (`Task` / `Agent`) from your current coding session to produce structured JSON judgments, and load them back into TracePress with `tp.from_rows(...)` for aggregation and drill-down!

## Step 2 — Load traces and inspect for free before making LM calls

All readers share the step convention (`source`, `message`, `reasoning`, `tool_calls`, `observation`, `timestamp`):

```python
import tracepress as tp

# Coding-agent sessions on disk (private — never commit raw user transcripts):
src = tp.read.claude_code(project="my-repo", limit=20)
src = tp.read.codex(project="my-repo", limit=20)
src = tp.read.cursor(project="my-repo", limit=20)

# Evaluation & observability formats:
src = tp.read.harbor("path/to/job_or_trials_dir")
src = tp.read.atif("path/to/trajectories")
src = tp.read.otel("path/to/otlp_spans.jsonl")
```

Always check row counts before running `lm_map` / `lm_fold` (especially over CLI subscriptions, where concurrency defaults to `cli_concurrency=8`):

```python
steps = src.steps.collect()
print(f"{len(steps)} steps")
```

## Step 3 — Build and run the pipeline

Start on a small slice (`limit=2`) to verify prompts, then scale up (cached prompts won't re-run):

```python
import dspy
import tracepress as tp

tp.configure(lm="subscription")
src = tp.read.harbor("jobs/my-job", limit=5)

failed_tests = (
    src.tests.filter(lambda r: r.status != "passed")
    .group_by("trial")
    .reduce(
        lambda rs: {"failed_tests": ", ".join(r.name.split("::")[-1] for r in rs)},
        columns={"failed_tests": str},
    )
)

notes = src.steps.lm_map(
    "step -> what_happened, is_mistake: bool",
    instructions="Summarize this step in one sentence and flag whether it derailed the task.",
)

per_trial = (
    notes.join(src.trials)
    .join(failed_tests)
    .group_by("trial")
    .lm_reduce(
        "instruction, reward, failed_tests, what_happened, is_mistake -> summary, failure_mode",
        instructions="Explain why the run succeeded or failed using the failed tests.",
        module=dspy.ChainOfThought,
    )
)

rows = per_trial.collect()
print(tp.last_run())
```

## Detailed Companion Skills

- **Batch failure modes, taxonomy & drill-down:** see `../tracepress-analyze-traces/SKILL.md`
- **Live streaming & `lm_fold` monitors:** see `../tracepress-stream-traces/SKILL.md`
- **Adding a new trace format adapter:** see `../tracepress-add-adapter/SKILL.md`
