# `/eval` Suite Format & Graders Reference

## Dataset Formats (`.jsonl`, `.json`, `.csv`)

Each evaluation case is an object (one JSON object per line in `.jsonl`, a JSON array or `{"cases": [...]}` in `.json`, or a row in `.csv`):

```json
{"id": "null_date", "input": "Parse 'N/A' as ISO date", "expected": "null", "grader": "exact"}
{"id": "sql_tz", "input": "Count orders placed yesterday in America/Chicago", "rubric": "Uses AT TIME ZONE 'America/Chicago' and filters a full 24h day boundary."}
```

### Supported Fields
- `id` *(optional)*: unique identifier (`case_001`, `case_002`, ... if omitted).
- `input` *(or `question` / `prompt`)*: string or JSON object passed to the prompt template (`{input}`, or named `{field}` placeholders when `input` is a dict) or piped to `--cmd` on `stdin`.
- `expected` *(or `target` / `ground_truth`, optional)*: reference value for deterministic graders or reference context for `llm_judge`.
- `rubric` *(optional)*: per-case natural-language grading rubric (overrides `--rubric`).
- `grader` *(optional)*: per-case grader override (`exact`, `contains`, `regex`, `json_match`, `llm_judge`).

## Graders

| Grader | When selected by `--grader auto` | Behavior |
|---|---|---|
| `exact` | Default when `expected` is a string/number and no `rubric` is set | Case-insensitive, whitespace-normalized equality |
| `contains` | Explicit `--grader contains` | Case-insensitive substring check (`expected in output`) |
| `regex` | Explicit `--grader regex` | `re.search(expected, output, re.S)` |
| `json_match` | Default when `expected` is a JSON object | Extracts JSON (fenced or bare) from output and checks all expected keys/values |
| `llm_judge` | Default whenever `rubric` or `--rubric` is provided | Grades via your subscription returning `{"pass": bool, "score": 0..1, "reason": "..."}` |

## Exporting `/eval` Into a Standalone Repository

To split `/eval` out into its own standalone skill repository:

```bash
bash skills/eval/scripts/export_standalone.sh ../sub-eval
cd ../sub-eval && git init
```
