#!/usr/bin/env bash
# Export the /eval subscription skill into its own standalone repository directory.
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <destination-directory>" >&2
  exit 1
fi

DEST="$1"
SKILL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "$SKILL_DIR/../.." && pwd)"

mkdir -p "$DEST/skills/eval" "$DEST/.agents/skills" "$DEST/.claude-plugin" "$DEST/.cursor-plugin" "$DEST/tests"
cp -R "$SKILL_DIR/"* "$DEST/skills/eval/"
ln -sfn "../../skills/eval" "$DEST/.agents/skills/eval"

if [[ -f "$REPO_ROOT/tests/test_sub_eval.py" ]]; then
  cp "$REPO_ROOT/tests/test_sub_eval.py" "$DEST/tests/test_sub_eval.py"
fi

cat > "$DEST/.gitignore" <<'EOF'
__pycache__/
.pytest_cache/
.eval-cache/
.venv/
EOF

cat > "$DEST/.claude-plugin/plugin.json" <<'EOF'
{
  "name": "sub-eval",
  "version": "0.1.0",
  "description": "Run /eval suites quickly using your existing Claude Code, Codex, or Cursor subscription.",
  "license": "MIT",
  "skills": "./skills/"
}
EOF

cp "$DEST/.claude-plugin/plugin.json" "$DEST/.cursor-plugin/plugin.json"

cat > "$DEST/pyproject.toml" <<'EOF'
[project]
name = "sub-eval"
version = "0.1.0"
description = "Slash-command /eval skill powered by your existing Claude Code, Codex, or Cursor subscription."
readme = "README.md"
license = "MIT"
requires-python = ">=3.10"
dependencies = []

[dependency-groups]
dev = ["pytest>=8.0"]

[tool.pytest.ini_options]
testpaths = ["tests"]
EOF

cat > "$DEST/README.md" <<'EOF'
# sub-eval (`/eval` with the AI subscription you already pay for)

Run quick, cached, reproducible evaluations directly from **Claude Code** (`/eval`), **Cursor** (`/eval`), or **OpenAI Codex** (`$eval`) using your existing subscription — zero API keys and zero pip dependencies required.

## Install

```bash
npx skills add <your-org>/sub-eval
```

## Usage

Inside Claude Code or Cursor, type:

```text
/eval skills/eval/examples/triage_cases.jsonl --prompt skills/eval/examples/prompt_v1.txt --compare skills/eval/examples/prompt_v2.txt
```

Or run the zero-dependency Python CLI directly:

```bash
python3 skills/eval/scripts/sub_eval.py doctor
python3 skills/eval/scripts/sub_eval.py run skills/eval/examples/triage_cases.jsonl \
  --prompt skills/eval/examples/prompt_v1.txt \
  --compare skills/eval/examples/prompt_v2.txt
```
EOF

echo "Exported standalone sub-eval project to $DEST"
