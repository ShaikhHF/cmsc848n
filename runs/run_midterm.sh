#!/usr/bin/env bash
# Run AgentAbstain task lists with Claude Sonnet 4.6 (built-in Claude Code
# tools switched off, see run_inference_clean.py), grade them with an Opus 4.6
# judge (as in the Oct 12 runs), and copy new results into the team repo.
#
# Usage: bash runs/run_midterm.sh task-list.yaml [task-list.yaml ...]
#        e.g. runs/check_no_builtin_tools.yaml runs/rerun_rest_no_builtin_tools.yaml
# Optional environment:
#   RESULTS_ROOT  results folder inside third_party/agentabstain (default: results_no_builtins)
#   TEAM_RESULTS  where finished runs are copied (default: agentabstain_results_clean)
#   PROMPT_SUFFIX text file appended to every task's system prompt (prompt-only baseline)
#
# Credentials go in third_party/agentabstain/.env: AWS_BEARER_TOKEN_BEDROCK
# (Bedrock route) or ANTHROPIC_API_KEY (Anthropic API route).
set -euo pipefail

REPO=$(cd "$(dirname "$0")/.." && pwd)
AA=$REPO/third_party/agentabstain
VENV=${VENV:-$(dirname "$REPO")/.venvs/agentabstain}
RESULTS_ROOT=${RESULTS_ROOT:-results_no_builtins}
TEAM_RESULTS=${TEAM_RESULTS:-$REPO/agentabstain_results_clean}

has_key() { grep -Eq "^$1=.+" "$AA/.env" 2>/dev/null; }

if has_key AWS_BEARER_TOKEN_BEDROCK; then
  export CLAUDE_CODE_USE_BEDROCK=1 LLM_PROVIDER=bedrock
  RUNTIME=$AA/src/configs/claudesdk_claude-sonnet-4-6.yaml
  JUDGE=$REPO/runs/judge_opus46_bedrock.yaml
  MODEL=us.anthropic.claude-sonnet-4-6
elif has_key ANTHROPIC_API_KEY; then
  unset CLAUDE_CODE_USE_BEDROCK
  export LLM_PROVIDER=native
  RUNTIME=$REPO/runs/claudesdk_claude-sonnet-4-6_api.yaml
  JUDGE=$REPO/runs/judge_opus46_api.yaml
  MODEL=claude-sonnet-4-6
else
  echo "Add AWS_BEARER_TOKEN_BEDROCK or ANTHROPIC_API_KEY to $AA/.env first." >&2
  exit 1
fi

configs=("$@")
if [ ${#configs[@]} -eq 0 ]; then
  echo "Usage: bash runs/run_midterm.sh task-list.yaml [task-list.yaml ...]" >&2
  exit 1
fi
# Resolve paths now: the script changes directory below.
for i in "${!configs[@]}"; do configs[$i]=$(realpath "${configs[$i]}"); done
if [ -n "${PROMPT_SUFFIX:-}" ]; then
  PROMPT_SUFFIX=$(realpath "$PROMPT_SUFFIX")
  export PROMPT_SUFFIX
  # With the default folders the runner would skip every task (already done
  # without the suffix) and copy the main runs as if they were the variant.
  if [ "$TEAM_RESULTS" = "$REPO/agentabstain_results_clean" ] || [ "$RESULTS_ROOT" = "results_no_builtins" ]; then
    echo "Set both TEAM_RESULTS and RESULTS_ROOT so prompt-variant runs stay separate from the main runs." >&2
    exit 1
  fi
fi

source "$VENV/bin/activate"  # also puts the bundled `claude` CLI on PATH
# Keep personal Claude Code settings, skills and plugins out of the benchmark agent.
export CLAUDE_CONFIG_DIR=$AA/logs/claude-config
cd "$AA"

# Same model settings, results written under $RESULTS_ROOT.
mkdir -p logs
RUNTIME_USED=logs/runtime_used.yaml
python - "$RUNTIME" "$RUNTIME_USED" "$RESULTS_ROOT/claudesdk" <<'EOF'
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1]))
cfg["results_root"] = sys.argv[3]
yaml.safe_dump(cfg, open(sys.argv[2], "w"), sort_keys=False)
EOF

for cfg in "${configs[@]}"; do
  # One task at a time: parallel workers caused the 429 crashes on Oct 12.
  python "$REPO/runs/run_inference_clean.py" --runtime-config "$RUNTIME_USED" \
    --task-config "$cfg" --workers 1
  # run_inference exits 0 even when tasks fail, so list failures explicitly.
  RESULTS_ROOT=$RESULTS_ROOT python - <<'EOF'
import json, os, pathlib
root = pathlib.Path(os.environ["RESULTS_ROOT"]) / "claudesdk"
latest = max(root.glob("*/batch_runs/*/summary.json"), key=lambda p: p.stat().st_mtime)
for f in json.loads(latest.read_text()).get("failed", []):
    print(f"FAILED {f['category']}/{f['task_id']}/{f['task_type']}: {f['error']}")
EOF
done

python -m eval.runner --provider claudesdk --model "$MODEL" --judge-config "$JUDGE" \
  --results-root "$RESULTS_ROOT"

# Copy finished runs only: no batch logs, no empty folders from failed starts.
mkdir -p "$TEAM_RESULTS/claudesdk"
rsync -a --prune-empty-dirs --exclude batch_runs --ignore-existing \
  "$RESULTS_ROOT/claudesdk/" "$TEAM_RESULTS/claudesdk/"
if [ "$TEAM_RESULTS" = "$REPO/agentabstain_results_clean" ]; then
  python "$REPO/analysis/cost_scoring.py"
else
  echo "Copied runs to $TEAM_RESULTS"
fi
