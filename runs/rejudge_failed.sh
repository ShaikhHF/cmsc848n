#!/usr/bin/env bash
# Re-grade runs whose LLM-judge call failed (e.g. the API credit ran out)
# without re-running the agent. AgentAbstain caches judge results in each
# run's eval.json, failed calls included, so only those files are removed;
# successful verdicts are reused and not paid for again.
#
# Usage: RESULTS_ROOT=results_prompt_four_options \
#        TEAM_RESULTS=baseline_runs/prompt_four_options bash runs/rejudge_failed.sh
#        (defaults: results_no_builtins and agentabstain_results_clean)
set -euo pipefail

REPO=$(cd "$(dirname "$0")/.." && pwd)
AA=$REPO/third_party/agentabstain
VENV=${VENV:-$(dirname "$REPO")/.venvs/agentabstain}
RESULTS_ROOT=${RESULTS_ROOT:-results_no_builtins}
TEAM_RESULTS=$(realpath "${TEAM_RESULTS:-$REPO/agentabstain_results_clean}")

has_key() { grep -Eq "^$1=.+" "$AA/.env" 2>/dev/null; }
if has_key AWS_BEARER_TOKEN_BEDROCK; then
  export LLM_PROVIDER=bedrock
  JUDGE=$REPO/runs/judge_opus46_bedrock.yaml
  MODEL=us.anthropic.claude-sonnet-4-6
elif has_key ANTHROPIC_API_KEY; then
  export LLM_PROVIDER=native
  JUDGE=$REPO/runs/judge_opus46_api.yaml
  MODEL=claude-sonnet-4-6
else
  echo "Add AWS_BEARER_TOKEN_BEDROCK or ANTHROPIC_API_KEY to $AA/.env first." >&2
  exit 1
fi

source "$VENV/bin/activate"
cd "$AA"

# Remove eval.json files whose judge call failed, here and in the team copy.
python - "$RESULTS_ROOT/claudesdk" "$TEAM_RESULTS/claudesdk" <<'EOF'
import json, sys
from pathlib import Path
src, team = Path(sys.argv[1]), Path(sys.argv[2])
removed = 0
for path in src.rglob("eval.json"):
    judge = json.loads(path.read_text())["metrics"].get("response_llm_judge", {})
    if judge.get("pass") is None and "error" in judge:
        path.unlink()
        (team / path.relative_to(src)).unlink(missing_ok=True)
        removed += 1
print(f"removed {removed} failed judge results; re-grading them now")
EOF

python -m eval.runner --provider claudesdk --model "$MODEL" --judge-config "$JUDGE" \
  --results-root "$RESULTS_ROOT"
rsync -a --prune-empty-dirs --exclude batch_runs --ignore-existing \
  "$RESULTS_ROOT/claudesdk/" "$TEAM_RESULTS/claudesdk/"
echo "Re-graded runs copied to $TEAM_RESULTS"
