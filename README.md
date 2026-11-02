# Ask, Act, or Abstain?

Our CMSC848N project (UMD, Fall 2026) by Hana Shaikh, Kolade Alabi and Parastoo Pilevar.

When an agent gets an unclear or risky request, it can go ahead, look around first, ask the user, or decline. We want to build a controller that picks between these based on what each choice costs. We build on the [AgentAbstain](https://github.com/AntiQuality/agentabstain) benchmark, which comes in pairs of tasks: in one version the agent should do the task, in the other it should stop or ask.

So far we've run Claude Sonnet 4.6 three times on 32 of those pairs (64 task versions, all 8 categories) and compared it with a few simple baselines. The controller itself isn't built yet.

## Results so far

Costs: a mistake costs 10, an unnecessary abstention costs 3, and a mistake that didn't change anything costs 5. Paired means the agent got both versions of a pair right.

| Policy | Paired, runs 1/2/3 | Mean paired (of 32) | Mean cost |
|---|---|---:|---:|
| Agent (Sonnet 4.6, as-is) | 20 / 17 / 18 | 18.3 | 1.62 |
| Prompt-only (told about the four options) | 17 / 18 / 19 | 18.0 | 1.49 |
| Confidence threshold, cutoffs from run 1 | 22 / 20 / 19 | 20.3 | 1.31 |
| Confidence threshold, re-tuned on each run | 22 / 10 / 9 | 13.7 | 1.52 |
| Confidence threshold, one cutoff that keeps the agent's pairs (exploratory) | 20.0 / 18.2 / 17.8 | 18.7 | 1.40 |
| Always abstain | 0 / 0 / 0 | 0 | 1.50 |
| Always act | 0 / 0 / 0 | 0 | 4.22 |

What we take from this so far:

- The plain agent and the prompt-only agent are tied. Which one comes out ahead changes from run to run.
- The clearest failures are in the ambiguous-action and missing-parameter categories. Six should-abstain versions there fail every run of both policies, because the agent goes ahead on a guess.
- A low confidence cutoff helps on these tasks, but it isn't a solid win. Half of its cost gain comes from one pair, and tuning on cost alone isn't reliable: cutoffs near 23 and 75 cost about the same, yet the higher one passes about half as many pairs.
- Tuning with one extra condition, the cheapest cutoff that gets at least as many pairs right as the plain agent, makes the choice much more stable. Its cost gain isn't clear, though: -0.22 (95% interval -0.54 to +0.03), and almost nothing without the five pairs with doubtful labels. It also stops adapting when mistakes get very expensive. We wrote this rule after seeing these pairs, so it still needs testing on unused ones.

The intervals are wide, and tuning and scoring use the same 64 versions, so treat all of this as a pilot. Figures, intervals and caveats are in [analysis/output/three_runs/summary.md](analysis/output/three_runs/summary.md).

## How the policies work

Only two of them actually run the model:

- **Agent** is the normal AgentAbstain agent. It gets the task's system prompt, the request and the sandbox tools, and decides on its own what to do.
- **Prompt-only** is the same run, with [runs/prompts/four_options.txt](runs/prompts/four_options.txt) added to the system prompt (set `PROMPT_SUFFIX`).

The rest are worked out afterwards from the saved runs:

- **Always act** and **always abstain** come straight from the task labels.
- **Confidence threshold** asks the model how sure it is before it starts. If it's below a cutoff, the agent's run is swapped for "abstain". Those stops are simulated, so they were never run or judged. Each category's cutoff is tuned on the other seven. "Cutoffs from run 1" tunes once and keeps those cutoffs, while "re-tuned" tunes again on each run's results. The exploratory row uses one cutoff for all categories, chosen on half the pairs in one run and scored on the other half in another run, so its run values are averages over many splits.

## What's in here

- `third_party/agentabstain/` is the benchmark, as a git submodule. We don't modify it.
- `agentabstain_results_clean/`, `agentabstain_results_repeat2/` and `agentabstain_results_repeat3/` hold the three agent runs.
- `baseline_runs/prompt_four_options/`, `prompt_four_options_repeat2/` and `prompt_four_options_repeat3/` hold the three prompt-only runs.
- `agentabstain_results/` holds our first runs, kept for comparison (see the note below).
- `runs/` has the scripts and task lists for running and grading.
- `analysis/` has the scoring scripts. `analysis/output/three_runs/` has the figures and tables for all three runs, and `analysis/output/summary.md` looks at run 1 on its own.

Run 1 used the Anthropic API, and runs 2 and 3 used AWS Bedrock, with the same models.

## Setup

```bash
git clone --recurse-submodules git@github.com:ShaikhHF/cmsc848n.git
cd cmsc848n

# The scripts look for the venv next to the repo (set VENV to use another one)
uv venv --python 3.11 ../.venvs/agentabstain
source ../.venvs/agentabstain/bin/activate
uv pip install -r third_party/agentabstain/requirements.txt "mcp<2" "anthropic<1"

# Benchmark data
cd third_party/agentabstain
python -c "from huggingface_hub import snapshot_download; snapshot_download('antiquality/agentabstain', repo_type='dataset', local_dir='data')"
cd ../..
```

You'll also need the Claude Code CLI (`claude`) on your PATH. Copying the binary into the venv's `bin/` works too.

Keys go in `third_party/agentabstain/.env`. Use either `ANTHROPIC_API_KEY`, or `AWS_BEARER_TOKEN_BEDROCK` plus `AWS_REGION` for Bedrock. If both are there, the run scripts use Bedrock. `confidence_threshold.py` only works with `ANTHROPIC_API_KEY`.

## Re-scoring (free, no API calls)

```bash
python analysis/three_run_figures.py   # figures and tables for all three runs
python analysis/cost_scoring.py        # one run in detail (run 1 by default)
```

Run them in that order, so `summary.md` picks up the latest three-run table. To look at another run, add `--results DIR` and `--out DIR`, and use a new output folder so you don't overwrite `analysis/output/`. Try `--mistake-cost` and `--abstain-cost` to change the costs.

## Running the agent (costs money)

```bash
# Agent
bash runs/run_midterm.sh runs/check_no_builtin_tools.yaml runs/rerun_rest_no_builtin_tools.yaml

# Prompt-only (separate folders, so it doesn't mix with the agent runs)
PROMPT_SUFFIX=runs/prompts/four_options.txt RESULTS_ROOT=results_prompt_four_options \
TEAM_RESULTS=$PWD/baseline_runs/prompt_four_options \
bash runs/run_midterm.sh runs/check_no_builtin_tools.yaml runs/rerun_rest_no_builtin_tools.yaml
```

To repeat a run, set `RESULTS_ROOT` and `TEAM_RESULTS` to new folders, because the runner skips tasks it has already finished.

The agent costs about $0.05 per task version, and grading with the Opus 4.6 judge costs extra. A batch of 64 takes 30 to 45 minutes. Tasks run one at a time, because parallel runs hit rate limits. To check progress, run `python3 runs/progress.py`. If grading fails partway (for example, you run out of credit), `runs/rejudge_failed.sh` re-grades only the failed runs.

The confidence scores come from `python analysis/confidence_threshold.py --samples 3`, which also makes API calls. Add `--dry-run` to see how many first.

## Things we learned the hard way

- AgentAbstain blocks Claude Code's built-in tools by name, but newer Claude Code versions have tools that list doesn't know about. So our first runs could see tools like ToolSearch. We re-ran everything with all built-in tools off. Only 4 of 64 results changed, as shown in [analysis/output/robustness_no_builtin_tools.md](analysis/output/robustness_no_builtin_tools.md).
- Newer `mcp` and `anthropic` packages break the benchmark's tool bridge and judge, hence the version pins above.
- The benchmark grades asking and refusing the same way. So "always ask" scores the same as "always abstain" for now, and we can't charge for questions yet.
- The grading has blind spots. The commit check matches tool names, so finishing a task with a differently named tool counts as incomplete. On 16 of the 32 should-act versions there's no commit tool to check, so "correct" only means the judge saw no abstention, and the judge reads the final message, not the tool results. Five failure labels look doubtful to us; they're listed in the summary and still count in our results.
- AgentAbstain skips grading when an agent hits the 30-turn limit. We count those should-act runs as incomplete acts rather than dropping them. So far this has happened once.
