"""Show progress of the runs started by run_midterm.sh.

Usage:
    python3 runs/progress.py                       # once
    watch -n 20 python3 runs/progress.py           # refresh every 20 s (Ctrl+C to stop)
    python3 runs/progress.py --tasks runs/check_no_builtin_tools.yaml \
        --results-root results_no_builtins --log logs/check_run.log
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent
AA = REPO / "third_party" / "agentabstain"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--tasks", nargs="+", type=Path,
                        default=[REPO / "runs" / "midterm_fill.yaml", REPO / "runs" / "midterm_sample.yaml"])
    parser.add_argument("--results-root", default="results", help="results folder inside third_party/agentabstain")
    parser.add_argument("--log", default="logs/midterm_run.log", help="log file inside third_party/agentabstain")
    args = parser.parse_args()
    LOG = AA / args.log

    planned = {
        (t["category"], t["task_id"], t["task_type"])
        for path in args.tasks if path.exists()
        for t in yaml.safe_load(path.read_text())["tasks"]
    }
    done, failed, cost = {}, {}, 0.0
    for rr_path in (AA / args.results_root / "claudesdk").glob("*/*/*/*/*/run_result.json"):
        rr = json.loads(rr_path.read_text())
        key = (rr["category"], rr["task_id"], rr["task_type"])
        if key not in planned:
            continue
        meta = (rr.get("provider_metadata") or {}).get("result_message") or {}
        cost += meta.get("total_cost_usd") or 0.0
        if rr.get("error"):
            failed[key] = str(rr["error"])[:120]
        else:
            # Graded only if the judge actually returned a verdict (a failed
            # judge call is cached in eval.json with pass = null).
            eval_path = rr_path.parent / "eval.json"
            done[key] = eval_path.exists() and json.loads(eval_path.read_text())[
                "metrics"].get("response_llm_judge", {}).get("pass") is not None
    failed = {k: v for k, v in failed.items() if k not in done}

    log = LOG.read_text(errors="ignore") if LOG.exists() else ""
    tasks = re.findall(r"[a-z_]+/preview_\d+/(?:act|abstain)", log)
    current = tasks[-1] if tasks else ""
    finished = "evaluated_runs" in log  # printed by the judge step at the very end

    n = len(planned)
    bar = "#" * round(30 * len(done) / n) if n else ""
    print(f"Runs finished: {len(done)}/{n}  [{bar:<30}]")
    print(f"Graded by judge: {sum(done.values())}   Failed: {len(failed)}   Agent cost so far: ${cost:.2f}")
    if current and len(done) < n:
        print(f"Working on: {current}")
    for (c, t, s), err in sorted(failed.items()):
        print(f"  FAILED {c}/{t}/{s}: {err}")
    by_cat: dict = {}
    for c, _, _ in planned:
        by_cat.setdefault(c, [0, 0])[1] += 1
    for c, _, _ in done:
        by_cat[c][0] += 1
    print("\nBy category:")
    for c, (d, total) in sorted(by_cat.items()):
        print(f"  {c:<32} {d}/{total}")
    if finished:
        print("\nAll batches done; see analysis/output/summary.md")


if __name__ == "__main__":
    main()
