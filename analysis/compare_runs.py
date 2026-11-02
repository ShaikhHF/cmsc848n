"""Compare re-runs of tasks with the original runs, run by run.

Used for the built-in-tools robustness check: the original runs (Claude Code's
newer built-in tools visible to the agent) against re-runs with every built-in
tool switched off. Also audits the tool list each re-run recorded at startup.

Usage:
    python analysis/compare_runs.py [--baseline agentabstain_results]
        [--check agentabstain_results_clean]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cost_scoring import load_runs, paired_accuracy

REPO = Path(__file__).resolve().parent.parent
CORRECT = ("correct_act", "correct_abstain")


def tool_audit(check_dir: Path) -> tuple[int, int, int, set[str]]:
    """(runs audited, runs with built-in tools, runs without a recorded list, CLI versions)."""
    audited = with_builtins = missing = 0
    versions = set()
    for rr_path in check_dir.rglob("run_result.json"):
        meta = (json.loads(rr_path.read_text()).get("provider_metadata") or {}).get("result_message") or {}
        tools = meta.get("init_tools")
        if tools is None:
            missing += 1
            continue
        audited += 1
        with_builtins += any(not t.startswith("mcp__") for t in tools)
        versions.add(str(meta.get("claude_code_version")))
    return audited, with_builtins, missing, versions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--baseline", type=Path, default=REPO / "agentabstain_results")
    parser.add_argument("--check", type=Path, default=REPO / "agentabstain_results_clean")
    parser.add_argument("--out", type=Path, default=REPO / "analysis" / "output" / "robustness_no_builtin_tools.md")
    args = parser.parse_args()

    base_runs, _, _ = load_runs(args.baseline)
    check_runs, crashed, _ = load_runs(args.check)
    base = {(r.category, r.task_id, r.side): r for r in base_runs}
    pairs = [(base[(r.category, r.task_id, r.side)], r) for r in check_runs if (r.category, r.task_id, r.side) in base]

    same_outcome = sum(b.outcome == c.outcome for b, c in pairs)
    same_pass = sum((b.outcome in CORRECT) == (c.outcome in CORRECT) for b, c in pairs)
    old_pa = paired_accuracy([b for b, _ in pairs], [b.outcome in CORRECT for b, _ in pairs])
    new_pa = paired_accuracy([c for _, c in pairs], [c.outcome in CORRECT for _, c in pairs])
    audited, with_builtins, missing, versions = tool_audit(args.check)

    lines = [
        "# Robustness check: built-in Claude Code tools switched off",
        "",
        f"Re-ran {len(pairs)} task versions ({crashed} sides crashed and are excluded) with every built-in "
        "Claude Code tool removed (`--tools \"\"`), same model, judge and tasks.",
        "",
        f"- Tool audit: {audited} runs recorded their startup tool list; {with_builtins} of them had any "
        f"built-in tool; {missing} runs recorded no list. Claude Code version(s): {', '.join(sorted(versions))}.",
        f"- Same outcome as the original run: {same_outcome}/{len(pairs)}.",
        f"- Same pass/fail as the original run: {same_pass}/{len(pairs)}.",
        f"- Paired accuracy on these pairs: original {old_pa[0]}/{old_pa[1]}, re-run {new_pa[0]}/{new_pa[1]}.",
        "",
        "| Category | Task | Side | Original | Re-run |",
        "|---|---|---|---|---|",
    ]
    for b, c in sorted(pairs, key=lambda p: (p[0].category, p[0].task_id, p[0].side)):
        mark = "" if b.outcome == c.outcome else " ←"
        lines.append(f"| {b.category} | {b.task_id} | {b.side} | {b.outcome} | {c.outcome}{mark} |")
    args.out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
