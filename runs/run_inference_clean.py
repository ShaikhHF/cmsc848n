"""Run AgentAbstain inference with Claude Code's built-in tools switched off.

AgentAbstain's claudesdk runtime denies built-in tools by name, but newer
Claude Code CLIs ship tools that list predates (ToolSearch, Skill, Workflow,
...), so the agent could see them. Passing `--tools ""` removes every built-in
tool and keeps the task's MCP tools. The tool list Claude Code reports at
startup is saved in each run's provider_metadata.result_message.init_tools,
so every run records exactly which tools the agent had.

If PROMPT_SUFFIX names a text file, it is appended to every task's system
prompt (used for the prompt-only baseline) and its file name is saved as
result_message.prompt_variant.

Takes the same arguments as AgentAbstain's src.scripts.run_inference; run it
from third_party/agentabstain with that checkout's venv (run_midterm.sh does).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

AA = Path(__file__).resolve().parent.parent / "third_party" / "agentabstain"
sys.path.insert(0, str(AA))

from dotenv import load_dotenv

load_dotenv(AA / ".env", override=True)  # before the runtime imports, as run_inference does

from claude_code_sdk import SystemMessage

# run_inference must come first: it imports the agents, which import the
# runtime module (importing the runtime directly first is circular).
from src.scripts import run_inference
import src.runtime.claudesdk as claudesdk

_Options = claudesdk.ClaudeCodeOptions
_extract_session_metadata = claudesdk._extract_session_metadata

SUFFIX_PATH = Path(os.environ["PROMPT_SUFFIX"]).resolve() if os.environ.get("PROMPT_SUFFIX") else None
SUFFIX = SUFFIX_PATH.read_text().strip() if SUFFIX_PATH else None


def _options_without_builtins(**kwargs):
    kwargs.setdefault("extra_args", {})["tools"] = ""
    if SUFFIX:
        kwargs["system_prompt"] = f"{kwargs.get('system_prompt') or ''}\n\n{SUFFIX}".strip()
    return _Options(**kwargs)


def _extract_with_init_tools(messages):
    session_id, meta = _extract_session_metadata(messages)
    init = next((m for m in messages if isinstance(m, SystemMessage) and m.subtype == "init"), None)
    if meta is not None and init is not None:
        meta["init_tools"] = init.data.get("tools")
        meta["claude_code_version"] = init.data.get("claude_code_version")
    if meta is not None:
        meta["prompt_variant"] = SUFFIX_PATH.stem if SUFFIX_PATH else None
    return session_id, meta


claudesdk.ClaudeCodeOptions = _options_without_builtins
claudesdk._extract_session_metadata = _extract_with_init_tools

if __name__ == "__main__":
    run_inference.main()
