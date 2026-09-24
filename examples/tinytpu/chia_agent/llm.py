"""CHIA's OpenCodeLLM with the MCP request timeout this design needs.

opencode's MCP client applies a per-server request timeout (`mcp.<name>.timeout`,
falling back to `experimental.mcp_timeout`) and CHIA's `_build_config` sets
neither. The 2026-09-19 smoke run measured the consequence: `MCP error -32001:
Request timed out` after 60 s. A cosim score takes 2-4 minutes, so the agent's
`score_cycles` could never return. The harness's own scoring was unaffected,
since it does not go through MCP.

Lives in its own module, not in loop.py, so Ray can import it by name on the
worker that runs `prompt`.
"""

from __future__ import annotations

from chia.models.opencode import OpenCodeLLM

#: Longer than the evaluator's worst case (gate 2 x 240 s + cosim 1800 s).
MCP_TIMEOUT_MS = 40 * 60 * 1000


class IsaOpenCodeLLM(OpenCodeLLM):
    def _build_config(self, tools):
        cfg = super()._build_config(tools)
        for entry in (cfg.get("mcp") or {}).values():
            entry["timeout"] = MCP_TIMEOUT_MS
        cfg.setdefault("experimental", {})["mcp_timeout"] = MCP_TIMEOUT_MS
        return cfg
