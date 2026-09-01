"""Narrow MCP tool surface for TinyTPU's Allo ISA specification."""

from __future__ import annotations

import subprocess
from pathlib import Path

from chia.base.tools.ChiaTool import ChiaTool


class AlloSpecTool(ChiaTool):
    """Let an agent inspect, patch, and validate *only* ``isa.py``."""

    def setup(self, repo: str, conda_exe: str) -> None:
        self.repo = Path(repo).resolve()
        self.spec = self.repo / "isa.py"
        self.conda_exe = conda_exe
        assert self.spec.is_file(), f"missing ISA specification: {self.spec}"
        self.mcp.add_tool(self.read_spec, name="tinytpu_read_spec")
        self.mcp.add_tool(self.apply_spec_patch, name="tinytpu_apply_spec_patch")
        self.mcp.add_tool(self.insert_after, name="tinytpu_insert_after")
        self.mcp.add_tool(self.run_compiler_check, name="tinytpu_run_compiler_check")

    def read_spec(self) -> str:
        """Return the complete Allo ISA specification the agent may edit."""
        return self.spec.read_text(encoding="utf-8")

    def apply_spec_patch(self, patch: str) -> str:
        """Apply a unified diff whose only modified path is ``isa.py``.

        The patch must use git-style ``a/isa.py`` / ``b/isa.py`` headers. No
        other repository path is accepted, which keeps the agent at the Allo
        ISA layer rather than letting it modify the compiler or microarchitecture.
        """
        headers = [line for line in patch.splitlines() if line.startswith(("--- ", "+++ "))]
        allowed = {"--- a/isa.py", "+++ b/isa.py"}
        if set(headers) != allowed or len(headers) != 2:
            return "Rejected: patch may modify only a/isa.py and b/isa.py."
        checked = subprocess.run(
            ["patch", "--dry-run", "--batch", "--forward", "-p1"],
            input=patch,
            text=True,
            cwd=self.repo,
            capture_output=True,
        )
        if checked.returncode:
            return f"Rejected: patch does not apply.\n{checked.stdout}{checked.stderr}"
        applied = subprocess.run(
            ["patch", "--batch", "--forward", "-p1"],
            input=patch,
            text=True,
            cwd=self.repo,
            capture_output=True,
        )
        assert applied.returncode == 0, applied.stdout + applied.stderr
        return "Patch applied to isa.py."

    def insert_after(self, anchor: str, content: str) -> str:
        """Insert ``content`` after a unique literal anchor in ``isa.py`` only."""
        source = self.spec.read_text(encoding="utf-8")
        if source.count(anchor) != 1:
            return "Rejected: anchor must occur exactly once in isa.py."
        self.spec.write_text(source.replace(anchor, anchor + content, 1), encoding="utf-8")
        return "Content inserted into isa.py."

    def run_compiler_check(self) -> str:
        """Run TinyTPU's direct-TOSA compiler regression suite in the Allo env."""
        result = subprocess.run(
            [
                "make",
                "compiler",
                f"CONDA={self.conda_exe}",
            ],
            cwd=self.repo,
            text=True,
            capture_output=True,
            timeout=300,
        )
        output = (result.stdout + result.stderr).strip()
        if len(output) > 24_000:
            output = output[-24_000:]
        return f"exit={result.returncode}\n{output}"
