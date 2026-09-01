"""Narrow MCP tool surface for TinyTPU's Allo co-design specification."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from chia.base.tools.ChiaTool import ChiaTool


class AlloSpecTool(ChiaTool):
    """Let an agent co-edit TinyTPU's ISA and composed hardware blocks only."""

    def setup(self, repo: str, conda_exe: str) -> None:
        self.repo = Path(repo).resolve()
        self.sources = {
            "isa.py": self.repo / "isa.py",
            "microarch.py": self.repo / "microarch.py",
        }
        self.conda_exe = conda_exe
        assert all(path.is_file() for path in self.sources.values()), self.sources
        self.mcp.add_tool(self.read_spec, name="tinytpu_read_spec")
        self.mcp.add_tool(self.apply_spec_patch, name="tinytpu_apply_spec_patch")
        self.mcp.add_tool(self.insert_after, name="tinytpu_insert_after")
        self.mcp.add_tool(self.run_compiler_check, name="tinytpu_run_compiler_check")
        self.mcp.add_tool(self.run_hardware_check, name="tinytpu_run_hardware_check")
        self.mcp.add_tool(self.score_access_cost, name="tinytpu_score_access_cost")

    def read_spec(self) -> str:
        """Return the complete ISA plus its writable composed hardware blocks."""
        return "\n\n".join(
            f"===== {name} =====\n{path.read_text(encoding='utf-8')}"
            for name, path in self.sources.items()
        )

    def apply_spec_patch(self, patch: str) -> str:
        """Apply a unified diff touching only ``isa.py`` and/or ``microarch.py``.

        The patch must use matching git-style headers such as ``a/isa.py`` /
        ``b/isa.py``. No other repository path is accepted: the generic ACT
        compiler, runtime, tests, and benchmark remain fixed.
        """
        headers = [line for line in patch.splitlines() if line.startswith(("--- ", "+++ "))]
        if not headers or len(headers) % 2:
            return "Rejected: patch needs paired git-style file headers."
        paths = []
        for old, new in zip(headers[::2], headers[1::2]):
            if not old.startswith("--- a/") or not new.startswith("+++ b/"):
                return "Rejected: use matching a/<path> and b/<path> headers."
            old_path, new_path = old.removeprefix("--- a/"), new.removeprefix("+++ b/")
            if old_path != new_path or old_path not in self.sources:
                return "Rejected: patches may touch only isa.py and microarch.py."
            paths.append(old_path)
        if len(paths) != len(set(paths)):
            return "Rejected: each writable file may appear only once per patch."
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
        return f"Patch applied to {', '.join(paths)}."

    def insert_after(self, path: str, anchor: str, content: str) -> str:
        """Insert text after one unique anchor in writable ``path``.

        ``path`` must be ``isa.py`` or ``microarch.py``. Prefer a unified diff
        when an architectural change must update both files atomically.
        """
        target = self.sources.get(path)
        if target is None:
            return "Rejected: path must be isa.py or microarch.py."
        source = target.read_text(encoding="utf-8")
        if source.count(anchor) != 1:
            return f"Rejected: anchor must occur exactly once in {path}."
        target.write_text(source.replace(anchor, anchor + content, 1), encoding="utf-8")
        return f"Content inserted into {path}."

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

    def _run_allo_module(self, module: str, *args: str) -> str:
        root = self.repo.parents[2]
        env = os.environ | {
            "PYTHONPATH": str(root),
            "SKBUILD_EDITABLE_SKIP": str(root / "build"),
        }
        result = subprocess.run(
            [self.conda_exe, "run", "-n", "allo", "python", "-m", module, *args],
            cwd=root,
            env=env,
            text=True,
            capture_output=True,
            timeout=300,
        )
        output = (result.stdout + result.stderr).strip()
        if len(output) > 24_000:
            output = output[-24_000:]
        return f"exit={result.returncode}\n{output}"

    def run_hardware_check(self) -> str:
        """Export the composed Allo-HLS microarchitecture as a feasibility gate."""
        return self._run_allo_module(
            "examples.accelerator.tinytpu.microarch", "--print-hls"
        )

    def score_access_cost(self) -> str:
        """Compile GEMM benchmarks, check numerics, and score VREG/VMEM traffic.

        The score is the sole optimization objective: VREG words cost 1 and
        VMEM/BRAM words cost 4 by default. DRAM traffic is reported but excluded.
        """
        return self._run_allo_module("examples.accelerator.tinytpu.feedback")
