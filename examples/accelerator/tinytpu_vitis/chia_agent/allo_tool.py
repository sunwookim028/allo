"""Narrow MCP tool surface for TinyTPU's Allo co-design specification."""

from __future__ import annotations

import ast
import os
import shlex
import shutil
import subprocess
import tempfile
from pathlib import Path

from chia.base.tools.ChiaTool import ChiaTool

from spec_policy import policy_violations

#: Sourced before any Vitis-invoking check so the synthesis gate works whether
#: the loop was launched from a Vitis-configured shell or a bare Ray worker.
VITIS_SETTINGS = os.environ.get(
    "TINYTPU_VITIS_SETTINGS", "/opt/xilinx/Vitis_HLS/2023.2/settings64.sh"
)


class AlloSpecTool(ChiaTool):
    """Let an agent co-edit TinyTPU's ISA and composed hardware blocks only."""

    def setup(self, repo: str, conda_exe: str) -> None:
        self.repo = Path(repo).resolve()
        self.sources = {
            "isa.py": self.repo / "isa.py",
            "microarch.py": self.repo / "microarch.py",
        }
        self.conda_exe = conda_exe
        # Per-instance, so parallel workers never share one HLS project dir.
        self.synth_project = Path(
            os.environ.get("TINYTPU_SYNTH_PROJECT", "/tmp/tinytpu_chia_synth")
        ).with_name(
            Path(
                os.environ.get("TINYTPU_SYNTH_PROJECT", "/tmp/tinytpu_chia_synth")
            ).name
            + f"_{self.name}"
        )
        assert all(path.is_file() for path in self.sources.values()), self.sources
        self.mcp.add_tool(self.read_spec, name=f"{self.name}_read_spec")
        self.mcp.add_tool(self.apply_spec_patch, name=f"{self.name}_apply_spec_patch")
        self.mcp.add_tool(self.insert_after, name=f"{self.name}_insert_after")
        self.mcp.add_tool(
            self.run_compiler_check, name=f"{self.name}_run_compiler_check"
        )
        self.mcp.add_tool(
            self.run_hardware_check, name=f"{self.name}_run_hardware_check"
        )
        self.mcp.add_tool(self.score_access_cost, name=f"{self.name}_score_access_cost")
        self.mcp.add_tool(self.score_cycles, name=f"{self.name}_score_cycles")

    # -- Variant bookkeeping. Deliberately *not* MCP tools: the search harness
    # -- accepts or rewinds a candidate, the agent does not get to choose.
    def snapshot(self) -> dict[str, bytes]:
        """Capture the current writable spec so a candidate can be rewound."""
        return {name: path.read_bytes() for name, path in self.sources.items()}

    def restore(self, snapshot: dict[str, bytes]) -> None:
        """Put the writable spec back to a captured snapshot."""
        for name, content in snapshot.items():
            self.sources[name].write_bytes(content)

    def diff_against(self, snapshot: dict[str, bytes]) -> str:
        """A unified diff from ``snapshot`` to the current spec, for the record."""
        import difflib

        chunks = []
        for name, path in self.sources.items():
            before = snapshot.get(name, b"").decode("utf-8").splitlines(keepends=True)
            after = path.read_text(encoding="utf-8").splitlines(keepends=True)
            chunks.extend(difflib.unified_diff(before, after, f"a/{name}", f"b/{name}"))
        return "".join(chunks)

    def _syntax_error(self, name: str, source: str) -> str | None:
        """``None`` if ``source`` parses, else the message to hand back.

        Both edit paths go through this. An edit that leaves a writable file
        unparseable would otherwise be discovered only by the compiler check,
        costing a whole debug round to a mistake visible at edit time.
        """
        try:
            ast.parse(source, filename=name)
        except SyntaxError as error:
            return (
                f"Rejected: the edit leaves {name} unparseable — "
                f"{error.msg} at line {error.lineno}. The file is unchanged."
            )
        problems = policy_violations(name, source)
        if problems:
            return (
                "Rejected: the edit is outside what a hardware spec may contain — "
                + "; ".join(problems)
                + ". Describe hardware; do not read, write, or execute. "
                "The file is unchanged."
            )
        return None

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
        headers = [
            line for line in patch.splitlines() if line.startswith(("--- ", "+++ "))
        ]
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
        # Never invoke patch in the real repository: GNU patch can leave a
        # partially-applied file and ``.orig`` backup after a malformed hunk.
        # First patch isolated copies, then commit all modified source files.
        with tempfile.TemporaryDirectory(prefix="tinytpu-patch-") as tmp:
            sandbox = Path(tmp)
            for name, source in self.sources.items():
                shutil.copy2(source, sandbox / name)
            applied = subprocess.run(
                ["patch", "--batch", "--forward", "--no-backup-if-mismatch", "-p1"],
                input=patch,
                text=True,
                cwd=sandbox,
                capture_output=True,
            )
            if applied.returncode:
                return (
                    f"Rejected: patch does not apply.\n{applied.stdout}{applied.stderr}"
                )
            patched = {name: (sandbox / name).read_bytes() for name in paths}
            for name, content in patched.items():
                broken = self._syntax_error(name, content.decode("utf-8"))
                if broken:
                    return broken
            for name, content in patched.items():
                self.sources[name].write_bytes(content)
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
        # Insert at the end of the anchor's line. An anchor that stops mid-line
        # would otherwise splice new code into that statement.
        cut = source.index(anchor) + len(anchor)
        line_end = source.find("\n", cut)
        cut = len(source) if line_end == -1 else line_end + 1
        updated = source[:cut] + content + source[cut:]
        broken = self._syntax_error(path, updated)
        if broken:
            return broken
        target.write_text(updated, encoding="utf-8")
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

    def _run_allo_module(
        self, module: str, *args: str, vitis: bool = False, timeout: int = 300
    ) -> str:
        root = self.repo.parents[2]
        env = os.environ | {
            "PYTHONPATH": str(root),
            "SKBUILD_EDITABLE_SKIP": str(root / "build"),
        }
        command = [self.conda_exe, "run", "-n", os.environ.get("TINYTPU_ENV", "allo"),
         "python", "-m", module, *args]
        if vitis and os.path.exists(VITIS_SETTINGS):
            # ``conda run`` does not read a profile, so Vitis has to be put on
            # PATH explicitly for the synthesis gate.
            quoted = " ".join(shlex.quote(part) for part in command)
            command = [
                "bash",
                "-c",
                f". {shlex.quote(VITIS_SETTINGS)} >/dev/null && {quoted}",
            ]
        result = subprocess.run(
            command,
            cwd=root,
            env=env,
            text=True,
            capture_output=True,
            timeout=timeout,
        )
        output = (result.stdout + result.stderr).strip()
        if len(output) > 24_000:
            output = output[-24_000:]
        return f"exit={result.returncode}\n{output}"

    def run_hardware_check(self) -> str:
        """Export the composed Allo-HLS microarchitecture as a feasibility gate."""
        return self._run_allo_module("examples.accelerator.tinytpu.verify", "--hls")

    def score_cycles(self) -> str:
        """Synthesize the candidate and score its benchmark cycle count.

        This is the optimization objective. Vitis HLS C-synthesis runs on the
        composed schedule, every unit's ``(ii, depth)`` is re-measured from that
        report, and the GEMM benchmarks are then costed under the measured
        table -- so a declared ``ISA.latency`` cannot influence the score. Area
        and Fmax come back in the same JSON but are not scored.
        """
        return self._run_allo_module(
            "examples.accelerator.tinytpu.ppa",
            "--project",
            str(self.synth_project),
            vitis=True,
            timeout=1800,
        )

    def score_access_cost(self) -> str:
        """Compile GEMM benchmarks, check numerics, and score VREG/VMEM traffic.

        The score is the sole optimization objective: VREG words cost 1 and
        VMEM/BRAM words cost 4 by default. DRAM traffic is reported but excluded.
        """
        return self._run_allo_module("examples.accelerator.tinytpu.feedback")
