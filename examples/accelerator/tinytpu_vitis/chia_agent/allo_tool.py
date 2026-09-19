"""Narrow MCP tool surface for co-designing TinyTPU-isa (tinytpu_vitis).

The agent may edit exactly two files -- `microarch_isa.py` (the hardware and its
ISA) and `isa_dsl.py` (the program generator) -- and only in a private spec
directory, never in the repository. Everything that decides the score is frozen
and outside its reach; see `evaluate.py` for the list and how it is enforced.
"""

from __future__ import annotations

import ast
import difflib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

from chia.base.tools.ChiaTool import ChiaTool

from spec_policy import policy_violations

EDITABLE = ("microarch_isa.py", "isa_dsl.py")
#: Read-only context the agent may look at. Served from git HEAD.
REFERENCE = {
    "cosim.py": "examples/accelerator/tinytpu_vitis/cosim.py",
    "bench_isa.py": "examples/accelerator/tinytpu_vitis/bench_isa.py",
    "stress.py": "examples/accelerator/tinytpu_vitis/chia_agent/stress.py",
    "evaluate.py": "examples/accelerator/tinytpu_vitis/chia_agent/evaluate.py",
    "RESULTS_ISA.md": "examples/accelerator/tinytpu_vitis/RESULTS_ISA.md",
    "COMPARISON.md": "examples/accelerator/tinytpu_vitis/COMPARISON.md",
}


class AlloSpecTool(ChiaTool):
    """Let an agent co-edit TinyTPU-isa's hardware and program generator only."""

    def setup(self, spec_dir: str, work_dir: str, agent_dir: str, repo: str,
              allo_python: str, llvm_build_dir: str) -> None:
        self.spec_dir = Path(spec_dir).resolve()
        self.sources = {name: self.spec_dir / name for name in EDITABLE}
        self.work_dir = Path(work_dir).resolve()
        # Absolute path to the checkout's own evaluator: the actor runs from a
        # Ray copy of chia_agent/, and the score must not come from that copy.
        self.evaluator = str(Path(agent_dir).resolve() / "evaluate.py")
        self.repo = str(Path(repo).resolve())
        # Captured here, in the driver, because a Ray actor inherits the
        # raylet's environment rather than the shell that launched the loop.
        self.eval_env = {
            "TINYTPU_ALLO_PYTHON": allo_python,
            "LLVM_BUILD_DIR": llvm_build_dir,
        }
        self._lock = None
        assert all(path.is_file() for path in self.sources.values()), self.sources
        self.mcp.add_tool(self.read_spec, name=f"{self.name}_read_spec")
        self.mcp.add_tool(self.read_reference, name=f"{self.name}_read_reference")
        self.mcp.add_tool(self.apply_spec_patch, name=f"{self.name}_apply_spec_patch")
        self.mcp.add_tool(self.insert_after, name=f"{self.name}_insert_after")
        self.mcp.add_tool(
            self.run_functional_check, name=f"{self.name}_run_functional_check")
        self.mcp.add_tool(self.score_cycles, name=f"{self.name}_score_cycles")

    def __getstate__(self):
        # The tool is re-pickled on every prompt; a Lock cannot be.
        state = super().__getstate__()
        state = dict(state) if isinstance(state, dict) else state
        if isinstance(state, dict):
            state["_lock"] = None
        return state

    # -- Variant bookkeeping. Deliberately *not* MCP tools: the search harness
    # -- accepts or rewinds a candidate, the agent does not get to choose.
    def snapshot(self) -> dict[str, bytes]:
        return {name: path.read_bytes() for name, path in self.sources.items()}

    def restore(self, snapshot: dict[str, bytes]) -> None:
        for name, content in snapshot.items():
            self.sources[name].write_bytes(content)

    def diff_against(self, snapshot: dict[str, bytes]) -> str:
        chunks = []
        for name, path in self.sources.items():
            before = snapshot.get(name, b"").decode("utf-8").splitlines(keepends=True)
            after = path.read_text(encoding="utf-8").splitlines(keepends=True)
            chunks.extend(difflib.unified_diff(before, after, f"a/{name}", f"b/{name}"))
        return "".join(chunks)

    def evaluate(self, gate_only: bool = False, shapes: str | None = None) -> dict:
        """Run the frozen evaluator on the spec dir; return its JSON verdict."""
        if self._lock is None:
            self._lock = threading.Lock()
        cmd = [sys.executable, self.evaluator, "--spec-dir", str(self.spec_dir),
               "--work", str(self.work_dir)]
        if gate_only:
            cmd.append("--gate-only")
        if shapes:
            cmd += ["--shapes", shapes]
        env = {k: v for k, v in os.environ.items() if not k.startswith("TPU_")}
        env.update(self.eval_env)
        with self._lock:  # one Vitis project per tool; never two at once
            p = subprocess.run(cmd, cwd=self.repo, env=env, text=True,
                               capture_output=True, timeout=5400)
        out = (p.stdout + p.stderr).strip()
        for line in reversed(out.splitlines()):
            line = line.strip()
            if line.startswith("{") and line.endswith("}"):
                try:
                    verdict = json.loads(line)
                    break
                except json.JSONDecodeError:
                    continue
        else:
            verdict = {"ok": False, "stage": "evaluator"}
        if not verdict.get("ok"):
            verdict["detail"] = out[-6000:]
        return verdict

    # -- MCP tools ---------------------------------------------------------
    def _check(self, name: str, source: str) -> str | None:
        try:
            ast.parse(source, filename=name)
        except SyntaxError as error:
            return (f"Rejected: the edit leaves {name} unparseable -- {error.msg} "
                    f"at line {error.lineno}. The file is unchanged.")
        problems = policy_violations(name, source)
        if problems:
            return ("Rejected: the edit is outside what a spec may contain -- "
                    + "; ".join(problems)
                    + ". Describe hardware; do not read, write, or execute. "
                    "The file is unchanged.")
        return None

    def read_spec(self) -> str:
        """Return both writable files: microarch_isa.py and isa_dsl.py."""
        return "\n\n".join(
            f"===== {name} =====\n{path.read_text(encoding='utf-8')}"
            for name, path in self.sources.items())

    def read_reference(self, name: str, start_line: int = 1,
                       max_lines: int = 400) -> str:
        """Read a FROZEN file for context (you cannot edit these).

        ``name`` is one of: cosim.py (the RTL cosim scorer and its testbench),
        bench_isa.py (the functional gate), stress.py (the extra semantic gate),
        evaluate.py (how the score is computed), RESULTS_ISA.md (design history
        and measurements), COMPARISON.md (the Gemmini comparison). Returns
        ``max_lines`` lines from ``start_line`` (1-based).
        """
        rel = REFERENCE.get(name)
        if rel is None:
            return f"Unknown reference; choose one of {sorted(REFERENCE)}."
        p = subprocess.run(["git", "show", f"HEAD:{rel}"], cwd=self.repo,
                           capture_output=True, text=True)
        lines = p.stdout.splitlines()
        start = max(1, int(start_line))
        chunk = lines[start - 1: start - 1 + max(1, min(int(max_lines), 1200))]
        return (f"{name}: lines {start}-{start + len(chunk) - 1} of {len(lines)}\n"
                + "\n".join(chunk))

    def apply_spec_patch(self, patch: str) -> str:
        """Apply a unified diff touching only microarch_isa.py and/or isa_dsl.py.

        Use git-style headers ``--- a/microarch_isa.py`` / ``+++ b/microarch_isa.py``
        (bare file names, no directories). Any other path is rejected: the
        evaluator, testbench, shapes, golden reference and Vitis settings are
        frozen.
        """
        headers = [l for l in patch.splitlines() if l.startswith(("--- ", "+++ "))]
        if not headers or len(headers) % 2:
            return "Rejected: patch needs paired git-style file headers."
        paths = []
        for old, new in zip(headers[::2], headers[1::2]):
            if not old.startswith("--- a/") or not new.startswith("+++ b/"):
                return "Rejected: use matching a/<file> and b/<file> headers."
            old_path = old.removeprefix("--- a/").split("\t")[0].strip()
            new_path = new.removeprefix("+++ b/").split("\t")[0].strip()
            if old_path != new_path or old_path not in self.sources:
                return ("Rejected: patches may touch only microarch_isa.py and "
                        "isa_dsl.py (bare names). Everything else is frozen.")
            paths.append(old_path)
        if len(paths) != len(set(paths)):
            return "Rejected: each writable file may appear only once per patch."
        with tempfile.TemporaryDirectory(prefix="tinytpu-isa-patch-") as tmp:
            sandbox = Path(tmp)
            for name, source in self.sources.items():
                shutil.copy2(source, sandbox / name)
            applied = subprocess.run(
                ["patch", "--batch", "--forward", "--no-backup-if-mismatch", "-p1"],
                input=patch, text=True, cwd=sandbox, capture_output=True)
            if applied.returncode:
                return f"Rejected: patch does not apply.\n{applied.stdout}{applied.stderr}"
            patched = {name: (sandbox / name).read_bytes() for name in paths}
            for name, content in patched.items():
                broken = self._check(name, content.decode("utf-8"))
                if broken:
                    return broken
            for name, content in patched.items():
                self.sources[name].write_bytes(content)
        return f"Patch applied to {', '.join(paths)}."

    def insert_after(self, path: str, anchor: str, content: str) -> str:
        """Insert text after the line containing one unique anchor.

        ``path`` must be microarch_isa.py or isa_dsl.py. Prefer a unified diff
        when a change must update both files together.
        """
        target = self.sources.get(path)
        if target is None:
            return "Rejected: path must be microarch_isa.py or isa_dsl.py."
        source = target.read_text(encoding="utf-8")
        if source.count(anchor) != 1:
            return f"Rejected: anchor must occur exactly once in {path}."
        cut = source.index(anchor) + len(anchor)
        line_end = source.find("\n", cut)
        cut = len(source) if line_end == -1 else line_end + 1
        updated = source[:cut] + content + source[cut:]
        broken = self._check(path, updated)
        if broken:
            return broken
        target.write_text(updated, encoding="utf-8")
        return f"Content inserted into {path}."

    def run_functional_check(self) -> str:
        """The gate, in ~10 s: bench_isa.py must print ALL EXACT and stress.py
        (full-range operands, sentinel-filled C, extra shapes) must pass.
        Functional (Allo simulator), not RTL. Run this before score_cycles."""
        return json.dumps(self.evaluate(gate_only=True), indent=1)

    def score_cycles(self) -> str:
        """The objective, in ~2-4 min: gate, then Vitis HLS csynth + RTL C/RTL
        cosim at 4x4x4 and 16x16x16, each testbench bit-exact against numpy.
        Score = sum of cosim cycles (lower is better). Also reports the csynth
        clock estimate (must meet 3.33 ns) and area (recorded, not scored)."""
        return json.dumps(self.evaluate(), indent=1)
