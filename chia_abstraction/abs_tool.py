# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The MCP surface for both dispositions. FROZEN to the agent.

The design-level loop's tool hands the agent two files in a private directory.
Here the editable surface is a set of PATHS in a checkout, so the agent gets a
private git worktree of `FROZEN_REF` (`<run>/<worker>/tree`) with its own
`mlir/build`, and every edit tool enforces `patch_policy`'s allowlist for the
worker's disposition before it writes. The candidate handed to the evaluator is
`git diff` of that worktree, applied into a SEPARATE evaluation slot -- so the
agent's tree is never what is measured, and a tree the agent left dirty in some
other way cannot become part of the score.

Why the agent gets a build directory at all: a `maintaining` candidate edits
C++, and a compiler diagnostic two minutes after the edit is the difference
between a candidate and a wasted session. `build_allo` is the cheapest tool and
the agent is told to use it first.

    read_source        any editable file, by path, with line numbers
    read_reference     a frozen file for context, from git
    replace_text       the preferred edit: exact, unique, allowlisted
    insert_after       insert after a unique anchor line
    apply_patch        a unified diff over allowlisted paths
    revert             undo everything and start again from FROZEN_REF
    build_allo         ninja + import check in the agent's own tree (~30 s)
    check_policy       the path/line/primitive policy on the current diff, free
    run_gates          the cheap correctness gates (no PPA), ~5 min
    score              the whole ladder including RTL cosim and csynth, ~8 min
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import threading
from pathlib import Path

import ray
from chia.base.tools.ChiaTool import ChiaTool

import patch_policy

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
DESIGN_PKG = "examples/accelerator/tinytpu_vitis"

#: As in the design loop: CHIA binds a tool server to the node's routable
#: address; the default here is loopback.
_TOOL_HOST = os.environ.get("CHIA_TOOL_HOST", "127.0.0.1")
if _TOOL_HOST != "node":
    ray.util.get_node_ip_address = lambda *a, **k: _TOOL_HOST

#: Read-only context, served from git at the ref. The pattern to copy, the
#: counter-example, the gates, and the register.
REFERENCE = {
    "customize.py": "allo/customize.py",
    "dataflow.py": "allo/dataflow.py",
    "builder.py": "allo/ir/builder.py",
    "infer.py": "allo/ir/infer.py",
    "ir_utils.py": "allo/ir/utils.py",
    "hls.py": "allo/backend/hls.py",
    "vitis.py": "allo/backend/vitis.py",
    "passes.py": "allo/passes.py",
    "EmitVivadoHLS.cpp": "mlir/lib/Translation/EmitVivadoHLS.cpp",
    "EmitBaseHLS.h": "mlir/lib/Translation/EmitBaseHLS.h",
    "EmitCatapultHLS.cpp": "mlir/lib/Translation/EmitCatapultHLS.cpp",
    "Utils.cpp": "mlir/lib/Support/Utils.cpp",
    "test_vhls.py": "tests/test_vhls.py",
    "microarch_isa.py": f"{DESIGN_PKG}/microarch_isa.py",
    "isa_dsl.py": f"{DESIGN_PKG}/isa_dsl.py",
    "cosim.py": f"{DESIGN_PKG}/cosim.py",
    "stress_isa.py": f"{DESIGN_PKG}/stress_isa.py",
    "isa_ref.py": f"{DESIGN_PKG}/isa_ref.py",
    "design_cases.py": "chia_abstraction/design_cases.py",
    "objective.py": "chia_abstraction/objective.py",
    "patch_policy.py": "chia_abstraction/patch_policy.py",
    "limitations.rst": "docs/source/developer/limitations.rst",
    "tinytpu_isa.rst": "docs/source/designs/tinytpu_isa.rst",
    "tinytpu_history.rst": "docs/source/designs/tinytpu_history.rst",
}


class AlloCompilerTool(ChiaTool):
    """Edit Allo (or the design), build it, and have the harness measure it."""

    def setup(self, tree: str, work_dir: str, agent_dir: str, repo: str,
              disposition: str, workload: str, ref: str, allo_python: str,
              llvm_build_dir: str) -> None:
        self.tree = Path(tree).resolve()
        self.work_dir = Path(work_dir).resolve()
        self.agent_dir = str(Path(agent_dir).resolve())
        self.repo = str(Path(repo).resolve())
        self.disposition = disposition
        self.workload = workload
        self.ref = ref
        self.allo_python = allo_python
        self.eval_env = {"TINYTPU_ALLO_PYTHON": allo_python,
                         "LLVM_BUILD_DIR": llvm_build_dir}
        self._lock = threading.Lock()
        self._slot = int(os.environ.get("CHIA_ABS_SLOT", "0"))
        # Pilots do NOT get the limitations register or the design history:
        # the first night's gap inventory came from the register, which
        # catalogues tooling defects, and aimed the agent at tooling.
        self.pilot = os.environ.get("CHIA_ABS_PILOT", "")
        if self.pilot:
            for k in ("limitations.rst", "tinytpu_history.rst", "tinytpu_isa.rst"):
                REFERENCE.pop(k, None)
        for name, fn in (
                ("read_source", self.read_source),
                ("read_reference", self.read_reference),
                ("replace_text", self.replace_text),
                ("insert_after", self.insert_after),
                ("apply_patch", self.apply_patch),
                ("revert", self.revert),
                ("check_policy", self.check_policy),
                ("build_allo", self.build_allo),
                ("run_gates", self.run_gates),
                ("score", self.score),
                ("list_probes", self.list_probes),
                ("declare_probe", self.declare_probe)):
            self.mcp.add_tool(fn, name=f"{self.name}_{name}")

    def __getstate__(self):
        state = dict(super().__getstate__())
        state["_lock"] = None
        return state

    def __setstate__(self, state):
        super().__setstate__(state)
        self._lock = threading.Lock()

    # -- not MCP tools: the harness decides, not the agent -------------------
    def diff(self) -> str:
        p = subprocess.run(["git", "diff", "HEAD", "--", *self._allowed_roots()],
                           cwd=self.tree, capture_output=True, text=True)
        return p.stdout

    def reset(self) -> None:
        subprocess.run(["git", "checkout", "-f", self.ref], cwd=self.tree,
                       capture_output=True)
        subprocess.run(["git", "clean", "-xffd", "-e", "/mlir/build"],
                       cwd=self.tree, capture_output=True)

    def apply(self, diff: str) -> None:
        """Put a known diff back into the tree (the loop's rewind)."""
        self.reset()
        if diff.strip():
            subprocess.run(["git", "apply", "-"], cwd=self.tree, input=diff,
                           text=True, capture_output=True, check=True)

    def _allowed_roots(self):
        return sorted({g.split("*")[0].rstrip("/") or "."
                       for g in patch_policy.EDITABLE[self.disposition]})

    def _resolve(self, path: str) -> tuple[Path | None, str]:
        p = path.strip().lstrip("./")
        bad = patch_policy.path_violations(self.disposition, [p])
        if bad:
            return None, ("Rejected: " + bad[0] + f". Editable in the "
                          f"`{self.disposition}` disposition: "
                          + ", ".join(patch_policy.EDITABLE[self.disposition]))
        target = (self.tree / p).resolve()
        if not str(target).startswith(str(self.tree)):
            return None, "Rejected: the path leaves the tree."
        return target, ""

    def _policy_now(self) -> list[str]:
        d = self.diff()
        if not d.strip():
            return []
        cust = self.tree / "allo/customize.py"
        problems, _ = patch_policy.check(
            self.disposition, d,
            cust.read_text() if cust.is_file() else None)
        return problems

    # -- reading -------------------------------------------------------------
    def read_source(self, path: str, start_line: int = 1,
                    max_lines: int = 300) -> str:
        """Read one editable file from YOUR tree, with line numbers.

        `path` is repository-relative, e.g. `allo/customize.py` or
        `mlir/lib/Translation/EmitVivadoHLS.cpp`. Returns `max_lines` lines
        from `start_line` (1-based). Use `read_reference` for a frozen file.
        """
        target, err = self._resolve(path)
        if target is None:
            return err
        if not target.is_file():
            return f"Rejected: {path} does not exist in the tree."
        lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
        start = max(1, int(start_line))
        n = max(1, min(int(max_lines), 1200))
        chunk = lines[start - 1:start - 1 + n]
        body = "\n".join(f"{start + i:6d}  {l}" for i, l in enumerate(chunk))
        return f"{path}: lines {start}-{start + len(chunk) - 1} of {len(lines)}\n{body}"

    def read_reference(self, name: str, start_line: int = 1,
                       max_lines: int = 300) -> str:
        """Read a FROZEN file for context, from git. You cannot edit these.

        Names: the compiler you may edit is also readable here at its
        UNMODIFIED state (customize.py, dataflow.py, builder.py, infer.py,
        ir_utils.py, hls.py, vitis.py, passes.py, EmitVivadoHLS.cpp,
        EmitBaseHLS.h, EmitCatapultHLS.cpp, Utils.cpp); the pattern's tests
        (test_vhls.py); the design cases you are scored on (microarch_isa.py,
        isa_dsl.py, design_cases.py) and their gates (cosim.py, stress_isa.py,
        isa_ref.py); the objective (objective.py) and the policy
        (patch_policy.py); and the register of measured limitations
        (limitations.rst) with the design's history (tinytpu_isa.rst,
        tinytpu_history.rst).
        """
        rel = REFERENCE.get(name)
        if rel is None:
            return f"Unknown reference; choose one of {sorted(REFERENCE)}."
        p = subprocess.run(["git", "show", f"{self.ref}:{rel}"], cwd=self.repo,
                           capture_output=True, text=True)
        lines = p.stdout.splitlines()
        start = max(1, int(start_line))
        n = max(1, min(int(max_lines), 1200))
        chunk = lines[start - 1:start - 1 + n]
        body = "\n".join(f"{start + i:6d}  {l}" for i, l in enumerate(chunk))
        return (f"{name} ({rel}): lines {start}-{start + len(chunk) - 1} of "
                f"{len(lines)}\n{body}")

    # -- editing -------------------------------------------------------------
    def _write_checked(self, target: Path, text: str, path: str, what: str) -> str:
        before = target.read_text(encoding="utf-8")
        target.write_text(text, encoding="utf-8")
        problems = self._policy_now()
        if problems:
            target.write_text(before, encoding="utf-8")
            return ("Rejected: the edit is outside what a candidate may "
                    "contain -- " + "; ".join(problems) +
                    ". The file is unchanged.")
        return f"{what} in {path}."

    def replace_text(self, path: str, old: str, new: str) -> str:
        """Replace one exact occurrence of `old` with `new`. The preferred edit.

        `old` must occur exactly once in `path` -- include enough surrounding
        lines, with exact indentation, to make it unique. Far more reliable
        than a unified diff. Nothing is written if the policy refuses the
        result. After editing C++, call `build_allo`.
        """
        target, err = self._resolve(path)
        if target is None:
            return err
        if not target.is_file():
            return f"Rejected: {path} does not exist in the tree."
        src = target.read_text(encoding="utf-8")
        n = src.count(old) if old else 0
        if n != 1:
            return (f"Rejected: `old` occurs {n} times in {path}; it must "
                    f"occur exactly once. The file is unchanged.")
        return self._write_checked(target, src.replace(old, new, 1), path,
                                   "Replaced 1 occurrence")

    def insert_after(self, path: str, anchor: str, content: str) -> str:
        """Insert `content` after the line containing the unique `anchor`."""
        target, err = self._resolve(path)
        if target is None:
            return err
        if not target.is_file():
            return f"Rejected: {path} does not exist in the tree."
        src = target.read_text(encoding="utf-8")
        if src.count(anchor) != 1:
            return f"Rejected: anchor must occur exactly once in {path}."
        cut = src.index(anchor) + len(anchor)
        nl = src.find("\n", cut)
        cut = len(src) if nl == -1 else nl + 1
        return self._write_checked(target, src[:cut] + content + src[cut:],
                                   path, "Content inserted")

    def apply_patch(self, patch: str) -> str:
        """Apply a unified diff. Use repository-relative `a/`/`b/` headers.

        Prefer `replace_text`: on this project unified diffs were the agents'
        main failure mode. Any path outside the allowlist is rejected, and the
        whole patch is rolled back if the result fails the policy.
        """
        try:
            paths = patch_policy.paths_in_diff(patch)
        except patch_policy.PatchError as e:
            return f"Rejected: {e}."
        bad = patch_policy.path_violations(self.disposition, paths)
        if bad:
            return "Rejected: " + "; ".join(bad)
        before = self.diff()
        p = subprocess.run(["git", "apply", "-p1", "-"], cwd=self.tree,
                           input=patch, text=True, capture_output=True)
        if p.returncode:
            return (f"Rejected: patch does not apply.\n{p.stdout}{p.stderr}"
                    [:2000])
        problems = self._policy_now()
        if problems:
            self.apply(before)
            return ("Rejected: " + "; ".join(problems) +
                    ". The tree is unchanged.")
        return f"Patch applied to {sorted(paths)}."

    def revert(self) -> str:
        """Discard every edit and start again from the unmodified checkout."""
        self.reset()
        return f"Tree reset to {self.ref[:8]}. No edits remain."

    def check_policy(self) -> str:
        """The path/line/primitive policy on your current diff. Free, instant.

        Run this before `build_allo` if you are unsure whether a change is
        allowed. It also reports NON-BLOCKING hints, e.g. that you taught one
        emitter to read a new attribute and left the others silently ignoring
        it.
        """
        d = self.diff()
        if not d.strip():
            return "No changes in the tree."
        cust = self.tree / "allo/customize.py"
        problems, hints = patch_policy.check(
            self.disposition, d, cust.read_text() if cust.is_file() else None)
        files = sorted(patch_policy.paths_in_diff(d))
        return json.dumps({"touches": files, "blocking": problems,
                           "hints": hints,
                           "diff_lines": len(d.splitlines())}, indent=1)

    # -- building and measuring ---------------------------------------------
    def _build_sync(self) -> str:
        env = {k: v for k, v in os.environ.items() if not k.startswith("TPU_")}
        env.update(self.eval_env, PYTHONPATH=str(self.tree),
                   PYTHONDONTWRITEBYTECODE="1",
                   PATH=f"{Path(self.allo_python).parent}:{env['PATH']}",
                   OMP_NUM_THREADS="8")
        bld = self.tree / "mlir" / "build"
        if not (bld / "build.ninja").exists():
            p = subprocess.run(
                ["cmake", "-G", "Ninja", "-S", "mlir", "-B", "mlir/build",
                 f"-DMLIR_DIR={env['LLVM_BUILD_DIR']}/lib/cmake/mlir",
                 f"-DPython3_EXECUTABLE={self.allo_python}",
                 f"-DPython_EXECUTABLE={self.allo_python}",
                 "-DMLIR_BINDINGS_PYTHON_NB_DOMAIN=allo"],
                cwd=self.tree, env=env, capture_output=True, text=True,
                timeout=1200)
            if p.returncode:
                return f"cmake FAILED:\n{(p.stdout + p.stderr)[-4000:]}"
        p = subprocess.run(["ninja", "-C", "mlir/build", "-j", "48"],
                           cwd=self.tree, env=env, capture_output=True,
                           text=True, errors="replace", timeout=2400)
        if p.returncode:
            return ("BUILD FAILED. The compiler said:\n"
                    + (p.stdout + p.stderr)[-6000:])
        q = subprocess.run(
            [self.allo_python, "-c",
             "import os, allo, allo.dataflow, allo.customize, "
             "allo.backend.hls, allo.backend.vitis; "
             "print('IMPORT OK', os.path.realpath(allo.__file__))"],
            cwd="/", env=env, capture_output=True, text=True, errors="replace",
            timeout=600)
        tail = (p.stdout or "")[-1200:]
        if q.returncode:
            return ("BUILD OK but IMPORT FAILED:\n"
                    + (q.stdout + q.stderr)[-4000:])
        return f"BUILD OK.\n{tail}\n{q.stdout.strip()}"

    def _evaluate(self, tier: str, gate_only: bool) -> dict:
        d = self.diff()
        run_dir = self.work_dir / ("gates" if gate_only else "score")
        patch_file = self.work_dir / f"candidate-{'g' if gate_only else 's'}.diff"
        patch_file.parent.mkdir(parents=True, exist_ok=True)
        patch_file.write_text(d)
        # ALLO_PYTHON, not sys.executable: this method runs in the CHIA
        # environment (py3.10, no numpy, no allo) and `evaluate_abs.py` cannot
        # even import there. With sys.executable EVERY candidate failed at
        # stage `evaluator` and the agent got no harness feedback at all.
        cmd = [self.allo_python, str(Path(self.agent_dir) / "evaluate_abs.py"),
               "--disposition", self.disposition, "--workload", self.workload,
               "--slot", str(self._slot), "--out", str(run_dir),
               "--tier", tier]
        if d.strip():
            cmd += ["--patch", str(patch_file)]
        if gate_only:
            cmd += ["--gate-only"]
        probe_file = self.work_dir / "probe_calls.json"
        if probe_file.exists():
            cmd += ["--probe-calls", str(probe_file)]
        env = {k: v for k, v in os.environ.items() if not k.startswith("TPU_")}
        env.update(self.eval_env)
        with self._lock:                  # one Vitis project per tool
            p = subprocess.run(cmd, cwd=self.repo, env=env, text=True,
                               errors="replace", capture_output=True,
                               timeout=9000)
        out = (p.stdout + p.stderr).strip()
        for line in reversed(out.splitlines()):
            line = line.strip()
            if line.startswith("{") and line.endswith("}"):
                try:
                    v = json.loads(line)
                    break
                except json.JSONDecodeError:
                    continue
        else:
            v = {"ok": False, "stage": "evaluator"}
        if not v.get("ok"):
            v["detail"] = out[-6000:]
        return v

    def list_probes(self) -> str:
        """The harness-owned probe sites your new primitive can be called at.

        Each is a FROZEN design with a FROZEN schedule. You cannot edit either;
        you declare ONE call of your new method per site with declare_probe,
        and the harness applies it and records accepted / refused / crashed.
        Where a site has a measured right answer it is shown.
        """
        # probe_meta imports NOTHING: `probes` needs numpy and allo, which
        # this environment does not have, and importing it here made both
        # probe tools raise.
        import probe_meta
        return json.dumps(probe_meta.describe(), indent=1)

    def declare_probe(self, probe: str, call: str) -> str:
        """Declare the ONE call of your new method to apply at probe site `probe`.

        `call` must be exactly `s.<method>(<literals>)`, where <method> is a
        Schedule method YOUR PATCH ADDS and every argument is a literal (str,
        int, float, bool, None, tuple/list of those). It is a declaration, not
        code; anything else is refused when the harness runs it. Declaring
        again for the same site replaces the earlier call.
        """
        import probe_meta
        if probe not in probe_meta.PROBES:
            return (f"Rejected: unknown probe; one of "
                    f"{sorted(probe_meta.PROBES)}.")
        f = self.work_dir / "probe_calls.json"
        f.parent.mkdir(parents=True, exist_ok=True)
        calls = json.loads(f.read_text()) if f.exists() else {}
        calls[probe] = call
        f.write_text(json.dumps(calls, indent=1))
        return f"Declared for {probe}: {call}. Now declared: {json.dumps(calls)}"

    async def build_allo(self) -> str:
        """Rebuild the C++ bindings in YOUR tree and import Allo. ~30 s warm.

        The cheapest feedback there is, and the first thing to run after any
        C++ edit. A build failure comes back with the compiler's own
        diagnostic. This builds YOUR tree; the harness rebuilds independently
        in its own checkout when it measures you.
        """
        return await asyncio.to_thread(self._build_sync)

    async def run_gates(self) -> str:
        """The cheap correctness gates, ~5 min. No PPA. Run this before `score`.

        In order: policy, build, import, Allo's own test suites against main's
        measured baseline, TinyTPU-isa's functional gates, every design case
        bit-exact under csim against numpy, and `tests/limits/` verdicts.
        """
        v = await asyncio.to_thread(self._evaluate, "loop", True)
        return json.dumps(v, indent=1)

    async def score(self) -> str:
        """The whole ladder including PPA, ~8-10 min.

        Everything `run_gates` does, then RTL C/RTL cosim cycles for
        TinyTPU-isa at the workload's scored shapes and csynth
        latency/interval/area/clock for each synthesisable design case, then
        the objective: cycles scored, resources CONSTRAINED, per design case.
        The harness re-scores your final tree independently either way.
        """
        v = await asyncio.to_thread(self._evaluate, "loop", False)
        return json.dumps(v, indent=1)
