"""Narrow MCP tool surface for co-designing TinyTPU-isa (examples/tinytpu).

The agent may edit exactly the files `design.EDITABLE` names -- the instruction
set (`isa_spec.json`, the `isa_encoding.py` generated from it and the
`isa_ref.py` built on that), the eight units under `ip/units/`, the
architecture that wires them, the assembler, the programs, `microarch_isa.py`
(the parameter set, and the `CHIA_CONFIG` declaration that proposes the
configuration to be scored at) and `isa_dsl.py` (the program generator) -- and
only in a private spec directory, never in the repository. Everything that
decides the score is frozen and outside its reach; see `evaluate.py` for the
list and how it is enforced.

`isa_encoding.py` is GENERATED. Editing `isa_spec.json` by itself leaves it
stale and `gen_isa.py --conform` refuses the candidate, so there is a tool --
`regenerate_isa` -- that runs the FROZEN generator on the spec and writes the
artefact back. It is the only tool that writes a file the agent did not type.
"""

from __future__ import annotations

import ast
import asyncio
import difflib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

import ray
from chia.base.tools.ChiaTool import ChiaTool

from design import EDITABLE, UNITS
from spec_policy import doc_violations, policy_violations

#: CHIA binds each tool server to `ray.util.get_node_ip_address()`, which on a
#: Ray worker is the host's routable address -- an unauthenticated server
#: anyone on the network can call, able to edit the candidate and start Vitis
#: runs. So the DEFAULT is loopback. A multi-host swarm, where opencode and the
#: tool server sit on different nodes, must opt in explicitly with
#: TINYTPU_TOOL_HOST=node (Ray's routable address) or a specific address -- and
#: should then add authentication or a firewall, which this module does not.
#: This module is imported in the tool's actor before the server starts.
_TOOL_HOST = os.environ.get("TINYTPU_TOOL_HOST", "127.0.0.1")
if _TOOL_HOST != "node":
    ray.util.get_node_ip_address = lambda *args, **kwargs: _TOOL_HOST

#: Read-only context the agent may look at, served from git HEAD: the frozen
#: code that judges a candidate, then the design's documentation. The docs
#: pages replaced RESULTS_ISA.md / COMPARISON.md when main moved its notes into
#: the docs site; the reST is served as source.
REFERENCE = {
    "cosim.py": "examples/tinytpu/cosim.py",
    # The five scored shapes, which cosim.py and bench_isa.py now import from
    # here rather than each spelling out.
    "shapes.py": "examples/tinytpu/shapes.py",
    "bench_isa.py": "examples/tinytpu/bench_isa.py",
    "stress_isa.py": "examples/tinytpu/stress_isa.py",
    # `isa_ref.py` is no longer here: it is EDITABLE now, and `read_spec` is
    # how a writable file is read. `gen_isa.py` took its place, because it is
    # what an ISA change is held to.
    "gen_isa.py": "examples/tinytpu/gen_isa.py",
    "evaluate.py": "examples/tinytpu/chia_agent/evaluate.py",
    "param_check.py": "examples/tinytpu/chia_agent/param_check.py",
    # The co-design loop's frozen half. Readable on purpose: the agent should
    # be able to see exactly what the mapper enumerates, what it refuses, and
    # by what rule it picks -- it just cannot change any of it.
    "mapspace.py": "examples/tinytpu/chia_agent/mapspace.py",
    "codesign_gate.py": "examples/tinytpu/chia_agent/codesign_gate.py",
    "codesign_cosim.py": "examples/tinytpu/chia_agent/codesign_cosim.py",
    "act.rst": "docs/source/extensions/act.rst",
    "tinytpu_isa.rst": "docs/source/designs/tinytpu_isa.rst",
    "tinytpu_history.rst": "docs/source/designs/tinytpu_history.rst",
    "gemmini_comparison.rst": "docs/source/designs/gemmini_comparison.rst",
    "limitations.rst": "docs/source/developer/limitations.rst",
    # Measured per-process cosim timeline of the shipped design at 16x16x16.
    "timeline_16x16x16": "dev/records/tinytpu/chia-evidence/timeline-476a70d8-16x16x16/README.md",
    "timeline_16x16x16.txt": "dev/records/tinytpu/chia-evidence/timeline-476a70d8-16x16x16/timeline.txt",
    "timeline_16x16x16_rle.txt": "dev/records/tinytpu/chia-evidence/timeline-476a70d8-16x16x16/rle.txt",
}


class AlloSpecTool(ChiaTool):
    """Let an agent co-edit TinyTPU-isa's hardware and program generator only."""

    def setup(self, spec_dir: str, work_dir: str, agent_dir: str, repo: str,
              allo_python: str, llvm_build_dir: str,
              codesign: bool = False) -> None:
        #: Co-design mode: the evaluator also enumerates the mapspace against
        #: the candidate's hardware and cosims the nest the FROZEN mapper
        #: chose, rather than the program the agent hand-wrote.
        self.codesign = bool(codesign)
        self.spec_dir = Path(spec_dir).resolve()
        self.sources = {rel: self.spec_dir / rel for rel in EDITABLE}
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
        self._lock = threading.Lock()
        assert all(path.is_file() for path in self.sources.values()), self.sources
        self.mcp.add_tool(self.read_spec, name=f"{self.name}_read_spec")
        self.mcp.add_tool(self.read_reference, name=f"{self.name}_read_reference")
        self.mcp.add_tool(self.replace_text, name=f"{self.name}_replace_text")
        self.mcp.add_tool(self.apply_spec_patch, name=f"{self.name}_apply_spec_patch")
        self.mcp.add_tool(self.insert_after, name=f"{self.name}_insert_after")
        self.mcp.add_tool(self.regenerate_isa, name=f"{self.name}_regenerate_isa")
        self.mcp.add_tool(
            self.run_functional_check, name=f"{self.name}_run_functional_check")
        self.mcp.add_tool(self.score_cycles, name=f"{self.name}_score_cycles")
        if self.codesign:
            self.mcp.add_tool(self.mapspace_report,
                              name=f"{self.name}_mapspace_report")

    def __getstate__(self):
        # The tool is re-pickled on every prompt; a Lock cannot be.
        state = dict(super().__getstate__())
        state["_lock"] = None
        return state

    def __setstate__(self, state):
        # A fresh lock per unpickled copy, made HERE rather than lazily in
        # evaluate(): two MCP calls arrive on two threads at once, and a lazy
        # `if self._lock is None` lets both create a lock, so two evaluations
        # would share one work directory -- each wiping the other's.
        super().__setstate__(state)
        self._lock = threading.Lock()

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

    def evaluate(self, gate_only: bool = False, shapes: str | None = None,
                 work: str = "agent") -> dict:
        """Run the frozen evaluator on the spec dir; return its JSON verdict.

        `work` names a subdirectory of the work dir. The agent's MCP calls run
        in the tool's actor and the loop's own re-scoring runs in the driver --
        two copies of this object, two locks -- so they get separate
        directories: an agent evaluation still running after its opencode call
        timed out must not be wiped by the harness's next one."""
        cmd = [sys.executable, self.evaluator, "--spec-dir", str(self.spec_dir),
               "--work", str(self.work_dir / work)]
        if self.codesign:
            cmd.append("--codesign")
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
        # `isa_spec.json` is data, not a module: the policy parses it as JSON
        # and holds the expressions its actions carry -- the strings `isa_ref`
        # evaluates -- to the same expression language `isa_ref` admits.
        if not name.endswith(".json"):
            try:
                ast.parse(source, filename=name)
            except SyntaxError as error:
                return (f"Rejected: the edit leaves {name} unparseable -- "
                        f"{error.msg} at line {error.lineno}. The file is "
                        f"unchanged.")
        base = subprocess.run(
            ["git", "show", f"HEAD:examples/tinytpu/{name}"],
            cwd=self.repo, capture_output=True, text=True).stdout
        problems = policy_violations(name, source) + (
            doc_violations(name, base, source) if base else [])
        if problems:
            return ("Rejected: the edit is outside what a spec may contain -- "
                    + "; ".join(problems)
                    + ". Describe hardware; do not read, write, or execute. "
                    "The file is unchanged.")
        return None

    def read_spec(self, path: str = "") -> str:
        """Read one writable file, or list them all.

        With no argument: every writable path with its line count -- the eight
        units under `ip/units/`, the composition `ip/tinytpu.py`, the ISA
        `ip/isa.py`, the assembler, the programs, `microarch_isa.py` (the
        parameter set), `isa_dsl.py` (the program generator) and the
        instruction set itself -- `isa_spec.json` (the source of truth),
        `isa_encoding.py` (generated from it: change it with `regenerate_isa`,
        never by hand) and `isa_ref.py` (the reference model). With a path:
        that file. `allo/compose.py` and `ip/params.py` are frozen machinery
        and are not writable; read them with `read_reference`.
        """
        if not path:
            rows = [f"  {rel:34s} "
                    f"{len(p.read_text(encoding='utf-8').splitlines()):4d} lines"
                    + ("   (a unit)" if rel in UNITS else "")
                    for rel, p in self.sources.items()]
            return ("Writable files (read one with read_spec(path=...)):\n"
                    + "\n".join(rows))
        target = self.sources.get(path)
        if target is None:
            return f"Unknown path; writable files are {list(self.sources)}."
        return f"===== {path} =====\n{target.read_text(encoding='utf-8')}"

    def read_reference(self, name: str, start_line: int = 1,
                       max_lines: int = 400) -> str:
        """Read a FROZEN file for context (you cannot edit these).

        ``name`` is one of: mapspace.py (THE MAPPER -- its exhaustive
        enumerator, the refusal histogram, and the rule by which it picks the
        nest that gets cosimmed; read this first in co-design mode),
        codesign_gate.py / codesign_cosim.py (how the mapper's choice reaches
        the RTL), act.rst (why the mapper looks like this, and the measured
        refusal counts on the shipped design), cosim.py (the RTL cosim scorer and its testbench),
        bench_isa.py (the published-setup functional check), stress_isa.py
        (the correctness gate: 492 runs at 476a70d8, full-range operands, 64 shapes, whole
        C compared, vector and random programs), isa_ref.py (what each
        instruction means -- the reference stress_isa checks against),
        evaluate.py (how the score is computed), param_check.py (the
        parametricity gate: the design rebuilt at MAXDIM 8 and 12 must be exact), tinytpu_isa.rst (the design,
        its ISA, and how to verify a change), tinytpu_history.rst (what was
        tried, measured, and reverted), gemmini_comparison.rst (the Gemmini
        comparison and where the gap comes from), limitations.rst (Allo
        frontend/simulator limitations and workarounds), timeline_16x16x16 /
        timeline_16x16x16.txt / timeline_16x16x16_rle.txt (the shipped
        design's measured per-process cosim timeline at 16x16x16). Returns
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

    def replace_text(self, path: str, old: str, new: str) -> str:
        """Replace one exact occurrence of ``old`` with ``new`` in ``path``.

        The preferred edit. ``path`` is one of the writable paths
        ``read_spec()`` lists (e.g. ``ip/units/pe.py``); ``old``
        must occur exactly once (include enough surrounding lines to make it
        unique, with exact indentation). The result must parse and pass the
        spec policy, or nothing is written.
        """
        target = self.sources.get(path)
        if target is None:
            return f"Rejected: path must be one of {list(self.sources)}."
        source = target.read_text(encoding="utf-8")
        n = source.count(old) if old else 0
        if n != 1:
            return (f"Rejected: `old` occurs {n} times in {path}; it must occur "
                    f"exactly once. The file is unchanged.")
        updated = source.replace(old, new, 1)
        broken = self._check(path, updated)
        if broken:
            return broken
        target.write_text(updated, encoding="utf-8")
        return f"Replaced 1 occurrence in {path}."

    def apply_spec_patch(self, patch: str) -> str:
        """Apply a unified diff touching only the writable files.

        Use git-style headers ``--- a/<path>`` / ``+++ b/<path>`` with the
        paths ``read_spec()`` lists (e.g. ``ip/units/pe.py``). Any other path
        is rejected: the evaluator, testbench, shapes, golden reference, Vitis
        settings and the composition machinery are frozen.
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
                return (f"Rejected: patches may touch only "
                        f"{list(self.sources)}. Everything else is frozen.")
            paths.append(old_path)
        if len(paths) != len(set(paths)):
            return "Rejected: each writable file may appear only once per patch."
        with tempfile.TemporaryDirectory(prefix="tinytpu-isa-patch-") as tmp:
            sandbox = Path(tmp)
            for name, source in self.sources.items():
                (sandbox / name).parent.mkdir(parents=True, exist_ok=True)
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

        ``path`` is one of the writable paths ``read_spec()`` lists. Prefer a
        unified diff when a change must update several files together.
        """
        target = self.sources.get(path)
        if target is None:
            return f"Rejected: path must be one of {list(self.sources)}."
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

    # The two evaluator tools are async and push the work to a thread: a sync
    # tool runs on the MCP server's event loop, and in the smoke run one hung
    # evaluation blocked every other request -- including tool listing for
    # the next session, which then reported that no tools existed.
    def regenerate_isa(self) -> str:
        """Regenerate `isa_encoding.py` from your `isa_spec.json`.

        Run this after every edit to `isa_spec.json`. `isa_encoding.py` is
        generated from the spec, and `run_functional_check` refuses a candidate
        whose generated module is not byte-identical to what its own spec
        produces (`gen_isa.py --conform`), so a spec edit on its own is always
        a refusal. The generator is FROZEN and is read from git, not from your
        spec directory: it writes what your spec says, and you cannot change
        what "says" means.

        Nothing else is touched, and the spec itself is never rewritten.
        """
        gen = subprocess.run(["git", "show", f"HEAD:examples/tinytpu/gen_isa.py"],
                             cwd=self.repo, capture_output=True, text=True)
        if gen.returncode:
            return f"Rejected: cannot read the frozen generator ({gen.stderr[:200]})."
        encoding = self.sources["isa_encoding.py"]
        with tempfile.TemporaryDirectory(prefix="tinytpu-isa-gen-") as tmp:
            # The whole spec directory, because `gen_isa.py` lists
            # `ip/units/` at import time -- a sandbox holding only the spec
            # raises before it reads a line of it.
            sandbox = Path(tmp) / "tinytpu"
            shutil.copytree(self.spec_dir, sandbox)
            (sandbox / "gen_isa.py").write_text(gen.stdout, encoding="utf-8")
            # `--write --no-doc` reads the spec and writes one file. It imports
            # nothing of the design, so it is fast and cannot run the
            # candidate's code.
            done = subprocess.run([sys.executable, "gen_isa.py", "--write",
                                   "--no-doc"], cwd=sandbox, text=True,
                                  capture_output=True, timeout=300)
            if done.returncode:
                return ("Rejected: the generator could not read your spec.\n"
                        + (done.stdout + done.stderr)[-2000:])
            produced = (sandbox / "isa_encoding.py").read_text(encoding="utf-8")
        before = encoding.read_text(encoding="utf-8")
        if produced == before:
            return ("isa_encoding.py already matches isa_spec.json; nothing "
                    "was written.")
        broken = self._check("isa_encoding.py", produced)
        if broken:
            return broken
        encoding.write_text(produced, encoding="utf-8")
        added = sum(1 for line in difflib.unified_diff(
            before.splitlines(), produced.splitlines()) if line.startswith("+"))
        removed = sum(1 for line in difflib.unified_diff(
            before.splitlines(), produced.splitlines()) if line.startswith("-"))
        return (f"Regenerated isa_encoding.py from isa_spec.json "
                f"(+{added} / -{removed} lines).")

    async def run_functional_check(self) -> str:
        """The gate, in ~15 s: bench_isa.py (the published [-4, 4] setup) and
        stress_isa.py (492 runs: full-range/corner/boundary int8, all 64
        shapes, C prefilled and compared in full, vector and random programs,
        many calls on one build) must both pass.
        Functional (Allo simulator), not RTL. A deadlocked dataflow fails after
        4 minutes. Run this before score_cycles."""
        verdict = await asyncio.to_thread(self.evaluate, True)
        return json.dumps(verdict, indent=1)

    async def score_cycles(self) -> str:
        """The objective, in ~2-4 min: gate, then Vitis HLS csynth + RTL C/RTL
        cosim at 4x4x4 and 16x16x16, each testbench bit-exact against numpy.

        In co-design mode the program under the RTL is the best nest the FROZEN
        mapper can encode on your hardware, chosen by exhaustive enumeration --
        not the program you hand-wrote. The verdict reports a PAIR and never
        collapses it: `cycles` per shape from the cosim, and `resources` (FF,
        LUT, BRAM18K, DSP, URAM and the estimated clock, which must meet
        3.33 ns) from the csynth of the same build. Buying cycles with block RAM
        is a trade, not a win, and it is reported as one. `mapspace` records how
        many nests your hardware made encodable and which constraint refused
        the rest."""
        verdict = await asyncio.to_thread(self.evaluate)
        return json.dumps(verdict, indent=1)

    async def mapspace_report(self) -> str:
        """The co-design signal, in ~30 s and no Vitis: the gate plus the
        exhaustive mapspace enumeration against your hardware.

        Reports, per scored shape, how many of the enumerated loop nests this
        hardware can ENCODE, which constraint refused each of the rest, and
        which nest the frozen mapper would choose. Use it to see whether a
        hardware change actually widened what the machine can say, before
        paying for a cosim. It is a count of nests, not a cycle count: nothing
        here is a performance number."""
        verdict = await asyncio.to_thread(self.evaluate, True)
        out = {k: verdict.get(k) for k in ("ok", "stage", "mapspace", "gate")}
        if not verdict.get("ok"):
            out["detail"] = verdict.get("detail", "")[-4000:]
        return json.dumps(out, indent=1)
