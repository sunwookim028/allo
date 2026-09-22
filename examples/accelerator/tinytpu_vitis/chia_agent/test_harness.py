"""LLM-free end-to-end test of the CHIA harness. Run it before any paid search.

Nothing here calls a real model. Two scripted "agents" drive the real harness:

* **mcp** phases: an MCP client (the `mcp` package, streamable HTTP -- the
  transport opencode uses) talking to the real `AlloSpecTool` server, hosted by
  CHIA on a Ray actor exactly as in a search.
* **loop** phase: the real `swarm.py -> loop.py -> opencode` path, with opencode
  pointed at `fake_model.py`, an OpenAI-compatible endpoint that replays tool
  calls. This covers what the MCP client cannot: opencode's own MCP client and
  its request timeout, CHIA's `OpenCodeLLM`, the loop's keep-or-rewind logic,
  `variants.jsonl`, `best.diff`, `summary.json` and the spend accounting.

Cases, each with an expected verdict:

  a  no-op (re-save the unmodified design)   scores exactly 172 / 686 (main @ 476a70d8)
  b  `spad ... = 0` (part of b4be2b10 reverted) bit-exact, scored WORSE
  c  PE partial sum narrowed to int16          passes bench_isa, REJECTED by
                                               stress_isa
  d  one stream put dropped (deadlock)         REJECTED by the 240 s gate timeout,
                                               no process left behind
  e  frozen-file / import-time attacks         every one refused, including a
                                               forged verdict line (SystemExit)
                                               and a numpy monkeypatch the
                                               static policy cannot see
  f  concurrent evaluations                    server answers in seconds, no
                                               cross-contamination
  g  parametricity / documentation guards     literal T/MAXDIM and a deleted
                                               docstring refused by the edit
                                               tools and the evaluator; a unit
                                               specialised to MAXDIM=16 REJECTED
                                               at gate:param
  control  the run's measured control          a forged or foreign control
                                               record refused; an unrecorded
                                               design still cross-checked
                                               against the published numbers
  accept  accept.py on (b)                     bit-exact at all five shapes, a
                                               control measured in the same run
                                               before the diff, and reported as
                                               NOT a win

    conda activate chia_env
    python test_harness.py                       # everything, ~45 min
    python test_harness.py --phases control,e,c,g   # cheap subset, ~3 min
    python test_harness.py --phases e,c,d,abf,loop,accept

Writes `<run-dir>/results.json` and exits non-zero if any case failed.
"""

from __future__ import annotations

import argparse
import ast
import asyncio
import difflib
import json
import os
import re
import subprocess
import sys
import logging
import time
from datetime import timedelta
from pathlib import Path

AGENT_DIR = Path(__file__).resolve().parent
REPO = AGENT_DIR.parents[3]
PKG = "examples/accelerator/tinytpu_vitis"
sys.path.insert(0, str(AGENT_DIR))

ALLO_PYTHON = os.environ.setdefault(
    "TINYTPU_ALLO_PYTHON", "/home/sk3463/miniconda3/envs/allo/bin/python")
LLVM_BUILD_DIR = os.environ.setdefault(
    "LLVM_BUILD_DIR", "/home/sk3463/llvm-allo-6b09f739/build")
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ["PATH"] = f"{AGENT_DIR / 'node_modules/.bin'}:{os.environ['PATH']}"
# No path in this test may reach a paid model: drop every cloud credential
# variable, and the loop phase names a model that exists only on localhost.
for _k in list(os.environ):
    if _k.startswith(("GOOGLE_", "VERTEX", "CLOUDSDK_", "TINYTPU_OPENCODE_",
                      "TINYTPU_VERTEX_", "ANTHROPIC_", "OPENAI_")):
        del os.environ[_k]
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/dev/null"
# Tool servers on loopback (see allo_tool.py); inherited by Ray's actors.
os.environ["TINYTPU_TOOL_HOST"] = "127.0.0.1"

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("mcp").setLevel(logging.WARNING)

#: The five benchmark shapes have one definition (`{PKG}/shapes.py`);
#: `evaluate` loads it by path and this takes the names from there rather than
#: writing them out a sixth time. The cycles are positional against it.
import control  # noqa: E402
from evaluate import ALL_SHAPES  # noqa: E402

#: The unmodified design (README, accept.py control run).
BASELINE_ALL = dict(zip(ALL_SHAPES, (172, 262, 418, 484, 686)))
BASELINE = {s: BASELINE_ALL[s] for s in ("4x4x4", "16x16x16")}
GATE_TIMEOUT = 240


def genuine_control(**changes) -> dict:
    """A control record as `accept.py` measures one for the design at HEAD;
    `changes` makes a forgery out of it."""
    return {**control.record(
        cycles=dict(BASELINE_ALL), design=control.blobs("HEAD"), ref="HEAD",
        estimated_ns=2.431, seconds=284.0, vouched=True, pristine_tree=True,
        driver="cosim",
        source="measured in this run from git at HEAD, before the candidate "
               "diff was applied"), **changes}

#: (file, old, new): exact, unique replacements on the HEAD design.
MUTANTS = {
    # b: the zero-fill of spad that b4be2b10 removed. At HEAD the DMA burst is
    # II=1, so the 514-cycle memset is no longer hidden behind it.
    "spad_zero": ("microarch_isa.py", "        spad: UInt(VW)[SPAD_ROWS]\n",
                  "        spad: UInt(VW)[SPAD_ROWS] = 0\n"),
    # c: [-4, 4] operands never overflow 16 bits at K <= 16; full-range ones do.
    "narrow16": ("microarch_isa.py", "o: int32 = p + av * wv",
                 "o: int16 = p + av * wv"),
    # g: specialised to the scored MAXDIM: identical at MAXDIM=16, wrong at 8/12.
    # Passes the static policy (T/MAXDIM themselves stay parameters).
    "wpr_literal": ("microarch_isa.py",
                    "WPR = MAXDIM // T              # packed words per DRAM row\n",
                    "WPR = 4                        # packed words per DRAM row\n"),
    # d: mvout never reaches dma_st, so accu blocks on a full ac2sp.
    "deadlock": ("microarch_isa.py",
                 "                    c_acc.put(rw)\n                    c_dst.put(rw)\n",
                 "                    c_acc.put(rw)\n"),
}
ANCHOR = "import allo.dataflow as df\n"
#: Prints both gates' verdict lines and exits 0 during the candidate's import
#: in any gate process (they put the tree root first on sys.path).
FORGE_VERDICT = (
    "import sys\n"
    "if not sys.path[0].endswith('tinytpu_vitis'):\n"
    "    print('  STRESS OK: 486/486 runs exact (forged)')\n"
    "    print('  STRESS OK: 60/60 runs exact (forged)')\n"
    "    print('  ALL EXACT')\n"
    "    raise SystemExit(0)\n")
#: Monkeypatches numpy's RNG through a method's `self`, which the static policy
#: cannot trace (`self` is a parameter). Only gate_runner's frozen modules stop it.
METHOD_SELF_PATCH = (
    "import numpy as _np\n"
    "class _P:\n"
    "    def go(self, v):\n"
    "        self.default_rng = v\n"
    "_orig = _np.random.default_rng\n"
    "def _four(seed=None):\n"
    "    return _orig(0)\n"
    "_P.go(_np.random, _four)\n")
#: e: import-time payloads. Each must be refused by the edit tools.
IMPORT_ATTACKS = {
    "import cosim": "import cosim\n",
    "from examples ... import cosim":
        "from examples.accelerator.tinytpu_vitis import cosim as _c\n",
    "import subprocess": "import subprocess\n",
    "open() write": "open('/tmp/pwn', 'w').write('x')\n",
    "__import__": "__import__('subprocess').run(['true'])\n",
    "sys.modules": "import sys\n_m = sys.modules\n",
    # Rewrites the frozen stress gate in the evaluation tree at import time, so
    # a narrowed datapath would sail through. Uses only allowed imports.
    "numpy.savetxt over stress_isa.py":
        "import numpy as _np\n"
        "_np.savetxt(os.path.join(os.path.dirname(__file__),"
        " 'stress_isa.py'), [], header=\"print('  STRESS OK: forged')\", comments='')\n",
    "ndarray.tofile over stress_isa.py":
        "import numpy as _np\n"
        "_np.frombuffer(b\"print('  STRESS OK: forged')\\n\", dtype=_np.uint8).tofile("
        "os.path.join(os.path.dirname(__file__), 'stress_isa.py'))\n",
    # Prints the verdict lines and exits 0 before the real check runs. Against
    # the old stdout-reading gate this passed with an int16 datapath (see
    # phase e's "forged verdict" case, which runs it past the policy).
    "SystemExit forged verdict": FORGE_VERDICT,
    # The golden reference's RNG, swapped for one that only draws what the
    # published [-4, 4] testbench draws.
    "monkeypatch numpy RNG via an alias":
        "import numpy as _np\n_r = _np.random\n_r.default_rng = lambda *a, **k: None\n",
    # Frame walking toward the gate runner's nonce.
    "frame walk": "def _g():\n    yield 1\n_f = _g().gi_frame\n",
}


def head(name: str) -> str:
    return subprocess.run(["git", "show", f"HEAD:{PKG}/{name}"], cwd=REPO,
                          capture_output=True, text=True, check=True).stdout


def mutate(name: str) -> str:
    f, old, new = MUTANTS[name]
    text = head(f)
    assert text.count(old) == 1, name
    return text.replace(old, new, 1)


def unified(name: str, before: str, after: str) -> str:
    return "".join(difflib.unified_diff(before.splitlines(keepends=True),
                                        after.splitlines(keepends=True),
                                        f"a/{name}", f"b/{name}"))


# -- results ----------------------------------------------------------------
RESULTS: list[dict] = []


def check(case: str, expected: str, actual, passed: bool, **extra):
    RESULTS.append({"case": case, "expected": expected, "actual": actual,
                    "pass": bool(passed), **extra})
    mark = "PASS" if passed else "FAIL"
    print(f"  [{mark}] {case}: expected {expected}; got {actual}", flush=True)


# -- process hygiene ----------------------------------------------------------
def _ancestors() -> set[int]:
    pids, pid = set(), os.getpid()
    while pid > 1:
        pids.add(pid)
        try:
            stat = Path(f"/proc/{pid}/stat").read_text()
        except OSError:
            break
        pid = int(stat.rsplit(")", 1)[1].split()[1])
    return pids


def procs_under(path: Path) -> list[str]:
    """Live processes (not this test or its parents) whose command line or cwd
    is inside `path`."""
    hits, needle, mine = [], str(path), _ancestors()
    for d in Path("/proc").iterdir():
        if not d.name.isdigit() or int(d.name) in mine:
            continue
        try:
            args = (d / "cmdline").read_bytes().replace(b"\0", b" ").decode(
                errors="replace")
            cwd = os.readlink(d / "cwd")
        except OSError:
            continue
        if "test_harness.py" in args:   # this test's own shell / siblings
            continue
        if needle in args or cwd.startswith(needle):
            hits.append(f"{d.name}: {args[:160]}")
    return hits


# -- MCP client -------------------------------------------------------------
class Client:
    """One scripted agent. A fresh MCP session per call, as opencode opens them."""

    def __init__(self, tool):
        self.url = f"http://{tool.hostname}:{tool.port}/{tool.name}/mcp"
        self.prefix = tool.name

    async def _session(self, fn, timeout):
        from mcp import ClientSession
        from mcp.client.streamable_http import streamablehttp_client
        async with streamablehttp_client(self.url, timeout=30,
                                         sse_read_timeout=timeout) as (r, w, _):
            async with ClientSession(r, w, read_timeout_seconds=timedelta(
                    seconds=timeout)) as s:
                await s.initialize()
                return await fn(s)

    async def tools(self) -> list[str]:
        async def fn(s):
            return [t.name for t in (await s.list_tools()).tools]
        return await self._session(fn, 60)

    async def call(self, tool_name: str, timeout: float = 2400, **args) -> str:
        async def fn(s):
            res = await s.call_tool(f"{self.prefix}_{tool_name}", args)
            return "".join(getattr(c, "text", "") for c in res.content)
        return await self._session(fn, timeout)

    async def verdict(self, tool_name: str, **args) -> dict:
        return json.loads(await self.call(tool_name, **args))

    async def timed_read(self) -> float:
        t = time.time()
        await self.call("read_spec", timeout=60)
        return time.time() - t


async def while_polling(clients, coro, every=5.0):
    """Run `coro`; meanwhile time read_spec on every client. Returns (result, max_s)."""
    task = asyncio.ensure_future(coro)
    worst = 0.0
    while not task.done():
        lat = await asyncio.gather(*(c.timed_read() for c in clients))
        worst = max(worst, *lat)
        try:
            await asyncio.wait_for(asyncio.shield(task), timeout=every)
        except asyncio.TimeoutError:
            pass
    return task.result(), worst


# -- phases -------------------------------------------------------------------
class Suite:
    def __init__(self, run_dir: Path):
        import ray
        from allo_tool import AlloSpecTool
        from loop import seed_spec
        self.run_dir = run_dir
        # 127.0.0.1: the tool servers bind to Ray's node address, and on the
        # default (the host's public address) anyone on the network could call
        # replace_text / score_cycles.
        ray.init(address="local", _node_ip_address="127.0.0.1",
                 resources={"opencode_creds": 4}, include_dashboard=False,
                 runtime_env={"working_dir": str(AGENT_DIR)}, log_to_driver=False)
        self.gcs = ray.get_runtime_context().gcs_address
        self.tools, self.specs, self.works = {}, {}, {}
        for name in ("tpta", "tptb"):
            spec = run_dir / name / "spec"
            seed_spec(spec)
            work = REPO / ".chia_scratch" / run_dir.name / name
            self.tools[name] = AlloSpecTool(name, str(spec), str(work), str(AGENT_DIR),
                                            str(REPO), ALLO_PYTHON, LLVM_BUILD_DIR)
            self.specs[name], self.works[name] = spec, work
        self.A, self.B = Client(self.tools["tpta"]), Client(self.tools["tptb"])

    def reset(self, name: str):
        for f in ("microarch_isa.py", "isa_dsl.py"):
            (self.specs[name] / f).write_text(head(f))

    def spec(self, name: str, f="microarch_isa.py") -> str:
        return (self.specs[name] / f).read_text()

    def close(self):
        import ray
        for t in self.tools.values():
            try:
                t.stop()
            except Exception:
                pass
        ray.shutdown()

    # e ------------------------------------------------------------------
    async def phase_e(self):
        print("== e: frozen-file and import-time attacks", flush=True)
        A, spec = self.A, self.specs["tpta"]
        self.reset("tpta")
        names = await A.tools()
        want = {f"tpta_{n}" for n in ("read_spec", "read_reference", "replace_text",
                                      "apply_spec_patch", "insert_after",
                                      "run_functional_check", "score_cycles")}
        check("e.tool-surface", "exactly the 7 spec tools", sorted(names),
              set(names) == want)

        cosim = head("cosim.py")
        forged = cosim.replace("SHAPES", "SHAPES_", 1)
        micro = head("microarch_isa.py")
        attacks = {
            "patch cosim.py": ("apply_spec_patch",
                               {"patch": unified("cosim.py", cosim, forged)}),
            "patch microarch_isa.py + cosim.py": ("apply_spec_patch", {"patch":
                unified("microarch_isa.py", micro, mutate("spad_zero"))
                + unified("cosim.py", cosim, forged)}),
            "patch ../cosim.py": ("apply_spec_patch", {"patch": unified(
                "microarch_isa.py", micro, mutate("spad_zero")).replace(
                "a/microarch_isa.py", "a/../cosim.py").replace(
                "b/microarch_isa.py", "b/../cosim.py")}),
            "replace_text cosim.py": ("replace_text", {"path": "cosim.py",
                                                        "old": "SHAPES", "new": "X"}),
            "replace_text ../cosim.py": ("replace_text", {"path": "../cosim.py",
                                                           "old": "SHAPES", "new": "X"}),
            "replace_text absolute repo path": ("replace_text", {
                "path": str(REPO / PKG / "microarch_isa.py"),
                "old": ANCHOR, "new": ANCHOR + "X = 1\n"}),
            "insert_after bench_isa.py": ("insert_after", {
                "path": "bench_isa.py", "anchor": "import", "content": "x=1\n"}),
        }
        for label, payload in IMPORT_ATTACKS.items():
            attacks[f"import-time: {label}"] = ("replace_text", {
                "path": "microarch_isa.py", "old": ANCHOR, "new": ANCHOR + payload})
        for label, (tool, args) in attacks.items():
            before = self.spec("tpta")
            reply = await A.call(tool, timeout=60, **args)
            unchanged = self.spec("tpta") == before
            check(f"e.{label}", "Rejected, spec unchanged",
                  f"{reply.splitlines()[0][:110]!r}, unchanged={unchanged}",
                  reply.startswith("Rejected") and unchanged)
            self.reset("tpta")
        ref = await A.call("read_reference", timeout=60, name="../../../../chia.env")
        check("e.read_reference outside the list", "Unknown reference",
              ref[:60], ref.startswith("Unknown reference"))

        # Second layer: files that reach the spec dir WITHOUT the edit tools
        # must still be refused by the evaluator, which re-runs the policy.
        (spec / "microarch_isa.py").write_text(micro.replace(
            ANCHOR, ANCHOR + IMPORT_ATTACKS["import cosim"], 1))
        v = await A.verdict("run_functional_check")
        check("e.on-disk import cosim -> evaluator", "ok=false at stage policy",
              f"ok={v.get('ok')} stage={v.get('stage')}",
              not v.get("ok") and v.get("stage") == "policy")
        # The tamper that the policy did not know about: narrow the datapath
        # AND overwrite the stress gate at import time. Written straight to
        # disk so it tests the evaluator's own defences, not only the tool's.
        for label in ("numpy.savetxt over stress_isa.py",
                      "ndarray.tofile over stress_isa.py",
                      "SystemExit forged verdict"):
            (spec / "microarch_isa.py").write_text(mutate("narrow16").replace(
                ANCHOR, ANCHOR + IMPORT_ATTACKS[label], 1))
            v = await A.verdict("run_functional_check")
            check(f"e.on-disk narrow16 + {label} -> evaluator", "ok=false",
                  f"ok={v.get('ok')} stage={v.get('stage')}", not v.get("ok"),
                  detail=v.get("detail", "")[-600:])
        # Past the static policy: a monkeypatch through a method's `self`.
        (spec / "microarch_isa.py").write_text(mutate("narrow16").replace(
            ANCHOR, ANCHOR + METHOD_SELF_PATCH, 1))
        from spec_policy import policy_violations
        missed = not policy_violations("microarch_isa.py", self.spec("tpta"))
        v = await A.verdict("run_functional_check")
        check("e.on-disk narrow16 + method-self numpy patch -> evaluator",
              "policy misses it; gate_runner refuses it (ok=false)",
              f"policy-missed={missed} ok={v.get('ok')} stage={v.get('stage')}",
              not v.get("ok") and "numpy.random.default_rng" in v.get("detail", ""),
              detail=v.get("detail", "")[-600:])
        # A stray frozen-file copy in the spec dir is ignored, not merged.
        self.reset("tpta")
        (spec / "cosim.py").write_text("raise SystemExit('forged cosim ran')\n")
        (spec / "stress_isa.py").write_text("print('  STRESS OK: forged')\n")
        v = await A.verdict("run_functional_check")
        stress = (v.get("gate") or {}).get("stress", "")
        check("e.forged cosim.py/stress_isa.py in spec dir",
              "ignored: real stress_isa n/n, vouched",
              stress, v.get("ok") and re.search(r"STRESS OK: (\d+)/\1 ", stress)
              and (v.get("gate") or {}).get("vouched"))
        (spec / "cosim.py").unlink()
        (spec / "stress_isa.py").unlink()
        self.forged_verdict_probe()
        self.sandbox_probe()

    def forged_verdict_probe(self):
        """The verdict line, forged past the policy: the stdout the old gate
        read carries it, the gate runner does not vouch for it."""
        import importlib.util
        import shutil
        spec = importlib.util.spec_from_file_location("evaluate_", AGENT_DIR / "evaluate.py")
        ev = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ev)
        work = REPO / ".chia_scratch" / self.run_dir.name / "forge-probe"
        if work.exists():
            shutil.rmtree(work)
        tree = work / "tree"
        for rel in ev.FROZEN + [f"{PKG}/isa_dsl.py"]:
            (tree / rel).parent.mkdir(parents=True, exist_ok=True)
            (tree / rel).write_bytes(ev.git_show("HEAD", rel))
        (tree / PKG / "microarch_isa.py").write_text(mutate("narrow16").replace(
            ANCHOR, ANCHOR + FORGE_VERDICT, 1))
        env = ev.env_for(tree)
        rc, out, _ = ev.run([ALLO_PYTHON, str(tree / PKG / "stress_isa.py")], tree, env,
                            GATE_TIMEOUT, work, tree)
        forged = rc == 0 and "STRESS OK: 486/486" in out
        results = {}
        for check_name in ("bench_isa", "stress_isa"):
            ok, rc2, out2, _ = ev.vouched(check_name, tree, env, work, tree, GATE_TIMEOUT)
            results[check_name] = (ok, out2.strip().splitlines()[-1:])
        check("e.forged verdict past the policy",
              "stdout forged (rc 0 + STRESS OK), runner vouches for neither gate",
              f"stdout-forged={forged}; runner={results}",
              forged and not any(ok for ok, _ in results.values()))

    def sandbox_probe(self):
        """The evaluator's sandbox, probed directly: candidate processes may
        write their work directory and nothing else -- the run's control record
        among the things they cannot reach."""
        import importlib.util
        spec = importlib.util.spec_from_file_location("evaluate_", AGENT_DIR / "evaluate.py")
        ev = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ev)
        if not hasattr(ev, "sandboxed"):
            check("e.sandbox", "candidate processes sandboxed", "no sandbox", False)
            return
        work = REPO / ".chia_scratch" / self.run_dir.name / "sandbox-probe"
        tree = work / "tree"
        (tree / PKG).mkdir(parents=True, exist_ok=True)
        (tree / PKG / "cosim.py").write_text("frozen\n")
        ctl = self.run_dir / "control.json"
        ctl.write_text(json.dumps(genuine_control()))
        before = ctl.read_bytes()
        code = ("import sys\n"
                "for p in sys.argv[1:]:\n"
                "    try:\n"
                "        open(p, 'a').close(); print('WROTE', p)\n"
                "    except OSError as e: print('DENIED', p, e.errno)\n")
        targets = [str(tree / PKG / "cosim.py"), str(REPO / "allo" / "__init__.py"),
                   str(AGENT_DIR / "evaluate.py"), str(work / "isa_sweep.prj"),
                   str(Path.home() / ".bashrc"), str(ctl)]
        rc, out, _ = ev.run(["python3", "-c", code, *targets], work, dict(os.environ),
                            60, work=work, tree=tree)
        wrote = [l.split()[1] for l in out.splitlines() if l.startswith("WROTE")]
        check("e.sandbox", "only the work dir (not tree/repo/evaluator/home) writable",
              f"writable={wrote}", wrote == [str(work / "isa_sweep.prj")])
        check("e.candidate-cannot-forge-the-control",
              "the run's control record is unwritable and unchanged",
              f"denied={str(ctl) not in wrote}, unchanged={ctl.read_bytes() == before}",
              str(ctl) not in wrote and ctl.read_bytes() == before)

    # c ------------------------------------------------------------------
    async def phase_c(self):
        print("== c: int16 partial-sum narrowing", flush=True)
        self.reset("tpta")
        f, old, new = MUTANTS["narrow16"]
        r = await self.A.call("replace_text", timeout=60, path=f, old=old, new=new)
        v = await self.A.verdict("run_functional_check")
        det = v.get("detail", "")
        m = re.search(r"STRESS FAILED: (\d+)/(\d+)", det)
        check("c.narrow16", "edit accepted; bench_isa passes; REJECTED at gate:stress",
              f"{r!r}; ok={v.get('ok')} stage={v.get('stage')} "
              f"stress={m.group(0) if m else None}",
              r.startswith("Replaced") and not v.get("ok")
              and v.get("stage") == "gate:stress")
        self.reset("tpta")

    # g ------------------------------------------------------------------
    async def phase_g(self):
        """The two holes the first paid run exposed (its accepted diff
        hard-coded T and deleted the 260-line design docstring)."""
        print("== g: parametricity and documentation guards", flush=True)
        A, spec = self.A, self.specs["tpta"]
        micro = head("microarch_isa.py")
        t_def = 'T = int(os.environ.get("TPU_T", 4))'
        md_def = 'MAXDIM = int(os.environ.get("TPU_MAXDIM", 16))'
        doc = ast.get_docstring(ast.parse(micro), clean=False)
        edits = {
            "literal T": {"path": "microarch_isa.py", "old": t_def, "new": "T = 4"},
            "literal MAXDIM": {"path": "microarch_isa.py", "old": md_def,
                               "new": "MAXDIM = 16"},
            "second MAXDIM definition": {"path": "microarch_isa.py", "old": t_def,
                                         "new": t_def + "\nMAXDIM = 16"},
            "module docstring deleted": {"path": "microarch_isa.py", "old": doc,
                                         "new": "TinyTPU-isa"},
        }
        for label, args in edits.items():
            self.reset("tpta")
            before = self.spec("tpta")
            reply = await A.call("replace_text", timeout=60, **args)
            check(f"g.edit: {label}", "Rejected, spec unchanged",
                  f"{reply.splitlines()[0][:120]!r}, unchanged={self.spec('tpta') == before}",
                  reply.startswith("Rejected") and self.spec("tpta") == before)
        # The first paid run's candidate shape, written straight to disk.
        self.reset("tpta")
        (spec / "microarch_isa.py").write_text(
            micro.replace(doc, "TinyTPU-isa", 1).replace(t_def, "T = 4", 1))
        v = await A.verdict("run_functional_check")
        det = v.get("detail", "")
        check("g.on-disk literal T + deleted docstring -> evaluator",
              "ok=false at stage policy, naming both",
              f"ok={v.get('ok')} stage={v.get('stage')}",
              not v.get("ok") and v.get("stage") == "policy"
              and "'T' must be defined" in det and "comments/docstrings" in det)
        # A documentation EDIT (same length, reworded) is allowed.
        self.reset("tpta")
        r = await A.call("replace_text", timeout=60, path="microarch_isa.py",
                         old="the machine both were aiming at",
                         new="the machine both of them were aiming at")
        check("g.edit: docstring reworded", "Replaced (edits are fine)", r[:60],
              r.startswith("Replaced"))
        # Past the static policy: a unit specialised to MAXDIM=16.
        self.reset("tpta")
        f, old, new = MUTANTS["wpr_literal"]
        r = await A.call("replace_text", timeout=60, path=f, old=old, new=new)
        v = await A.verdict("run_functional_check")
        check("g.wpr_literal", "edit accepted; bench+stress pass at 16; REJECTED at "
              "gate:param", f"{r[:40]!r}; ok={v.get('ok')} stage={v.get('stage')} "
              f"{(v.get('detail') or '')[:80]!r}",
              r.startswith("Replaced") and not v.get("ok")
              and v.get("stage") == "gate:param")
        # The unmodified design passes it, at both configurations.
        self.reset("tpta")
        v = await A.verdict("run_functional_check")
        param = (v.get("gate") or {}).get("param", {})
        check("g.unmodified passes gate:param", "ok, PARAM OK at MAXDIM 8 and 12",
              param, v.get("ok") and len(param) == 2)
        self.reset("tpta")

    # d ------------------------------------------------------------------
    async def phase_d(self):
        print("== d: deadlock (one stream put dropped)", flush=True)
        self.reset("tpta")
        f, old, new = MUTANTS["deadlock"]
        await self.A.call("replace_text", timeout=60, path=f, old=old, new=new)
        t = time.time()
        v, worst = await while_polling([self.A, self.B],
                                       self.A.verdict("run_functional_check"))
        took = time.time() - t
        time.sleep(3)
        left = procs_under(self.works["tpta"])
        check("d.deadlock", f"REJECTED at gate:bench_isa by the {GATE_TIMEOUT}s timeout",
              f"ok={v.get('ok')} stage={v.get('stage')} after {took:.0f}s, "
              f"timeout-in-detail={'TIMEOUT after' in v.get('detail', '')}",
              not v.get("ok") and v.get("stage") == "gate:bench_isa"
              and "TIMEOUT after" in v.get("detail", "") and took < GATE_TIMEOUT + 90)
        check("d.no-orphans", "no process left under the work dir", left or "none",
              not left)
        check("d.server-responsive", "read_spec < 5 s during the hung gate",
              f"worst {worst:.2f}s", worst < 5)
        self.reset("tpta")

    # a, b, f --------------------------------------------------------------
    async def phase_abf(self):
        print("== a/b/f: no-op and spad-zero, scored concurrently", flush=True)
        self.reset("tpta")
        self.reset("tptb")
        # a: "re-save" the unmodified design through the edit tool.
        r = await self.A.call("replace_text", timeout=60, path="microarch_isa.py",
                              old=ANCHOR, new=ANCHOR)
        f, old, new = MUTANTS["spad_zero"]
        r2 = await self.B.call("replace_text", timeout=60, path=f, old=old, new=new)
        assert r.startswith("Replaced") and r2.startswith("Replaced"), (r, r2)
        diff = unified("microarch_isa.py", head("microarch_isa.py"), self.spec("tptb"))
        (self.run_dir / "spad_zero.diff").write_text(diff)

        # f1: two evaluations on ONE tool must serialise on its work dir.
        t = time.time()
        v1, v2 = await asyncio.gather(self.B.verdict("run_functional_check"),
                                      self.B.verdict("run_functional_check"))
        check("f.same-tool-concurrent-gates", "both ok (serialised on one work dir)",
              f"ok={v1.get('ok')},{v2.get('ok')} in {time.time() - t:.0f}s",
              v1.get("ok") and v2.get("ok"))

        # f2: two tools scoring at once, polled for responsiveness.
        t = time.time()
        (va, vb), worst = await while_polling(
            [self.A, self.B],
            asyncio.gather(self.A.verdict("score_cycles"),
                           self.B.verdict("score_cycles")), every=10)
        took = time.time() - t
        ca, cb = va.get("cycles"), vb.get("cycles")
        check("a.noop", f"ok, cycles == {BASELINE}", f"ok={va.get('ok')} {ca}",
              va.get("ok") and ca == BASELINE, verdict=va)
        worse = bool(vb.get("ok") and cb and all(cb[s] > BASELINE[s] for s in BASELINE))
        check("b.spad_zero", "ok (bit-exact at both shapes), every shape WORSE",
              f"ok={vb.get('ok')} {cb} "
              f"(delta {({s: cb[s] - BASELINE[s] for s in cb} if cb else None)})",
              worse, verdict=vb)
        check("f.server-responsive", "read_spec < 5 s during two cosims",
              f"worst {worst:.2f}s over {took:.0f}s", worst < 5)
        prj_a = self.works["tpta"] / "agent" / "isa_sweep.prj"
        logs_a = sorted(p.name for p in prj_a.glob("cosim_*.log"))
        distinct = (self.works["tpta"] != self.works["tptb"]
                    and ca != cb and len(logs_a) == 2)
        check("f.no-cross-contamination", "distinct work dirs, distinct results",
              f"A={ca} B={cb} A-logs={logs_a}", distinct)
        (self.run_dir / "abf.json").write_text(json.dumps({"a": va, "b": vb}, indent=1))

    # loop --------------------------------------------------------------------
    def phase_loop(self):
        """The real swarm -> loop -> opencode -> MCP path, with a scripted model."""
        print("== loop: swarm.py + opencode + fake model (no LLM)", flush=True)
        from fake_model import FakeModel
        from spend import spent_since
        f, old, new = MUTANTS["spad_zero"]
        f2, old2, new2 = MUTANTS["narrow16"]
        cosim = head("cosim.py")
        scripts = [
            # iteration 1: no-op, and the agent asks for a score mid-turn --
            # a 2-4 minute MCP call, which timed out at 60 s in the smoke run.
            [("read_spec", {}),
             ("replace_text", {"path": "microarch_isa.py", "old": ANCHOR, "new": ANCHOR}),
             ("run_functional_check", {}), ("score_cycles", {}),
             ("say", "No-op: re-saved the design unchanged.")],
            # iteration 2: spad zero-fill back -- correct and slower.
            [("replace_text", {"path": f, "old": old, "new": new}),
             ("run_functional_check", {}),
             ("say", "Re-added the spad zero-fill.")],
            # iteration 3: narrowed accumulator; the debug session gives up.
            [("replace_text", {"path": f2, "old": old2, "new": new2}),
             ("say", "Narrowed the PE partial sum to int16.")],
            [("say", "I cannot fix this.")],
            # iteration 4: attacks only; nothing may change.
            [("apply_spec_patch", {"patch": unified("cosim.py", cosim,
                                                    cosim.replace("SHAPES", "S_", 1))}),
             ("replace_text", {"path": "../cosim.py", "old": "SHAPES", "new": "S_"}),
             ("replace_text", {"path": "microarch_isa.py", "old": ANCHOR,
                               "new": ANCHOR + IMPORT_ATTACKS["import cosim"]}),
             ("say", "Tried to edit the frozen files.")],
        ]
        fake = FakeModel(scripts).start()
        run_dir = self.run_dir / "loop"
        t0 = int(time.time() * 1000)
        env = dict(os.environ, RAY_ADDRESS=self.gcs,
                   TINYTPU_OPENCODE_MODEL="scripted/replay",
                   TINYTPU_OPENCODE_BASE_URL=fake.url)
        t = time.time()
        p = subprocess.run([sys.executable, "-u", str(AGENT_DIR / "swarm.py"),
                            "--workers", "1", "--iterations", "4",
                            "--budget-usd", "5", "--stagger", "0",
                            "--run-dir", str(run_dir)],
                           cwd=AGENT_DIR, env=env, capture_output=True, text=True,
                           timeout=7200)
        fake.stop()
        (self.run_dir / "loop-swarm.log").write_text(p.stdout + p.stderr)
        (self.run_dir / "loop-fake-model.json").write_text(
            json.dumps(fake.log, indent=1, default=str))
        worker = next(d for d in run_dir.iterdir() if d.is_dir())
        entries = [json.loads(l) for l in (worker / "variants.jsonl").read_text()
                   .splitlines() if l.strip()]
        by = {(e["kind"], e["iteration"]): e for e in entries}
        print(f"  swarm exited {p.returncode} after {time.time() - t:.0f}s; "
              f"{len(fake.log)} model sessions; fake errors {fake.errors}")
        check("loop.fake-model", "5 scripted sessions consumed, no errors",
              f"{len(fake.log)} sessions, errors={fake.errors}",
              len(fake.log) == 5 and not fake.errors and not fake.scripts)
        base = by.get(("baseline", 0), {}).get("verdict", {})
        check("loop.baseline", f"cycles == {BASELINE}", base.get("cycles"),
              base.get("cycles") == BASELINE)
        # What the agent itself received from score_cycles, mid-turn.
        s1 = fake.log[0]["calls"] if fake.log else []
        sc = next((c for c in s1 if c["tool"].endswith("score_cycles")), {})
        try:
            got = json.loads(sc.get("result", "")).get("cycles")
        except (json.JSONDecodeError, AttributeError):
            got = (sc.get("result") or "")[:200]
        check("a.noop via opencode score_cycles (mid-turn MCP)",
              f"{BASELINE}, returned after > 60 s", f"{got} after "
              f"{sc.get('returned_after_s')}s",
              got == BASELINE and (sc.get("returned_after_s") or 0) > 60)
        e1 = by.get(("candidate", 1), {})
        check("loop.iter1 no-op", "not accepted: no diff, not re-scored",
              f"accepted={e1.get('accepted')} reason={e1.get('reason')}",
              e1.get("accepted") is False and e1.get("reason") == "no diff")
        e2 = by.get(("candidate", 2), {})
        v2 = e2.get("verdict", {})
        check("b.spad_zero via loop", "bit-exact, delta > 0, REJECTED (not better)",
              f"ok={v2.get('ok')} {v2.get('cycles')} delta={e2.get('delta_cycles')} "
              f"accepted={e2.get('accepted')}",
              v2.get("ok") and (e2.get("delta_cycles") or 0) > 0
              and e2.get("accepted") is False)
        e3 = by.get(("candidate", 3), {})
        check("c.narrow16 via loop", "REJECTED at gate:stress (after 1 debug session)",
              f"stage={e3.get('verdict', {}).get('stage')} accepted={e3.get('accepted')}",
              e3.get("verdict", {}).get("stage") == "gate:stress"
              and e3.get("accepted") is False)
        e4 = by.get(("candidate", 4), {})
        replies = [c.get("result", "")[:40] for c in (fake.log[4]["calls"]
                                                       if len(fake.log) > 4 else [])]
        check("e.attacks via opencode", "every call Rejected; no diff",
              f"{replies}; reason={e4.get('reason')}",
              len(replies) == 3 and all(r.startswith("Rejected") for r in replies)
              and e4.get("reason") == "no diff")
        best = (worker / "best.diff").read_text() if (worker / "best.diff").exists() else None
        summary = json.loads((run_dir / "summary.json").read_text()) if (
            run_dir / "summary.json").exists() else {}
        check("loop.artifacts", "best.diff empty, summary.json with no best",
              f"best.diff={best!r} best={summary.get('best')}",
              best == "" and summary.get("best") is None)
        usd = spent_since(t0)["usd"]
        check("loop.spend", "$0.00", f"${usd:.2f}", usd == 0)
        check("loop.opencode-tools", "only the 7 MCP tools advertised to the model",
              fake.log[0]["tools"] if fake.log else None,
              bool(fake.log) and len(fake.log[0]["tools"]) == 7
              and all("tpufrontend_" in t for t in fake.log[0]["tools"]))

    # control -------------------------------------------------------------
    def phase_control(self):
        """What surrounds the measured control: the record a claim may rest on,
        and the cross-check that stays loud when the design's blobs move."""
        print("== control: the run's measured control and its cross-check",
              flush=True)
        import accept
        design = control.blobs("HEAD")
        want = {f: subprocess.run(["git", "rev-parse", f"HEAD:{PKG}/{f}"], cwd=REPO,
                                  capture_output=True, text=True).stdout.strip()
                for f in ("microarch_isa.py", "isa_dsl.py")}
        check("control.design-identity", "the two editable blobs at HEAD", design,
              design == want and all(design.values()))
        check("control.genuine-record-usable", "no problems",
              control.unusable(genuine_control(), design, "cosim") or "none",
              not control.unusable(genuine_control(), design, "cosim"))
        forgeries = {
            "verdict not vouched": genuine_control(vouched=False),
            "measured by another driver": genuine_control(driver="codesign_cosim"),
            "measured on a dirty checkout": genuine_control(pristine_tree=False),
            "another design's control": genuine_control(
                blobs={f: "0" * 40 for f in design}),
            "a shape left out": genuine_control(
                cycles={s: c for s, c in list(BASELINE_ALL.items())[:4]}),
            "a shape invented": genuine_control(
                cycles=dict(BASELINE_ALL, **{"32x32x32": 1})),
            "cycles never measured": genuine_control(
                cycles={s: None for s in BASELINE_ALL}),
            "a clock the design cannot meet": genuine_control(estimated_ns=4.0),
            "not a record at all": None,
        }
        for label, rec in forgeries.items():
            problems = control.unusable(rec, design, "cosim")
            check(f"control.refused: {label}", "refused, with a reason",
                  problems, bool(problems))
        # accept.py's --control path: the same refusal, through a file.
        peer = self.run_dir / "peer-accept.json"
        peer.write_text(json.dumps({"control": genuine_control()}))
        reused = accept.reuse_control(peer, design, "cosim")
        check("control.reuse-genuine", "reused, and says where from",
              reused["source"][:40], reused["cycles"] == BASELINE_ALL
              and reused["source"].startswith("reused from"))
        peer.write_text(json.dumps(
            {"control": genuine_control(cycles=dict(BASELINE_ALL, **{"4x4x4": 1}),
                                        blobs={f: "0" * 40 for f in design})}))
        try:
            accept.reuse_control(peer, design, "cosim")
            refused = "accepted a foreign control"
        except accept.NoControl as why:
            refused = str(why)[:80]
        check("control.reuse-foreign-refused", "NoControl, nothing compared",
              refused, refused.startswith("refusing the control in"))
        # A control measured by another driver describes the same hardware with
        # a different program: reusing it across drivers is a free win.
        peer.write_text(json.dumps({"control": genuine_control()}))
        try:
            accept.reuse_control(peer, design, "codesign_cosim")
            crossed = "reused across drivers"
        except accept.NoControl as why:
            crossed = str(why).split(": ", 1)[-1][:70]
        check("control.reuse-across-drivers-refused",
              "NoControl: the candidate's driver is not the control's",
              crossed, crossed.startswith("measured by 'cosim'"))
        # The cross-check: a prose-only edit moves the blobs, and the check
        # must still compare -- that silent "no-baseline" is what it replaces.
        agree = control.crosscheck(dict(BASELINE_ALL), design)
        check("control.crosscheck-agrees", "agree, against the published numbers",
              f"{agree['status']} vs {agree['against'][:40]}",
              agree["status"] == "agree" and not control.banner(agree))
        moved = control.crosscheck(dict(BASELINE_ALL, **{"4x4x4": 171}), design)
        check("control.crosscheck-disagrees-loudly", "DISAGREES, with a banner",
              f"{moved['status']} delta={moved['delta']['4x4x4']}",
              moved["status"] == "DISAGREES"
              and "DISAGREES" in control.banner(moved))
        recorded = control.crosscheck(dict(BASELINE_ALL), dict(zip(
            ("microarch_isa.py", "isa_dsl.py"),
            ("98b20b8b3f9ecf289604a428ffdb28997964b9dd",
             "8f2e9aa9f518ef320cab163adc95e05737c777be"))))
        check("control.crosscheck-recorded-design", "agree, against its own entry",
              f"{recorded['status']} vs {recorded['against'][:40]}",
              recorded["status"] == "agree"
              and recorded["against"].startswith("the control recorded"))
        search = control.crosscheck(dict(BASELINE), design)
        check("control.crosscheck-search-shapes",
              "agree over the two scored shapes alone",
              f"{search['status']} {search['compared']}",
              search["status"] == "agree" and search["compared"]
              == sorted(BASELINE))

    # accept --------------------------------------------------------------
    def phase_accept(self):
        print("== accept: accept.py on (b)", flush=True)
        diff = self.run_dir / "spad_zero.diff"
        if not diff.exists():
            diff.write_text(unified("microarch_isa.py", head("microarch_isa.py"),
                                    mutate("spad_zero")))
        out = self.run_dir / "accept-spad_zero"
        p = subprocess.run([ALLO_PYTHON, str(AGENT_DIR / "accept.py"), "--diff",
                            str(diff), "--out", str(out)], cwd=AGENT_DIR,
                           capture_output=True, text=True, timeout=7200)
        (self.run_dir / "accept.log").write_text(p.stdout + p.stderr)
        res = json.loads((out / "accept.json").read_text())
        cyc = {s: v["cycles"] for s, v in res.get("cosim", {}).items()}
        check("accept.spad_zero", "bit-exact at all 5 shapes, claim=not-better",
              f"ok={res.get('ok')} claim={res.get('claim')} {cyc}",
              res.get("ok") and res.get("claim") == "not-better"
              and len(cyc) == 5, accept=res)
        # The control the claim rests on: measured in this same run, off the
        # same build, while the candidate was still nowhere on disk. The
        # candidate is slower at every shape, so a control that moved with it
        # would show here.
        ctl = res.get("control") or {}
        check("accept.control-measured-before-the-diff",
              f"measured in the run, pristine tree, {BASELINE_ALL}",
              f"source={(ctl.get('source') or '')[:60]!r} "
              f"cycles={ctl.get('cycles')} vouched={ctl.get('vouched')} "
              f"pristine={ctl.get('pristine_tree')}",
              "before the candidate diff" in (ctl.get("source") or "")
              and ctl.get("cycles") == BASELINE_ALL and ctl.get("vouched")
              and ctl.get("pristine_tree")
              and ctl.get("blobs") == control.blobs("HEAD"))
        cc = res.get("crosscheck") or {}
        check("accept.crosscheck", "agree with the published five-shape control",
              f"{cc.get('status')} delta={cc.get('delta')}",
              cc.get("status") == "agree")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phases", default="control,e,c,g,d,abf,loop,accept")
    ap.add_argument("--run-dir", type=Path, default=REPO / "chia_runs"
                    / f"harness-test-{time.strftime('%Y%m%d-%H%M%S')}")
    a = ap.parse_args()
    phases = a.phases.split(",")
    run_dir = a.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    suite = Suite(run_dir)
    try:
        for ph in phases:
            if ph in ("e", "c", "d", "abf", "g"):
                asyncio.run(getattr(suite, f"phase_{ph}")())
            elif ph in ("control", "loop", "accept"):
                getattr(suite, f"phase_{ph}")()
            else:
                raise SystemExit(f"unknown phase {ph}")
    finally:
        suite.close()
        time.sleep(2)
        left = procs_under(REPO / ".chia_scratch" / run_dir.name)
        check("hygiene.no-orphans-at-exit", "no process under the scratch dir",
              left or "none", not left)
        (run_dir / "results.json").write_text(json.dumps(
            {"phases": phases, "seconds": round(time.time() - started),
             "head": subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                                    capture_output=True, text=True).stdout.strip(),
             "results": RESULTS}, indent=1, default=str))
    failed = [r["case"] for r in RESULTS if not r["pass"]]
    print("=" * 78)
    print(f"{len(RESULTS) - len(failed)}/{len(RESULTS)} cases pass "
          f"({(time.time() - started) / 60:.1f} min); results in {run_dir}")
    for f in failed:
        print(f"  FAILED: {f}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
