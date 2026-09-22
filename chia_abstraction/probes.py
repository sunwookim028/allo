# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The probe rung: a new abstraction exercised at a HARNESS-OWNED call site. FROZEN.

Why it exists. Last night's held-out candidate built in 28.3 s, passed 291/291
tests with zero regressions, was bit-exact on three design cases, left every
resource at ratio 1.000 -- and aborted the compiler the first time anything
called it (`IntegerType.get_unsigned` on the Python side, `IntegerAttr::getInt`
on the C++ side). The objective could not REWARD it and the gates could not
CATCH it, for one reason: a new abstraction has no callers, and every gate
exercised the compiler only through existing designs. Any ladder built that way
cannot see a new capability in either direction.

What this rung does, and what it does not claim.

* The DESIGN and the SITE are the harness's. Each probe is a frozen design
  defined in this file (or one of `design_cases.py`'s), with its own frozen
  schedule. The agent cannot edit either -- which keeps the original guard,
  "an agent must not rewrite a design to flatter its own primitive".
* The CALL is the agent's, and is restricted to a declaration: exactly one
  call expression `s.<name>(...)` whose arguments are literals (str, int,
  float, bool, None, tuples/lists of those), where `<name>` is a `Schedule`
  method THE PATCH ADDED, and whose arguments bind to that method's signature
  (`inspect.signature(...).bind`). Anything else is refused before it runs.
  So the agent says what it would declare; it cannot supply code.
* The outcome is recorded as `refused` (the call or the subsequent emission
  raised) or `accepted`, together with the emitted C++. Where a probe has a
  KNOWN RIGHT ANSWER -- established by measuring the RTL, not by assertion --
  the outcome is graded against it.

Stated plainly: the call site is harness-authored; the call's arguments are
agent-declared. "Derived by introspection" means the call is validated against
the candidate's own signature, not invented by the harness.

The port-requirement probes (Pilot B's target). Ground truth, measured with
Vitis HLS 2023.2 csynth on 2026-09-22 and re-measurable with `ground_truth()`:

    ports_dual    `buf` -> ONE RAM module named `..._RAM_AUTO_1R1W` containing
                  TWO write statements (`ram[address0] <= d0`, `ram[address1]
                  <= d1`): a true dual-write-port RAM wearing a 1R1W name. On
                  an FPGA the second write port is free; standard cells have no
                  dual-write primitive, and Design Compiler refuses the
                  equivalent RTL (ELAB-366, multi-driver) on the burst-widened
                  TinyTPU's `rbA`.
    ports_banked  the same kernel with `buf` cyclically partitioned by 2 ->
                  TWO instances of that module, each with ONE write statement.

So an abstraction by which a memory declares its write ports and the compiler
checks the access pattern against it must REFUSE `ports_dual` declared 1W and
ACCEPT `ports_banked` declared 1W. The counter distinguishes a write
(`ram[a] <= d`) from a read port (`q <= ram[a]`): a second always-block that
only READS is a legal read port, and a crude count flagged two false positives
on the real design before it was corrected.

    printf NONCE | python abs_gate_runner.py probe --calls FILE --work DIR
    python probes.py --ground-truth --work DIR        # re-measure the RTL
"""

from __future__ import annotations

import argparse
import ast
import inspect
import json
import os
import re
import traceback
from pathlib import Path

import numpy as np

import allo
from allo.ir.types import int32

HERE = Path(__file__).resolve().parent
VITIS_BIN = os.environ.get("CHIA_VITIS_BIN", "/opt/xilinx/Vitis_HLS/2023.2/bin")

# -- the port probes ----------------------------------------------------------
PN = 16


def ports_kernel(A: int32[PN], B: int32[PN]):
    buf: int32[PN] = 0
    for i in range(PN // 2):
        buf[2 * i] = A[2 * i] + 1
        buf[2 * i + 1] = A[2 * i + 1] + 1
    for j in range(PN):
        B[j] = buf[j]


def _ports_dual(s):
    s.pipeline("i")
    return s


def _ports_banked(s):
    s.pipeline("i")
    s.partition(s.buf, partition_type=2, dim=1, factor=2)   # 2 = Cyclic
    return s


def _ports_golden():
    rng = np.random.default_rng(20260922)
    A = rng.integers(-1000, 1000, (PN,)).astype(np.int32)
    return A, (A + 1).astype(np.int32)


#: name -> (kind, fn, frozen schedule, buffer the probe is about, expected
#: outcome of a 1-write-port declaration on that buffer, or None if ungraded)
PROBES = {
    "ports_dual": ("kernel", ports_kernel, _ports_dual, "buf", "refused"),
    "ports_banked": ("kernel", ports_kernel, _ports_banked, "buf", "accepted"),
}
#: The measured RTL, recorded so a candidate run need not re-synthesise.
#: write statements per instance / instances, for the probe's buffer.
GROUND_TRUTH = {
    "ports_dual": {"writes_per_instance": 2, "instances": 1},
    "ports_banked": {"writes_per_instance": 1, "instances": 2},
}


# -- the declared call --------------------------------------------------------
_LITERAL = (ast.Constant,)


def _literal(node) -> bool:
    if isinstance(node, _LITERAL):
        return True
    if isinstance(node, (ast.Tuple, ast.List)):
        return all(_literal(e) for e in node.elts)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return isinstance(node.operand, ast.Constant)
    return False


def parse_call(src: str, allowed: set[str]):
    """(method name, args, kwargs) for one declared call, or raise ValueError.

    Exactly `s.<name>(<literals>)`. `<name>` must be a Schedule method the
    patch added. This is a declaration, not code: anything else is refused.
    """
    try:
        tree = ast.parse(src.strip(), mode="eval")
    except SyntaxError as e:
        raise ValueError(f"not one expression: {e.msg}")
    call = tree.body
    if not isinstance(call, ast.Call):
        raise ValueError("not a call")
    fn = call.func
    if not (isinstance(fn, ast.Attribute) and isinstance(fn.value, ast.Name)
            and fn.value.id == "s"):
        raise ValueError("the call must be `s.<method>(...)`")
    if fn.attr not in allowed:
        raise ValueError(f"`s.{fn.attr}` is not a Schedule method this patch "
                         f"added (added: {sorted(allowed) or 'none'})")
    for a in call.args:
        if not _literal(a):
            raise ValueError("arguments must be literals")
    for k in call.keywords:
        if k.arg is None or not _literal(k.value):
            raise ValueError("keyword arguments must be named literals")
    args = [ast.literal_eval(a) for a in call.args]
    kwargs = {k.arg: ast.literal_eval(k.value) for k in call.keywords}
    return fn.attr, args, kwargs


# -- RTL ground truth ---------------------------------------------------------
_WRITE = re.compile(r"\bram\s*\[[^\]]*\]\s*<=")
_READ = re.compile(r"<=\s*ram\s*\[")


def rtl_write_ports(prj: Path, buffer: str) -> dict:
    """Write statements per instance, and instances, of `buffer`'s RAM module.

    Counts WRITE statements (`ram[a] <= d`), not always-blocks: an always-block
    that only reads (`q1 <= ram[address1]`) is a legal read port."""
    vdir = prj / "out.prj/solution1/syn/verilog"
    mods = sorted(vdir.glob(f"*_{buffer}_RAM*.v"))
    if not mods:
        # Fail LOUD: an empty count here would read as "no dual-writer".
        raise RuntimeError(f"no RAM module for {buffer!r} in {vdir}")
    stem = mods[0].stem
    text = mods[0].read_text(errors="replace")
    inst = sum(len(re.findall(rf"^\s*{re.escape(stem)}\b", p.read_text(
        errors="replace"), re.M)) for p in vdir.glob("*.v") if p != mods[0])
    return {"module": mods[0].name,
            "writes_per_instance": len(_WRITE.findall(text)),
            "reads_per_instance": len(_READ.findall(text)),
            "instances": inst}


def ground_truth(work: Path) -> dict:
    if VITIS_BIN not in os.environ.get("PATH", ""):
        os.environ["PATH"] = VITIS_BIN + os.pathsep + os.environ.get("PATH", "")
    out = {}
    for name, (_, fn, sched, buffer, _) in PROBES.items():
        s = allo.customize(fn)
        sched(s)
        prj = work / f"{name}.prj"
        s.build(target="vitis_hls", mode="csyn", project=str(prj))()
        out[name] = rtl_write_ports(prj, buffer)
    return out


# -- running a declared call at a harness-owned site --------------------------
def run_probe(name: str, call_src: str, allowed: set[str], work: Path) -> dict:
    kind, fn, sched, buffer, expected = PROBES[name]
    rec = {"probe": name, "call": call_src, "expected": expected,
           "ground_truth": GROUND_TRUTH.get(name)}
    try:
        meth, args, kwargs = parse_call(call_src, allowed)
    except ValueError as e:
        rec.update(outcome="invalid-call", why=str(e))
        return rec
    s = allo.customize(fn)
    sched(s)
    bound = getattr(s, meth)
    try:
        inspect.signature(bound).bind(*args, **kwargs)
    except TypeError as e:
        rec.update(outcome="invalid-call",
                   why=f"does not bind to s.{meth}{inspect.signature(bound)}: {e}")
        return rec
    try:
        bound(*args, **kwargs)
        code = s.build(target="vhls").hls_code
    except BaseException as e:                              # noqa: BLE001
        # A refusal is any exception, at the declaration or at emission --
        # INCLUDING a C++ abort surfacing as a Python error. Whether it is the
        # RIGHT refusal is graded separately below, by its message: an
        # assertion in MLIR is not a legality rule, it is a crash.
        msg = f"{type(e).__name__}: {e}"
        crash = bool(re.search(r"Assertion|Aborted|core dumped|"
                               r"must be signless|Segmentation", msg))
        rec.update(outcome="crashed" if crash else "refused",
                   why=msg[-800:], traceback=traceback.format_exc()[-1500:])
        return rec
    (work / f"{name}.probe.cpp").write_text(code)
    rec.update(outcome="accepted", emitted_bytes=len(code))
    if name == "ports_banked":
        # The ACCEPTED design must be CONFIRMED, not merely accepted: bit-exact
        # through the emitter (csim), since "encodable is not runnable".
        A, golden = _ports_golden()
        B = np.zeros_like(golden)
        s2 = allo.customize(fn)
        sched(s2)
        getattr(s2, meth)(*args, **kwargs)
        if VITIS_BIN not in os.environ.get("PATH", ""):
            os.environ["PATH"] = VITIS_BIN + os.pathsep + os.environ["PATH"]
        mod = s2.build(target="vitis_hls", mode="csim",
                       project=str(work / f"{name}.csim.prj"))
        mod(A, B)
        rec["confirmed_csim_exact"] = bool(np.array_equal(B, golden))
    return rec


def grade(records: list[dict]) -> dict:
    """The port-requirement grade: refuse dual, accept AND confirm banked."""
    by = {r["probe"]: r for r in records}
    d, b = by.get("ports_dual", {}), by.get("ports_banked", {})
    ok = (d.get("outcome") == "refused" and b.get("outcome") == "accepted"
          and b.get("confirmed_csim_exact") is True)
    return {
        "ports": ok,
        "dual": d.get("outcome"), "banked": b.get("outcome"),
        "banked_confirmed": b.get("confirmed_csim_exact"),
        "note": ("refuses the dual-writer against a 1W declaration and accepts "
                 "the banked design, which is bit-exact" if ok else
                 "does not separate the two ground truths"),
    }


def run(argv) -> tuple[int, dict]:
    """Called by abs_gate_runner.py in the vouched process."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--calls", required=True,
                    help="JSON: {'added': [...], 'calls': {probe: 's.x(...)'}}")
    ap.add_argument("--work", required=True)
    a = ap.parse_args(argv)
    work = Path(a.work).resolve()
    work.mkdir(parents=True, exist_ok=True)
    spec = json.loads(Path(a.calls).read_text())
    allowed = set(spec.get("added", []))
    recs = [run_probe(name, call, allowed, work)
            for name, call in sorted(spec.get("calls", {}).items())
            if name in PROBES]
    rep = {"records": recs, "grade": grade(recs)}
    # rc 0 = the rung RAN. Whether the candidate passed is the grade, decided
    # by the evaluator, so a refusal on the dual probe is not a gate failure.
    return 0, rep


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ground-truth", action="store_true")
    ap.add_argument("--work", required=True)
    a = ap.parse_args()
    if a.ground_truth:
        print(json.dumps(ground_truth(Path(a.work)), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
