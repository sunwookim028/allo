# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Mutation testing of the TinyTPU-isa HARNESS: does it catch a broken design?

A harness that has only ever seen the correct design has never been shown to
fail. This applies one deliberate single-point bug at a time to
`microarch_isa.py` -- in a copy under `.mutants/`, never in place -- runs each
verification level against the mutant, and prints which level caught it:

    bench_isa    the published functional sweep ([-4, 4] operands, seed 0)
    stress_isa   the stress gate (full range, boundaries, prefilled C, vector
                 and random programs, the validator)
    cosim        `cosim.py` with `TPU_TB=stress` -- with `--cosim` for the
                 mutants named (minutes each), and ALWAYS for the RTL-only
                 mutants below

A mutant no level catches is a hole in the harness (or an equivalent mutant,
which the table has to say). `none` is the control: the unmodified source
through the same loader, which must pass everything -- it is what shows the
loader really runs the file it was given.

    python mutate.py                         # every mutant (~15 min: one cosim)
    python mutate.py --no-rtl                # functional levels only (~5 min)
    python mutate.py pe_psum_int16 none      # just these
    python mutate.py --cosim pe_psum_int16   # add cosim (TPU_SHAPES applies)

**RTL-only mutants** (`RTL_ONLY`) break something no simulator models, so they
pass `bench_isa` and `stress_isa` by construction and only cosim can catch
them. `vr_claim_false` is the one there is: it makes the dependence claim
`schedule()` emits on `vpu`'s `vreg` (`#pragma HLS dependence ... inter false`)
untrue for programs the assembler accepts, and the `TPU_TB=stress` testbench's
`ar_distance_program` case is what fails. Their cosim runs at 4x4x4 unless
`TPU_SHAPES` says otherwise. With `--no-rtl` they are reported as not run, not
as caught.

Every mutant's `old` text must occur exactly once after its anchor, so a
refactor of the design makes this script fail loudly instead of silently
testing nothing.
"""

import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
SRC = os.path.join(HERE, "microarch_isa.py")
WORK = os.path.join(HERE, ".mutants")
MOD = "examples.accelerator.tinytpu_vitis.microarch_isa"

# (name, what it models, anchor, old, new). `old` is replaced once, at its
# first occurrence after `anchor`; `anchor` must be unique in the file.
MUTANTS = [
    ("none", "control: the design unmodified", None, None, None),
    # --- the array ---
    ("pe_psum_int16", "PE partial sum narrowed int32 -> int16",
     "o: int32 = p + av * wv", "o: int32", "o: int16"),
    ("pe_weight_lane_swapped", "wld(i,j) hands its PE lane i of the weight word, not j",
     "q[0:8] = ww[8 * j : 8 * (j + 1)]", "ww[8 * j : 8 * (j + 1)]", "ww[8 * i : 8 * (i + 1)]"),
    ("pe_act_lane_swapped", "column-0 PE taps activation lane j, not i",
     "a = aw[8 * i : 8 * (i + 1)]", "aw[8 * i : 8 * (i + 1)]", "aw[8 * j : 8 * (j + 1)]"),
    ("pe_shadow_not_swapped", "weight double buffer: the PE keeps its first nonzero "
     "weight instead of taking the next mm's from wq",
     "w = q[0:8]", "w = q[0:8]", "if w == 0:\n                    w = q[0:8]"),
    ("vru_flag_sticky", "the weight-switch flag is never cleared, so every pushed "
     "row switches weights",
     "                    pend = 0\n", "pend = 0", "pend = pend"),
    # --- VMEM, and vld ---
    ("vmu_vld_off_by_one", "vld reads vmem one row late",
     "                ra: int32 = f1 + r\n                if op == OP_VMEMST:",
     "ra: int32 = f1 + r", "ra: int32 = f1 + r + 1"),
    # --- the vreg file: both array ports, vld, the lane conversions ---
    ("vpu_read_off_by_one", "vmatload/vmatpush/vst read vreg rows one late",
     "                if op != OP_VRELU:\n                    ra = f0 + rr",
     "ra = f0 + rr", "ra = f0 + rr + 1"),
    ("vpu_vld_dst_ignored", "vld writes vreg[r], ignoring its destination base f0",
     "v8: UInt(VW) = vm2vr.get()", "v8: UInt(VW) = vm2vr.get()",
     "v8: UInt(VW) = vm2vr.get()\n                wa = rr"),
    ("vpu_vld_zero_extends", "vld zero-extends int8 lanes instead of sign-extending",
     "b8: int8 = v8[8 * e0", "b8: int8", "b8: UInt(8)"),
    ("vpu_narrow_high_byte", "the array gets bits 8..15 of each lane, not the low 8",
     "nv[8 * e5 : 8 * (e5 + 1)] = rv[32 * e5 : 32 * e5 + 8]",
     "rv[32 * e5 : 32 * e5 + 8]", "rv[32 * e5 + 8 : 32 * e5 + 16]"),
    # --- DMA ---
    ("dma_ld_space_swapped", "vmemld reads B's beats for A's and A's for B's",
     "            if bt < BEATS:", "bt < BEATS:", "bt >= BEATS:"),
    ("dma_ld_stride_ignored", "vmemld ignores its stride (beat base + r)",
     "            bt: int32 = f1 + r * f2\n            pw", "f1 + r * f2", "f1 + r"),
    ("dma_ld_base_ignored", "vmemld ignores its DRAM base (A path: beat r * stride)",
     "                pw = rbA[bt]", "rbA[bt]", "rbA[bt - f1]"),
    ("dma_st_stride_ignored", "vmemst ignores its stride",
     "bt: int32 = f1 + r * f2     # C beat", "f1 + r * f2", "f1 + r"),
    ("dma_st_accumulates_C", "dma_st adds into C (relies on C arriving zeroed)",
     "lC[bt * T + e] = ov", "= ov", "= ov + lC[bt * T + e]"),
    # --- the sequencer: prefetch, AGU, loops, per-unit words ---
    ("prefetch_lane7_dup", "the 8-wide program prefetch copies word 6 into word 7 of "
     "every group",
     "ib[8 * i + e8] = l_imem[8 * i + e8]", "l_imem[8 * i + e8]",
     "l_imem[8 * i + e8 - e8 // 7]"),
    ("agu_stride_halved", "AGU stride field read one bit high (stride/2)",
     "with allo.meta_for(AGU_TERMS) as _t:",
     "sw: int32 = w1[19 * _t + 7 :", "sw: int32 = w1[19 * _t + 8 :"),
    ("agu_f3_dropped", "AGU terms targeting f3 are ignored",
     "if tw == AGU_F3:", "f3 = f3 + d", "f3 = f3 + 0 * d"),
    ("loop_extra_trip", "loop back-edge test <= (one extra iteration)",
     "nxt: int32 = lp_iv[sp - 1] + 1", "if nxt < lp_trip", "if nxt <= lp_trip"),
    ("vrelu_rows_short", "the sequencer's vrelu carries one row fewer",
     "                if op == OP_VRELU:\n                    c_vpu.put(rw)",
     "c_vpu.put(rw)", "rw[54:62] = nr - 1\n                    c_vpu.put(rw)"),
    # --- the accumulator and the vector ALU ---
    ("pop_ors_old_row", "vmatpop ORs the result into the old vreg row (right only "
     "if the file arrives zeroed)",
     "z = mxo.get()", "z = mxo.get()", "z = mxo.get() | rv"),
    ("pop_dst_base_ignored", "vmatpop writes vreg[r], ignoring its f0 base",
     "z = mxo.get()", "z = mxo.get()", "z = mxo.get()\n                wa = rr"),
    ("vrelu_dst_is_src", "vrelu writes its source row, not f0",
     "            wa: int32 = f0 + rr\n", "wa: int32 = f0 + rr\n",
     "wa: int32 = f0 + rr\n            if op == OP_VRELU:\n                wa = f1 + rr\n"),
    ("vadd_dst_is_src1", "vadd writes its first source, not f0",
     "            wa: int32 = f0 + rr\n", "wa: int32 = f0 + rr\n",
     "wa: int32 = f0 + rr\n            if op == OP_VADD:\n                wa = f1 + rr\n"),
    ("relu_off_by_one", "ReLU threshold off by one (-1 survives)",
     "ue: int32 = rv[32 * e3", "if re < 0:", "if re < -1:"),
    ("vadd_subtracts", "vadd computes x - y",
     "xy: int32 = xe + ye", "xe + ye", "xe - ye"),
    ("vadd_src2_is_src1", "vadd reads its first source twice (x + x)",
     "if ph == 1:", "ra = f2 + rr", "ra = f1 + rr"),
    ("vadd_holds_stale_x", "vadd's first operand register is never loaded",
     "xr = rv", "xr = rv", "xr = xr"),
    ("vrelu_src_base_ignored", "vrelu reads vreg[r], ignoring its f1 base",
     "            rv: UInt(AW) = vreg[ra]", "rv: UInt(AW) = vreg[ra]",
     "if op == OP_VRELU:\n                ra = rr\n            rv: UInt(AW) = vreg[ra]"),
    ("vst_src_base_ignored", "vst reads vreg[r], ignoring its f0 base",
     "            rv: UInt(AW) = vreg[ra]", "rv: UInt(AW) = vreg[ra]",
     "if op == OP_VST:\n                ra = rr\n            rv: UInt(AW) = vreg[ra]"),
    # --- VMEM: the output path (vst in, vmemst out) ---
    ("vmu_vst_dst_ignored", "a vst row lands at vmem[r], ignoring its f1 base",
     "                    wv = vr2vm.get()\n", "wv = vr2vm.get()\n",
     "wv = vr2vm.get()\n                    wa = r\n"),
    ("vmu_vmemst_off_by_one", "vmemst reads VMEM one row late",
     "                if op == OP_VMEMST:\n                    ra = f3 + r",
     "ra = f3 + r", "ra = f3 + r + 1"),
    ("clip_hi_off_by_one", "vst saturation upper bound 128, not 127",
     "te: int32 = rv[32 * e4", "if te > 127:", "if te > 128:"),
    ("clip_lo_off_by_one", "vst saturation lower bound -129, not -128",
     "te: int32 = rv[32 * e4", "if te < -128:", "if te < -129:"),
    # --- the assembler, and the dependence claim it makes true ---
    ("assembler_span_short", "assemble() bursts one DRAM beat too few",
     "def span(src):", "b - lo + 1 for b in beats", "b - lo for b in beats"),
    ("dma_race_unenforced", "check_program stops refusing a VMEM access that "
     "races an unfenced descriptor",
     "if hit and (dw or writes):", "if hit and (dw or writes):",
     "if hit and (dw or writes) and False:"),
    ("busy_channel_unenforced", "check_program lets a descriptor onto a busy channel",
     "            if dma[f0] is not None:", "is not None:", "is not None and False:"),
    ("idle_wait_unenforced", "check_program lets a wait name an idle channel",
     "                    if dma[ch] is None:", "is None:", "is None and False:"),
    ("outq_unenforced", "check_program stops enforcing the output FIFO bound",
     'if q["out"] > OUTQ:', "> OUTQ:", "> 10 * OUTQ:"),
    ("pop_underflow_unenforced", "check_program lets a pop take more rows than "
     "are pushed",
     'if nr > q["out"]:', 'if nr > q["out"]:', 'if nr > q["out"] + MAXDIM:'),
    ("vr_contract_unenforced", "check_program stops enforcing the vreg "
     "distance contract",
     "if at - vr_wrote[row] < VR_RAW_DIST:", "< VR_RAW_DIST:", "< 1:"),
    ("vr_claim_false", "FALSE DEPENDENCE CLAIM: the contract admits a vreg read one "
     "vpu iteration after its write, so the `inter false` pragma on vreg is untrue "
     "for programs the assembler accepts",
     "VR_RAW_DIST = ", "VR_RAW_DIST = 4", "VR_RAW_DIST = 1"),
]


# Caught only in RTL (see the module docstring); cosim runs for these by default.
RTL_ONLY = {"vr_claim_false"}

LEVELS = {
    # name: (script, args, success marker, timeout s)
    "bench_isa": ("bench_isa.py", [], "ALL EXACT", 240),
    "stress_isa": ("stress_isa.py", [], "STRESS OK", 480),
    "cosim": ("cosim.py", [], "COSIM OK", 7200),
}

SHIM = """
import importlib.util, runpy, sys
sys.path.insert(0, {repo!r})
import examples.accelerator.tinytpu_vitis as pkg
spec = importlib.util.spec_from_file_location({mod!r}, {path!r})
m = importlib.util.module_from_spec(spec)
sys.modules[{mod!r}] = m
spec.loader.exec_module(m)
pkg.microarch_isa = m
sys.argv = [{script!r}] + {args!r}
runpy.run_path({script!r}, run_name="__main__")
"""


def mutant_source(name):
    src = open(SRC).read()
    _, _, anchor, old, new = next(m for m in MUTANTS if m[0] == name)
    if anchor is None:
        return src
    assert src.count(anchor) == 1, f"{name}: anchor found {src.count(anchor)}x"
    a = src.index(anchor)
    i = src.find(old, a)
    assert i >= 0, f"{name}: {old!r} not found after its anchor"
    return src[:i] + new + src[i + len(old):]


def run_level(name, level):
    """-> ('pass' | 'CAUGHT' | 'CAUGHT (hang)', log path)"""
    d = os.path.join(WORK, name)
    path = os.path.join(d, "microarch_isa.py")
    script, args, marker, timeout = LEVELS[level]
    env = dict(os.environ)
    if level == "cosim":
        env.setdefault("TPU_TB", "stress")
        if name in RTL_ONLY:
            env.setdefault("TPU_SHAPES", "4x4x4")
        env["TPU_PRJ"] = os.path.join(d, "cosim.prj")
    shim = SHIM.format(repo=REPO, mod=MOD, path=path,
                       script=os.path.join(HERE, script), args=args)
    log = os.path.join(d, f"{level}.log")
    with open(log, "w") as f:
        try:
            # On timeout `run` SIGKILLs: the simulator ignores SIGTERM while
            # a PE blocks on a stream, which is what a hang looks like.
            rc = subprocess.run([sys.executable, "-c", shim], cwd=HERE, env=env,
                                stdout=f, stderr=subprocess.STDOUT,
                                timeout=timeout).returncode
        except subprocess.TimeoutExpired:
            return "CAUGHT (hang)", log
    out = open(log, errors="replace").read()
    return ("pass" if rc == 0 and marker in out else "CAUGHT"), log


def evaluate(name, levels):
    d = os.path.join(WORK, name)
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d)
    open(os.path.join(d, "microarch_isa.py"), "w").write(mutant_source(name))
    return {lv: run_level(name, lv)[0] for lv in levels}


def main(argv):
    cosim = "--cosim" in argv
    no_rtl = "--no-rtl" in argv
    names = [a for a in argv if not a.startswith("--")] or [m[0] for m in MUTANTS]
    known = {m[0] for m in MUTANTS}
    assert set(names) <= known, f"unknown mutant(s): {set(names) - known}"
    for n in names:                       # fail fast on a stale anchor
        mutant_source(n)
    functional = ["bench_isa", "stress_isa"]

    def levels_of(n):
        if cosim or (n in RTL_ONLY and not no_rtl):
            return functional + ["cosim"]
        return functional
    # Functional levels in parallel; cosim ones one at a time (they are
    # heavy, and Vitis is happier without company).
    quick = [n for n in names if "cosim" not in levels_of(n)]
    heavy = [n for n in names if "cosim" in levels_of(n)]
    res = {}
    with ThreadPoolExecutor(max_workers=8) as ex:
        res.update(zip(quick, ex.map(lambda n: evaluate(n, levels_of(n)), quick)))
    for n in heavy:
        res[n] = evaluate(n, levels_of(n))

    levels = functional + (["cosim"] if heavy else [])
    w = max(len(n) for n in names)
    print(f"\n  {'mutant':{w}s}  " + "  ".join(f"{lv:13s}" for lv in levels)
          + "  caught first by")
    holes, not_run, bad_control = [], [], False
    for n in names:
        r = res[n]
        first = next((lv for lv in levels if r.get(lv, "pass") != "pass"), None)
        if n == "none":
            bad_control = first is not None
            verdict = "control passes" if not bad_control else "CONTROL FAILED"
        elif first is None and n in RTL_ONLY and "cosim" not in r:
            verdict = "NOT RUN (RTL-only: needs cosim)"
            not_run.append(n)
        else:
            verdict = first or "SURVIVED"
            if first is None:
                holes.append(n)
        print(f"  {n:{w}s}  " + "  ".join(f"{r.get(lv, '-'):13s}" for lv in levels)
              + f"  {verdict}")
    print()
    for n in names:
        print(f"  {n:{w}s}  {next(m[1] for m in MUTANTS if m[0] == n)}")
    print(f"\n  logs: {WORK}/<mutant>/<level>.log")
    if bad_control:
        print("  MUTATE FAILED: the unmodified design did not pass through the loader")
        return 2
    if holes:
        print(f"  MUTATE: {len(holes)} mutant(s) SURVIVED every level: {holes}")
        return 1
    run = [n for n in names if n != "none" and n not in not_run]
    print(f"  MUTATE OK: all {len(run)} mutants run were caught"
          + (f"; {len(not_run)} RTL-only mutant(s) not run (--no-rtl): {not_run}"
             if not_run else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
