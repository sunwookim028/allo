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
    cosim        `cosim.py` with `TPU_TB=stress` -- only with `--cosim`, and
                 only for the mutants named, because it costs minutes each

A mutant no level catches is a hole in the harness (or an equivalent mutant,
which the table has to say). `none` is the control: the unmodified source
through the same loader, which must pass everything -- it is what shows the
loader really runs the file it was given.

    python mutate.py                         # every mutant, functional levels
    python mutate.py pe_psum_int16 none      # just these
    python mutate.py --cosim pe_psum_int16   # add cosim (TPU_SHAPES applies)

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
    ("pe_psum_int16", "PE partial sum narrowed int32 -> int16",
     "o: int32 = p + av * wv", "o: int32", "o: int16"),
    ("mm_acc_dropped", "mm ignores the accumulate flag",
     "# `f2` selects overwrite (0)", "if f2 == 1:", "if f2 == 2:"),
    ("mm_always_acc", "mm always accumulates (relies on ar arriving zeroed)",
     "# `f2` selects overwrite (0)", "if f2 == 1:", "if f2 <= 1:"),
    ("mm_dst_base_ignored", "mm writes ar[r], ignoring its f1 base",
     "# `f2` selects overwrite (0)", "ar[f1 + r] = z", "ar[r] = z"),
    ("vrelu_loop_short", "off-by-one loop bound: vrelu skips its last row",
     "if op == OP_VRELU:\n                for r in range(nr):", "for r in range(nr):", "for r in range(nr - 1):"),
    ("vrelu_dst_is_src", "vrelu writes its source row, not f0",
     "if op == OP_VRELU:\n                for r in range(nr):", "ar[f0 + r] = zr", "ar[f1 + r] = zr"),
    ("relu_off_by_one", "ReLU threshold off by one (-1 survives)",
     "if op == OP_VRELU:\n                for r in range(nr):", "if re < 0:", "if re < -1:"),
    ("vadd_dst_is_src1", "vadd writes its first source, not f0",
     "if op == OP_VADD:\n                for r in range(nr):", "ar[f0 + r] = z", "ar[f1 + r] = z"),
    ("vadd_subtracts", "vadd computes x - y",
     "if op == OP_VADD:\n                for r in range(nr):", "se: int32 = xe + ye", "se: int32 = xe - ye"),
    ("vadd_src2_is_src1", "vadd reads its first source twice (x + x)",
     "if op == OP_VADD:\n                for r in range(nr):", "y: UInt(AW) = ar[f2 + r]",
     "y: UInt(AW) = ar[f1 + r]"),
    ("vrelu_src_base_ignored", "vrelu reads ar[r], ignoring its f1 base",
     "if op == OP_VRELU:\n                for r in range(nr):", "u: UInt(AW) = ar[f1 + r]",
     "u: UInt(AW) = ar[r]"),
    ("mvout_src_base_ignored", "mvout reads ar[r], ignoring its f0 base",
     "if op == OP_MVOUT:\n                for r in range(nr):", "t: UInt(AW) = ar[f0 + r]",
     "t: UInt(AW) = ar[r]"),
    ("clip_hi_off_by_one", "mvout clip upper bound 128, not 127",
     "if op == OP_MVOUT:\n                for r in range(nr):", "if te > 127:", "if te > 128:"),
    ("clip_lo_off_by_one", "mvout clip lower bound -129, not -128",
     "if op == OP_MVOUT:\n                for r in range(nr):", "if te < -128:", "if te < -129:"),
    ("agu_stride_halved", "AGU stride field read one bit high (stride/2)",
     "with allo.meta_for(AGU_TERMS) as _t:",
     "sw: int32 = w1[19 * _t + 7 :", "sw: int32 = w1[19 * _t + 8 :"),
    ("agu_f3_dropped", "AGU terms targeting f3 are ignored",
     "if tw == AGU_F3:", "f3 = f3 + d", "f3 = f3 + 0 * d"),
    ("loop_extra_trip", "loop back-edge test <= (one extra iteration)",
     "nxt: int32 = lp_iv[sp - 1] + 1", "if nxt < lp_trip", "if nxt <= lp_trip"),
    ("pe_weight_lane_swapped", "PE(i,j) latches lane i of its weight word, not j",
     "w = ww[8 * j : 8 * (j + 1)]", "ww[8 * j : 8 * (j + 1)]", "ww[8 * i : 8 * (i + 1)]"),
    ("pe_act_lane_swapped", "column-0 PE taps activation lane j, not i",
     "a = aw[8 * i : 8 * (i + 1)]", "aw[8 * i : 8 * (i + 1)]", "aw[8 * j : 8 * (j + 1)]"),
    ("vru_act_off_by_one", "vru activation address off by one row",
     "va: int32 = f0 + r - T - 1", "f0 + r - T - 1", "f0 + r - T"),
    ("dma_ld_src_swapped", "dma_ld reads B for src 0 and A for src 1",
     "pw: UInt(VW) = 0\n            if f0 == 0:", "if f0 == 0:", "if f0 != 0:"),
    ("dma_ld_row_ignored", "dma_ld (A path) ignores its DRAM row f1",
     "pw = rbA[(f1 + r) * WPR + f2]", "(f1 + r) * WPR", "(r) * WPR"),
    ("spm_vld_off_by_one", "vld reads spad one row late",
     "lw: UInt(VW) = spad[f1 + r]", "spad[f1 + r]", "spad[f1 + r + 1]"),
    ("mvout_row_ignored", "dma_st ignores the DRAM row f1",
     "lC[(f1 + r) * MAXDIM + f2 * T + e] = ov", "(f1 + r) * MAXDIM", "(r) * MAXDIM"),
    ("dma_st_accumulates_C", "dma_st adds into C (relies on C arriving zeroed)",
     "lC[(f1 + r) * MAXDIM + f2 * T + e] = ov", "= ov",
     "= ov + lC[(f1 + r) * MAXDIM + f2 * T + e]"),
    ("assembler_span_short", "assemble() bursts one DRAM row too few",
     "def span(src):", "e[3] + e[1] for e in ev", "e[3] + e[1] - 1 for e in ev"),
]

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
    names = [a for a in argv if not a.startswith("--")] or [m[0] for m in MUTANTS]
    known = {m[0] for m in MUTANTS}
    assert set(names) <= known, f"unknown mutant(s): {set(names) - known}"
    for n in names:                       # fail fast on a stale anchor
        mutant_source(n)
    levels = ["bench_isa", "stress_isa"] + (["cosim"] if cosim else [])
    # Functional levels in parallel; cosim ones one at a time (they are
    # heavy, and Vitis is happier without company).
    with ThreadPoolExecutor(max_workers=1 if cosim else 8) as ex:
        res = dict(zip(names, ex.map(lambda n: evaluate(n, levels), names)))

    w = max(len(n) for n in names)
    print(f"\n  {'mutant':{w}s}  " + "  ".join(f"{lv:13s}" for lv in levels)
          + "  caught first by")
    holes, bad_control = [], False
    for n in names:
        r = res[n]
        first = next((lv for lv in levels if r[lv] != "pass"), None)
        if n == "none":
            bad_control = first is not None
            verdict = "control passes" if not bad_control else "CONTROL FAILED"
        else:
            verdict = first or "SURVIVED"
            if first is None:
                holes.append(n)
        print(f"  {n:{w}s}  " + "  ".join(f"{r[lv]:13s}" for lv in levels)
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
    print(f"  MUTATE OK: all {len([n for n in names if n != 'none'])} mutants caught")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
