# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Mutation testing of the TinyTPU-isa HARNESS: does it catch a broken design?

A harness that has only ever seen the correct design has never been shown to
fail. This applies one deliberate single-point bug at a time to
the design -- in a copy under `.mutants/`, never in place -- runs each
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
them. `ar_claim_false` is the one there is: it makes the dependence claim
`schedule()` emits on `accu`'s `ar` (`#pragma HLS dependence ... inter false`)
untrue for programs the assembler accepts, and the `TPU_TB=stress` testbench's
`ar_distance_program` case is what fails. Their cosim runs at 4x4x4 unless
`TPU_SHAPES` says otherwise. With `--no-rtl` they are reported as not run, not
as caught.

Every mutant's anchor must occur exactly once across the whole design
(`microarch_isa.py` plus the `ip` unit library, `DESIGN` below), and its `old`
text must occur after it, so a refactor that moves or renames anchored code
makes this script fail loudly instead of silently testing nothing.
"""

import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
WORK = os.path.join(HERE, ".mutants")

# The design: the shipped instantiation, and the unit library it composes. A
# mutant names no file -- its anchor has to occur exactly once across all of
# them, which is a stronger claim than uniqueness within one file, and is what
# locates the mutation.
DESIGN = ["microarch_isa.py"] + sorted(
    os.path.relpath(os.path.join(d, f), HERE)
    for d, _, fs in os.walk(os.path.join(HERE, "ip")) for f in fs
    if f.endswith(".py"))

# (name, what it models, anchor, old, new). `old` is replaced once, at its
# first occurrence after `anchor`; `anchor` must be unique across `DESIGN`.
MUTANTS = [
    ("none", "control: the design unmodified", None, None, None),
    # --- the array ---
    ("pe_psum_int16", "PE partial sum narrowed int32 -> int16",
     "psum: int32 = psum_north + activation16 * weight16",
     "psum: int32", "psum: int16"),
    ("pe_weight_lane_swapped", "wld(i,j) hands its PE lane i of the weight word, not j",
     "pe_word[0:8] = weight_word[8 * j : 8 * (j + 1)]",
     "weight_word[8 * j : 8 * (j + 1)]", "weight_word[8 * i : 8 * (i + 1)]"),
    ("pe_act_lane_swapped", "column-0 PE taps activation lane j, not i",
     "activation = activation_word[8 * i : 8 * (i + 1)]",
     "activation_word[8 * i : 8 * (i + 1)]",
     "activation_word[8 * j : 8 * (j + 1)]"),
    ("pe_shadow_not_swapped", "weight double buffer: the PE keeps its first nonzero "
     "weight instead of taking the next mm's from wq",
     "weight = pe_word[0:8]", "weight = pe_word[0:8]",
     "if weight == 0:\n                weight = pe_word[0:8]"),
    ("wld_rows_from_weight", "wld sends the weight lane as the PE's row count",
     "pe_word[8:20] = header[0:12]", "header[0:12]", "weight_word[0:12]"),
    # --- the scratchpad: weights for mm, and vld ---
    ("spm_weight_off_by_one", "spm streams weight rows f3+1.. instead of f3..",
     "read_row = spad_base + row - 1", "spad_base + row - 1",
     "spad_base + row"),
    ("spm_vld_off_by_one", "vld reads spad one row late",
     "read_row: int32 = f1 + row\n        if op == OP_MM:",
     "f1 + row", "f1 + row + 1"),
    # --- the vregs: the A path ---
    ("vru_act_off_by_one", "vru streams activation rows one row late",
     "activation: UInt(VW) = vr[vr_base + row]", "vr[vr_base + row]",
     "vr[vr_base + row + 1]"),
    ("vru_dma_ignores_f3", "a dma_ld into the vregs lands at f0 + r, not f3 + r",
     "write_row = dma_base + row", "dma_base + row", "vr_base + row"),
    # --- DMA ---
    ("dma_ld_src_swapped", "dma_ld reads B for src A and A for src B",
     "if (route & DMA_SRC_B) == 0:", "== 0:", "!= 0:"),
    ("dma_ld_row_ignored", "dma_ld (A path) ignores its DRAM row f1",
     "packed = a_onchip[(dram_row0 + row) * WPR + col_block]",
     "(dram_row0 + row) * WPR", "(row) * WPR"),
    ("mvout_row_ignored", "dma_st ignores the DRAM row f1",
     "dram_c[(dram_row0 + row) * MAXDIM + col_block * T + lane] = lane_value",
     "(dram_row0 + row) * MAXDIM", "(row) * MAXDIM"),
    ("dma_st_accumulates_C", "dma_st adds into C (relies on C arriving zeroed)",
     "dram_c[(dram_row0 + row) * MAXDIM + col_block * T + lane] = lane_value",
     "= lane_value",
     "= lane_value + dram_c[(dram_row0 + row) * MAXDIM + col_block * T + lane]"),
    # --- the sequencer: prefetch, AGU, loops, per-unit words ---
    ("prefetch_lane7_dup", "the 8-wide program prefetch copies word 6 into word 7 of "
     "every group",
     "program[8 * group + word_in_group] = dram_imem[8 * group + word_in_group]",
     "dram_imem[8 * group + word_in_group]",
     "dram_imem[8 * group + word_in_group - word_in_group // 7]"),
    ("agu_stride_halved", "AGU stride field read one bit high (stride/2)",
     "with allo.meta_for(AGU_TERMS) as term:",
     "stride: int32 = agu_word[19 * term + 7 :",
     "stride: int32 = agu_word[19 * term + 8 :"),
    ("agu_f3_dropped", "AGU terms targeting f3 are ignored",
     "if target == AGU_F3:", "f3 = f3 + offset", "f3 = f3 + 0 * offset"),
    ("loop_extra_trip", "loop back-edge test <= (one extra iteration)",
     "next_iter: int32 = loop_iter[loop_sp - 1] + 1",
     "if next_iter < loop_trip", "if next_iter <= loop_trip"),
    ("vrelu_rows_short", "the sequencer's vrelu carries one row fewer",
     "            if op == OP_VRELU:\n                c_acc.put(resolved)",
     "c_acc.put(resolved)",
     "resolved[54:62] = nr - 1\n                c_acc.put(resolved)"),
    # --- the accumulator and the vector ALU ---
    ("mm_acc_dropped", "mm ignores the accumulate flag",
     "base: UInt(AW) = 0", "if f2 == 1:", "if f2 == 2:"),
    ("mm_always_acc", "mm always accumulates (relies on ar arriving zeroed)",
     "base: UInt(AW) = 0", "if f2 == 1:", "if f2 <= 1:"),
    ("mm_dst_base_ignored", "mm writes ar[r], ignoring its f1 base",
     "write_row = f1 + row", "write_row = f1 + row", "write_row = row"),
    ("vrelu_dst_is_src", "vrelu writes its source row, not f0",
     "        if op == OP_MM:\n            write_row = f1 + row",
     "if op == OP_MM:", "if op != OP_VADD:"),
    ("vadd_dst_is_src1", "vadd writes its first source, not f0",
     "        if op == OP_MM:\n            write_row = f1 + row",
     "if op == OP_MM:", "if op != OP_VRELU:"),
    ("relu_off_by_one", "ReLU threshold off by one (-1 survives)",
     "before: int32 = read_word[32 * relu_lane",
     "if rectified < 0:", "if rectified < -1:"),
    ("vadd_subtracts", "vadd computes x - y",
     "added: int32 = first + second", "first + second", "first - second"),
    ("vadd_src2_is_src1", "vadd reads its first source twice (x + x)",
     "if phase == 1:", "read_row = f2 + row", "read_row = f1 + row"),
    ("vadd_holds_stale_x", "vadd's first operand register is never loaded",
     "vadd_first = read_word", "vadd_first = read_word",
     "vadd_first = vadd_first"),
    ("vrelu_src_base_ignored", "vrelu reads ar[r], ignoring its f1 base",
     "        if op == OP_MVOUT:\n            read_row = f0 + row",
     "read_row = f0 + row",
     "read_row = f0 + row\n        if op == OP_VRELU:\n            read_row = row"),
    ("mvout_src_base_ignored", "mvout reads ar[r], ignoring its f0 base",
     "        if op == OP_MVOUT:\n            read_row = f0 + row",
     "read_row = f0 + row", "read_row = row"),
    ("clip_hi_off_by_one", "mvout clip upper bound 128, not 127",
     "retiring: int32 = read_word[32 * clip_lane",
     "if retiring > 127:", "if retiring > 128:"),
    ("clip_lo_off_by_one", "mvout clip lower bound -129, not -128",
     "retiring: int32 = read_word[32 * clip_lane",
     "if retiring < -128:", "if retiring < -129:"),
    # --- the assembler, and the dependence claim it makes true ---
    ("assembler_span_short", "assemble() bursts one DRAM row too few",
     "def dram_span(source):", "issue.f1 + issue.nr for issue in issues",
     "issue.f1 + issue.nr - 1 for issue in issues"),
    ("ar_contract_unenforced", "check_program stops enforcing the accumulator "
     "distance contract",
     "if step - ar_written_at[row] < self.ar_raw_dist:",
     "< self.ar_raw_dist:", "< 1:"),
    ("ar_claim_false", "FALSE DEPENDENCE CLAIM: the contract admits an ar read one "
     "accu iteration after its write, so the `inter false` pragma on ar is untrue "
     "for programs the assembler accepts",
     "AR_RAW_DIST = ", "AR_RAW_DIST = 4", "AR_RAW_DIST = 1"),
]


# Caught only in RTL (see the module docstring); cosim runs for these by default.
RTL_ONLY = {"ar_claim_false"}

LEVELS = {
    # name: (script, args, success marker, timeout s)
    "bench_isa": ("bench_isa.py", [], "ALL EXACT", 240),
    "stress_isa": ("stress_isa.py", [], "STRESS OK", 480),
    "cosim": ("cosim.py", [], "COSIM OK", 7200),
}

# The mutant tree shadows the design: its copy of every design file goes in
# front of the real one on the package's search path, so `microarch_isa` and
# every `ip` submodule resolve to the mutated copy while the harness scripts
# themselves still come from the checkout.
SHIM = """
import runpy, sys
sys.path.insert(0, {repo!r})
import examples.tinytpu as pkg
pkg.__path__ = [{overlay!r}] + list(pkg.__path__)
sys.argv = [{script!r}] + {args!r}
runpy.run_path({script!r}, run_name="__main__")
"""


def locate(name, anchor):
    """The one design file the anchor occurs in, exactly once."""
    counts = {f: open(os.path.join(HERE, f)).read().count(anchor)
              for f in DESIGN}
    hits = [f for f, n in counts.items() if n]
    assert len(hits) == 1 and counts[hits[0]] == 1, (
        f"{name}: anchor must occur exactly once in the design, found "
        + ", ".join(f"{n}x in {f}" for f, n in counts.items() if n))
    return hits[0]


def mutant_tree(name):
    """-> {relative design path: source text}, one file of it mutated."""
    tree = {f: open(os.path.join(HERE, f)).read() for f in DESIGN}
    _, _, anchor, old, new = next(m for m in MUTANTS if m[0] == name)
    if anchor is None:
        return tree
    path = locate(name, anchor)
    src = tree[path]
    i = src.find(old, src.index(anchor))
    assert i >= 0, f"{name}: {old!r} not found after its anchor in {path}"
    tree[path] = src[:i] + new + src[i + len(old):]
    return tree


def run_level(name, level):
    """-> ('pass' | 'CAUGHT' | 'CAUGHT (hang)', log path)"""
    d = os.path.join(WORK, name)
    script, args, marker, timeout = LEVELS[level]
    env = dict(os.environ)
    if level == "cosim":
        env.setdefault("TPU_TB", "stress")
        if name in RTL_ONLY:
            env.setdefault("TPU_SHAPES", "4x4x4")
        env["TPU_PRJ"] = os.path.join(d, "cosim.prj")
    shim = SHIM.format(repo=REPO, overlay=d,
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
    for rel, text in mutant_tree(name).items():
        path = os.path.join(d, rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        open(path, "w").write(text)
    return {lv: run_level(name, lv)[0] for lv in levels}


def main(argv):
    cosim = "--cosim" in argv
    no_rtl = "--no-rtl" in argv
    names = [a for a in argv if not a.startswith("--")] or [m[0] for m in MUTANTS]
    known = {m[0] for m in MUTANTS}
    assert set(names) <= known, f"unknown mutant(s): {set(names) - known}"
    for n in names:                       # fail fast on a stale anchor
        mutant_tree(n)
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
