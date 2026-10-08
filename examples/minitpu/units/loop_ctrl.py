# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""U4: ``loop_ctrl``, the sequencer's loop stack and loop buffer:
``sequencer_loop_ctrl`` (8 frames of ``{body_start, iv, hi, step}``,
``STACK_DEPTH = 1 << LEVEL_SEL_W``) and ``sequencer_loop_buffer`` (``LB_CAP``
= 24 bundles of LUTRAM that capture the innermost body's first pass and replay
it), wired as ``sequencer.sv:141-179`` wires them (``units/rtl/u4_loop_ctrl.sv``;
the only change is that ``iv_by_level_o`` is flattened, level k at
``[32k +: 32]``).

Ports: the decoded ``loop.begin`` (``body_start``, ``lo``, ``hi``, ``step``;
``may_skip`` + ``skip`` for ``loop.begin.r``), ``loop.end``, the issued pulse,
the bundle the fetch queue offers (``capture_data``). Out: the refetch request
(``branch_taken``/``branch_target``), every level's induction variable (read
by the X-slot AGU and the scalar AGU), the loop-buffer controls, the replayed
bundle, depth and the two error flags (left open in ``sequencer.sv``).

Reference: ``harness/ref_ctrl_front.LoopCtrl``, a register-level cycle model.
Traces are *programs* run closed-loop on that model, as the sequencer would
run them: a bundle issues each cycle unless stalled (a delay field), two empty
cycles follow a taken branch (the fetch queue's flush, ``units/fetch.py``),
and a warm body is served from ``replay_data``. An independent program-level
interpreter checks the model's branches and replayed words as it goes (events
``target mismatch`` / ``replay mismatch``, which must stay zero on a legal
program). Plus raw random stimulus, illegal by construction.

Instances: ``shipped`` only (no geometry parameter: ``STACK_DEPTH`` and
``LB_CAP`` are package localparams).
"""

import os

from examples.minitpu.harness import ref_ctrl_front, rtl
from examples.minitpu.harness.traces import Trace, rng_for, word

HERE = os.path.dirname(os.path.abspath(__file__))
SOURCES = ["src/pkg/minitpu_config_pkg.sv", "src/core/vpu/vpu_pkg.sv", "src/core/sequencer/sequencer_pkg.sv",
           "src/core/sequencer/sequencer_loop_buffer.sv", "src/core/sequencer/sequencer_loop_ctrl.sv",
           os.path.join(HERE, "rtl", "u4_loop_ctrl.sv")]
LB_CAP, STACK_DEPTH = ref_ctrl_front.LB_CAP, ref_ctrl_front.STACK_DEPTH
INPUTS = [("rst_n", 1), ("loop_begin_valid", 1), ("body_start", 12), ("lo", 8), ("hi", 16), ("step", 8),
          ("may_skip", 1), ("skip", 12), ("loop_end_valid", 1), ("bundle_issued", 1), ("capture_data", 128)]
OUTPUTS = [(p, w) for p, w in ref_ctrl_front.LOOP_OUT_W.items()]

INSTANCES = {"shipped": rtl.RtlUnit(top="u4_loop_ctrl", sources=SOURCES, inputs=INPUTS, outputs=OUTPUTS,
                                    shape="trace", clk="clk", rst_n="rst_n", assertions=True)}
DEFAULT = "shipped"
RTL = INSTANCES[DEFAULT]
LATENCY_SOURCE = ("sequencer_pkg.sv:61,64 STACK_DEPTH 8, LB_CAP 24; isa_latency.json C row "
                  "('resumes ... 3 cycles after the loop.begin.r'); resources.loop_buffer ('2 more cycles per iteration')")


def REF(inst, cmd):
    return ref_ctrl_front.loop_ctrl_trace(cmd)


# ---------------------------------------------------------------------------
# programs
# ---------------------------------------------------------------------------

class Op:
    """One program bundle: ``kind`` in op | begin | end."""

    def __init__(self, kind, lo=0, hi=0, step=1, r=False):
        self.kind, self.lo, self.hi, self.step, self.r = kind, lo, hi, step, r
        self.skip = 0  # begin.r: end index - begin index (asm._resolve_loop_skips)
        self.word = 0


def loop(body, lo=0, trips=2, step=1, r=False, hi=None):
    hi = lo + trips * step if hi is None else hi
    return [Op("begin", lo, hi, step, r)] + body + [Op("end")]


def ops(k):
    return [Op("op") for _ in range(k)]


def finish(prog, rng):
    """Resolve skips and give every bundle a distinct 128-bit word."""
    stack = []
    for i, b in enumerate(prog):
        b.word = (rng.getrandbits(100) << 28) | (i << 12) | 0xABC
        if b.kind == "begin":
            stack.append(i)
        elif b.kind == "end" and stack:
            j = stack.pop()
            prog[j].skip = i - j
    return prog


def run(prog, rng, p_stall=0.15, max_cycles=6000, tail=6):
    """Run ``prog`` closed-loop on the reference model; returns the trace."""
    m = ref_ctrl_front.LoopCtrl()
    t = Trace({p: 0 for p, _ in INPUTS} | {"rst_n": 1})
    rows = []

    def drive(**kw):
        r = dict(t.defaults) | kw
        o, _ = m.step(r)
        t.cycle(**kw)
        rows.append(r)
        return o

    for _ in range(2):
        drive(rst_n=0)
    pc, bubbles, stack = 0, 0, []  # stack: interpreter frames [begin_pc, iv, hi, step]
    while pc < len(prog) and len(t) < max_cycles:
        junk = dict(capture_data=rng.getrandbits(128), body_start=rng.getrandbits(12),
                    lo=rng.getrandbits(8), hi=rng.getrandbits(16), step=rng.getrandbits(8),
                    skip=rng.getrandbits(12))
        if bubbles or rng.random() < p_stall:
            bubbles = max(0, bubbles - 1)
            drive(**junk)
            continue
        b = prog[pc]
        kw = dict(junk, bundle_issued=1, capture_data=b.word, body_start=(pc + 1) & 0xFFF)
        if b.kind == "begin":
            kw.update(loop_begin_valid=1, lo=b.lo, hi=b.hi, step=b.step, may_skip=int(b.r), skip=b.skip)
        elif b.kind == "end":
            kw.update(loop_end_valid=1)
        o = drive(**kw)
        if o["lb_replay_en"] and o["replay_data"] != b.word:
            m.event("replay mismatch")
        # the program's own semantics
        nxt = pc + 1
        if b.kind == "begin":
            if b.r and b.hi <= b.lo:
                nxt = pc + b.skip + 1
            else:
                stack.append([pc, b.lo, b.hi, b.step])
        elif b.kind == "end" and stack:
            f = stack[-1]
            f[1] += f[3]
            if f[1] < f[2]:
                nxt = f[0] + 1
            else:
                stack.pop()
        if o["branch_taken"]:
            if o["branch_target"] != nxt:
                m.event("target mismatch")
            bubbles = 2
        pc = nxt
    for _ in range(tail):
        drive()
    return t.cmd(), m.ev


def random_block(rng, depth, budget):
    """A random body: straight ops and nested loops, at most ``budget`` bundles."""
    out = []
    while budget > 0 and rng.random() < 0.95:
        if depth < STACK_DEPTH and rng.random() < 0.35 and budget > 3:
            inner = random_block(rng, depth + 1, min(budget - 2, rng.choice([4, 10, 22, 23, 24, 30])))
            trips = rng.choice([1, 1, 2, 2, 3, 4]) if depth < 3 else rng.choice([1, 2])
            lo, step = rng.randrange(4), rng.choice([1, 1, 2, 3])
            r = rng.random() < 0.3
            hi = lo + trips * step
            if r and rng.random() < 0.4:
                hi = rng.randrange(lo + 1)  # zero trip: skipped
            out += loop(inner, lo=lo, step=step, r=r, hi=hi)
            budget -= len(inner) + 2
        else:
            k = rng.randrange(1, 5)
            out += ops(k)
            budget -= k
    return out


def directed():
    rng = rng_for("loop-directed")
    progs = []
    # eight nested loops, two trips each, then a ninth level (overflow) separately
    body = ops(2)
    for _ in range(STACK_DEPTH):
        body = loop(ops(1) + body, trips=2)
    progs.append(("nest8", ops(1) + body + ops(2), True))
    body = ops(1)
    for _ in range(STACK_DEPTH + 1):
        body = loop(body, trips=1)
    progs.append(("nest9-overflow", body + ops(2), False))
    # the same at two trips: the ninth loop's loop.end closes the eighth frame -- silently
    body = ops(1)
    for _ in range(STACK_DEPTH + 1):
        body = loop(body, trips=2)
    progs.append(("nest9-overflow-2trips", body + ops(2), False))
    progs.append(("underflow", ops(2) + [Op("end")] + ops(2), False))
    # bodies of exactly LB_CAP bundles (loop.end included: replays) and LB_CAP + 1 (refetches)
    progs.append(("body-24-warm", ops(1) + loop(ops(LB_CAP - 1), trips=4) + ops(2), True))
    progs.append(("body-25-cold", ops(1) + loop(ops(LB_CAP), trips=4) + ops(2), True))
    # exits: cold (one trip: the first end exits while capturing), warm (many trips), nested inner warm
    progs.append(("exit-cold", loop(ops(3), trips=1) + loop(ops(30), trips=2) + ops(1), True))
    progs.append(("exit-warm", loop(ops(5), trips=6) + loop(loop(ops(2), trips=3), trips=3) + ops(1), True))
    # loop.begin.r: zero trip (skip), one trip, a skip inside a warm body, a skip at the top level
    progs.append(("skip", ops(1) + loop(ops(4), r=True, lo=3, hi=3) + loop(ops(2), r=True, lo=0, hi=1)
                  + loop(ops(1) + loop(ops(2), r=True, lo=2, hi=1) + ops(1), trips=4)
                  + loop(ops(3), r=True, lo=5, hi=0) + ops(2), True))
    # step and lo at their 4-bit maxima, a large hi
    progs.append(("wide-iv", loop(ops(2), lo=15, step=15, hi=15 + 15 * 40) + ops(1), True))
    return [(name, finish(p, rng), legal) for name, p, legal in progs]


def raw_random(seed, n=4000):
    """Raw stimulus: begins and ends at random, both at once now and then,
    any issued pattern -- the unit's whole function, no program."""
    rng = rng_for("loop-raw", seed)
    t = Trace({p: 0 for p, _ in INPUTS} | {"rst_n": 1})
    t.idle(2, rst_n=0)
    for _ in range(n):
        r = rng.random()
        t.cycle(loop_begin_valid=int(r < 0.12), loop_end_valid=int(0.08 < r < 0.35),
                body_start=rng.getrandbits(12), lo=rng.choice([0, 1, 7, 15, rng.getrandbits(8)]),
                hi=rng.choice([0, 3, 9, 40, rng.getrandbits(16)]), step=rng.choice([1, 2, 15, rng.getrandbits(8)]),
                may_skip=int(rng.random() < 0.3), skip=rng.getrandbits(12),
                bundle_issued=int(rng.random() < 0.8), capture_data=word(rng, 128))
    return t.cmd()


RUN_EVENTS = {}  # trace -> the closed-loop run's events (interpreter checks)


def _check_run(name, ev, legal):
    """A legal program must run as the interpreter says: every taken branch
    lands where the program goes next, every replayed word is the program's."""
    RUN_EVENTS[name] = dict(ev)
    bad = {k: v for k, v in ev.items() if "mismatch" in k}
    if legal and bad:
        raise AssertionError(f"{name}: model disagrees with the program interpreter: {bad}")


def traces(inst):
    out = []
    rng = rng_for("loop-run")
    for name, prog, legal in directed():
        cmd, ev = run(prog, rng)
        _check_run(name, ev, legal)
        out.append((name, cmd, legal))
    for s in range(6):
        r = rng_for("loop-prog", s)
        prog = finish(ops(1) + random_block(r, 0, 400) + ops(2), r)
        cmd, ev = run(prog, r)
        _check_run(f"program-{s}", ev, True)
        out.append((f"program-{s}", cmd, True))
    out += [(f"raw-{s}", raw_random(s), False) for s in range(2)]
    return out


def probes(inst):
    u = INSTANCES[inst]
    rng = rng_for("loop-probe")
    res = []

    def prog_cmd(prog):
        return run(finish(prog, rng), rng, p_stall=0.0)[0]

    def first(cmd, port_name, value, start=0):
        return next(i for i in range(start, len(cmd[port_name])) if cmd[port_name][i] == value)

    # loop.begin -> its frame visible (depth, iv_by_level): one edge
    cmd = prog_cmd(ops(3) + loop(ops(3), lo=5, trips=2) + ops(2))
    ev = first(cmd, "loop_begin_valid", 1)
    res.append(("loop.begin -> depth", 1, rtl.probe_trace(u, cmd, "depth", ev)))
    res.append(("loop.begin -> iv_by_level (iv = lo)", 1, rtl.probe_trace(u, cmd, "iv_by_level", ev)))
    # loop.end -> branch_taken: combinational (a cold continue)
    ev = first(cmd, "loop_end_valid", 1)
    res.append(("loop.end (cold continue) -> branch_taken", 0, rtl.probe_trace(u, cmd, "branch_taken", ev)))
    # loop.begin.r with hi <= lo -> branch_taken: combinational; the fetch queue then adds 3
    # (units/fetch.py), so the bundle after loop.end issues 3 cycles after the loop.begin.r
    cmd = prog_cmd(ops(3) + loop(ops(3), r=True, lo=4, hi=2) + ops(4))
    ev = first(cmd, "loop_begin_valid", 1)
    res.append(("loop.begin.r skip -> branch_taken (+3 in fetch = isa_latency.json's 3)", 0,
                rtl.probe_trace(u, cmd, "branch_taken", ev)))
    # completing loop.end -> warm (replay_en): one edge
    cmd = prog_cmd(ops(2) + loop(ops(4), trips=4) + ops(2))
    ev = first(cmd, "loop_end_valid", 1)
    res.append(("completing loop.end -> lb_replay_en", 1, rtl.probe_trace(u, cmd, "lb_replay_en", ev)))
    # measured refetch cost per iteration, cold vs warm (cycles per trip, by run length)
    res.append(("refetch cost per iteration, body 25 vs 24 (+2 cycles)", 2, refetch_cost()))
    return res


def refetch_cost():
    """Cycles per trip of a 25-bundle body (cold every pass) minus a 24-bundle
    body's (warm from the second pass), per extra bundle: closed-loop runs,
    no stalls, so it counts what the fetch flush costs each iteration."""
    rng = rng_for("loop-refetch")

    def cycles(n_body, trips):
        prog = finish(loop(ops(n_body - 1), trips=trips), rng)
        cmd, _ = run(prog, rng, p_stall=0.0, tail=0)
        return len(cmd["rst_n"]) - 2

    per24 = (cycles(LB_CAP, 6) - cycles(LB_CAP, 2)) / 4
    per25 = (cycles(LB_CAP + 1, 6) - cycles(LB_CAP + 1, 2)) / 4
    return int(per25 - per24 - 1)  # minus the one extra bundle


# ---------------------------------------------------------------------------
# seeds: the sequencer-level tbs, ports of dut.u_loop_ctrl / dut.u_loop_buffer
# ---------------------------------------------------------------------------

SEQ_SOURCES = (["src/pkg/minitpu_config_pkg.sv", "src/core/vpu/vpu_pkg.sv", "src/core/sequencer/sequencer_pkg.sv",
                "src/core/sequencer/sequencer_decoder.sv", "src/core/sequencer/sequencer_iram.sv",
                "src/core/sequencer/sequencer_fetch_queue.sv", "src/core/sequencer/sequencer_loop_buffer.sv",
                "src/core/sequencer/sequencer_loop_ctrl.sv", "src/core/sequencer/sequencer_agu_resolve.sv",
                "src/core/sequencer/sequencer_scalar_agu.sv", "src/core/sequencer/dma_desc_adapter.sv",
                "src/core/sequencer/sequencer_vpu_adapter.sv", "src/core/sequencer/sequencer.sv",
                "tb/isa_latency_pkg.sv"])
LC = "dut.u_loop_ctrl"
LB = "dut.u_loop_buffer"
SEED_MAP = {  # wrapper port -> (scope, name)
    "rst_n": (LC, "rst_n"), "loop_begin_valid": (LC, "loop_begin_valid_i"), "body_start": (LC, "loop_body_start_i"),
    "lo": (LC, "loop_lo_i"), "hi": (LC, "loop_hi_i"), "step": (LC, "loop_step_i"),
    "may_skip": (LC, "loop_may_skip_i"), "skip": (LC, "loop_skip_i"), "loop_end_valid": (LC, "loop_end_valid_i"),
    "bundle_issued": (LC, "bundle_issued_i"), "capture_data": (LB, "capture_data_i"),
    "branch_taken": (LC, "branch_taken_o"), "branch_target": (LC, "branch_target_o"), "iv_tos": (LC, "iv_tos_o"),
    "level_tos": (LC, "level_tos_o"), "lb_reset": (LC, "lb_reset_o"), "lb_capture_en": (LC, "lb_capture_en_o"),
    "lb_replay_en": (LC, "lb_replay_en_o"), "lb_replay_idx": (LC, "lb_replay_idx_o"),
    "lb_capture_overflow": (LB, "capture_overflow_o"), "replay_data": (LB, "replay_data_o"),
    "depth": (LC, "loop_depth_o"), "overflow": (LC, "loop_overflow_o"), "underflow": (LC, "loop_underflow_o"),
}
SEED_TBS = ("tb_bundle_loop", "tb_loop_begin_r", "tb_loop_buffer_cap")


def _extract(tb):
    from examples.minitpu.harness import vcd_seed

    vcd = vcd_seed._build_and_run(tb, SEQ_SOURCES)
    ch = {}
    for scope in (LC, LB):
        names = sorted({n for s, n in SEED_MAP.values() if s == scope} | ({"clk"} if scope == LC else set()))
        for n, v in vcd_seed.read_vcd(vcd, f"tb.{scope}", names).items():
            ch[(scope, n)] = v
    clk = ch[(LC, "clk")]
    edges = [t for (t, v), (_, p) in zip(clk[1:], clk[:-1]) if v == 1 and p == 0]
    times = edges + [float("inf")]

    def before(c):
        out, cur, i = [], 0, 0
        for t in times:
            while i < len(c) and c[i][0] < t:
                cur = c[i][1]
                i += 1
            out.append(cur)
        return out

    vals = {p: before(ch[k]) for p, k in SEED_MAP.items()}
    cmd = {p: vals[p] for p, _ in INPUTS}
    seen = {p: vals[p] if p in vals else [None] * len(times) for p, _ in OUTPUTS}
    return cmd, seen


def seeds():
    return [(tb, "shipped", *_extract(tb), True) for tb in SEED_TBS]


VARIANTS = {}
