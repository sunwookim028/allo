# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""U3: ``mxu``, MiniTPU's matrix unit, checked against its push / commit /
pop contract.

``mxu.sv``: a shared input FIFO (``vpu_fifo``, ``1 + DIM*16`` b x 4,
drained every cycle), the activation skew lines, the two-bank weight-load
and commit-skew logic, the ``mxu_systolic_array``, and per lane a gather
register (``pack_bf16`` per result row) feeding an output FIFO of
``MXU_OUTPUT_FIFO_DEPTH / NUM_SUBLANES`` groups. Its interface is the only
"declared" one in MiniTPU (``UNITS.md`` §2.2): push / kind / data in with
``input_ready_o``/``input_accept_o``; ``weight_commit_i``; ``output_pop_i``
with ``output_valid_o`` / ``output_data_o`` (a whole VREG per pop).

The reference is a **contract model** (``harness/ref_mxu.py``
``mxu_trace``): what a client may rely on, with one derived timing constant
(``push -> output_valid = 2 + DIM*(MXU_PE_LATENCY + 1)``), and none of
``mxu.sv``'s registers. Its outputs are all ``"pre"`` (combinational from
registered state), held on every cycle; ``output_data_o`` while valid.

Traces are *programs*, laid out as MiniTPU's stream and pop engines lay them
out (``mxu_stream_engine.sv``, ``mxu_pop_engine.sv``): a load is ``DIM``
RHS rows back to back, bottom row first, with ``weight_commit_i`` in the
cycle after the last row is pushed; a push is ``NUM_SUBLANES`` LHS rows back
to back; a pop is either one cycle where the group is valid (the pop
engine's ``beat_fires``) or ``output_pop_i`` held from before the group is
valid until it is (a waiting pop: ignored while not valid). Legal programs
keep the two rules the assembler keeps (``asm.py`` / ``isa_latency.json``):
the weight-switch span and the output-FIFO capacity. Illegal programs break
one of them on purpose.

Instances: ``dim2`` (``tb_mxu_single_port``'s DIM, default package),
``dim4`` (iteration size) and ``dim16`` (shipped). ``DIM`` is a real module
parameter; ``NUM_SUBLANES = 4`` and the output depth (64 rows = 16 groups)
come from ``vpu_pkg`` defaults.
"""

from examples.minitpu.harness import ref_mxu, rtl
from examples.minitpu.harness.traces import Trace, rng_for
from examples.minitpu.units.mxu_pe import bf16

SOURCES = ["src/core/vpu/vpu_pkg.sv", "src/core/vpu/vpu_fifo.sv",
           "src/core/mxu/mxu_bf16_mul_acc24.sv", "src/core/mxu/mxu_acc24_add_pipe.sv",
           "src/core/mxu/mxu_pe.sv", "src/core/mxu/mxu_systolic_array.sv", "src/core/mxu/mxu.sv"]
SUB = ref_mxu.NUM_SUBLANES
ENTRIES = 64 // SUB  # MXU_OUTPUT_FIFO_DEPTH / NUM_SUBLANES
LHS, RHS = 0, 1


def _unit(dim):
    return rtl.RtlUnit(
        top="mxu",
        sources=SOURCES,
        inputs=[("rst_ni", 1), ("input_push_i", 1), ("input_kind_i", 1),
                ("input_data_i", 16 * dim), ("weight_commit_i", 1), ("output_pop_i", 1)],
        outputs=[("input_ready_o", 1, "pre"), ("input_accept_o", 1, "pre"),
                 ("output_valid_o", 1, "pre"), ("output_data_o", 16 * dim * SUB, "pre")],
        shape="trace",
        params={"DIM": dim},
        assertions=True,
    )


DIMS = {"dim2": 2, "dim4": 4, "dim16": 16}
INSTANCES = {k: _unit(d) for k, d in DIMS.items()}
DEFAULT = "dim4"
RTL = INSTANCES[DEFAULT]
LATENCY_SOURCE = ("derived: 2 + DIM*(MXU_PE_LATENCY + 1) (vpu_pkg.sv:106); at DIM 16 "
                  "isa_latency.json matrix.result_latency.vmatpush 85 = 4 + 82 - 1")


def REF(inst, cmd):
    return ref_mxu.mxu_trace(cmd, DIMS[inst])


def _join16(vals):
    return sum(int(v) << (16 * i) for i, v in enumerate(vals))


class Program:
    """A cycle-indexed MXU program, laid out as the stream and pop engines
    would. Tracks what the legality rules need (banks, tile starts, group
    visibility) from the derived timing in ``ref_mxu``."""

    def __init__(self, dim, rng):
        self.D, self.rng = dim, rng
        self.rows = {}
        self.t = 2  # stream-engine time; cycles 0, 1 are reset
        self.load_bank, self.loaded_bank, self.waiting = 0, 1, False
        self.tile_start = {}  # bank -> cycle its last tile started (first LHS accepted)
        self.groups = []  # visible cycle of each group
        self.fires = []  # pop cycle of each group
        self.lhs_count = 0
        self.span = ref_mxu.switch_span(dim)
        self.p2v = ref_mxu.push_to_valid(dim)

    def put(self, c, **kv):
        self.rows.setdefault(c, {}).update(kv)

    def gap(self, k):
        self.t += k

    def load(self, W=None, legal=True, at=None):
        """DIM RHS rows, bottom first; commit the cycle after the last."""
        D = self.D
        W = W or [[bf16(self.rng, 0.03, 0.02) for _ in range(D)] for _ in range(D)]
        b = self.load_bank
        if at is not None:
            self.t = at
        elif legal and b in self.tile_start:  # first RHS accepted >= start + span
            self.t = max(self.t, self.tile_start[b] + self.span - 1)
        for i in range(D):
            self.put(self.t + i, input_push_i=1, input_kind_i=RHS, input_data_i=_join16(W[D - 1 - i]))
        self.put(self.t + D, weight_commit_i=1)
        self.t += D
        self.load_bank, self.loaded_bank, self.waiting = 1 - b, b, True
        return W

    def push(self, rows=SUB, legal=True, act=None):
        """``rows`` LHS rows back to back (a vmatpush is NUM_SUBLANES);
        ``act`` gives them (default random)."""
        D = self.D
        if legal:  # the lane-0 FIFO must have room when this group arrives
            g = len(self.groups)
            if g >= ENTRIES and len(self.fires) > g - ENTRIES:
                arrive0 = self.t + rows - 1 + 1 + 1 + D * ref_mxu.PE_LATENCY
                need = self.fires[g - ENTRIES] + 1
                self.t += max(0, need - arrive0)
        for i in range(rows):
            if self.waiting and self.lhs_count >= 0 and i == 0:
                self.tile_start[self.loaded_bank] = self.t + 1
                self.waiting = False
            a = act[i] if act else [bf16(self.rng, 0.03, 0.02) for _ in range(D)]
            self.put(self.t + i, input_push_i=1, input_kind_i=LHS, input_data_i=_join16(a))
            self.lhs_count += 1
            if self.lhs_count % SUB == 0:
                self.groups.append(self.t + i + self.p2v)
        self.t += rows

    def pop(self, delay=0, early=0, not_before=0):
        """Pop the oldest unpopped group: fire ``delay`` cycles after it is
        visible (and after the previous pop, and not before ``not_before``);
        ``early`` > 0 holds ``output_pop_i`` that many cycles before it fires
        (a waiting pop)."""
        k = len(self.fires)
        assert k < len(self.groups)
        prev = self.fires[-1] if self.fires else -1
        fire = max(self.groups[k] + delay, prev + 1, not_before)
        if early:
            fire = max(self.groups[k], prev + 1 + early)  # nothing valid while held
            for c in range(fire - early, fire):
                self.put(c, output_pop_i=1)
        self.put(fire, output_pop_i=1)
        self.fires.append(fire)
        return fire

    def pop_all(self, rng=None, max_delay=0, p_early=0.0, not_before=0):
        while len(self.fires) < len(self.groups):
            if rng and rng.random() < p_early:
                self.pop(early=rng.randrange(1, 8))
            else:
                self.pop(delay=rng.randrange(0, max_delay + 1) if rng else 0,
                         not_before=not_before)

    def after_last_group(self):
        """The first cycle every lane holds every group pushed so far."""
        return (self.groups[-1] if self.groups else self.t) + 1

    def cmd(self, tail=8):
        end = max([self.t] + list(self.rows) + self.fires + self.groups) + tail
        t = Trace({p: 0 for p, _ in _unit(self.D).inputs} | {"rst_ni": 1})
        t.idle(2, rst_ni=0)
        for c in range(2, end):
            t.cycle(**self.rows.get(c, {}))
        return t.cmd()


def random_program(inst, n_cmds, seed, max_gap=3):
    """A legal random program: loads and pushes on the stream engine at
    random gaps, pops interleaved (prompt, delayed or waiting)."""
    D = DIMS[inst]
    rng = rng_for("mxu", inst, seed)
    p = Program(D, rng)
    p.load()
    for _ in range(n_cmds):
        p.gap(rng.choice([0, 0, 0, 1, 2, max_gap]))
        if rng.random() < 0.2:
            p.load()
        else:
            p.push()
        # pop what is outstanding now and then, with a random policy
        if rng.random() < 0.6 and len(p.fires) < len(p.groups):
            if rng.random() < 0.3:
                p.pop(early=rng.randrange(1, 10))
            else:
                p.pop(delay=rng.randrange(0, 30))
    p.pop_all(rng, max_delay=5, p_early=0.2)
    return p.cmd()


def directed(inst):
    """``[(label, cmd, legal)]``."""
    D = DIMS[inst]
    out = []
    rng = rng_for("mxu-directed", inst)
    I = [[0x3F80 if r == c else 0 for c in range(D)] for r in range(D)]
    # identity, as tb_mxu_single_port: one load, one push, one pop
    p = Program(D, rng)
    p.load(W=I)
    p.push()
    p.pop()
    out.append(("identity", p.cmd(), True))
    # back to back: load, 2 pushes, the next load at the earliest legal cycle
    # into the other bank, 2 pushes, a third load refilling bank 0 exactly at
    # the switch span, pops after everything
    p = Program(D, rng)
    p.load()
    p.push(rows=1)  # tile A starts (bank 0)
    p.load()  # bank 1, streams while A switches
    p.load()  # refills bank 0 at exactly start(A) + span (the legal rule)
    p.push(rows=3), p.push(), p.push()
    p.pop_all()
    out.append(("banks-at-span", p.cmd(), True))
    # fill the output FIFO to exactly its capacity, then drain it
    p = Program(D, rng)
    p.load()
    for _ in range(ENTRIES):
        p.push()
    p.pop_all(not_before=p.after_last_group() + 3)
    out.append(("fifo-full-exact", p.cmd(), True))
    # waiting pops: every pop raised before its group is valid
    p = Program(D, rng)
    p.load()
    for _ in range(4):
        p.push()
        p.pop(early=6)
    out.append(("waiting-pops", p.cmd(), True))
    # LHS rows not in groups of NUM_SUBLANES: a group straddles a reload
    p = Program(D, rng)
    p.load()
    p.push(rows=6)
    p.load()
    p.push(rows=2)
    p.push()
    p.pop_all()
    out.append(("group-straddles-load", p.cmd(), True))
    # LHS before any commit: the active weights are reset zeros
    p = Program(D, rng)
    p.push()
    p.load()
    p.push()
    p.pop_all()
    out.append(("push-before-load", p.cmd(), True))
    # mid-trace reset with groups in the FIFOs and in flight
    p = Program(D, rng)
    p.load()
    for _ in range(3):
        p.push()
    p.pop()
    rc = p.groups[2] + 2
    c = p.cmd(tail=40)
    for k in (rc, rc + 1):
        c["rst_ni"][k] = 0
    q = Program(D, rng)  # then a fresh program after the reset
    q.t = 0
    q.load()
    q.push()
    q.pop()
    c2 = q.cmd()
    for k in c:
        c[k] = c[k] + c2[k][2:]
    out.append(("reset-mid-stream", c, True))
    # --- illegal programs -------------------------------------------------
    # output FIFO overflow: ENTRIES + 1 groups with no pop (mxu.sv:44: dropped)
    p = Program(D, rng)
    p.load()
    for _ in range(ENTRIES + 1):
        p.push(legal=False)
    p.pop_all(not_before=p.after_last_group() + 3)  # the reference drops the 17th
    out.append(("overflow", p.cmd(), False))
    # a load refilling a bank one cycle before the switch span ends
    p = Program(D, rng)
    W, act = sensitive(D)
    p.load(W=W)
    p.push(rows=1, act=act)
    p.load()
    p.load(at=p.tile_start[0] + p.span - 2, legal=False)  # accepted at start + span - 1
    p.push(rows=3)
    p.pop_all()
    out.append(("bank-overwrite", p.cmd(), False))
    # weight_commit_i before the last RHS row is accepted
    p = Program(D, rng)
    p.load()
    del p.rows[p.t]["weight_commit_i"]
    p.put(p.t - 1, weight_commit_i=1)
    p.push()
    p.pop()
    out.append(("commit-early", p.cmd(), False))
    return out


def traces(inst):
    tr = directed(inst)
    n = {"dim2": 3000, "dim4": 2000, "dim16": 300}[inst]
    tr += [(f"random-{s}", random_program(inst, n, s), True) for s in range(3)]
    return tr


def probes(inst):
    """``[(label, declared, measured)]``, step probes on the RTL."""
    D = DIMS[inst]
    u = INSTANCES[inst]
    rng = rng_for("mxu-probe", inst)
    res = []
    # push -> input_accept_o
    p = Program(D, rng)
    p.gap(4)
    ev = p.t
    p.push(rows=1)
    res.append(("push -> input_accept_o", 1, rtl.probe_trace(u, p.cmd(), "input_accept_o", ev)))
    # last LHS push of a group -> output_valid_o
    p = Program(D, rng)
    p.load()
    p.gap(4)
    p.push()
    ev = p.t - 1
    res.append(("last push of a group -> output_valid_o", ref_mxu.push_to_valid(D),
                rtl.probe_trace(u, p.cmd(tail=p.p2v + 8), "output_valid_o", ev)))
    # pop -> output_valid_o falls (one group outstanding)
    p = Program(D, rng)
    p.load()
    p.push()
    ev = p.pop(delay=3)
    res.append(("pop -> output_valid_o", 1, rtl.probe_trace(u, p.cmd(), "output_valid_o", ev)))
    # weight_commit_i -> the next push's result: switch span, measured by
    # function (monotone suffix as tb_vpu_latency_probe): the smallest offset
    # from a tile's start at which a load may refill its bank with every later
    # offset also leaving the tile's results exact
    res.append(("weight switch span (by function)", ref_mxu.switch_span(D), measure_span(inst, below=4)))
    return res


def sensitive(D):
    """Weights and activations that make any early refill visible: column
    ``D-1`` holds a distinct power of two in every row (zero elsewhere) and
    every activation is 1.0, so lane ``D-1`` is exactly the column's sum and a
    PE that latches a shifted pending row changes it (a refill shifts the
    column down: row ``r`` would take row ``r-1``'s weight)."""
    W = [[0] * D for _ in range(D)]
    for r in range(D):
        W[r][D - 1] = (127 + r - D // 2) << 7  # 2**(r - D//2), exact in bf16 and acc24
    return W, [[0x3F80] * D for _ in range(SUB)]


def measure_span(inst, below=2, above=2):
    """Refill a switching bank at offsets around the declared span; return the
    smallest offset from which the tile's results are exact at every larger
    offset tried (``isa_latency.json`` ``weight_switch.span``: 75 at DIM 16)."""
    D = DIMS[inst]
    u = INSTANCES[inst]
    span = ref_mxu.switch_span(D)
    ok = {}
    below = min(below, span - D - 1)  # two loads must fit before the refill
    for off in range(span - below, span + above + 1):
        p = Program(D, rng_for("mxu-span", inst))
        W, act = sensitive(D)
        p.load(W=W)
        p.push(rows=1, act=act)  # tile A starts on bank 0 at `start`
        start = p.tile_start[0]
        p.load()  # bank 1
        assert p.t <= start + off - 1, "offset below what two loads allow"
        p.load(at=start + off - 1, legal=False)  # bank 0: first row accepted at start + off
        p.push(rows=3)
        p.pop_all()
        cmd = p.cmd()
        packed = {q: rtl.pack(cmd[q], w) for q, w in u.inputs}
        got = rtl.run_trace(u, packed)
        want, why, _ = ref_mxu.mxu_trace(packed, D)
        fire = p.fires[0]  # the first tile's group: the one the refill can break
        ok[off] = bool((got["output_data_o"][fire] == want["output_data_o"][fire]).all())
    good = [o for o in sorted(ok) if all(ok[x] for x in ok if x >= o)]
    last_span_scan.clear()
    last_span_scan.update(ok)
    return good[0] if good else None


last_span_scan = {}  # offset -> tile exact, of the latest measure_span()


def seeds():
    """``tb_mxu_single_port`` (DIM = 2): identity weights, one push, a pop."""
    from examples.minitpu.harness import vcd_seed

    cmd, seen = vcd_seed.extract("tb_mxu_single_port", SOURCES, dut="dut", clk="clk_i",
                                 unit=INSTANCES["dim2"])
    return [("tb_mxu_single_port", "dim2", cmd, seen, True)]


VARIANTS = {}
