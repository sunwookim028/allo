# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""U3: ``mxu_systolic_array``, MiniTPU's DIM x DIM weight-stationary array.

``mxu_systolic_array.sv``: a grid of ``mxu_pe``. Activations (and their
valid and the weight commit) enter row ``r`` on the west edge
(``lhs_i[r]``) and move east one PE per cycle; weights enter column ``c`` on
the north edge (``rhs_i[c]``, replicated into both pending banks; the bank is
chosen per column by ``rhs_valid_i[c][b]``, **broadcast to every row** of the
column, so every row shifts its pending bank down at once); partial sums move
south, row 0 starting from ``'0`` with ``psum_valid = lhs_valid``; the bottom
row's ``psum_o``/``psum_valid_o`` are ``result_o[c]``/``result_valid_o[c]``.
``weight_commit_bank_i[r][c]`` is per PE (``mxu.sv`` skews it).

``DIM`` is a real module parameter (``vpu_pkg::NUM_LANES`` by default), so
small arrays need no package define. Instances: ``dim2`` (the DIM of
``tb_mxu_single_port``), ``dim4`` (the iteration size) and ``dim16`` (the
shipped array, run once, shorter traces). The skew that makes this a matrix
multiply lives in ``mxu.sv``, not here: this unit is checked as a cycle
model (``harness/ref_mxu.py`` ``mxu_systolic_array_trace``, the PE model
wired as the ``.sv`` wires it) on random per-cycle stimulus and on
correctly skewed tiles; the matrix contract is checked on ``mxu``.

Declared latencies (derived from ``MXU_PE_LATENCY = 4``, ``vpu_pkg.sv:106``):
an activation entering row ``r`` reaches ``result_o[0]`` after
``4 * (DIM - r)`` edges (its partial sum passes ``DIM - r`` PEs); entering
the bottom row, after 4.
"""

from examples.minitpu.harness import ref_mxu, rtl
from examples.minitpu.harness.traces import Trace, rng_for
from examples.minitpu.units.mxu_pe import bf16

SOURCES = ["src/core/vpu/vpu_pkg.sv", "src/core/mxu/mxu_bf16_mul_acc24.sv",
           "src/core/mxu/mxu_acc24_add_pipe.sv", "src/core/mxu/mxu_pe.sv",
           "src/core/mxu/mxu_systolic_array.sv"]
BANKS = ref_mxu.BANKS


def _unit(dim):
    return rtl.RtlUnit(
        top="mxu_systolic_array",
        sources=SOURCES,
        inputs=[("rst_ni", 1), ("weight_commit_i", dim), ("weight_commit_bank_i", dim * dim),
                ("lhs_i", 16 * dim), ("lhs_valid_i", dim), ("rhs_i", 16 * dim),
                ("rhs_valid_i", BANKS * dim)],
        outputs=[("result_o", 24 * dim, "post"), ("result_valid_o", dim, "post")],
        shape="trace",
        params={"DIM": dim},
        assertions=True,
    )


DIMS = {"dim2": 2, "dim4": 4, "dim16": 16}
INSTANCES = {k: _unit(d) for k, d in DIMS.items()}
DEFAULT = "dim4"
RTL = INSTANCES[DEFAULT]
LATENCY_SOURCE = "derived from vpu_pkg.sv:106 MXU_PE_LATENCY = 4"


def REF(inst, cmd):
    return ref_mxu.mxu_systolic_array_trace(cmd, DIMS[inst])


def _defaults(dim):
    return {p: 0 for p, _ in _unit(dim).inputs} | {"rst_ni": 1}


def _join16(vals):
    return sum(int(v) << (16 * i) for i, v in enumerate(vals))


def random_trace(inst, n, seed, p_rst=0.002):
    D = DIMS[inst]
    rng = rng_for("mxu_array", inst, seed)
    t = Trace(_defaults(D))
    t.idle(2, rst_ni=0)
    for _ in range(n):
        t.cycle(rst_ni=int(rng.random() > p_rst),
                weight_commit_i=sum(int(rng.random() < 0.1) << r for r in range(D)),
                weight_commit_bank_i=rng.getrandbits(D * D),
                lhs_i=_join16(bf16(rng) for _ in range(D)),
                lhs_valid_i=sum(int(rng.random() < 0.7) << r for r in range(D)),
                rhs_i=_join16(bf16(rng) for _ in range(D)),
                rhs_valid_i=sum(int(rng.random() < 0.2) << i for i in range(BANKS * D)))
    return t.cmd()


def tile_trace(inst, seed, tiles=3, rows=8):
    """Correctly skewed matrix tiles, as ``mxu.sv`` drives the array: load a
    bank over DIM beats (rows in reverse), then per tile commit each row
    ``r*PE`` cycles after row 0, and enter each activation row ``r`` skewed by
    ``r*PE``; consecutive tiles alternate banks. Returns the command trace."""
    D = DIMS[inst]
    PE = ref_mxu.PE_LATENCY
    rng = rng_for("mxu_array_tile", inst, seed)
    t = Trace(_defaults(D))
    t.idle(2, rst_ni=0)
    sched = {}

    def put(c, **kv):
        row = sched.setdefault(c, {})
        for k, v in kv.items():
            row[k] = row.get(k, 0) | v

    c = 2
    for k in range(tiles):
        bank = k % 2
        W = [[bf16(rng, 0.02, 0.0) for _ in range(D)] for _ in range(D)]
        for i in range(D):  # bottom row first
            put(c + i, rhs_i=_join16(W[D - 1 - i]), rhs_valid_i=sum(1 << (2 * col + bank) for col in range(D)))
        c += D
        start = c + 1  # the tile's first activation enters row 0
        for r in range(D):
            put(start - 1 + r * PE, weight_commit_i=1 << r)
            # PE (r, col) commits r*PE + col cycles after row 0's commit
            for col in range(D):
                put(start - 1 + r * PE + col, weight_commit_bank_i=bank << (r * D + col))
        for j in range(rows):
            a = [bf16(rng, 0.02, 0.0) for _ in range(D)]
            for r in range(D):
                put(start + j + r * PE, lhs_i=a[r] << (16 * r), lhs_valid_i=1 << r)
        c = start + rows + (D - 1) * PE + D  # the switch wave has passed the last PE
    end = max(sched) + 4 * D + 8
    for cyc in range(2, end):
        t.cycle(**sched.get(cyc, {}))
    return t.cmd()


def traces(inst):
    n = {"dim2": 20000, "dim4": 20000, "dim16": 3000}[inst]
    tr = [(f"tiles-{s}", tile_trace(inst, s), True) for s in range(2)]
    tr += [(f"random-{s}", random_trace(inst, n, s), True) for s in range(2)]
    return tr


def probes(inst):
    """``[(label, declared, measured)]``: an activation entering row ``r``
    reaches ``result_o`` after ``PE * (DIM - r)`` edges (weights committed)."""
    D = DIMS[inst]
    PE = ref_mxu.PE_LATENCY
    res = []
    for r in (0, D - 1):
        t = Trace(_defaults(D))
        t.idle(2, rst_ni=0)
        for i in range(D):
            t.cycle(rhs_i=_join16([0x3F80] * D), rhs_valid_i=sum(1 << (2 * col) for col in range(D)))
        # commit all rows together, bank 0 everywhere; hold until it has swept east
        t.cycle(weight_commit_i=(1 << D) - 1)
        hold = dict(lhs_i=_join16([0x3F80] * D), lhs_valid_i=(1 << D) - 1)
        t.idle(2 * D + 4 * D + 4, **hold)
        ev = len(t)
        lhs = [0x3F80] * D
        lhs[r] = 0x4000
        t.idle(4 * D + 8, lhs_i=_join16(lhs), lhs_valid_i=(1 << D) - 1)
        res.append((f"lhs_i[{r}] -> result_o", PE * (D - r),
                    rtl.probe_trace(INSTANCES[inst], t.cmd(), "result_o", ev)))
    return res


VARIANTS = {}
