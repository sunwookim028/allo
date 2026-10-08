"""U4 track E: freeze each unit's Phase 0 oracle to an ``.npz`` the open-HLS
tools' environments can read (their MLIR builds cannot import this checkout).

    python dump_oracles.py <outdir> [unit ...]

For every unit: the joined command trace (``check._trace_all``: every trace of
the default instance + its seeds, exactly what ``check.py`` drives), the RTL's
response (Verilator, ``harness/rtl.py``, seed 1) and the reference's
defined-slot mask. Asserts RTL == reference on every defined slot before
writing, as ``check.py`` prints it. Wide ports are stored as ``uint32`` lanes
(lane 0 = bits 31:0), ``<port>`` of shape ``[n, lanes]``; ``def_<port>`` is the
defined mask (bool[n]). ``dma_addr_gen`` (``comb`` shape) stores its
``stimulus()`` and the RTL's ``word_addr``.
"""
import importlib, os, sys, time
import numpy as np

from examples.minitpu.harness import rtl
from examples.minitpu.harness.check import _trace_all

TRACE_UNITS = {  # unit -> instance
    "seq_decoder": "base", "agu_resolve": "base", "vpu_adapter": "base",
    "scalar_agu": "lat2", "dma_desc_adapter": "base",
}


def lanes(vals, width):
    k = (width + 31) // 32
    return np.array([[(int(v) >> (32 * j)) & 0xFFFFFFFF for j in range(k)] for v in vals],
                    dtype=np.uint32).reshape(len(vals), k)


def dump_trace(name, inst, out):
    u = importlib.import_module(f"examples.minitpu.units.{name}")
    unit = u.INSTANCES[inst]
    cmd, spans = _trace_all(u, inst)
    n = len(next(iter(cmd.values())))
    packed = {p: rtl.pack(cmd[p], w) for p, w in unit.inputs}
    t = time.time()
    rtl_out = rtl.run_trace(unit, packed, seed=1)
    want, reason, _ = u.REF(inst, packed)
    arrs = {}
    ndef = bad = 0
    for p, w in unit.inputs:
        arrs[p] = lanes(cmd[p], w)
    for o in unit.outputs:
        p, w = o[0], o[1]
        r = rtl.unpack(rtl_out[p])
        d = np.asarray(reason[p] == "")
        ndef += int(d.sum())
        bad += int((d & (rtl_out[p] != want[p]).any(axis=1)).sum())
        arrs[p] = lanes(r, w)
        arrs["def_" + p] = d
    arrs["span_labels"] = np.array([s[0] for s in spans])
    arrs["span_bounds"] = np.array([(s[1], s[2]) for s in spans], dtype=np.int64)
    np.savez_compressed(os.path.join(out, f"{name}.npz"), **arrs)
    print(f"DUMP {name}:{inst} {n} cycles, {len(spans)} traces, {ndef} defined slots, "
          f"RTL vs reference on defined {ndef - bad}/{ndef} ({time.time() - t:.1f}s)", flush=True)
    assert bad == 0


def dump_addr(out):
    u = importlib.import_module("examples.minitpu.units.dma_addr_gen")
    stim = u.stimulus()
    want, cyc = rtl.run(u.RTL, stim.astype(np.uint64))
    ref = u.REF(stim[:, 0], stim[:, 1], stim[:, 2])
    w = want[:, 0].astype(np.uint32)
    assert (w.astype(np.int64) == (ref & 0xFFFFFFFF)).all()
    np.savez_compressed(os.path.join(out, "dma_addr_gen.npz"), base=stim[:, 0].astype(np.uint32),
                        row=stim[:, 1].astype(np.uint32), stride=stim[:, 2].astype(np.uint32), want=w)
    print(f"DUMP dma_addr_gen {len(stim)} vectors, latency {sorted(set(cyc.tolist()))}, RTL == reference", flush=True)


if __name__ == "__main__":
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    units = sys.argv[2:] or ["dma_addr_gen"] + list(TRACE_UNITS)
    for nm in units:
        if nm == "dma_addr_gen":
            dump_addr(out)
        else:
            dump_trace(nm, TRACE_UNITS[nm], out)
