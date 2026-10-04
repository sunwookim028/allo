"""Finding 3, measured: the MXU with its output FIFOs as per-lane Streams,
on the simulator and in SystemC csim, against the contract reference.
Compared: pop data in pop order (the consumer-visible contract) and
``output_valid_o`` per cycle at shift 0."""
import os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import allo.dataflow as df
from allo.compose import Architecture, Memory
from examples.minitpu.harness import rtl, ref_mxu
from examples.minitpu.units import mxu as U
from examples.minitpu.units.mxu_pe_unit import pe_channels, pe_unit
from examples.minitpu.units.mxu_unit import mxu_front
from mxu_streams_units import gather_try, pop_engine, probe_channels

inst, D = "dim2", 2
backend = sys.argv[1] if len(sys.argv) > 1 else "simulator"
labels = sys.argv[2:] or ["identity", "fifo-full-exact", "waiting-pops", "overflow", "random-0"]
tr = {l: c for l, c, _ in U.traces(inst)}
for lab in labels:
    cmd = tr[lab]
    n = len(cmd["rst_ni"])
    packed = {p: rtl.pack(cmd[p], w) for p, w in U.INSTANCES[inst].inputs}
    want, why, ev = ref_mxu.mxu_trace(packed, D)
    mems = (Memory("RST", "UInt(1)[N]"), Memory("PUSH", "UInt(1)[N]"), Memory("KIND", "UInt(1)[N]"),
            Memory("DATA", "UInt(16)[N * D]"), Memory("CMT", "UInt(1)[N]"), Memory("RDY", "UInt(1)[N]"),
            Memory("ACC", "UInt(1)[N]"), Memory("DROP", "UInt(16)[N]"), Memory("POP", "UInt(1)[N]"),
            Memory("VLD", "UInt(1)[N]"), Memory("ODATA", "UInt(16)[N * D * SUB]"))
    params = {"N": n, "D": D, "PE": 4, "SUB": U.SUB, "ENTRIES": U.ENTRIES, "SKEW": (D - 1) * 4 + 1,
              "SPAN": ref_mxu.switch_span(D), "ROW0_PSUM_ZERO": 1, "EDGE_OUT": 0}
    arch = Architecture(name=f"mxu_streams_{inst}", parameters=params, memories=mems,
                        channels=probe_channels(pe_channels()), units=(mxu_front, pe_unit, gather_try, pop_engine))
    prj = os.path.join(os.environ.get("U3B_PRJ", "/tmp/minitpu_harness_prj"), f"streams_fifo_{lab}")
    mod = df.build(arch.region(), target=backend) if backend == "simulator" else \
        df.build(arch.region(), target="systemc", mode="csim", project=prj)
    data = [(int(v) >> (16 * c)) & 0xFFFF for v in cmd["input_data_i"] for c in range(D)]
    ins = [np.asarray(cmd["rst_ni"], np.uint8), np.asarray(cmd["input_push_i"], np.uint8),
           np.asarray(cmd["input_kind_i"], np.uint8), np.asarray(data, np.uint16),
           np.asarray(cmd["weight_commit_i"], np.uint8)]
    rdy, acc, vld = (np.zeros(n, np.uint8) for _ in range(3)); drop = np.zeros(n, np.uint16)
    od = np.zeros(n * D * U.SUB, np.uint16)
    mod(*ins, rdy, acc, drop, np.asarray(cmd["output_pop_i"], np.uint8), vld, od)
    k = D * U.SUB
    got_data = [sum(int(od[t * k + i]) << (16 * i) for i in range(k)) for t in range(n)]
    wv = rtl.unpack(want["output_valid_o"]); wd = rtl.unpack(want["output_data_o"])
    pops_w = [wd[t] for t in range(n) if cmd["output_pop_i"][t] and wv[t]]
    pops_g = [got_data[t] for t in range(n) if cmd["output_pop_i"][t] and vld[t]]
    k_ok = sum(int(a == b) for a, b in zip(pops_w, pops_g))
    v_ok = sum(int(int(vld[t]) == wv[t]) for t in range(n) if why["output_valid_o"][t] == "")
    v_tot = sum(1 for t in range(n) if why["output_valid_o"][t] == "")
    rise_w = next((t for t in range(n) if wv[t]), None); rise_g = next((t for t in range(n) if vld[t]), None)
    print(f"{backend:9s} {lab:16s} pops: rtl {len(pops_w)} allo {len(pops_g)} data-in-order {k_ok}/{len(pops_w)} | "
          f"output_valid per cycle {v_ok}/{v_tot} | first valid rtl {rise_w} allo {rise_g} | drops allo {int((drop != 0).sum())} ref {ev.get('drop', 0)}", flush=True)
