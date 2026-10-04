import sys, numpy as np
sys.path.insert(0, "/tmp/claude-1772902/-work-shared-users-phd-sk3463-allo/3b1c24a0-f6d1-4c37-88e0-31db609cddf4/scratchpad/u3b")
import allo.dataflow as df
from allo.compose import Architecture, Memory
from examples.minitpu.units import mxu as U
from examples.minitpu.units.mxu_pe_unit import pe_channels
from examples.minitpu.units.mxu_unit import mxu_front, mxu_channels
from examples.minitpu.harness import ref_mxu
from front_sink import front_sink
inst, D = "dim2", 2
cmd = dict((l, c) for l, c, _ in U.directed(inst))["identity"]
n = len(cmd["rst_ni"]); PE = 4
mems = (Memory("RST", "UInt(1)[N]"), Memory("PUSH", "UInt(1)[N]"), Memory("KIND", "UInt(1)[N]"),
        Memory("DATA", "UInt(16)[N * D]"), Memory("CMT", "UInt(1)[N]"), Memory("RDY", "UInt(1)[N]"), Memory("ACC", "UInt(1)[N]"),
        Memory("WT", "UInt(32)[N * D]"), Memory("NT", "UInt(64)[N * D]"))
params = {"N": n, "D": D, "PE": PE, "SKEW": (D - 1) * PE + 1, "SPAN": ref_mxu.switch_span(D)}
arch = Architecture(name="front_probe", parameters=params, memories=mems, channels=tuple(c for c in mxu_channels(pe_channels()) if c.name != "px"), units=(mxu_front, front_sink))
mod = df.build(arch.region(), target="simulator")
data = [(int(v) >> (16 * c)) & 0xFFFF for v in cmd["input_data_i"] for c in range(D)]
ins = [np.asarray(cmd["rst_ni"], np.uint8), np.asarray(cmd["input_push_i"], np.uint8), np.asarray(cmd["input_kind_i"], np.uint8),
       np.asarray(data, np.uint16), np.asarray(cmd["weight_commit_i"], np.uint8)]
rdy = np.zeros(n, np.uint8); acc = np.zeros(n, np.uint8); wt = np.zeros(n * D, np.uint32); nt = np.zeros(n * D, np.uint64)
mod(*ins, rdy, acc, wt, nt)
for t in range(n):
    row = [f"{int(wt[t*D+r]):06x}" for r in range(D)]
    col = [f"{int(nt[t*D+c]):010x}" for c in range(D)]
    print(t, "push", cmd["input_push_i"][t], cmd["input_kind_i"][t], f"{cmd['input_data_i'][t]:08x}", "cmt", cmd["weight_commit_i"][t], "acc", acc[t], "west", row, "north", col)
