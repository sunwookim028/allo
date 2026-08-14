# Allo testbench generation

This node converts a backend-independent frozen workload plus realized HLS
artifacts into backend-specific simulation collateral. Backend selection is a
small dispatcher in `generate_testbench.py`; `vitis` and `catapult` are
registered, and unsupported backends fail explicitly.

The Vitis generator derives the positional mapping from the generated C++ top
signature, obtains each argument's `m_axi` bundle from its HLS pragma, reads
port widths from the generated top RTL, and reads pointer-register addresses
from the generated AXI-Lite control RTL. It then emits a self-checking
SystemVerilog testbench, AXI memory and AXI-Lite BFMs, copied hexadecimal
vectors, a machine-readable testbench contract/report, and a VCS RTL file list.

The initial implementation supports the predictable Allo/Vitis contract used
by the scaling example: array arguments on `m_axi`, pointer registers on
`s_axi_control`, direct `ap_start`/`ap_done`, one array element per AXI beat,
and bit-exact expected-memory checks. These restrictions are checked rather
than silently guessed.

The Catapult generator consumes `asic-manifest-final.json` rather than guessing
numeric names from RTL. Each argument is matched against one exact realized
port-family schema: packed direct array, synchronous read memory, synchronous
write memory, or synchronous read/write memory. Incomplete, ambiguous, and
unknown families fail explicitly. Mixed protocols in one top are supported.

Direct arrays retain the row-major, element-zero-at-LSB packed behavior.
Synchronous memory arguments use the same hexadecimal workload vectors to
initialize/check a SystemVerilog memory model. Reads return data one cycle
after read enable, writes are captured on the rising edge, and enabled
out-of-range accesses fail. Data width, address capacity, declared depth, and
workload shape are validated before simulation.

Output `triosy` pulses are currently latched as completion evidence so outputs
completing on different cycles are handled correctly. This completion adapter
is intentionally separate from the argument data protocol and is provisional;
a future compiler-authored transaction contract can replace it without
changing direct-array or memory modeling.

Catapult calls currently require `reset_before=true`. This is intentional: the
observed direct-array top has no separate start/restart port, so reset release
is the only proven transaction launch event. The generated testbench retains
the same clock/input/output-delay plusargs used by RTL, FFGL, and BAGL runs.
