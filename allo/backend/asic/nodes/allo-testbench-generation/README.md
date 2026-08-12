# Allo testbench generation

This node converts a backend-independent frozen workload plus realized HLS
artifacts into backend-specific simulation collateral. Backend selection is a
small dispatcher in `generate_testbench.py`; only `vitis` is registered today.
An unsupported backend fails explicitly.

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
