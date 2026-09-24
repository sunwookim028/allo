# Allo behavioral RTL simulation

Compiles the normalized, pre-synthesis Allo RTL with the generated workload
testbench. Vitis runs include its AXI BFMs; Catapult direct-array runs do not.
A passing `ALLO_TEST_PASS` marker and a nonempty
VCD are required before the unchanged `design.v` is passed to macro planning.

This early check catches protocol, finite-FIFO, backpressure, and liveness
problems before macro synthesis and physical implementation.

Backend-specific collateral checks are conditional: Vitis BFMs remain required
for Vitis and automatically pass for Catapult, while a Catapult testbench
contract is required for Catapult and automatically passes for Vitis.
