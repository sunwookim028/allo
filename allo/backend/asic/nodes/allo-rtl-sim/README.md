# Allo behavioral RTL simulation

Compiles the normalized, pre-synthesis Allo RTL with the generated workload
testbench and Vitis AXI BFMs. A passing `ALLO_TEST_PASS` marker and a nonempty
VCD are required before the unchanged `design.v` is passed to macro planning.

This early check catches protocol, finite-FIFO, backpressure, and liveness
problems before macro synthesis and physical implementation.
