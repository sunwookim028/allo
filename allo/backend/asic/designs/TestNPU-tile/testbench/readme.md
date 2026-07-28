Legacy `compute_tile` testbench for the mflowgen post-route VCS node. This
harness deliberately targets the `compute_tile` RTL/netlist already used by
the flow; it does not target the newer `tile_top` hierarchy.

`test_vectors.txt` is read with `$readmemh` as 256-bit rows. The initial
workload is `matmul.tile + flush`:

1. rows 0..1: 64-bit IRAM program words in the low bits
2. rows 2..17: L1 preload rows, W at rows 0..7 and X at rows 8..15
3. rows 18..25: expected fp32 output rows at L1 rows 16..23

The SystemVerilog testbench loads the program through the 10-bit instruction
write port, loads L1 through the DMA write port, pulses `start`, waits for
`done`, then reads back L1 rows 16..23 through the DMA read port.

`generate_test_vectors.py` intentionally uses only the Python standard library
so vector generation does not depend on NumPy being installed on an ASIC host.
The workload uses small integer-valued fp16 inputs, making every input and
expected fp32 result exactly representable.
