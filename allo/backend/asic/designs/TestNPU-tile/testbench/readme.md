`compute_tile` wrapper testbench for the mflowgen post-route VCS node. This
harness deliberately targets the already-built post-route module, which is
named `compute_tile` but wraps the integrated `tile_top` compute plane.

`test_vectors.txt` is read with `$readmemh` as 256-bit rows. The initial
workload is `vreg.load` x8 + `mxu.load.w` + `mxu.matmul` + `halt`:

1. rows 0..10: 64-bit IRAM program words in the low bits
2. rows 11..26: SPAD preload rows, BF16 W at rows 0..7 and BF16 X at rows 8..15
3. rows 27..34: expected FP32 output rows at SPAD rows 16..23

The SystemVerilog testbench loads the program through the 12-bit instruction
write port, loads SPAD through the DMA write port, pulses `start`, waits for
`done`, then reads back SPAD rows 16..23 through the DMA read port. The eight
`vreg.load` instructions are required because this compute plane sources MXU
activations from VREG rather than directly from SPAD.

`generate_test_vectors.py` intentionally uses only the Python standard library
so vector generation does not depend on NumPy being installed on an ASIC host.
The workload uses small integer-valued BF16 inputs, making every input and
expected fp32 result exactly representable.
