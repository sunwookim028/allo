Compute tile testbench for the mflowgen VCS node.

`test_vectors.txt` is read with `$readmemh` as 256-bit rows. The initial
workload is `matmul.tile + flush`:

1. rows 0..1: 64-bit IRAM program words in the low bits
2. rows 2..17: L1 preload rows, W at rows 0..7 and X at rows 8..15
3. rows 18..25: expected fp32 output rows at L1 rows 16..23

The SystemVerilog testbench loads the program through the instruction write
port, loads L1 through the DMA write port, pulses `start`, waits for `done`,
then reads back L1 rows 16..23 through the DMA read port.
