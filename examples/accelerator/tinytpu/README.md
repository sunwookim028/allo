# TinyTPU

TinyTPU is a small, runnable accelerator flow. `isa.py` describes the buffers
and instruction semantics; the generic DSA compiler maps TOSA into those
instructions; `microarch.py` is the separately authored Allo-HLS instruction
interpreter. The ISA description does **not** generate the hardware decoder.

## Reproduce

From this directory, with the local project built in the `allo` Conda
environment:

```bash
make oracle       # 8 hand-written assembly examples
make compiler     # direct-TOSA compiler and spill checks
make hls          # generate and sanity-check Vitis HLS C++
```

Set `CONDA=/path/to/conda` or `ENV=name` when needed. The compiler check uses
TOSA text directly, so it does not require PyTorch or torch-mlir.

## Verified results

| Check | Result |
| --- | --- |
| Hand-written ISA programs | 8/8 pass against NumPy |
| TOSA vector add | 7 emitted instructions; output matches NumPy |
| TOSA pressure graph | 73 emitted instructions, 4 `vstore`s; output matches NumPy |
| ISA extension: TOSA negate | 5 emitted instructions; output matches NumPy |
| Hardware generation | Vitis HLS C++ exports successfully without Vitis installed |

The pressure graph has 20 vector inputs and 10 partial sums, deliberately
exceeding the eight vector-register slots. It exercises allocation, movement,
and spilling.

## Map of the flow

- `isa.py`: memories, vector-register shape, and instruction access/compute
  semantics. Add a compiler-visible operation here.
- `microarch.py`: opcode numbers, decode, datapaths, schedules, and HLS export.
  Add the matching hardware behavior here too.
- `oracle.py`: hand-written instruction-stream examples and functional checks.
- `program.py`: optional PyTorch-to-TOSA examples; requires `torch_mlir`.
- `verify.py`: dependency-light direct-TOSA checks used by the Makefile.

The `vneg` extension is intentionally small but crosses every layer: its ISA
semantics select `tosa.negate`, its opcode is 9 (leaving existing encodings
unchanged), its VPU behavior is synthesizable, and both an oracle program and a
direct-TOSA test validate it. Use the same four edits for a new operation.

The compiler accepts static-shape TOSA and performs pattern selection, exact
shape fitting, buffer allocation, routing, and spilling. It currently has fixed
8-element vectors and fixed 4x4 matmul tiles; it does not tile larger operators.

## Important integration rule

Keep instruction operand order identical in `isa.py` and `microarch.py`.
`dma_store` is the one intentional adapter: the ISA encodes `(bram source,
dram destination, length)`, while its hardware helper is named `(dram address,
bram address, length)`, so the decoder swaps the first two operands.
