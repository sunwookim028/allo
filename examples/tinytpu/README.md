<!--- Copyright Allo authors. All Rights Reserved. -->
<!--- SPDX-License-Identifier: Apache-2.0  -->

# TinyTPU-isa

TinyTPU-isa is an int8, instruction-programmable tiled-GEMM accelerator
written in grid Allo (`@df.region` / `@df.kernel`) and taken through Vitis HLS
to RTL co-simulation. Its eight units are the modules under `ip/units/`, wired
by `ip/tinytpu.py` and composed into one region by `allo/compose.py`;
`isa_spec.json` is the ISA, from which `isa_encoding.py` (the encoder, the
decoder and the reference model's meaning of each instruction) and the ISA
reference page are generated; `microarch_isa.py` is the parameter set the
published numbers were measured on. The design is frozen as the fork's
regression reference (README decision D-4): a change to it is kept beside it
as a patch, not made to the shipped design.

Documentation: the machine, its results and how to verify a change,
[`docs/source/designs/tinytpu_isa.rst`](../../docs/source/designs/tinytpu_isa.rst)
(https://sunwookim028.github.io/allo/designs/tinytpu_isa.html); the decomposition
into composable units,
[`tinytpu_library.rst`](../../docs/source/designs/tinytpu_library.rst)
(https://sunwookim028.github.io/allo/designs/tinytpu_library.html); the
PyTorch-to-TinyTPU flow and its models,
[`workload_suite.rst`](../../docs/source/designs/workload_suite.rst)
(https://sunwookim028.github.io/allo/designs/workload_suite.html).

This README is the walk-through. A two-layer PyTorch MLP is compiled onto the
machine by ACT and run on the design itself (1); the machine is then changed
twice -- a larger systolic array (2) and a new fused instruction (3) -- and the
unchanged model is compiled and run again after each change; then the gates
that hold the shipped design to its published behaviour are run (4). Every
command below is shown with its **complete** standard output, as run on
zhang-21 on 2026-10-08. Nothing in sections 1 to 3 needs Vitis.

```text
PyTorch nn.Module                 workloads/models.py
  -> AlloTracer + torch.fx        workloads/extract.py     shapes and ops
  -> one workload spec per layer                           act/corpus/'s JSON
  -> ACT mapspace search          act/search.py, act_target.py
  -> a TinyTPU program                                     checked on isa_ref
  -> the design, simulated        ip/, df.build(target="simulator")
  -> compared with PyTorch        workloads/run.py
```

## Setup

The `allo` conda environment with this checkout's own MLIR bindings, `torch`
(an optional dependency upstream, required by the PyTorch front end here), and
`LLVM_BUILD_DIR` set the way the host wants it. From the repository root:

```bash
source $(conda info --base)/etc/profile.d/conda.sh && conda activate allo
export LLVM_BUILD_DIR=/home/sk3463/llvm-allo-6b09f739/build   # ace-01 only; on zhang-21 the env sets it: don't override it there
export OMP_NUM_THREADS=8
examples/tinytpu/reproduce.sh --no-cosim   # once per checkout: builds mlir/build, runs the gates (section 4)
pip install --index-url https://download.pytorch.org/whl/cpu torch==2.14.0   # if the env has no torch
cd examples/tinytpu
```

On zhang-21 the shared `allo` env has no torch, and the pinned install is the
clone `allo-torch214` (`dev/toolchains.rst`, "Installed 2026-10-02"): use
`conda activate allo-torch214` in place of `conda activate allo` for `make
mlp`. The `Makefile` runs the shell's `python` whenever it can import `allo`
with the checkout on `sys.path`, as the scripts do, and falls back to `conda
run -n allo` otherwise; it exports `OMP_NUM_THREADS` and, on ace-01,
`LLVM_BUILD_DIR` if they are unset. `make -n mlp` shows which python it picked.
The first line of every `make` output below is `make` echoing the command it
runs.

## 1. The model

An ordinary `nn.Module` with nothing in it about Allo or the TinyTPU
(`workloads/models.py`, `MlpSmall`), registered with its batch size and input
width in the `MODELS` table there, which is how `make mlp MODEL=<name>` finds
it. `models.build` fills the weights with integers in [-8, 8), so casting the
model to int8 is exact: the walk-through measures a mapping, not a
quantisation scheme.

`make mlp` runs `workloads/demo.py`, one section per stage; each stage is a
function in that file, so a script can reuse any of them. About 30 s:

```bash
make mlp
```

```text
python workloads/demo.py mlp_small --top 3

[1] PyTorch: mlp_small, input (8, 32)
-------------------------------------
# workloads/models.py
class MlpSmall(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(32, 32, bias=False)
        self.fc2 = nn.Linear(32, 16, bias=False)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))

[2] Trace: AlloTracer + torch.fx ShapeProp
------------------------------------------
graph():
    %x : [num_users=1] = placeholder[target=x]
    %fc1 : [num_users=1] = call_module[target=fc1](args = (%x,), kwargs = {})
    %relu : [num_users=1] = call_function[target=torch.relu](args = (%fc1,), kwargs = {})
    %fc2 : [num_users=1] = call_module[target=fc2](args = (%relu,), kwargs = {})
    return fc2

[3] Workload specs: 2 layers
----------------------------
  mlp_small_l0   mk,kn->mn  m=8 k=32 n=32  epilogue=relu+saturate
  mlp_small_l1   mk,kn->mn  m=8 k=32 n=16  epilogue=saturate

[4] Compile with ACT: search the mapspace onto the TinyTPU
----------------------------------------------------------

  mlp_small_l0: 2036 loop nests, 3 legal, 2033 refused
      refused  1722  acc-split
      refused   242  acc-position
      refused    61  ar-distance
      refused     8  AGU_TERMS
    rank  mapping        instrs est. cycles
       1  N8>K8              14        1186
       2  M2>N8>K8           16        1588
       3  N8>M2>K8           16        1790
    picked N8>K8; the program against the spec's einsum (isa_ref, 4 operand distributions): BIT-EXACT

  mlp_small_l1: 1175 loop nests, 3 legal, 1172 refused
      refused   990  acc-split
      refused   141  acc-position
      refused    37  ar-distance
      refused     4  AGU_TERMS
    rank  mapping        instrs est. cycles
       1  N4>K8              13         682
       2  M2>N4>K8           15         884
       3  N4>M2>K8           15         984
    picked N4>K8; the program against the spec's einsum (isa_ref, 4 operand distributions): BIT-EXACT

[5] The program ACT emitted for mlp_small_l0 (14 instructions)
--------------------------------------------------------------
    0  loop x8
    1    dma_ld rows=8 mode=2 dram_row0=0 col_block=0 dst_row0=0   agu: col_block+=iv0*1, dst_row0+=iv0*64
    2  endloop
    3  loop x8
    4    dma_ld rows=32 mode=1 dram_row0=0 col_block=0 dst_row0=0   agu: col_block+=iv0*1, dst_row0+=iv0*64
    5  endloop
    6  loop x8
    7    mm rows=8 vr_a=0 ar0=0 acc=0 spad_w=0   agu: spad_w+=iv0*64
    8    loop x7
    9      mm rows=8 vr_a=64 ar0=0 acc=1 spad_w=4   agu: vr_a+=iv1*64, spad_w+=iv0*64, spad_w+=iv1*4
   10    endloop
   11    vrelu rows=8 ar_d=0 ar_s=0
   12    mvout rows=8 ar0=0 dram_row0=0 col_block=0   agu: col_block+=iv0*1
   13  endloop

[6] Build the design: df.build(tinytpu_isa, target='simulator')
---------------------------------------------------------------
  built: T=4 (4x4 array), MAXDIM=64, DMA_WORDS=1

[7] Run the model on the design, one program per layer
------------------------------------------------------
  mlp_small_l0   8x32x32   design vs isa_ref over all 4096 bytes of C: 0 differ
  mlp_small_l1   8x32x16   design vs isa_ref over all 4096 bytes of C: 0 differ

[✓] Against PyTorch
-------------------
  0 of 384 output bytes differ
  torch  [127, -128, 127, -128, -128, -128, -128, -128, 127, -128, -128, 127, 127, -128, -128, -128]
  tpu    [127, -128, 127, -128, -128, -128, -128, -128, 127, -128, -128, 127, 127, -128, -128, -128]

Summary (cycles are the cost model's estimate; `make mlp-cosim` measures)
  layer          mapping        instrs est. cycles
  mlp_small_l0   N8>K8              14        1186
  mlp_small_l1   N4>K8              13         682
  model                                       1868
```

What each section shows:

- **[2] Trace, [3] Specs.** `torch.fx` gives the graph with shapes. Each
  `nn.Linear` becomes one GEMM spec, and the ReLU that follows it becomes that
  spec's epilogue.
- **[4] Compile.** ACT lists every loop nest that tiles the GEMM onto the 4x4
  array, lowers each to a TinyTPU program, refuses the nests the instruction
  word cannot express (with the reason), and ranks the rest with its cost
  model. The winner is run on `isa_ref` (the ISA as numpy) against the spec's
  einsum.
- **[5] The program.** `disasm.py` prints it with the operand names from the
  ISA spec. Note the `vrelu` then `mvout` at the end: the ReLU is a pass over
  the accumulator of its own, and it is what section 3 removes.
- **[6], [7] Build and run.** `df.build(tinytpu_isa, target="simulator")`
  builds the eight-unit dataflow design from `ip/` (about 5 s). Each layer's
  program runs on it with the real weights in DRAM, and each layer's output
  becomes the next layer's input.

Two checks are made, and they catch different faults. The design against
`isa_ref`, over the whole 4096-byte `C` buffer, catches the hardware. The
model's output against PyTorch catches the compiler and the flow. Most output
bytes are saturated at 127 or -128, because weights and inputs in [-8, 8)
summed over K=32 overflow int8, so the row comparison is weak evidence on its
own and the per-layer full-buffer check is the strong one. The cycle column is
the cost model's estimate; section 4 has the measurement.

Other models: `make mlp MODEL=mlp_deep TOP=5` (four layers, five mappings
each); `make mlp-compile` stops after ACT, with no design build. A model the
machine cannot run is refused with the reason, not approximated (`make` exits
1 and reports that on stderr):

```bash
make mlp MODEL=mlp_bias
```

```text
python workloads/demo.py mlp_bias --top 3

[1] PyTorch: mlp_bias, input (4, 16)
------------------------------------
# workloads/models.py
class MlpBias(nn.Module):
    """The probe: a bias the ISA has no epilogue for and a sigmoid it has no
    unit for. It exists so the report names refusals instead of implying the
    suite covers everything an MLP can contain."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(16, 16, bias=True)
        self.fc2 = nn.Linear(16, 16, bias=False)

    def forward(self, x):
        return self.fc2(torch.sigmoid(self.fc1(x)))

[2] Trace: AlloTracer + torch.fx ShapeProp
------------------------------------------
graph():
    %x : [num_users=1] = placeholder[target=x]
    %fc1 : [num_users=1] = call_module[target=fc1](args = (%x,), kwargs = {})
    %sigmoid : [num_users=1] = call_function[target=torch.sigmoid](args = (%fc1,), kwargs = {})
    %fc2 : [num_users=1] = call_module[target=fc2](args = (%sigmoid,), kwargs = {})
    return fc2
  REFUSED fc1: nn.Linear with bias: the ISA has vadd but no mapping that broadcasts a bias row into the accumulator, so the baseline cannot express it
  REFUSED sigmoid: call_function torch.sigmoid is not an int8 GEMM or a fused ReLU; this build computes int8 x int8 -> int32 with a ReLU epilogue

The ISA has no instruction for the refused nodes, so the model does not compile.
```

## 2. A larger array

The design's parameters live in `ip/params.py` (`TpuParams`). `T` is the array
dimension and the only parameter that changes the *shape* of the design:
`T*T` processing elements and `T`- and `T*T`-wide stream arrays; every size is
derived from it. The shipped build reads `TPU_T` from the environment, so an
8x8 array is one variable. ACT searches again and finds different tilings
with half the loop trips; the design is still bit-exact and the model still
matches PyTorch. About 65 s, the design being four times larger:

```bash
TPU_T=8 make mlp
```

```text
python workloads/demo.py mlp_small --top 3

[1] PyTorch: mlp_small, input (8, 32)
-------------------------------------
# workloads/models.py
class MlpSmall(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(32, 32, bias=False)
        self.fc2 = nn.Linear(32, 16, bias=False)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))

[2] Trace: AlloTracer + torch.fx ShapeProp
------------------------------------------
graph():
    %x : [num_users=1] = placeholder[target=x]
    %fc1 : [num_users=1] = call_module[target=fc1](args = (%x,), kwargs = {})
    %relu : [num_users=1] = call_function[target=torch.relu](args = (%fc1,), kwargs = {})
    %fc2 : [num_users=1] = call_module[target=fc2](args = (%relu,), kwargs = {})
    return fc2

[3] Workload specs: 2 layers
----------------------------
  mlp_small_l0   mk,kn->mn  m=8 k=32 n=32  epilogue=relu+saturate
  mlp_small_l1   mk,kn->mn  m=8 k=32 n=16  epilogue=saturate

[4] Compile with ACT: search the mapspace onto the TinyTPU
----------------------------------------------------------

  mlp_small_l0: 680 loop nests, 3 legal, 677 refused
      refused   495  acc-split
      refused   141  acc-position
      refused    37  ar-distance
      refused     4  AGU_TERMS
    rank  mapping        instrs est. cycles
       1  N4>K4              14         607
       2  M2>N4>K4           16         833
       3  N4>M2>K4           16         833
    picked N4>K4; the program against the spec's einsum (isa_ref, 4 operand distributions): BIT-EXACT

  mlp_small_l1: 185 loop nests, 3 legal, 182 refused
      refused   129  acc-split
      refused    40  acc-position
      refused    13  ar-distance
    rank  mapping        instrs est. cycles
       1  N2>K4              13         393
       2  M2>N2>K4           15         506
       3  N2>M2>K4           15         506
    picked N2>K4; the program against the spec's einsum (isa_ref, 4 operand distributions): BIT-EXACT

[5] The program ACT emitted for mlp_small_l0 (14 instructions)
--------------------------------------------------------------
    0  loop x4
    1    dma_ld rows=8 mode=2 dram_row0=0 col_block=0 dst_row0=0   agu: col_block+=iv0*1, dst_row0+=iv0*64
    2  endloop
    3  loop x4
    4    dma_ld rows=32 mode=1 dram_row0=0 col_block=0 dst_row0=0   agu: col_block+=iv0*1, dst_row0+=iv0*64
    5  endloop
    6  loop x4
    7    mm rows=8 vr_a=0 ar0=0 acc=0 spad_w=0   agu: spad_w+=iv0*64
    8    loop x3
    9      mm rows=8 vr_a=64 ar0=0 acc=1 spad_w=8   agu: vr_a+=iv1*64, spad_w+=iv0*64, spad_w+=iv1*8
   10    endloop
   11    vrelu rows=8 ar_d=0 ar_s=0
   12    mvout rows=8 ar0=0 dram_row0=0 col_block=0   agu: col_block+=iv0*1
   13  endloop

[6] Build the design: df.build(tinytpu_isa, target='simulator')
---------------------------------------------------------------
  built: T=8 (8x8 array), MAXDIM=64, DMA_WORDS=1

[7] Run the model on the design, one program per layer
------------------------------------------------------
  mlp_small_l0   8x32x32   design vs isa_ref over all 4096 bytes of C: 0 differ
  mlp_small_l1   8x32x16   design vs isa_ref over all 4096 bytes of C: 0 differ

[✓] Against PyTorch
-------------------
  0 of 384 output bytes differ
  torch  [127, -128, 127, -128, -128, -128, -128, -128, 127, -128, -128, 127, 127, -128, -128, -128]
  tpu    [127, -128, 127, -128, -128, -128, -128, -128, 127, -128, -128, 127, 127, -128, -128, -128]

Summary (cycles are the cost model's estimate; `make mlp-cosim` measures)
  layer          mapping        instrs est. cycles
  mlp_small_l0   N4>K4              14         607
  mlp_small_l1   N2>K4              13         393
  model                                       1000
```

The cost model is fitted to cosim at `T=4`, so the fall from 1868 to 1000
estimated cycles is its extrapolation, not a measurement. The measurement is
`TPU_T=8 make mlp-cosim MODEL=mlp_small` (Vitis: one `csynth`, then one cosim
per layer; minutes), the `T=4` form of which is in section 4.

To change the *shipped* default rather than one run, edit it in both
`isa_spec.json` (the `T` parameter) and `microarch_isa.py`, then `python
gen_isa.py --write`. Editing only `microarch_isa.py` fails: `isa_encoding.py`,
which `isa_ref` reads, is generated from the spec and keeps modelling a 4x4
machine, and `gen_isa.py --check` reports "the reference model would be
modelling a different machine". The two defaults are kept apart on purpose, so
the design is checked against an ISA written down independently of it.

## 3. A fused instruction

The `vrelu` and `mvout` at the end of every ReLU layer are two passes over the
accumulator rows. `mvout` already clips each lane to int8 on its way out, so it
can apply the ReLU in the same step: `mvoutrelu`. The whole change is one
git-tracked patch beside the design, `mvoutrelu.patch`, which stays off the
shipped design (D-4). It applies from the repository root and prints nothing:

```bash
cd ../..                                   # the repository root
git apply examples/tinytpu/mvoutrelu.patch
git apply --stat examples/tinytpu/mvoutrelu.patch
```

```text
 docs/source/designs/tinytpu_isa_spec.rst |   66 +++++++++++++++++++++++-
 examples/tinytpu/act_machine.py          |    9 +++
 examples/tinytpu/act_target.py           |   16 +++++-
 examples/tinytpu/ip/assembler.py         |    8 +--
 examples/tinytpu/ip/isa.py               |    4 +
 examples/tinytpu/ip/units/accumulator.py |   15 +++++
 examples/tinytpu/ip/units/dma_store.py   |    3 +
 examples/tinytpu/ip/units/sequencer.py   |    5 +-
 examples/tinytpu/isa_dsl.py              |    6 ++
 examples/tinytpu/isa_encoding.py         |   12 ++++
 examples/tinytpu/isa_spec.json           |   84 ++++++++++++++++++++++++++++++
 examples/tinytpu/microarch_isa.py        |    4 +
 examples/tinytpu/stress_isa.py           |    3 +
 examples/tinytpu/units_isa.py            |   16 +++++-
 14 files changed, 230 insertions(+), 21 deletions(-)
```

Eleven of the files are the designer's change, in the four steps below; the
other three (`isa_encoding.py`, `units_isa.py` and the ISA reference page
`docs/source/designs/tinytpu_isa_spec.rst`) are what `python gen_isa.py
--write` and `python lift_units.py` regenerate from the spec and the units,
included so that the patched tree is complete as applied and `gen_isa.py
--check` has something to hold it to.

**Step 1: declare it in the ISA** (`isa_spec.json`). One entry: an opcode
number, the operand fields (the same as `mvout`'s) and its *actions*, the
per-unit steps that say what the instruction does. Compared with `mvout` it
adds one `max0` on the accumulator's ALU before the clip:

```json
{
  "name": "mvoutrelu",
  "value": 11,
  "software_constant": "OP_MVOUTRELU",
  "rows": "nr accumulator rows",
  "operands": [ {"field": "f0", "name": "ar0", ...},
                {"field": "f1", "name": "dram_row0", ...},
                {"field": "f2", "name": "col_block", ...} ],
  "actions": [
    {"unit": "accu",   "kind": "read",    "port": "ar.read", "state": "ar", "base": "ar0", "into": "value", ...},
    {"unit": "accu",   "kind": "compute", "port": "alu", "compute": "max0",       "args": ["value"],     "into": "rectified"},
    {"unit": "accu",   "kind": "compute", "port": "alu", "compute": "to_operand", "args": ["rectified"], "into": "clipped"},
    {"unit": "accu",   "kind": "emit",    "port": "ac2sp", "args": ["clipped"]},
    {"unit": "dma_st", "kind": "receive", "port": "ac2sp", "into": "clipped_in", "args": ["clipped"]},
    {"unit": "dma_st", "kind": "write",   "port": "dram.write", "state": "C", "base": "dram_row0", "args": ["clipped_in"], "offset": "col_block * T"}
  ]
}
```

**Step 2: build it in the hardware** (`ip/isa.py`, `ip/units/accumulator.py`,
`ip/units/sequencer.py`, `ip/units/dma_store.py`). The opcode constant goes in
`ip/isa.py`. The accumulator, an Allo `@df.kernel`, reads the source row from
`f0` as `mvout` does and rectifies in the step that already clips. The
sequencer sends the instruction to the same two units as `mvout`. `dma_st`
only gets a comment, because it receives rows that `accu` has already
rectified:

```diff
         if op == OP_MVOUT:
             read_row = f0 + row
+        if op == OP_MVOUTRELU:     # retires from f0, exactly as `mvout`
+            read_row = f0 + row
 ...
         else:
             do_write = 0
+            # `mvoutrelu` rectifies in the same step `mvout` clips in.
+            rectify: int32 = 0
+            if op == OP_MVOUTRELU:
+                rectify = 1
             clipped_word: UInt(VW) = 0
             with allo.meta_for(T) as clip_lane:
                 retiring: int32 = read_word[32 * clip_lane : 32 * (clip_lane + 1)]
                 ...
                 if retiring < -128:
                     retiring = -128
+                if rectify == 1:
+                    if retiring < 0:
+                        retiring = 0
                 clipped: int8 = retiring
```

```diff
             if op == OP_MVOUT:
                 c_acc.put(resolved)
                 c_dst.put(resolved)
+            if op == OP_MVOUTRELU:
+                c_acc.put(resolved)
+                c_dst.put(resolved)
```

**Step 3: teach the toolchain** (`ip/assembler.py`, `isa_dsl.py`,
`microarch_isa.py`, `stress_isa.py`). The assembler checks the new
instruction's accumulator reads and adds its rows to the per-unit work counts
in the program header; the program DSL gets an `mvoutrelu()` method; and the
stress fuzzer mixes it into its random programs half the time:

```diff
-            elif op == OP_MVOUT:
+            elif op in (OP_MVOUT, OP_MVOUTRELU):
                 for i, row in enumerate(span("ar", f0, nr)):
                     ar_read(row, accu_step + i, "the value to retire")
```

```diff
+    def mvoutrelu(self, ar, dram_row=0, col_block=0, rows=0):
+        """`vrelu` then `mvout`, in one pass of the accumulator."""
+        self._ins(OP_MVOUTRELU, ar, dram_row, col_block, nr=rows)
```

```diff
-        k.mvout(s, dram_row=ri(0, MAXDIM - n), col_block=ri(0, WPR - 1), rows=n)
+        retire = k.mvoutrelu if ri(0, 1) else k.mvout
+        retire(s, dram_row=ri(0, MAXDIM - n), col_block=ri(0, WPR - 1), rows=n)
```

**Step 4: let the compiler use it** (`act_machine.py`, `act_target.py`). ACT's
cost model gets the opcode with the same unit loads as `mvout`, and the
lowering gets one rule: if a layer's epilogue ends in an op the retiring
instruction can apply (`FUSED_RETIRE`), fold it into the retire. Nothing else
in ACT changes; the mapspace search, ranking and checks are the same:

```diff
+MVOUTRELU = Opcode(
+    code=OP_MVOUTRELU, name="mvoutrelu",
+    loads=MVOUT.loads, reads=MVOUT.reads, writes=MVOUT.writes)
```

```diff
+# An epilogue op the retiring instruction can apply on its way out. When the
+# LAST op of the epilogue is one of these, it is folded into the `mvout` and
+# costs no accumulator pass of its own.
+FUSED_RETIRE = {
+    "relu": lambda k: k.mvoutrelu,
+}
 ...
-            for op in workload.epilogue:
+            epilogue, retire = list(workload.epilogue), k.mvout
+            if epilogue and epilogue[-1] in FUSED_RETIRE:
+                retire = FUSED_RETIRE[epilogue.pop()](k)
+            for op in epilogue:
                 EPILOGUE[op](k, result, result, rows)
-            k.mvout(result, dram_row=walked(0, row_terms),
-                    col_block=walked(0, column_terms), rows=rows)
+            retire(result, dram_row=walked(0, row_terms),
+                   col_block=walked(0, column_terms), rows=rows)
```

**Check the instruction itself.** Two gates, both run in `examples/tinytpu`.
`gen_isa.py --check` holds the spec, the generated files and the nine
consumers (the design's constants and bit slices, the encoder, the assembler's
header, the reference model, the action model, the sequencer's dispatch, ...)
to each other; the counts in its report grow by one instruction. About 70 s:

```bash
cd examples/tinytpu
python gen_isa.py --check
```

```text
TinyTPU-isa ISA conformance, against examples/tinytpu/isa_spec.json

  examples/tinytpu/isa_encoding.py: up to date (32661 bytes, byte-identical)
  docs/source/designs/tinytpu_isa_spec.rst: up to date (43955 bytes, byte-identical)
  design constants: 26 held to the spec by value
  design layout: 15 names in ip/isa.py held to the field table by value
  design parameters: 8 in range, read from their environment variables, constraints hold
  parameter agreement: the design and the generated module land on the same 20 values under 6 configurations, defaults included
  hardware bit slices: 9 distinct slices of the 13 64-bit words ip/units declares (accu_copy, agu_word, array_counts_word, control_word, count_word, dram_imem, fused_copy, header, program, resolved, span_word, spm_copy, word), all defined by the spec
  encoder: 2900 words identical between microarch_isa.enc/enc_agu and the spec's
  control flow and AGU: the design's `expand` and the spec's agree on all 73 programs
  imem header: microarch_isa.assemble's 8 words equal the spec's on all 73 programs
  numerics: operand and accumulator lane widths and the mvout saturation bounds match the int8 configuration
  reference model: takes its opcodes, field layout, control flow and numerics from the generated module, not from the design
  action model: 51 actions over 7 units compose 10 instructions; the rule accepts and leaves 62 obligation(s)
    status: 51 description (description is what an undecorated action is worth, and checked must name its checker; nothing here is guaranteed)
  dispatch rewrites: the actions imply 3, and ip/units/sequencer.py performs exactly those -- mm/spm -> 5, vadd/accu -> 6, vaddrelu/accu -> 6
  sequencer dispatch: 8 opcodes feed exactly the queues their actions name, read out of ip/units/sequencer.py
  unit ISA namespaces: 5 units decode exactly the opcodes they have actions for
  ports against the composed region: 20 of the spec's ports ARE ip/tinytpu.py's own channels and memories and 4 more stand for them; 3 are arithmetic no composition carries and 1 carries nothing at all; 23 composed ports are not modelled, each with a reason; and 1 unit(s) are several composed units: array = wld + pe
  write-before-read surface: 11 read actions name a memory the assembler must have seen written; the contract is carried as an obligation, not as a property of any one instruction
  allo.encoding.TINYTPU_ISA: 2 budget(s) held to the spec by value
  derived properties: 5 recomputed from the spec and confirmed against the design --
      every operand field is an AGU target, `acc` included (resolved acc [0, 1]);
      terms are additively monotone, so `acc` at base 0 stride 1 is drivable for 2 trips and rejected at 3;
      the shipped accumulating mm already spends 3/3 terms, so a term on `acc` needs 4 and the budget refuses it first
      the encoding's MAXDIM ceilings, computed at T=4: addressing 88, cubic_header 76 (binding 76); the design 4 accepts, 72 accepts, 76 accepts, 80 refuses in assemble, 88 refuses in assemble, 92 refuses at import -- each ceiling confirmed by the stage it fires at
      the program limits predict exactly which of 30 GEMM shapes the design assembles

  ISA OK: the spec, its 2 generated artefacts and all 9 consumers agree
```

`stress_isa.py` runs 640 programs on the built design against `isa_ref`,
with the random programs now retiring through `mvoutrelu` half the time. About
5 min on a loaded host, 3 min otherwise:

```bash
python stress_isa.py
```

```text
  validator: 18 crafted bad programs rejected, 582 generated programs accepted
  STRESS OK: 640/640 runs exact (full: full-range/corner/boundary operands, prefilled C compared in full, GEMM at 96 shapes, vector and random programs)
```

Then the same model, with the same command as section 1. ACT now retires the
ReLU layer with the new instruction, one instruction and one accumulator pass
shorter; `mlp_small_l1` has no ReLU and is unchanged; the estimate falls by
what the `vrelu` pass cost:

```bash
make mlp
```

```text
python workloads/demo.py mlp_small --top 3

[1] PyTorch: mlp_small, input (8, 32)
-------------------------------------
# workloads/models.py
class MlpSmall(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(32, 32, bias=False)
        self.fc2 = nn.Linear(32, 16, bias=False)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))

[2] Trace: AlloTracer + torch.fx ShapeProp
------------------------------------------
graph():
    %x : [num_users=1] = placeholder[target=x]
    %fc1 : [num_users=1] = call_module[target=fc1](args = (%x,), kwargs = {})
    %relu : [num_users=1] = call_function[target=torch.relu](args = (%fc1,), kwargs = {})
    %fc2 : [num_users=1] = call_module[target=fc2](args = (%relu,), kwargs = {})
    return fc2

[3] Workload specs: 2 layers
----------------------------
  mlp_small_l0   mk,kn->mn  m=8 k=32 n=32  epilogue=relu+saturate
  mlp_small_l1   mk,kn->mn  m=8 k=32 n=16  epilogue=saturate

[4] Compile with ACT: search the mapspace onto the TinyTPU
----------------------------------------------------------

  mlp_small_l0: 2036 loop nests, 3 legal, 2033 refused
      refused  1722  acc-split
      refused   242  acc-position
      refused    61  ar-distance
      refused     8  AGU_TERMS
    rank  mapping        instrs est. cycles
       1  N8>K8              13        1085
       2  M2>N8>K8           15        1588
       3  N8>M2>K8           15        1790
    picked N8>K8; the program against the spec's einsum (isa_ref, 4 operand distributions): BIT-EXACT

  mlp_small_l1: 1175 loop nests, 3 legal, 1172 refused
      refused   990  acc-split
      refused   141  acc-position
      refused    37  ar-distance
      refused     4  AGU_TERMS
    rank  mapping        instrs est. cycles
       1  N4>K8              13         682
       2  M2>N4>K8           15         884
       3  N4>M2>K8           15         984
    picked N4>K8; the program against the spec's einsum (isa_ref, 4 operand distributions): BIT-EXACT

[5] The program ACT emitted for mlp_small_l0 (13 instructions)
--------------------------------------------------------------
    0  loop x8
    1    dma_ld rows=8 mode=2 dram_row0=0 col_block=0 dst_row0=0   agu: col_block+=iv0*1, dst_row0+=iv0*64
    2  endloop
    3  loop x8
    4    dma_ld rows=32 mode=1 dram_row0=0 col_block=0 dst_row0=0   agu: col_block+=iv0*1, dst_row0+=iv0*64
    5  endloop
    6  loop x8
    7    mm rows=8 vr_a=0 ar0=0 acc=0 spad_w=0   agu: spad_w+=iv0*64
    8    loop x7
    9      mm rows=8 vr_a=64 ar0=0 acc=1 spad_w=4   agu: vr_a+=iv1*64, spad_w+=iv0*64, spad_w+=iv1*4
   10    endloop
   11    mvoutrelu rows=8 ar0=0 dram_row0=0 col_block=0   agu: col_block+=iv0*1
   12  endloop

[6] Build the design: df.build(tinytpu_isa, target='simulator')
---------------------------------------------------------------
  built: T=4 (4x4 array), MAXDIM=64, DMA_WORDS=1

[7] Run the model on the design, one program per layer
------------------------------------------------------
  mlp_small_l0   8x32x32   design vs isa_ref over all 4096 bytes of C: 0 differ
  mlp_small_l1   8x32x16   design vs isa_ref over all 4096 bytes of C: 0 differ

[✓] Against PyTorch
-------------------
  0 of 384 output bytes differ
  torch  [127, -128, 127, -128, -128, -128, -128, -128, 127, -128, -128, 127, 127, -128, -128, -128]
  tpu    [127, -128, 127, -128, -128, -128, -128, -128, 127, -128, -128, 127, 127, -128, -128, -128]

Summary (cycles are the cost model's estimate; `make mlp-cosim` measures)
  layer          mapping        instrs est. cycles
  mlp_small_l0   N8>K8              13        1085
  mlp_small_l1   N4>K8              13         682
  model                                       1767
```

Both gates earned their keep while the change was written; the first version
of it failed each of them. `gen_isa.py --check` failed twice: the
accumulator's clip no longer read as a literal saturation at -128, which is how
the check verifies it; and `dma_st`, now reached by two opcodes, looked like it
should decode them -- it should not, since both do the same thing there, and
that was a gap in the check, which now lets a unit skip decoding when every
opcode that reaches it has identical actions. `stress_isa.py` first deadlocked
in `kpn_model`, a channel-level model of the design that had its own
hand-written "only `mvout` sends a row to `dma_st`" (that rule is now derived
from the spec's actions), and then found a real hardware bug: the accumulator
read the source row from `f1`, the DRAM-row field, instead of `f0`. `make mlp`
had passed anyway, because in its programs both fields are 0; only the random
programs exposed it. That is the line `if op == OP_MVOUTRELU: read_row = f0 +
row` in step 2. The two fixes to the checks are part of the tree, so the patch
is just the designer's change.

Undo it. The tree is byte-identical to before; `git status --short` prints
nothing:

```bash
cd ../..
git apply -R examples/tinytpu/mvoutrelu.patch
git status --short
cd examples/tinytpu
```

## 4. The gates

`reproduce.sh` is the one command from a clean checkout to the published
behaviour: it builds this checkout's bindings in-tree, then runs every
functional gate against the shipped design and, without `--no-cosim`, Vitis
cosim at the five published shapes. With `--no-cosim`, about 3 min once
`mlir/build` is warm (each gate prints its verdict line; `reproduce.sh --help`
lists what each one holds):

```bash
./reproduce.sh --no-cosim
```

```text
== building mlir/build (incremental)
   allo -> /work/shared/users/phd/sk3463/scratch/wt-ttex/allo/__init__.py
== gen_isa.py --check (the ISA spec and both its consumers)
  ISA OK: the spec, its 2 generated artefacts and all 9 consumers agree
== lift_units.py --check (units_isa.py against what ip/ composes to)
  UNITS OK: units_isa.py is byte-identical to what ip/ composes to
== bench_isa.py (published functional setup)
  ALL EXACT
== stress_isa.py (correctness gate)
  STRESS OK: 492/492 runs exact (full: full-range/corner/boundary operands, prefilled C compared in full, GEMM at 64 shapes, vector and random programs)
== act_compile.py --gate (every mapping the search accepts, verified)
ACT GATE OK: 12/12 problems, every encodable mapping verified
REPRODUCED (functional only)
```

**The cosim cycle counts.** The published figures are `4x4x4=175`,
`8x8x8=265`, `12x12x12=421`, `16x16x8=482` and `16x16x16=674` cycles at
`TPU_MAXDIM=16`, measured by `cosim.py` with the default testbench; the full
`reproduce.sh` (about 6 min) runs that cosim and exits nonzero if any number
differs. For the model of section 1 the measurement is `make mlp-cosim
MODEL=mlp_small`: one `csynth` of the shipped design, then one bounded cosim
per layer, each checked bit-exact against `isa_ref`.

That run is not reproduced on this page: on the host this page was captured
on, the cosim's testbench link fails (`/usr/bin/ld: unable to initialize
decompress status for section .debug_info`: the system linker that
`cosim.py`'s `-B/usr/bin` selects, 2.30 here, cannot read what Vitis
2023.2's compiler emits; the fix was written on a host with 2.42, see
`docs/source/backends/vitis.rst`). The published measurement of this model,
taken with the same command (`workload_suite.rst`, "Per model"), is
`mlp_small_l0` 1 636 and `mlp_small_l1` 1 145 cycles, **2 781** for the model
against the cost model's 1 868, at the shipped `TPU_QD=16` and `DMA_WORDS=1`:
the estimates on this page are known to sit about a third below the RTL.

**Does the harness catch a broken design?** `reproduce.sh --with-mutants`
adds `mutate.py` (about 15 min, one cosim among them; `--no-rtl` for the
functional levels only): it applies one deliberate bug at a time to a copy of
the design under `.mutants/` and reports which level caught it. A mutant's
anchor must occur exactly once across the design, so a refactor that moves
anchored code fails loudly rather than silently testing nothing. `bench_isa.py`
and `stress_isa.py` cite it as the evidence for their own power; the default
`reproduce.sh` runs the gates against the correct design only.

## What is and is not established

- **Cycles in sections 1 to 3 are estimates** from ACT's cost model, fitted to
  cosim at `T=4`, and the model is known to sit a third below the RTL on
  `mlp_small` (`workload_suite.rst`). The `T=8` figure is an extrapolation,
  and the fused instruction has **not** been through RTL cosim. `make
  mlp-cosim MODEL=mlp_small` measures either one.
- **The fused instruction is not on `main`.** `mvoutrelu.patch` is the whole
  of it. With the patch applied, the `gemm.relu` cases of
  `tests/act/test_tinytpu.py` fail, and should: they pin ReLU layers to the
  shipped `vrelu; mvout` programs, which cosim has measured. Merging
  `mvoutrelu` would mean measuring the new programs with `TPU_TB=stress`
  cosim, re-pinning those tests and re-running `mutate.py`.
- **Untested combination:** `TPU_T=8` with the patch applied.
- `pytest tests/act/test_tutorial.py` (seconds, from the repository root)
  checks that the walk-through still compiles and matches PyTorch, that
  `mlp_bias` is still refused, and that the patch still applies to the tree;
  if a file the patch edits moves on, that test fails first.
