# 1D vector-adder array

`allo_design.py` computes `C[i] = A[i] + B[i]` using signed 32-bit integers.
It keeps the addition from Vincent's datapath and removes the sequencer,
instruction memory, opcodes, enable stream, and program-loading calls.

Start with `N = 4`: a length-4 vector and four parallel adder kernels in a
logical 1D array. Change this one constant to `32` to use Vincent's vector
length with 32 parallel adders. Rebuild in a fresh build directory after
changing it. This is a logical array; physical placement is chosen by the PD
flow, not constrained to a literal row by this source.

The hardware portion is just `top`: one input loader, `mapping=[N]` copies of
the `adder` kernel, and one output collector, connected by small FIFOs.
Each adder reads one A value and one B value and writes their sum. In contrast,
Vincent's `mapping=[1]` kernel processes N values in a loop without explicitly
requesting N spatial copies. The functions below `top` provide the existing
flow's build and testbench interfaces; they are not extra hardware kernels.

`construct-commercial.py` reuses the existing `allo-example-systemc` graph:

`Allo -> SystemC/Connections -> Catapult -> hierarchical ASIC PD`

It changes the design source, report name, `min_macro_reuse = 1`, and
`stop_after_step = "floorplan"`. The reference graph is a dependency of this
small wrapper. Its other settings are inherited unchanged.

## What min_macro_reuse means

This is a macro-selection threshold, not the number of times an adder executes.
A macro is a block implemented separately and then instantiated in the
full-chip design. Setting the threshold to 1 allows an eligible implementation
class with just one member to be selected. Thus the loader and collector can
also be selected, not only the repeated adders. Actual grouping and selection
are recorded in the generated manifest and macro plan.

## Check the design without launching HLS or PD

On the research server:

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate allo
source /work/shared/common/allo/setup-llvm-main.sh
cd ~/allo
python allo/backend/asic/designs/vadd-1d/allo_design.py
```

The software simulator checks the deterministic flow workload, all-zero
inputs, and a seeded random vector pair against NumPy. The flow also has a
generated RTL testbench transaction. Arithmetic uses signed 32-bit results;
the supplied inputs stay within range.

## Configure a separate PD build

Use the same mflowgen environment you use for EVA. The Allo interpreter must
be the environment containing this checkout's built native compiler.

```bash
export ALLO_PYTHON=/home/sjc364/miniconda3/envs/allo/bin/python
export ALLO_SETUP_SCRIPT=/work/shared/common/allo/setup-llvm-main.sh
mkdir -p /work/shared/users/ugrad/sjc364/vadd-1d-build
cd /work/shared/users/ugrad/sjc364/vadd-1d-build
mflowgen run --design \
  /home/sjc364/allo/allo/backend/asic/designs/vadd-1d/construct-commercial.py
make list
```

Choose the target for the milestone you want:

```bash
# Generate the HLS RTL and ASIC manifests first.
make allo-asic-compilation

# Or continue through full-chip synthesis, including hierarchical prerequisites.
make commercial-full-chip-synthesis

# Or continue to the full-chip floorplan checkpoint.
make commercial-full-chip-pnr
```

Run targets in order as you inspect each result; do not launch them concurrently.
The compilation target uses Catapult and needs the site's tool setup/licenses.
The hierarchical synthesis/floorplan targets also build and harden selected
macros as prerequisites. The floorplan stop applies to full-chip PNR, not
macro hardening. Avoid an unqualified `make` for this initial experiment:
downstream signoff/power stages expect a completed PNR run.

Inspect `*-allo-asic-compilation/outputs/backend-rtl/concat_rtl.v` for the
generated RTL and `asic-manifest-final.json` beside `backend-rtl` for kernel
instances. The floorplan checkpoint is under
`*-commercial-full-chip-pnr/outputs/design.checkpoint/`; the PNR script saves
`save.enc` there for Innovus. No HLS or commercial PD result is implied by the
software simulation check.
