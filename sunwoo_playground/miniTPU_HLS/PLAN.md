# miniTPU_HLS Execution Plan

## Scope
- Implement a Vitis HLS-compatible `top_module` supporting load, matmul, vector add, and scale commands over a shared on-chip BRAM.
- Develop a C++/HLS testbench that emulates a host issuing commands and validates results against software reference.
- Provide automation tooling (`setup_env.sh`, `Makefile`) to configure Vitis HLS and run tests end-to-end.

## Assumptions
1. Toolchain: Xilinx Vitis HLS 2020.2+ (adjust paths in `setup_env.sh` as needed).
2. Data model: 32-bit signed integers (`ap_int<32>`), word-addressed BRAM sized to hold all operands.
3. Target BRAM: single-port block RAM inferred via `#pragma HLS RESOURCE variable=shared_bram core=RAM_1P_BRAM`.
4. Host-side tests compiled with `g++` (or `clang++`) supporting C++17 and linked against the HLS simulation model.

## Milestones
### M1. Directory & Boilerplate Setup
1. Create directory layout under `sunwoo_playground/miniTPU_HLS/`:
   - `include/` for shared headers (`bram_defs.h`, `instructions.h`).
   - `src/` for HLS kernel sources (`top_module.cpp`, helper kernels).
   - `host/` for simulation/testbench (`host_test.cpp`, utilities).
   - `scripts/` for environment setup.
   - Project root files: `Makefile`, `README.md`.
2. Add `.gitignore` entries if required (e.g., build artifacts, `.log`, `xsim/`).

### M2. Shared Data Structures
1. Draft `include/bram_defs.h`:
   - Define constants (`BRAM_WORDS`, address ranges for A/B/s/E regions).
   - Declare instruction opcodes enum (`OP_LOAD`, `OP_MATMUL`, `OP_VECADD`, `OP_SCALE`, `OP_COPY_OUT` if needed).
   - Specify argument struct(s) using fixed-size fields for addresses/lengths.
2. Draft `include/instructions.h`:
   - Provide pack/unpack helpers for command words (if encoded as struct -> raw array).
   - Document command sequence for load, matmul, vecadd, scale, copy.

### M3. HLS Kernel Implementation
1. Implement `src/top_module.cpp`:
   - Declare `static ap_int<32> shared_bram[BRAM_WORDS];` with `RESOURCE`/`INTERFACE` pragmas enforcing BRAM inference.
   - Expose top-level function `void top_module(const Command *cmds, int num_cmds)` (or similar) with AXI-Lite/stream interfaces as required.
2. Within `top_module`, iterate over command list and dispatch to operation-specific helpers in `src/kernels.cpp`.
3. Implement helpers:
   - `kernel_load(ap_int<32>* bram, const LoadArgs&)` to copy from host buffer to BRAM. Host provides data via AXI memory-mapped array argument; implement burst-friendly loops with HLS pragmas (`PIPELINE`, `LOOP_TRIPCOUNT`).
   - `kernel_matmul(ap_int<32>* bram, const MatMulArgs&)` performing 2x2 multiply using addresses from args.
   - `kernel_vecadd(...)` for element-wise addition.
   - `kernel_scale(...)` for scaling by scalar `s`.
4. Ensure each helper uses direct BRAM pointer arithmetic, enforcing sequential (single-port) access.
5. Guard all loops with explicit bounds derived from arguments; apply `#pragma HLS INLINE` for control helpers as needed.

### M4. Host Simulation & Validation
1. Implement `host/host_test.cpp`:
   - Seed RNG (deterministic by default; allow overriding with CLI).
   - For each of 10 trials:
     - Randomize arrays `A`, `B`, and scalar `s` within 32-bit signed range (optionally reduce magnitude to avoid overflow).
     - Compute `C_host = A * B`, `D_host = C_host + A`, `E_host = s * D_host`.
     - Construct command list to:
       1. Load `A`, `B`, `s` into disjoint BRAM regions.
       2. Invoke matmul, vecadd, scale commands with appropriate addresses.
       3. Copy result region back to host buffer.
     - Call `top_module` (C-sim) with command sequence; capture resulting `E`.
     - Compare `E` vs `E_host`, assert equality.
   - Collect and print summary (`PASS`/`FAIL`, mismatches with addresses).
2. Provide optional CLI flags (`--trials N`, `--seed S`).

### M5. Tooling & Automation
1. Write `scripts/setup_env.sh`:
   - Detect/install environment variables for Vitis (`VITIS_HLS`, `XILINX_VITIS`, etc.).
   - Source Xilinx setup scripts if available; fallback instructions otherwise.
2. Create `Makefile` with targets:
   - `setup`: invokes `scripts/setup_env.sh`.
   - `build_host`: compiles `host/host_test.cpp` against HLS simulation (include paths to Vitis headers, link flags).
   - `csim`: runs host simulation binary (depends on `build_host`).
   - `hls`: launches `vitis_hls -f scripts/run_hls.tcl` (add Tcl script optionally).
   - `clean`: removes build outputs (`build/`, `*.log`, etc.).
3. Add optional `scripts/run_hls.tcl` to automate HLS synthesis (project creation, Synthesize, Export RTL).

### M6. Documentation & Review
1. Draft `README.md`:
   - Overview, directory structure, build/test instructions, dependency setup.
   - Describe command encoding and BRAM layout.
   - Note limitations (single-port BRAM, 2x2 matmul only).
2. Provide comments in code to clarify address computations and instruction flow.
3. Run lint/style checks if available; document verification status.

### M7. Validation Checklist
- [ ] `make csim` completes with 10 trials passing.
- [ ] (Optional) `make hls` generates Vitis HLS report with single BRAM instance.
- [ ] README instructions verified on clean environment.

## Timeline Guidance
1. Day 1: Milestones M1–M3.
2. Day 2: Milestone M4.
3. Day 3: Milestones M5–M7, integration testing, polish.


