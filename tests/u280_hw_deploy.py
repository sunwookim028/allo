#!/usr/bin/env python3
"""
Deploy an existing Allo dataflow kernel on the Xilinx Alveo U280.

Uses the producer-consumer design from tests/dataflow/test_producer_consumer.py
as the target kernel — a minimal streaming dataflow example (A[16,16] → B=A+1).

Purposes
--------
1. End-to-end smoke test of the Allo → Vitis HLS → xclbin → U280 flow.
2. Produce a ready-to-use project directory (kernel.cpp, host.cpp, Makefile,
   xcl2.hpp/cpp) as a harness skeleton for custom U280 projects.

Usage
-----
  # Generate project files only (no Vitis required, fast):
  ./run_allo.sh python tests/u280_hw_deploy.py --codegen-only

  # Software emulation — functional check without FPGA (~1 min):
  source /work/shared/common/allo/vitis_2023.2_u280.sh
  ./run_allo.sh python tests/u280_hw_deploy.py --sw-emu

  # Full hardware build + run on attached U280 (~4-6 hrs first build):
  source /work/shared/common/allo/vitis_2023.2_u280.sh
  ./run_allo.sh python tests/u280_hw_deploy.py --hw

Manual build (after --codegen-only):
  source /work/shared/common/allo/vitis_2023.2_u280.sh
  cd u280_producer_consumer_hw.prj
  make all TARGET=sw_emu PLATFORM=$XDEVICE        # sw_emu sanity check
  make all TARGET=hw     PLATFORM=$XDEVICE        # full bitstream
  ./top build_dir.hw.xilinx_u280_gen3x16_xdma_1_202211_1/top.xclbin
"""

import argparse
import os
import sys
import numpy as np

import allo
from allo.ir.types import float32, Stream
import allo.dataflow as df
from allo.backend import hls

# ---------------------------------------------------------------------------
# Kernel — identical to test_producer_consumer.py::test_producer_consumer()
# Using fixed (non-closure) sizes so it can be a module-level df.region.
# ---------------------------------------------------------------------------
M, N = 16, 16
Ty = float32


@df.region()
def top(A: Ty[M, N], B: Ty[M, N]):
    pipe: Stream[Ty, 4]

    @df.kernel(mapping=[1], args=[A])
    def producer(local_A: Ty[M, N]):
        for i, j in allo.grid(M, N):
            out: Ty = local_A[i, j]
            pipe.put(out)

    @df.kernel(mapping=[1], args=[B])
    def consumer(local_B: Ty[M, N]):
        for i, j in allo.grid(M, N):
            data = pipe.get()
            local_B[i, j] = data + 1


# ---------------------------------------------------------------------------
# Reference data
# ---------------------------------------------------------------------------
def make_inputs():
    rng = np.random.default_rng(0)
    A = rng.random((M, N)).astype(np.float32)
    B = np.zeros((M, N), dtype=np.float32)
    return A, B


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def verify_simulator():
    """Functional check via Allo's Python simulator (no Vitis needed)."""
    A, B = make_inputs()
    sim_mod = df.build(top, target="simulator")
    sim_mod(A, B)
    np.testing.assert_allclose(B, A + 1, rtol=1e-5)
    print("[OK] Allo simulator passed.")


def build_project(mode, project):
    print(f"\n==> Generating project '{project}'  (mode={mode}) ...")
    mod = df.build(top, target="vitis_hls", mode=mode, project=project)
    return mod


def print_project_files(project):
    key_files = [
        "kernel.h",
        "kernel.cpp",
        "host.cpp",
        "xcl2.hpp",
        "xcl2.cpp",
        "Makefile",
        "utils.mk",
        "xrt.ini",
        "run.tcl",
        "description.json",
    ]
    print()
    print("  Generated files:")
    for f in key_files:
        path = os.path.join(project, f)
        size = f"  ({os.path.getsize(path):,} B)" if os.path.exists(path) else "  [missing]"
        print(f"    {project}/{f}{size}")


def print_next_steps(project):
    xsa = "xilinx_u280_gen3x16_xdma_1_202211_1"
    print(f"""
==> Next steps:

  # 1. Load Vitis 2023.2 + XRT + XDEVICE for U280:
  source /work/shared/common/allo/vitis_2023.2_u280.sh

  # 2. (Optional) software-emulation sanity check (~1 min):
  cd {project}
  make all TARGET=sw_emu PLATFORM=$XDEVICE
  XCL_EMULATION_MODE=sw_emu ./top build_dir.sw_emu.{xsa}/top.xclbin

  # 3. Full hardware build (~4-6 hrs, produces bitstream):
  make all TARGET=hw PLATFORM=$XDEVICE

  # 4. Program and run on U280:
  ./top build_dir.hw.{xsa}/top.xclbin

==> To adapt as a custom U280 skeleton:
  - Replace the kernel body in  {project}/kernel.cpp
  - Adjust buffer sizes/types in {project}/host.cpp
  - Keep Makefile / utils.mk / xcl2.hpp / xcl2.cpp as-is
""")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Allo producer-consumer → U280")
    group = ap.add_mutually_exclusive_group()
    group.add_argument("--codegen-only", action="store_true",
                       help="Write project files; do not invoke Vitis.")
    group.add_argument("--sw-emu", action="store_true",
                       help="Software emulation (XCL_EMULATION_MODE=sw_emu).")
    group.add_argument("--hw-emu", action="store_true",
                       help="Hardware emulation (XCL_EMULATION_MODE=hw_emu).")
    group.add_argument("--hw", action="store_true",
                       help="Full hardware build + execute on U280.")
    args = ap.parse_args()

    if not any([args.codegen_only, args.sw_emu, args.hw_emu, args.hw]):
        ap.print_help()
        sys.exit(0)

    # 1. Always verify in simulator first
    verify_simulator()

    # 2. Determine target mode and project directory
    if args.codegen_only:
        mode, project = "hw", "u280_producer_consumer_hw.prj"
    elif args.sw_emu:
        mode, project = "sw_emu", "u280_producer_consumer_swemu.prj"
    elif args.hw_emu:
        mode, project = "hw_emu", "u280_producer_consumer_hwemu.prj"
    else:
        mode, project = "hw", "u280_producer_consumer_hw.prj"

    # 3. Check Vitis / XDEVICE for modes that need them
    if not args.codegen_only:
        if not hls.is_available("vitis_hls"):
            sys.exit(
                "\n[ERROR] vitis_hls not in PATH.\n"
                "Run: source /work/shared/common/allo/vitis_2023.2_u280.sh"
            )
        if not os.environ.get("XDEVICE"):
            sys.exit(
                "\n[ERROR] XDEVICE not set.\n"
                "Run: source /work/shared/common/allo/vitis_2023.2_u280.sh"
            )

    # 4. Codegen (always)
    mod = build_project(mode, project)
    print_project_files(project)

    if args.codegen_only:
        print_next_steps(project)
        return

    # 5. Build + execute (triggers make run TARGET=<mode>)
    A, B = make_inputs()
    print(f"\n==> Running on hardware (mode={mode}) ...")
    mod(A, B)

    # 6. Verify result
    np.testing.assert_allclose(B, A + 1, rtol=1e-5)
    print(f"[OK] Result verified against A+1  (mode={mode}).")


if __name__ == "__main__":
    main()
