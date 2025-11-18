#!/usr/bin/env python3

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
HLS_DIR = PROJECT_ROOT / "hls"
MLIR_FILE = PROJECT_ROOT / "mlir" / "accumulate.mlir"
TRANSLATE_BIN = PROJECT_ROOT.parent / "mlir" / "build" / "bin" / "mlir-translate"


def run_command(args, cwd=None, check=True):
  """Run a subprocess, streaming stdout/stderr, and optionally raise on failure."""
  process = subprocess.run(
      args,
      cwd=cwd,
      stdout=subprocess.PIPE,
      stderr=subprocess.PIPE,
      text=True,
  )
  if check and process.returncode != 0:
    sys.stderr.write(f"Command failed: {' '.join(args)}\n")
    if process.stdout:
      sys.stderr.write(process.stdout + "\n")
    if process.stderr:
      sys.stderr.write(process.stderr + "\n")
    raise RuntimeError(f"Command {' '.join(args)} exited with {process.returncode}")
  return process


def verify_hls():
  print("==> Building and running HLS testbench")
  run_command(["make", "clean"], cwd=HLS_DIR)
  run_command(["make"], cwd=HLS_DIR)
  result = run_command([str(HLS_DIR / "tb")], cwd=HLS_DIR)
  if "Test PASSED" not in result.stdout:
    raise RuntimeError("HLS testbench did not report success")
  print("HLS testbench passed")


def verify_mlir():
  print("==> Emitting HLS C++ from MLIR")
  if not MLIR_FILE.exists():
    raise FileNotFoundError(f"Missing MLIR file: {MLIR_FILE}")
  if not TRANSLATE_BIN.exists():
    raise FileNotFoundError(
        f"Missing mlir-translate binary at {TRANSLATE_BIN}. "
        "Rebuild it via 'cmake --build /home/sk3463/allo/mlir/build --target mlir-translate'."
    )

  emit = run_command(
      [str(TRANSLATE_BIN), "--emit-vivado-hls", str(MLIR_FILE)],
      cwd=PROJECT_ROOT,
  ).stdout

  checks = [
      ("static int32_t acc_state", "static qualifier on accumulator global"),
      ("acc_state = ", "store back into persistent state"),
      ("accumulate_kernel(", "top-level kernel signature"),
      (" + ", "addition produced in generated C++"),
  ]
  for token, description in checks:
    if token not in emit:
      raise RuntimeError(f"Expected '{token}' ({description}) in translator output")

  print("MLIR translation checks passed")


def main():
  parser = argparse.ArgumentParser(
      description="Verify the HLS and MLIR flows for the static accumulator kernel."
  )
  parser.add_argument(
      "--run-hls",
      action="store_true",
      help="Run only the HLS C++ build and testbench",
  )
  parser.add_argument(
      "--run-mlir",
      action="store_true",
      help="Run only the MLIR → HLS translation checks",
  )
  args = parser.parse_args()

  run_hls = args.run_hls or not (args.run_hls or args.run_mlir)
  run_mlir = args.run_mlir or not (args.run_hls or args.run_mlir)

  if run_hls:
    verify_hls()
  if run_mlir:
    verify_mlir()

  print("All requested verification steps completed successfully.")


if __name__ == "__main__":
  main()




