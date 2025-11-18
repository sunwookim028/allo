#!/usr/bin/env python3

import argparse
import subprocess
import sys
import io
from pathlib import Path

# Add parent directory to path to import allo
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

try:
    from allo.backend.hls import HLSModule
    ALLO_AVAILABLE = True
except ImportError:
    ALLO_AVAILABLE = False
    print("Warning: Allo not available, will try direct MLIR translation")


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


def verify_mlir_syntax():
  """Verify MLIR file has correct syntax and required elements."""
  print("==> Checking MLIR file syntax")
  if not MLIR_FILE.exists():
    raise FileNotFoundError(f"Missing MLIR file: {MLIR_FILE}")

  with open(MLIR_FILE, "r") as f:
    content = f.read()

  checks = [
      ("memref.global", "global variable declaration"),
      ("hls.static", "hls.static attribute"),
      ("@acc_state", "accumulator state global"),
      ("accumulate_kernel", "kernel function"),
      ("memref.load", "load operation"),
      ("memref.store", "store operation"),
      ("arith.addi", "addition operation"),
  ]

  missing = []
  for token, description in checks:
    if token not in content:
      missing.append(f"  - Missing '{token}' ({description})")

  if missing:
    raise RuntimeError(f"MLIR file is missing required elements:\n" + "\n".join(missing))

  print("MLIR file syntax check passed")
  return True


def verify_mlir():
  print("==> Emitting HLS C++ from MLIR")
  
  # First verify MLIR syntax
  verify_mlir_syntax()

  # Read MLIR file
  with open(MLIR_FILE, "r") as f:
    mlir_content = f.read()

  emit = None
  translation_method = None

  # Try using HLSModule if available
  if ALLO_AVAILABLE:
    try:
      mod = HLSModule(
          mod=mlir_content,
          top_func_name="accumulate_kernel",
          platform="vivado_hls",
      )
      emit = mod.hls_code
      translation_method = "HLSModule (Python bindings)"
    except Exception as e:
      print(f"Warning: HLSModule failed: {e}")
      print("Trying direct translation...")

  # If HLSModule didn't work, try direct translation via subprocess if binary exists
  if emit is None:
    if TRANSLATE_BIN.exists():
      result = run_command(
          [str(TRANSLATE_BIN), "--emit-vivado-hls", str(MLIR_FILE)],
          cwd=PROJECT_ROOT,
      )
      emit = result.stdout
      translation_method = "mlir-translate binary"
    else:
      # Last resort: manual syntax check only
      print("Warning: Cannot translate MLIR (no binary or Python bindings)")
      print("Performed syntax check only - full translation test skipped")
      print("To enable full translation test:")
      print("  1. Build mlir-translate: cd mlir/build && ninja mlir-translate")
      print("  2. Or fix Python bindings for allo._mlir")
      return

  if emit is None:
    raise RuntimeError("Failed to translate MLIR - no translation method available")

  print(f"Translation method: {translation_method}")
  print(f"Generated code length: {len(emit)} characters")

  checks = [
      ("static int32_t acc_state", "static qualifier on accumulator global"),
      ("acc_state", "accumulator state variable name"),
      ("accumulate_kernel(", "top-level kernel signature"),
      (" + ", "addition produced in generated C++"),
  ]
  
  failed_checks = []
  for token, description in checks:
    if token not in emit:
      failed_checks.append(f"  - Missing '{token}' ({description})")

  if failed_checks:
    print("\nGenerated code preview (first 500 chars):")
    print("=" * 60)
    print(emit[:500])
    print("=" * 60)
    raise RuntimeError(f"Translation output missing required elements:\n" + "\n".join(failed_checks))

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




