"""
HLS Synthesis Script: float16 (half) Type Verification
=======================================================
Verifies that float16 kernels synthesize through Vitis HLS 2023.2.
csyn-only (no csim/testbench required).

Test A (top_fp16_arith): elementwise add + scale — confirms `half` synthesizes at all.
Test B (top_fp16_exp):   elementwise exp        — checks if exp(half) works via hls_math.h.

Usage: conda run -n allo python tests/dataflow/hls_synth_fp16.py
       or: ./run_allo.sh python tests/dataflow/hls_synth_fp16.py
"""
from __future__ import annotations
import os
import json
import allo
import allo.dataflow as df
from allo.ir.types import float16
from allo.backend import hls

# ─── Constants ────────────────────────────────────────────────────────────────

N = 4

# ─── Design Definitions ───────────────────────────────────────────────────────

@df.region()
def top_fp16_arith(a: float16[N], b: float16[N], out: float16[N]):
    """Test A: elementwise add + scale on float16 arrays."""

    @df.kernel(mapping=[1], args=[a, b, out])
    def arith_kernel(a_buf: float16[N], b_buf: float16[N], out_buf: float16[N]):
        for i in range(N):
            out_buf[i] = a_buf[i] + b_buf[i] * 2.0


@df.region()
def top_fp16_exp(a: float16[N], out: float16[N]):
    """Test B: elementwise exp on float16 arrays via allo.exp / hls_math.h."""

    @df.kernel(mapping=[1], args=[a, out])
    def exp_kernel(a_buf: float16[N], out_buf: float16[N]):
        for i in range(N):
            out_buf[i] = allo.exp(a_buf[i])


# ─── CSYN: HLS Synthesis ──────────────────────────────────────────────────────

def run_csyn(region_fn, project_dir, desc):
    """Run Vitis HLS synthesis and return resource dictionary."""
    if not hls.is_available("vitis_hls"):
        print(f"[SKIP] vitis_hls not available, skipping CSYN for {desc}")
        return None
    print(f"\n[CSYN] {desc} → {project_dir}")
    mod = df.build(region_fn, target="vitis_hls", mode="csyn",
                   project=project_dir)
    try:
        mod()  # runs vitis_hls -f run.tcl
    except RuntimeError as e:
        print(f"  [FAIL] Synthesis failed: {e}")
        return {"status": "FAIL", "error": str(e)}
    from allo.backend.report import parse_xml
    try:
        result = parse_xml(project_dir, "Vitis HLS",
                           top=mod.top_func_name, print_flag=True)
        result["status"] = "PASS"
        return result
    except Exception as e:
        print(f"  [WARN] Could not parse report: {e}")
        return {"status": "PASS_NO_REPORT"}


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    base = "/scratch/sk3463/hls_projects"
    os.makedirs(base, exist_ok=True)

    results = {}

    print("\n" + "="*60)
    print("  HLS SYNTHESIS: float16 (half) Type Verification")
    print("="*60)

    # Test A: basic float16 arithmetic (add + scale)
    r_arith = run_csyn(top_fp16_arith,
                       f"{base}/fp16_arith_csyn.prj",
                       "float16 arith kernel (add + scale)")
    results["fp16_arith"] = r_arith

    # Test B: float16 exp via hls_math.h
    r_exp = run_csyn(top_fp16_exp,
                     f"{base}/fp16_exp_csyn.prj",
                     "float16 exp kernel (allo.exp / hls_math.h)")
    results["fp16_exp"] = r_exp

    # ── Summary ────────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  SUMMARY")
    print("="*60)
    for name, r in results.items():
        if r:
            print(f"\n  {name}:")
            for k, v in r.items():
                print(f"    {k}: {v}")
        else:
            print(f"\n  {name}: [no report / skipped]")

    summary_path = f"{base}/fp16_synth_summary.json"
    with open(summary_path, "w") as f:
        serializable = {}
        for name, r in results.items():
            if r:
                serializable[name] = {k: str(v) for k, v in r.items()}
        json.dump(serializable, f, indent=2)
    print(f"\n[INFO] Report saved to {summary_path}")


if __name__ == "__main__":
    main()
