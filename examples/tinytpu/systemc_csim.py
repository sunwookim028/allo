# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""TinyTPU-isa through ``target="systemc"`` csim, on the stress cases.

Runs the first N ``stress_isa`` cases through the SystemC emission's
behavioural csim and compares the GEMM region with the reference. It is a
functional check of the emission, not of Catapult synthesis.

The libraries come from the environment: Catapult's own on a licence host,
or ``scripts/systemc-csim-setup.sh``'s stand-in elsewhere.

``clobbered_outside`` is expected to be non-zero. The SystemC testbench
preloads only input arrays, so ``C`` starts at zero here rather than
prefilled, unlike the simulator and Vitis flows.

Usage::

    python examples/tinytpu/systemc_csim.py [N] [--project DIR]
"""

import argparse
import os
import sys
import time

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
)
from allo.dataflow import customize  # noqa: E402
from examples.tinytpu.microarch_isa import tinytpu_isa, schedule  # noqa: E402
from examples.tinytpu import stress_isa  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("n", nargs="?", type=int, default=3)
    ap.add_argument("--project", default="tinytpu_systemc_csim.prj")
    args = ap.parse_args()

    s = customize(tinytpu_isa)
    schedule(s)
    mod = s.build(target="systemc", mode="csim", project=args.project)
    bad_cases = 0
    for i, (tag, prog, A, B, C0, gold, M, N) in enumerate(stress_isa.cases(True)):
        if i >= args.n:
            break
        t = time.time()
        got = stress_isa.execute(mod, prog, A, B, C0)
        bad = stress_isa.compare(tag, got, gold, M, N)
        verdict = "EXACT" if bad is None else bad
        bad_cases += "wrong=0/" not in str(verdict) and verdict != "EXACT"
        print(f"CASE {tag} -> {verdict} {time.time() - t:.1f}s", flush=True)
    print("SYSTEMC CSIM OK" if bad_cases == 0 else f"SYSTEMC CSIM FAIL {bad_cases}")
    sys.exit(1 if bad_cases else 0)


if __name__ == "__main__":
    main()
