# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""M-R1: run MiniTPU kernels on the wrapped core in the Allo region and hold the drain to the testbench digest.

Each launch is staged by ``oracle.py``'s machinery (the pinned clone's ``asm.py`` through its own builders,
with ``tools/sim_kernel.py``'s operands -- reused, not re-derived; the image and operand checksums are checked
against ``oracle.json``), run through ``minitpu_rtl.py``'s region on ``target="simulator"`` (Verilator inside
the RTLModule), and judged by the sha256 of the drain against ``oracle.json``'s ``rtl.drain_sha256`` (MiniTPU's
own ``tb_kernel_image``). Cycles are the core's own ``perf_cnt_cycles`` beside the testbench's.

    python examples/minitpu-rtl/run_kernel.py                          # the looped GEMM (the M-R1 pass check)
    python examples/minitpu-rtl/run_kernel.py --only gemm_varying_gelu softmax_8
    python examples/minitpu-rtl/run_kernel.py --all --json out.json     # every oracle launch
    python examples/minitpu-rtl/run_kernel.py --sfu-negative-control    # run from the wrong cwd once
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import pathlib
import sys
import tempfile
import time

sys.dont_write_bytecode = True
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import oracle  # noqa: E402  (M-R0's machinery: export, staging)
import minitpu_rtl  # noqa: E402

PASS_CHECK = "gemm_structured"   # the plan's "one looped GEMM": sim_kernel's --operands structured
WORK = pathlib.Path(os.environ.get("MR1_WORK", "/work/shared/users/phd/sk3463/scratch/mr1_out/work"))


@contextlib.contextmanager
def captured_fds():
    """Captures the process's fd 1 and 2 (the Verilated model prints there, not through Python)."""
    sys.stdout.flush()
    sys.stderr.flush()
    with tempfile.TemporaryFile(mode="w+b") as sink:
        saved = os.dup(1), os.dup(2)
        os.dup2(sink.fileno(), 1)
        os.dup2(sink.fileno(), 2)
        box = {}
        try:
            yield box
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            os.dup2(saved[0], 1)
            os.dup2(saved[1], 2)
            os.close(saved[0])
            os.close(saved[1])
            sink.seek(0)
            box["text"] = sink.read().decode(errors="replace")


def sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--only", nargs="*", default=None, help="launch or group names (default: the GEMM)")
    parser.add_argument("--all", action="store_true", help="every launch in oracle.json")
    parser.add_argument("--skip", nargs="*", default=[], help="launch or group names to leave out")
    parser.add_argument("--json", type=pathlib.Path, default=None, help="write the per-launch results here")
    parser.add_argument("--sfu-negative-control", action="store_true",
                        help="run the selected launches from a directory without the SFU tables")
    parser.add_argument("--build-jobs", type=int, default=16)
    args = parser.parse_args()

    manifest = json.loads((HERE / "oracle.json").read_text())
    expected = {k["name"]: k for k in manifest["kernels"]}
    tree = oracle.export_clone(WORK)
    t0 = time.monotonic()
    groups = oracle.collect(tree)
    import asm  # the pinned clone's, on the path collect() set
    depth = asm.mxu_output_fifo()[0]
    assert depth == manifest["mxu_output_fifo_depth"], (depth, manifest["mxu_output_fifo_depth"])
    cases = [c for group in groups.values() for c in group]
    names = {c.name for c in cases} | set(groups)
    wanted = None if args.all else set(args.only or [PASS_CHECK])
    if wanted and wanted - names:
        raise SystemExit(f"unknown launches: {sorted(wanted - names)}")
    group_of = {c.name: g for g, cs in groups.items() for c in cs}
    cases = [c for c in cases if (wanted is None or c.name in wanted or group_of[c.name] in wanted)
             and c.name not in args.skip and group_of[c.name] not in args.skip]
    print(f"staged {len(cases)} launch(es) in {time.monotonic() - t0:.1f} s", flush=True)

    t0 = time.monotonic()
    mod, ip = minitpu_rtl.build(tree, depth, build_jobs=args.build_jobs)
    build_s = time.monotonic() - t0
    print(f"region built (Verilator model + simulator) in {build_s:.0f} s", flush=True)

    cwd = tree
    if args.sfu_negative_control:
        cwd = pathlib.Path(tempfile.mkdtemp(prefix="mr1_nosfu_"))
    results = []
    for case in cases:
        want = expected[case.name]
        staged = {
            "image": sha(case.image) == want["image_sha256"],
            "operands": sha(b"".join(len(w).to_bytes(8, "little") + w for w in case.windows)) == want["operands_sha256"],
        }
        t0 = time.monotonic()
        with captured_fds() as out:
            status, drain, _ = minitpu_rtl.run(mod, case.image, case.windows, case.offsets, case.values,
                                               case.drain_argument, case.drain_bytes, case.max_cycles, cwd)
        wall = time.monotonic() - t0
        text = out["text"]
        readmem = [ln for ln in text.splitlines() if "readmem" in ln.lower()]
        assertions = [ln for ln in text.splitlines() if "Assertion" in ln or "%Error" in ln][:3]
        digest = sha(drain)
        tb = want["rtl"]
        halted = bool(status["flags"] & 1) and not (status["flags"] >> 4) & 1
        row = {
            "name": case.name, "staged_identical": staged, "halted": halted, "flags": status["flags"],
            "drain_sha256": digest, "tb_drain_sha256": tb["drain_sha256"],
            # A run that printed a $readmem warning ran with zeroed SFU tables: never a pass.
            "identical": digest == tb["drain_sha256"] and not readmem,
            "cycles": status["perf_cnt_cycles"], "tb_cycles": tb["cycles"],
            "bundles": status["perf_cnt_instrs"], "tb_bundles": tb["bundles_issued"],
            "dma_err": status["dma_err"], "mem_errors": status["shim_mem_errors"],
            "partial_wstrb": status["shim_partial_wstrb"], "magic_ok": status["magic"] == 0x4D523101,
            "readmem_warnings": len(readmem), "assertions": assertions, "wall_s": round(wall, 1),
            "status": status,
        }
        results.append(row)
        print(f"{case.name:26} {'IDENTICAL' if row['identical'] else 'DIFFERS':9} drain {digest[:8]} tb {tb['drain_sha256'][:8]}"
              f"  cycles {row['cycles']:>7} tb {row['tb_cycles']:>7} ({row['cycles'] - row['tb_cycles']:+d})"
              f"  bundles {row['bundles']}/{row['tb_bundles']}  halted={int(halted)} dma_err={row['dma_err']:#x}"
              f" readmem_warnings={row['readmem_warnings']} staged={'ok' if all(staged.values()) else staged}"
              f"  {wall:.1f}s", flush=True)
        for line in readmem[:2] + assertions:
            print(f"    | {line}", flush=True)

    same = sum(r["identical"] for r in results)
    print(f"{same}/{len(results)} drains bit-identical to tb_kernel_image; build {build_s:.0f} s; "
          f"pins: MiniTPU {manifest['minitpu_commit'][:8]}, oracle round_trip_cycles={manifest['round_trip_cycles']}")
    if args.json:
        args.json.write_text(json.dumps({"build_s": round(build_s, 1), "negative_control": args.sfu_negative_control,
                                         "launches": results}, indent=1) + "\n")
    return 0 if same == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
