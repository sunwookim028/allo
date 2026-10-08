# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""M-R0 oracle: MiniTPU's own RTL testbench and its own emulator, run on the same kernel images.

For every kernel in ``KERNELS`` this assembles the image with the pinned clone's ``asm.py`` (through the
clone's own builders, with the operands ``tools/sim_kernel.py`` stages), replays it on the real
``minitpu.sv`` through ``tb/run_kernel_image.sh`` (Verilator), runs it through ``tools/emulate_image.py``
(via ``compiler/backends.py:emulate``), and records the image checksum, both drain digests, whether they
are bit-identical, and the RTL cycle and bundle counters. Nothing here is Allo yet; this is the reference
M-R1/M-R2 must reproduce (``dev/records/minitpu/minitpu_rtl_plan_2026-10-08.rst`` section 4).

The pinned clone is read-only: its tracked tree at ``PIN`` is exported (``git archive``) into ``--work``
and everything -- Python imports, the Verilator build, the testbench's cwd-relative SFU ROMs -- runs there.

    python examples/minitpu-rtl/oracle.py                  # every kernel, writes oracle.json
    python examples/minitpu-rtl/oracle.py --only gemm_structured --no-write
"""

from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import dataclasses
import hashlib
import io
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile
import time

sys.dont_write_bytecode = True

HERE = pathlib.Path(__file__).resolve().parent
PIN = "b3ba0a4d4fb69d39091c55f5f00d1f237082a4f1"
CLONE = pathlib.Path("/work/shared/users/phd/sk3463/minitpu")
WORK = pathlib.Path("/work/shared/users/phd/sk3463/scratch/mr0")
VERILATOR_BIN = pathlib.Path("/work/shared/users/phd/sk3463/tools/verilator/bin")
GCC_BIN = pathlib.Path("/opt/rh/gcc-toolset-13/root/usr/bin")
ROUND = re.compile(r"KERNEL_IMAGE_ROUND round=0 status=(\w+) cycles=(\d+) bundles=(\d+) "
                   r"launch_status=0x([0-9a-fA-F]+)")


def sh(*command: str, cwd=None, env=None) -> str:
    return subprocess.run(command, cwd=cwd, env=env, check=True, capture_output=True, text=True).stdout


def tool_env() -> dict[str, str]:
    """The testbench's environment: the pinned Verilator and gcc-toolset-13's g++ first (not Catapult's 10.3)."""
    env = dict(os.environ)
    env["PATH"] = os.pathsep.join([str(GCC_BIN), str(VERILATOR_BIN), "/usr/bin", "/bin"])
    env.pop("VERILATOR_ROOT", None)
    return env


def export_clone(work: pathlib.Path) -> pathlib.Path:
    """Exports the pinned clone's tracked tree once; refuses a clone that is not at the pin or is dirty."""
    head = sh("git", "-C", str(CLONE), "rev-parse", "HEAD").strip()
    if head != PIN:
        raise SystemExit(f"{CLONE} is at {head}, not the pin {PIN}")
    if sh("git", "-C", str(CLONE), "status", "--porcelain", "--untracked-files=no").strip():
        raise SystemExit(f"{CLONE} has modified tracked files; the oracle runs only the pinned tree")
    tree = work / f"minitpu-{PIN[:8]}"
    marker = tree / ".allo-export-pin"
    if not (marker.is_file() and marker.read_text().strip() == PIN):
        if tree.exists():
            shutil.rmtree(tree)
        tree.mkdir(parents=True)
        archive = subprocess.run(["git", "-C", str(CLONE), "archive", "--format=tar", PIN],
                                 check=True, capture_output=True).stdout
        subprocess.run(["tar", "-x", "-C", str(tree)], input=archive, check=True)
        marker.write_text(PIN + "\n")
    return tree


@dataclasses.dataclass
class Case:
    """One launch, in tb_kernel_image's terms: argument windows, pointer offsets, register values, drain."""

    name: str
    label: str
    image: bytes
    windows: list[bytes]
    offsets: list[int]
    values: list[int | None]
    drain_argument: int
    drain_bytes: int
    max_cycles: int


def _parse_extra(extra) -> tuple[list[int], list[int | None], int]:
    offsets, values, drain = [0] * 4, [None] * 4, 2
    for item in extra:
        key, value = item.lstrip("+").split("=")
        if key.startswith("ARG_OFFSET"):
            offsets[int(key[-1])] = int(value)
        elif key.startswith("ARG_VALUE"):
            values[int(key[-1])] = int(value)
        elif key == "DUMP_ARG":
            drain = int(value)
        else:
            raise ValueError(f"plusarg {item} is not modelled by the oracle")
    return offsets, values, drain


def collect(tree: pathlib.Path) -> dict[str, list[Case]]:
    """Builds every kernel's launches with sim_kernel.py's own operand code.

    sim_kernel.py's ``run_*`` functions stage operands and call ``run_kernel_image``; that one function is
    swapped for a recorder, so the images and blobs are exactly the ones MiniTPU's own gate replays.
    The GEMM and row-softmax paths bypass ``run_kernel_image`` and are staged here the way sim_kernel does.
    """
    sys.path[:0] = [str(tree / "tools"), str(tree)]
    import numpy as np
    import sim_kernel as sk  # the pinned clone's

    recorded: list[Case] = []

    def recorder(kernel, *, blobs, out_bytes, label, max_cycles=2_000_000, extra=(), raw=False):
        offsets, values, drain = _parse_extra(extra)
        windows = [blobs.get(f"ARG{i}", b"") for i in range(4)]
        recorded.append(Case("", label, kernel.image.to_bytes(pad_to=4), windows, offsets, values,
                             drain, out_bytes, max_cycles))
        # Hand back the drain window as staged (what a launch that wrote nothing would dump), so the caller's
        # own judging runs on and records its remaining launches instead of raising on the first.
        staged = windows[drain][:out_bytes].ljust(out_bytes, b"\0")
        words = np.frombuffer(staged, dtype="<u2").copy()
        return words if raw else sk.from_bf16_words(words=words)

    sk.run_kernel_image = recorder

    def gemm(column_tiles=1, groups=1, operands="varying", activate=False, fuse=False, seed=0):
        tokens, depth = sk._BLOCK_TOKENS, groups * sk._LOOP_BLOCKS * sk._BLOCK_DEPTH
        left, right = sk.build_operands(tokens, depth, column_tiles * sk._TILE, kind=operands, seed=seed)
        drained = 1 if fuse else groups
        drain_bytes = (sk._ZERO_TILE_WORDS + column_tiles * drained * sk._BLOCK_TOKENS * sk._TILE) * sk._ELEMENT_BYTES
        kernel = sk.k.build_looped_gemm(column_tiles=column_tiles, groups=groups, activate=activate,
                                        fuse_groups=fuse, loop_blocks=sk._LOOP_BLOCKS)
        with tempfile.TemporaryDirectory() as scratch:
            paths = sk.write_operands(pathlib.Path(scratch), "r0", left, right, drain_bytes)
            windows = [p.read_bytes() for p in paths] + [b""]
        recorded.append(Case("", f"looped gemm [{tokens},{depth}]@[{depth},{column_tiles * sk._TILE}]",
                             kernel.image.to_bytes(pad_to=4), windows, [0] * 4, [None] * 4, 2, drain_bytes,
                             200_000 + column_tiles * groups * 60_000))

    def softmax_rows(rows, seed=0):
        rng = np.random.default_rng(seed)
        scores = sk.k._on_bf16_grid(values=(rng.standard_normal((rows, sk.k._VREG_ELEMENTS)) * 4.0).astype(np.float32))
        per_chunk = sk.k._MAX_DESCRIPTOR_ROWS // sk.k._TILE_VREGS
        chunks, remainder = divmod(rows, per_chunk)
        if remainder or chunks == 0:
            chunks, per_chunk = 1, rows
        kernel = sk.k.build_softmax_rows(per_chunk, chunks=chunks)
        out_bytes = rows * scores.shape[1] * sk._ELEMENT_BYTES
        windows = [sk.k.as_bf16_words(values=scores).reshape(-1).tobytes(), b"",
                   np.full(rows * scores.shape[1], 0x7F80, dtype="<u2").tobytes(), b""]
        recorded.append(Case("", f"softmax {rows}x{scores.shape[1]}", kernel.image.to_bytes(pad_to=4), windows,
                             [0] * 4, [None] * 4, 2, out_bytes, 200_000))

    # The set tb/run_all.sh runs through sim_kernel.py by default (its lines 71-137), minus the 2-round and
    # MINITPU_GEMM_STAGING variants; the long tier (rope 14, gqa 2 7, ffn, attention, flash --full) is left out.
    plan = {
        "gemm_structured": lambda: gemm(operands="structured"),
        "gemm_varying": lambda: gemm(),
        "gemm_varying_gelu": lambda: gemm(activate=True),
        "gemm_c4g2_fuse": lambda: gemm(column_tiles=4, groups=2, fuse=True),
        "gemm_c4g2": lambda: gemm(column_tiles=4, groups=2),
        "softmax_8": lambda: softmax_rows(8),
        "layernorm_4": lambda: sk.run_layernorm(4, 0),
        "layernorm_32_packed": lambda: sk.run_layernorm(32, 0, packed=True),
        "add": lambda: sk.run_elementwise_add(0),
        "add_bias": lambda: sk.run_elementwise_add(0, broadcast=True),
        "add_bias_gelu_3072": lambda: sk.run_elementwise_add(0, features=3072, broadcast=True, gelu=True),
        "softmax_packed_12": lambda: sk.run_softmax_packed(12, 0),
        "softmax_packed_2_span128": lambda: sk.run_softmax_packed(2, 0, span=128),
        "rmsnorm_4_w896": lambda: sk.run_rmsnorm(4, 896, 0),
        "rmsnorm_32_w896_packed": lambda: sk.run_rmsnorm(32, 896, 0, packed=True),
        "rope_2": lambda: sk.run_rope(2, 0),
        "swiglu_2": lambda: sk.run_swiglu(2, 0),
        "gqa_2_2": lambda: sk.run_gqa(2, 2, 0),
        "head_scatter": lambda: sk.run_head_scatter(0),
        "loop_begin_r": lambda: sk.run_loop_begin_r(),
        "flash_attention": lambda: sk.run_flash_attention(full=False),
    }
    groups: dict[str, list[Case]] = {}
    for group, build in plan.items():
        recorded.clear()
        with contextlib.redirect_stdout(io.StringIO()):
            try:
                build()
            except Exception as error:  # the caller judging an unwritten drain may raise after recording
                print(f"note: {group}: sim_kernel's judging raised {type(error).__name__} after "
                      f"{len(recorded)} launch(es) were recorded", file=sys.stderr)
        if not recorded:
            raise RuntimeError(f"{group}: no launch was recorded")
        for index, case in enumerate(recorded):
            case.name = group if len(recorded) == 1 else f"{group}.{index}"
        groups[group] = list(recorded)
    return groups


def run_rtl(tree: pathlib.Path, build_dir: pathlib.Path, case: Case, round_trip: int, depth: int) -> dict:
    """One launch on tb_kernel_image, as compiler/harness.py:run stages it."""
    stage = pathlib.Path(tempfile.mkdtemp(prefix=f"oracle.{case.name}.", dir=build_dir.parent))
    try:
        (stage / "kernel.bin").write_bytes(case.image)
        plusargs = [f"+KERNEL_BIN={stage / 'kernel.bin'}", f"+DUMP={stage / 'dump.bin'}",
                    f"+DUMP_BYTES={case.drain_bytes}", f"+DUMP_ARG={case.drain_argument}",
                    f"+MAX_CYCLES={case.max_cycles}"]
        for index, window in enumerate(case.windows):
            if window:
                (stage / f"arg{index}.bin").write_bytes(window)
                plusargs.append(f"+ARG{index}={stage / f'arg{index}.bin'}")
            if case.offsets[index]:
                plusargs.append(f"+ARG_OFFSET{index}={case.offsets[index]}")
            if case.values[index] is not None:
                plusargs.append(f"+ARG_VALUE{index}={case.values[index]}")
        env = {**tool_env(), "KERNEL_IMAGE_BUILD_DIR": str(build_dir), "MINITPU_FIFO_DEPTH": str(depth),
               "MINITPU_ROUND_TRIP_CYCLES": str(round_trip), "TB_TIMEOUT": os.environ.get("TB_TIMEOUT", "7200")}
        start = time.monotonic()
        done = subprocess.run(["bash", str(tree / "tb" / "run_kernel_image.sh"), *plusargs], cwd=tree, env=env,
                              capture_output=True, text=True)
        wall = time.monotonic() - start
        transcript = done.stdout + done.stderr
        found = ROUND.search(transcript)
        result = {"rc": done.returncode, "wall_s": round(wall, 1),
                  "pass_banner": "=== TB_KERNEL_IMAGE PASS ===" in transcript,
                  "assertions": [ln for ln in transcript.splitlines() if "Assertion failed" in ln][:3]}
        if found:
            result.update(status=found.group(1), cycles=int(found.group(2)), bundles=int(found.group(3)),
                          launch_status=int(found.group(4), 16))
        dump = stage / "dump.bin"
        result["drain"] = dump.read_bytes() if dump.is_file() else None
        if not result["pass_banner"]:
            result["tail"] = transcript[-1500:]
        return result
    finally:
        if not os.environ.get("ORACLE_KEEP"):
            shutil.rmtree(stage, ignore_errors=True)


def run_emulator(case: Case) -> dict:
    from compiler.backends import EmulationError, emulate  # the pinned clone's
    from compiler.operands import Launch

    launch = Launch(windows=tuple(case.windows), offsets=tuple(case.offsets), drain_bytes=case.drain_bytes,
                    label=case.label, values=tuple(case.values), drain_argument=case.drain_argument)
    try:
        outcome = emulate(case.image, launch)
    except EmulationError as error:
        return {"error": str(error), "drain": None}
    return {"drain": outcome.drained, "bundles": outcome.bundles}


def compare(rtl: bytes | None, emu: bytes | None) -> dict:
    import numpy as np

    if rtl is None or emu is None:
        return {"bit_identical": False}
    a = np.frombuffer(rtl, dtype="<u2")
    b = np.frombuffer(emu, dtype="<u2")
    differ = int(np.count_nonzero(a != b))
    # BF16 NaN is exponent 0xFF with a non-zero mantissa; the RTL writes +qNaN (0x7FC0), numpy's NaN is often -qNaN.
    nan_a = ((a & 0x7F80) == 0x7F80) & ((a & 0x007F) != 0)
    nan_b = ((b & 0x7F80) == 0x7F80) & ((b & 0x007F) != 0)
    differ_not_nan = int(np.count_nonzero((a != b) & ~(nan_a & nan_b)))
    fa = (a.astype(np.uint32) << 16).view(np.float32).astype(np.float64)
    fb = (b.astype(np.uint32) << 16).view(np.float32).astype(np.float64)
    finite = np.isfinite(fa) & np.isfinite(fb)
    norm = float(np.linalg.norm(fb[finite])) or 1.0
    # Distance in BF16 units in the last place: map sign-magnitude words onto one ordered integer line.
    ordered = lambda w: np.where(w & 0x8000, -(w.astype(np.int64) & 0x7FFF), w.astype(np.int64) & 0x7FFF)  # noqa: E731
    ulp = np.abs(ordered(a) - ordered(b))[finite]
    return {"bit_identical": differ == 0, "words": int(a.size), "words_differ": differ,
            "words_differ_except_nan_payload": differ_not_nan,
            "nonfinite_mismatch": int(np.count_nonzero(np.isfinite(fa) != np.isfinite(fb))),
            "max_ulp_finite": int(ulp.max()) if ulp.size else 0,
            "rel_error_finite": float(np.linalg.norm(fa[finite] - fb[finite]) / norm)}


def sha(payload: bytes | None) -> str | None:
    return None if payload is None else hashlib.sha256(payload).hexdigest()


def versions() -> dict:
    env = tool_env()
    return {"verilator": sh(str(VERILATOR_BIN / "verilator"), "--version", env=env).strip(),
            "gxx": sh("g++", "--version", env=env).splitlines()[0],
            "python": sys.version.split()[0]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--work", type=pathlib.Path, default=WORK, help="scratch for the export and the build")
    parser.add_argument("--only", nargs="*", default=None, help="kernel groups to run (default: all)")
    parser.add_argument("--list", action="store_true", help="list the kernel groups and their launches")
    parser.add_argument("--jobs", type=int, default=4, help="testbench launches in parallel")
    parser.add_argument("--round-trip", type=int, default=0, help="memory model round trip (ROUND_TRIP_CYCLES)")
    parser.add_argument("--out", type=pathlib.Path, default=HERE / "oracle.json")
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()

    tree = export_clone(args.work)
    groups = collect(tree)
    import asm  # the pinned clone's, on the path collect() set
    depth = asm.mxu_output_fifo()[0]
    if args.list:
        for group, cases in groups.items():
            print(group, ", ".join(f"{c.name} ({len(c.image) // 16} bundles)" for c in cases))
        return 0
    if args.only:
        unknown = set(args.only) - set(groups)
        if unknown:
            raise SystemExit(f"unknown kernel groups: {sorted(unknown)}")
        groups = {g: groups[g] for g in args.only}
    cases = [case for group in groups.values() for case in group]

    build_dir = args.work / "build" / f"kernel_image_fifo{depth}"
    build_dir.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    sh("bash", str(tree / "tb" / "build_kernel_image.sh"), str(build_dir), cwd=tree,
       env={**tool_env(), "MINITPU_FIFO_DEPTH": str(depth)})
    build_wall = time.monotonic() - start
    print(f"tb_kernel_image build ready in {build_wall:.0f} s ({build_dir})", file=sys.stderr)

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        rtl_runs = dict(zip([c.name for c in cases],
                            pool.map(lambda c: run_rtl(tree, build_dir, c, args.round_trip, depth), cases)))
    emu_runs = {c.name: run_emulator(c) for c in cases}

    entries = []
    for case in cases:
        rtl, emu = rtl_runs[case.name], emu_runs[case.name]
        entry = {
            "name": case.name, "label": case.label, "bundles_in_image": len(case.image) // 16,
            "image_sha256": sha(case.image),
            "operands_sha256": sha(b"".join(len(w).to_bytes(8, "little") + w for w in case.windows)),
            "offsets": case.offsets, "values": case.values, "drain_argument": case.drain_argument,
            "drain_bytes": case.drain_bytes, "max_cycles": case.max_cycles,
            "rtl": {"halted": rtl.get("status") == "HALTED" and rtl["pass_banner"], "cycles": rtl.get("cycles"),
                    "bundles_issued": rtl.get("bundles"), "launch_status": rtl.get("launch_status"),
                    "assertions": rtl["assertions"], "drain_sha256": sha(rtl["drain"])},
            "emulator": {"bundles_issued": emu.get("bundles"), "error": emu.get("error"),
                         "drain_sha256": sha(emu["drain"])},
            "compare": compare(rtl["drain"], emu["drain"]),
        }
        entry["agree"] = entry["compare"]["bit_identical"]
        entry["bundles_match"] = entry["rtl"]["bundles_issued"] == entry["emulator"]["bundles_issued"]
        entries.append((entry, rtl))

    manifest = {
        "schema": "allo/minitpu-rtl/oracle/1",
        "minitpu_commit": PIN,
        "harness": "tb/run_kernel_image.sh (tb_kernel_image on src/minitpu.sv, Verilator --binary --timing)",
        "emulator": "tools/emulate_image.py via compiler/backends.py:emulate",
        "round_trip_cycles": args.round_trip,
        "mxu_output_fifo_depth": depth,
        "tools": versions(),
        "kernels": [entry for entry, _ in entries],
    }
    if not args.no_write:
        args.out.write_text(json.dumps(manifest, indent=1) + "\n")

    print(f"{'kernel':26} {'image':9} {'rtl drain':9} {'emu drain':9} {'agree':6} {'differ':>6} "
          f"{'rel err':>8} {'cycles':>7} {'bundles':>7} {'emu':>4} {'wall':>5}")
    for entry, rtl in entries:
        c = entry["compare"]
        short = lambda h: (h or "-")[:8]  # noqa: E731
        agree = ("yes" if entry["agree"] else
                 "nan" if c.get("words_differ_except_nan_payload") == 0 else "NO")
        print(f"{entry['name']:26} {short(entry['image_sha256']):9} {short(entry['rtl']['drain_sha256']):9} "
              f"{short(entry['emulator']['drain_sha256']):9} {agree:6} "
              f"{c.get('words_differ', '-'):>6} {c.get('rel_error_finite', float('nan')):>8.1e} "
              f"{entry['rtl']['cycles'] or '-':>7} {entry['rtl']['bundles_issued'] or '-':>7} "
              f"{'=' if entry['bundles_match'] else 'DIFF':>4} {rtl['wall_s']:>4.0f}s")
        if not entry["rtl"]["halted"]:
            print(f"   RTL did not halt: {rtl.get('tail', '')[-600:]}")
        if entry["emulator"]["error"]:
            print(f"   emulator: {entry['emulator']['error']}")
    halted = all(e["rtl"]["halted"] for e, _ in entries)
    agree = sum(e["agree"] for e, _ in entries)
    nan = sum(not e["agree"] and e["compare"].get("words_differ_except_nan_payload") == 0 for e, _ in entries)
    bundles = sum(e["bundles_match"] for e, _ in entries)
    print(f"{len(entries)} launches; RTL halted on {'all' if halted else 'NOT all'}; RTL == emulator drain "
          f"bit-identical on {agree}, equal up to NaN sign/payload on {nan} more; issued bundles equal on "
          f"{bundles}/{len(entries)}")
    print("agree: yes = bit-identical, nan = only NaN encodings differ, NO = values differ (emulator is functional:"
          " float64 MXU, exact SFU functions)")
    return 0 if halted else 1


if __name__ == "__main__":
    raise SystemExit(main())
