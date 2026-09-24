# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run one frozen check and vouch for its verdict, when ALLO is the candidate. FROZEN.

`chia_agent/gate_runner.py` does this for the design-level loop, and this module
reuses its security-critical primitive verbatim -- `_Frozen`, `_snapshot`,
`_changed` are imported from it, not copied. One canonical implementation of
"refuse to let a module's functions be rebound, and notice if they were".

What it CANNOT reuse is the order. `gate_runner.py` imports `numpy` AND `allo`,
then snapshots, then freezes, then imports the candidate -- correct when the
candidate is two design files and `allo` is trusted. Here `allo` IS the
candidate. Its import-time code runs inside `import allo`, so a snapshot taken
after that point would record whatever the candidate had already done to numpy.
This runner therefore:

1. reads the nonce from stdin, before anything else, and closes stdin;
2. imports `numpy`, `numpy.random` and `builtins` ONLY, snapshots them, and
   freezes them -- while `allo` is still unimported;
3. runs the check, which is what imports the candidate's `allo`;
4. takes the check's RETURN VALUE (or `SystemExit(0)` raised from the check
   script's own module frame) as the verdict, never a printed line;
5. compares the numpy/builtins snapshot again;
6. only then prints `CHIA-GATE <check> OK <nonce>`.

The nonce lives in one local. The candidate has no stdin (closed), a read-only
filesystem (bubblewrap), its own PID namespace, and `patch_policy.py` refuses
`sys.modules`, frame walking and `builtins` in added lines -- none of which
`allo/` uses today, so refusing them costs nothing.

Checks:

    build_import   import allo and its HLS backends; print the resolved path
    pytest         Allo's own suites; the verdict is pytest's exit status and
                   the per-test outcomes a frozen in-process plugin collects
    bench_isa      \\ the TinyTPU-isa design case, run through the DESIGN's own
    stress_isa      > frozen gate scripts. `chia_agent/gate_runner.py` cannot be
    cosim          / reused for these because of the ordering above.
    design_case    one entry of design_cases.py: build, numpy-golden, csynth

    printf '%s\\n' NONCE | python abs_gate_runner.py <check> [args...]
"""

import importlib
import json
import os
import runpy
import sys
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
DESIGN = os.path.join(REPO, "examples", "tinytpu")
DESIGN_AGENT = os.path.join(DESIGN, "chia_agent")

#: Frozen scripts run as `__main__`; the verdict is SystemExit(0) from their
#: own module frame.
SCRIPTS = {
    "bench_isa": os.path.join(DESIGN, "bench_isa.py"),
    "cosim": os.path.join(DESIGN, "cosim.py"),
}
#: Frozen modules with a `main(argv) -> int`; the verdict is the return value.
MODULES = {
    "stress_isa": "examples.tinytpu.stress_isa",
    "param_check": None,   # chia_agent/param_check.py, run as a script below
}
#: What the checks compute their golden references with. NOT `allo`: allo is
#: the candidate.
WATCHED = ("numpy", "builtins")


def _fail(check, why):
    print(f"CHIA-GATE {check} FAIL {why}", flush=True)
    return 1


def _freeze_machinery():
    """`_Frozen` / `_snapshot` / `_changed` from the design loop's gate runner.

    Imported rather than copied: it is the one security-critical primitive in
    the harness and there should be exactly one of it. `WATCHED` is narrowed
    here because `allo` is the candidate in this loop.
    """
    sys.path.insert(0, DESIGN_AGENT)
    import gate_runner                                    # noqa: E402
    gate_runner.WATCHED = WATCHED
    return gate_runner


def _exit_status_of(exc, script):
    """(ok, why) for a SystemExit: it must come from `script`'s module frame."""
    tb = exc.__traceback__
    while tb is not None and tb.tb_next is not None:
        tb = tb.tb_next
    if tb is None:
        return False, "SystemExit with no traceback"
    code = tb.tb_frame.f_code
    own = (os.path.abspath(code.co_filename) == script
           and code.co_name == "<module>")
    if not own:
        return False, (f"SystemExit raised from {code.co_filename}:"
                       f"{code.co_name}, not from {os.path.basename(script)}")
    return exc.code in (0, None), f"exit status {exc.code!r}"


def _run_script(script, args):
    sys.argv = [script, *args]
    try:
        runpy.run_path(script, run_name="__main__")
        return False, f"{os.path.basename(script)} returned without an exit status"
    except SystemExit as e:
        return _exit_status_of(e, script)


def main():
    # 1. The nonce, before anything else can run.
    nonce = sys.stdin.readline().strip()
    if len(nonce) < 32 or not all(c in "0123456789abcdef" for c in nonce):
        print("CHIA-GATE usage: a >= 32-hex-digit nonce on stdin", flush=True)
        return 2
    try:
        sys.stdin.close()
    except OSError:
        pass
    check = sys.argv[1] if len(sys.argv) > 1 else ""
    args = sys.argv[2:]
    sys.path.insert(0, REPO)

    # 2. numpy and builtins, snapshotted and frozen while `allo` is still
    #    unimported. This is the ordering `gate_runner.py` cannot give us.
    gr = _freeze_machinery()
    import numpy            # noqa: F401
    import numpy.random     # noqa: F401
    if "allo" in sys.modules:
        return _fail(check, "allo was imported before the snapshot")
    before = gr._snapshot()
    gr._freeze()

    ok, why = False, "not run"
    try:
        if check == "build_import":
            import allo
            import allo.dataflow          # noqa: F401
            import allo.customize         # noqa: F401
            import allo.backend.hls       # noqa: F401
            import allo.backend.vitis     # noqa: F401
            import allo.ir.builder        # noqa: F401
            import allo.ir.infer          # noqa: F401
            import allo.ir.types          # noqa: F401
            print("ALLO " + os.path.realpath(allo.__file__), flush=True)
            ok, why = True, "imported"
        elif check == "pytest":
            sys.path.insert(0, HERE)
            import suite_runner
            rc, report = suite_runner.run(args)
            print("SUITE " + json.dumps(report, sort_keys=True), flush=True)
            ok, why = rc == 0, f"suite runner returned {rc}"
        elif check == "probe":
            sys.path.insert(0, HERE)
            import probes
            rc, report = probes.run(args)
            print("PROBE " + json.dumps(report, sort_keys=True), flush=True)
            ok, why = rc == 0, f"probe runner returned {rc}"
        elif check == "design_case":
            sys.path.insert(0, HERE)
            import design_cases
            rc, report = design_cases.run(args)
            print("CASE " + json.dumps(report, sort_keys=True), flush=True)
            ok, why = rc == 0, f"design case returned {rc}"
        elif check == "stress_isa":
            mod = importlib.import_module(MODULES["stress_isa"])
            sys.stdout.flush()
            rc = mod.main(args)
            ok, why = rc == 0, f"stress_isa.main returned {rc!r}"
        elif check == "param_check":
            script = os.path.join(DESIGN_AGENT, "param_check.py")
            ok, why = _run_script(script, args)
        elif check in SCRIPTS:
            ok, why = _run_script(SCRIPTS[check], args)
        else:
            return _fail(check or "?", f"unknown check; one of "
                                       f"{sorted(set(SCRIPTS) | set(MODULES) | {'build_import', 'pytest', 'design_case', 'probe'})}")
    except BaseException as e:                      # noqa: BLE001 -- on purpose
        traceback.print_exc()
        sys.stdout.flush()
        return _fail(check, f"{type(e).__name__} during the check: {e}"[:400])
    sys.stdout.flush()
    if not ok:
        return _fail(check, why)
    # 5. numpy and builtins as they were.
    bad = gr._changed(before)
    if bad:
        return _fail(check, f"tamper: numpy/builtins changed: {bad[:8]}")
    print(f"CHIA-GATE {check} OK {nonce}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
