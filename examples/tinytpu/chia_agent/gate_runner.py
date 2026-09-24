# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run one frozen check in-process and vouch for its verdict. FROZEN.

`bench_isa.py`, `stress_isa.py` and `cosim.py` import the candidate's two
files, so the candidate's code runs in the process whose output the evaluator
used to read. A printed verdict line (`ALL EXACT`, `STRESS OK: n/n ...`) is
then only as trustworthy as that code: a module that prints the line and raises
`SystemExit(0)` at import time "passes" any check that reads stdout, with an
int16 datapath underneath. The spec policy now refuses the obvious spellings of
that, but a policy is a blacklist. This runner is what makes the verdict
unforgeable:

1. The evaluator writes a fresh random nonce on stdin. It is read HERE, before
   anything imports the candidate, and lives only in this function's locals
   -- not in the environment, the command line, or a file. The candidate has no
   file or stdin access (policy + sandbox), no `sys.modules`, no frames (policy),
   and cannot see the evaluator's process (own PID namespace).
2. The checks compute their golden references and operands with `numpy` and
   build the design with `allo`, in the candidate's process. Before the
   candidate is imported, every loaded `numpy*` / `allo*` module (and
   `builtins`) is switched to a module class that refuses to REBIND an existing
   function, class or submodule attribute, and records the attempt -- so a
   `default_rng` that only draws [-4, 4], swapped in for one call and swapped
   back, is refused and reported as tamper, not passed. The identity of every
   such attribute is also snapshotted and compared after the check, for any
   route that bypasses `__setattr__`.
3. The candidate is imported first, on its own: anything it raises -- including
   `SystemExit(0)` -- is a failure.
4. The verdict is the check's RETURN VALUE, not its output:
   `stress_isa.main(argv)` must return 0; `bench_isa.py` / `cosim.py` run as
   `__main__` must end in `SystemExit(0)` raised from that script's own
   module-level frame. A `SystemExit` from anywhere else (the candidate's
   functions) is a failure.

Only then is `CHIA-GATE <check> OK <nonce>` printed; the evaluator requires
that exact line. The check's own output passes through unchanged, for the log.

    printf '%s\\n' NONCE | python gate_runner.py \\
        {bench_isa|stress_isa|cosim|param_check|codesign|codesign_cosim} [args...]
"""

import importlib
import os
import runpy
import sys
import traceback

PKG = "examples.tinytpu"
HERE = os.path.dirname(os.path.abspath(__file__))
DESIGN = os.path.dirname(HERE)
CHECKS = {
    "bench_isa": os.path.join(DESIGN, "bench_isa.py"),
    "stress_isa": os.path.join(DESIGN, "stress_isa.py"),
    "cosim": os.path.join(DESIGN, "cosim.py"),
    "param_check": os.path.join(HERE, "param_check.py"),
    # The co-design loop's two frozen stages: the exhaustive mapspace
    # enumeration with its refusal histogram, and cosim of the program the
    # mapper chose for this hardware. Both live here rather than in the design
    # directory because the agent may not edit them.
    "codesign": os.path.join(HERE, "codesign_gate.py"),
    "codesign_cosim": os.path.join(HERE, "codesign_cosim.py"),
    # The model term of the objective: the PyTorch workload suite, layer by
    # layer, through one csynth of the candidate's own hardware. Frozen and run
    # from here for the same reason cosim.py is -- the candidate supplies the
    # machine, never the workload it is measured on.
    "workloads": os.path.join(DESIGN, "workloads", "run.py"),
}
#: Module-name prefixes whose attributes the checks compute with.
WATCHED = ("numpy", "allo", "builtins")


#: Rebinding attempts refused by `_Frozen`, for the verdict. Only this module
#: and the refused writer see it.
_REFUSED = []


def _guarded(val):
    return callable(val) or isinstance(val, type(os))


class _Frozen(type(os)):
    """A module that may gain attributes but not have its functions, classes or
    submodules rebound or deleted."""

    def __setattr__(self, name, value):
        cur = self.__dict__.get(name, _REFUSED)
        if cur is not _REFUSED and cur is not value and _guarded(cur):
            _REFUSED.append(f"{self.__name__}.{name}")
            raise AttributeError(f"CHIA gate: {self.__name__}.{name} is frozen")
        super().__setattr__(name, value)

    def __delattr__(self, name):
        if _guarded(self.__dict__.get(name)):
            _REFUSED.append(f"del {self.__name__}.{name}")
            raise AttributeError(f"CHIA gate: {self.__name__}.{name} is frozen")
        super().__delattr__(name)


def _freeze():
    for mod in _watched_modules().values():
        try:
            if type(mod) is type(os):
                mod.__class__ = _Frozen
        except TypeError:
            pass


def _watched_modules():
    return {name: mod for name, mod in list(sys.modules.items())
            if mod is not None and name.split(".")[0] in WATCHED}


def _snapshot():
    """{(module, attr): id} for every callable / class / module attribute."""
    snap = {}
    for name, mod in _watched_modules().items():
        try:
            items = list(vars(mod).items())
        except TypeError:
            continue
        for attr, val in items:
            if attr.startswith("__") or not _guarded(val):
                continue
            snap[(name, attr)] = id(val)
    return snap


def _changed(before):
    mods = _watched_modules()
    bad = [f"refused rebinding {r}" for r in _REFUSED]
    for (name, attr), ident in before.items():
        mod = mods.get(name)
        if mod is None:
            bad.append(f"{name} removed from sys.modules")
            continue
        val = vars(mod).get(attr, _changed)
        if val is _changed or id(val) != ident:
            bad.append(f"{name}.{attr}")
    return sorted(set(bad))


def _fail(check, why):
    print(f"CHIA-GATE {check} FAIL {why}", flush=True)
    return 1


def main():
    # 1. The nonce, before any candidate code can run.
    nonce = sys.stdin.readline().strip()
    if len(nonce) < 32 or not all(c in "0123456789abcdef" for c in nonce):
        print("CHIA-GATE usage: a >= 32-hex-digit nonce on stdin", flush=True)
        return 2
    try:
        sys.stdin.close()
    except OSError:
        pass
    check = sys.argv[1] if len(sys.argv) > 1 else ""
    if check not in CHECKS:
        return _fail(check or "?", f"unknown check; one of {sorted(CHECKS)}")
    script, args = CHECKS[check], sys.argv[2:]
    # The tree root first, exactly as the scripts do themselves.
    sys.path.insert(0, os.path.abspath(os.path.join(DESIGN, "..", "..")))

    # 2. Snapshot what the checks compute with.
    import numpy  # noqa: F401
    import numpy.random  # noqa: F401
    import allo  # noqa: F401
    import allo.dataflow  # noqa: F401
    before = _snapshot()
    _freeze()

    # 3. The candidate, on its own.
    try:
        for name in ("microarch_isa", "isa_dsl"):
            importlib.import_module(f"{PKG}.{name}")
    except BaseException as e:  # noqa: BLE001 -- SystemExit included, on purpose
        traceback.print_exc()
        bad = _changed(before)
        return _fail(check, f"importing the candidate raised {type(e).__name__}"
                     + (f"; tamper: {bad[:8]}" if bad else ""))
    bad = _changed(before)
    if bad:
        return _fail(check, f"tamper: the candidate's import replaced {bad[:8]}")

    # 4. The check; its return value is the verdict.
    ok = False
    try:
        if check == "stress_isa":
            mod = importlib.import_module(f"{PKG}.stress_isa")
            sys.stdout.flush()
            ok = mod.main(args) == 0
        else:
            sys.argv = [script, *args]
            try:
                runpy.run_path(script, run_name="__main__")
                why = f"{check} returned without an exit status"
            except SystemExit as e:
                tb = e.__traceback__
                while tb.tb_next is not None:
                    tb = tb.tb_next
                code = tb.tb_frame.f_code
                own = (os.path.abspath(code.co_filename) == script
                       and code.co_name == "<module>")
                ok = own and e.code in (0, None)
                why = (f"exit status {e.code!r}" if own else
                       f"SystemExit raised from {code.co_filename}:{code.co_name}, "
                       f"not from {os.path.basename(script)}")
            if not ok:
                sys.stdout.flush()
                return _fail(check, why)
    except BaseException as e:  # noqa: BLE001
        traceback.print_exc()
        sys.stdout.flush()
        return _fail(check, f"{type(e).__name__} during the check")
    sys.stdout.flush()
    if not ok:
        return _fail(check, "the check reported failures")
    bad = _changed(before)
    if bad:
        return _fail(check, f"tamper: replaced during the check: {bad[:8]}")
    print(f"CHIA-GATE {check} OK {nonce}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
