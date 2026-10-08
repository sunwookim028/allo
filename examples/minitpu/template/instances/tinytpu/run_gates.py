# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run TinyTPU's frozen gates, unedited, on the instance.

    python -m examples.minitpu.template.instances.tinytpu.run_gates GATE [args]

GATE is ``stress`` (``stress_isa.py``), ``bench`` (``bench_isa.py``),
``cosim`` (``cosim.py``; ``TPU_PRJ`` defaults to a scratch project, never
``examples/tinytpu/``), ``gen_isa`` (``gen_isa.py --check``, plus the
instance's own ``isa_slots`` arm) or ``isa`` (the slot comparison alone).
``TPU_INSTANCE=template`` (default) or ``frozen`` selects the build; every
``TPU_*`` knob means what it means to the frozen glue.

The mount is ``mutate.py``'s: ``glue/`` goes in front of
``examples.tinytpu``'s search path, so ``examples.tinytpu.microarch_isa``
resolves to the instance-aware copy while every other module of the harness
(``isa_dsl``, ``isa_ref``, ``kpn_model``, ``stress_isa``, ``cosim``, ...)
and the ``ip`` unit library come from the frozen checkout. ``gen_isa.py``
imports the design by its bare name after putting its own directory first,
so that name is pre-bound to the mounted module.
"""

from __future__ import annotations

import importlib
import json
import os
import runpy
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
GLUE = os.path.join(HERE, "glue")
REPO = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", ".."))
TINYTPU = os.path.join(REPO, "examples", "tinytpu")


def mount():
    """Put the instance-aware glue in front of ``examples.tinytpu``."""
    sys.path.insert(0, REPO)
    import examples.tinytpu as pkg  # noqa: PLC0415
    if GLUE not in pkg.__path__:
        pkg.__path__ = [GLUE] + list(pkg.__path__)
    for stale in [m for m in sys.modules if m == "examples.tinytpu.microarch_isa"]:
        del sys.modules[stale]
    U = importlib.import_module("examples.tinytpu.microarch_isa")
    assert U.__file__.startswith(GLUE), U.__file__
    sys.modules["microarch_isa"] = U       # gen_isa's bare `import microarch_isa`
    return U


def isa_arm(U, spec=None) -> list:
    """The instance's ISA against the spec: ``compose.isa_slots`` of the
    composed instance against the opcodes the spec declares and the module
    each belongs to. Returns the failures, printing one line as the other
    arms do."""
    from examples.minitpu.template.instances.tinytpu.instance import compare_isa  # noqa: PLC0415
    if getattr(U, "GEOMETRY", None) is None:
        print("  isa slots: skipped (TPU_INSTANCE=frozen composes no options)")
        return []
    r = compare_isa(U.TPU.architecture, spec)
    fails = []
    for kind in ("missing_in_instance", "extra_in_instance", "module_differs"):
        for s in r[kind]:
            fails.append(f"isa slot {s!r}: {kind.replace('_', ' ')} "
                         f"(instance {r['instance'].get(s)!r}, spec {r['spec'].get(s)!r})")
    mods = {}
    for s, m in r["instance"].items():
        mods.setdefault(m, []).append(s)
    print(f"  isa slots: {len(r['instance'])} of the composed instance == the "
          f"spec's {len(r['spec'])} opcodes, by module: "
          + "; ".join(f"{m}: {', '.join(v)}" for m, v in mods.items()))
    return fails


def main(argv):
    if not argv or argv[0] not in ("stress", "bench", "cosim", "gen_isa", "isa"):
        print(__doc__)
        return 2
    gate, args = argv[0], argv[1:]
    if gate == "cosim":
        os.environ.setdefault("TPU_PRJ", os.path.join(
            tempfile.gettempdir(), f"ttinst_cosim_{os.environ.get('TPU_INSTANCE', 'template')}.prj"))
        print(f"  cosim project: {os.environ['TPU_PRJ']}")
    U = mount()
    print(f"  build under test: TPU_INSTANCE={U.INSTANCE} "
          f"({'template instance' if U.GEOMETRY is not None else 'frozen ip/tinytpu.py'}), "
          f"T={U.T} MAXDIM={U.MAXDIM} QD={U.QD}", flush=True)
    if gate == "isa":
        fails = isa_arm(U)
        for f in fails:
            print("  ISA FAIL", f)
        print("  ISA SLOTS " + ("OK" if not fails else "FAILED"))
        return 1 if fails else 0
    if gate == "gen_isa":
        # gen_isa's arms, with the instance's module bound; then this
        # instance's own arm. The record names which arms read source
        # files of the frozen tree and so do not see the instance.
        sys.path.insert(0, TINYTPU)
        G = importlib.import_module("gen_isa")
        spec = G.load()
        print("TinyTPU-isa ISA conformance on the instance\n")
        E = G.spec_module(spec)
        D = importlib.import_module("isa_dsl")
        fails = []
        for arm in (lambda: G.check_design_constants(spec, U),
                    lambda: G.check_design_layout(spec),
                    lambda: G.check_design_parameters(spec, U, E),
                    lambda: G.check_design_slices(spec),
                    lambda: G.check_behaviour(spec, U, D, E),
                    lambda: G.check_actions(spec, U, E),
                    lambda: G.check_derived_properties(spec, U, D, E),
                    lambda: isa_arm(U, spec)):
            fails += arm()
        if "--conform" in args:
            fails += G.check_emitted(spec)
        print()
        for f in fails:
            print("  ISA FAIL", f)
        print("  ISA OK: the instance agrees with the spec on every arm run"
              if not fails else f"  ISA FAILED: {len(fails)} disagreement(s)")
        return 1 if fails else 0
    script = os.path.join(TINYTPU, {"stress": "stress_isa.py", "bench": "bench_isa.py",
                                    "cosim": "cosim.py"}[gate])
    sys.argv = [script] + args
    if gate == "cosim":
        return cosim(script)
    try:
        runpy.run_path(script, run_name="__main__")
    except SystemExit as e:
        return int(e.code or 0)
    return 0


def linker_dir():
    """Where ``cosim_design -ldflags "-B..."`` finds a linker that reads
    this host's objects. ``cosim.py`` points the Vitis compiler driver at
    ``/usr/bin`` (a 2.42 ``ld`` on ace-01; ``dev/toolchains.rst``). On
    zhang-21 ``/usr/bin/ld`` is 2.30 and refuses the compressed
    ``.debug_info`` Vitis's gcc emits (``unable to initialize decompress
    status``), on the frozen design as much as on the instance. ``TPU_LD_DIR``
    names a directory with an ``ld``; otherwise the ``allo`` env's binutils
    (2.44) is linked into a scratch directory; otherwise ``/usr/bin``."""
    chosen = os.environ.get("TPU_LD_DIR")
    if chosen:
        return chosen
    conda = os.environ.get("CONDA_PREFIX", "")
    ld = os.path.join(conda, "bin", "x86_64-conda-linux-gnu-ld")
    if not os.path.exists(ld):
        return "/usr/bin"
    d = os.path.join(tempfile.gettempdir(), "ttinst_ld")
    os.makedirs(d, exist_ok=True)
    link = os.path.join(d, "ld")
    if not os.path.exists(link):
        os.symlink(ld, link)
    return d


def cosim(script):
    """``cosim.py``'s ``main()`` with its ``-B`` pointed at ``linker_dir()``:
    the script formats ``LDFLAGS`` into ``TCL_COSIM`` at import, so it is
    loaded as a module and the one string patched before ``main`` runs."""
    g = runpy.run_path(script, run_name="ttinst_cosim")
    where = linker_dir()
    print(f"  cosim linker: -B{where}", flush=True)
    g["TCL_COSIM"] = g["TCL_COSIM"].replace('-ldflags "-B/usr/bin', f'-ldflags "-B{where}')
    assert f"-B{where}" in g["TCL_COSIM"]
    return int(g["main"]() or 0)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
