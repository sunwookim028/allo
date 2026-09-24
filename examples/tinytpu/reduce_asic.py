# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Take the adder tree through the ASIC flow, not only through csynth.

`reduce_csynth.py` answers "what does it cost on an FPGA, and is it free of
the ELAB-366 shape that standard cells refuse". That audit is a *static* scan
of the generated Verilog, and a scan that finds nothing is not a synthesis:
the project has already shipped one export whose 223 files and 223 matching
manifest entries contained no top module at all, with every internal check
passing. The only thing that settles "this unit is synthesisable" is Design
Compiler mapping it to standard cells.

So this exports the emitted RTL and runs the same flow
`asic_synthesis/README.md` documents for TinyTPU -- DC W-2024.09, FreePDK45
`view-standard`, 3.33 ns, topographical, flatten effort 3 -- at two tops:

``tree``    the unit alone (`..._reduce_tree_0_Pipeline_VITIS_LOOP_*`), which
            is the unit-library claim. It pulls in one submodule, the flow
            control block, and nothing else.
``region``  the whole composed `dot_tree_<lanes>_<groups>`, which is what the
            TinyTPU rows in that README are and is therefore the only figure
            comparable with them.

Both are the SAME RTL, so the tree's own area appears twice -- once alone and
once as a line of the region's hierarchical area report -- and the two must
agree. They are a cross-check on the export, not two measurements.

The flow itself is vendored at `allo/backend/asic/`, which the generated
construct graph resolves from its own location (override with
`ALLO_ASIC_FLOW`). Running it needs zhang-21: DC W-2024.09 and mflowgen 0.8.0
live there and ace-01 has neither. Builds go in `/scratch` there, never on NFS.

    python reduce_asic.py --export            # RTL + manifests from a kept prj
    python reduce_asic.py --launch tree       # rsync and start DC remotely
    python reduce_asic.py --collect tree      # pull the reports back
"""

import json
import os
import re
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)

from examples.tinytpu.export_rtl import compile_order

ASIC = os.path.join(HERE, "asic_reduce")
REMOTE = "zhang-21.ece.cornell.edu"
REMOTE_RTL = "/scratch/users/sk3463/reduce_asic"
CLOCK_NS = 3.33

#: The tree's own pipeline module is named for the SOURCE LINE of the work
#: loop, which moves whenever the docstring above it does. Finding it by
#: pattern rather than by a written-down name is deliberate: a hard-coded
#: `VITIS_LOOP_272_1` becomes a top module that does not exist, and DC's
#: failure for a missing top looks like DC's failure for a broken design.
_TREE_TOP = re.compile(r"^module\s+(\w*reduce_tree_\d+_Pipeline_VITIS_LOOP_\w+)\s*\(",
                       re.M)


class ExportError(Exception):
    pass


def _modules(text):
    return set(re.findall(r"^\s*module\s+([A-Za-z_]\w*)", text, re.M))


def _needs(text, known):
    """Which known modules this file instantiates."""
    out = set()
    for name in known:
        if re.search(r"^\s*%s\s+(?:#\s*\([^;]*?\)\s*)?\w+\s*\(" % re.escape(name),
                     text, re.M):
            out.add(name)
    return out


def cone(files, top):
    """Every file in `top`'s instantiation cone, leaves first.

    A manifest built from "every file in the directory" would hand DC the
    whole region when the top is the tree, and DC links what it is given: the
    area would silently be the region's. The cone is what makes the tree-only
    number a number about the tree."""
    owner = {}
    for name, text in files.items():
        for mod in _modules(text):
            owner[mod] = name
    if top not in owner:
        raise ExportError(
            f"no file defines `module {top}`. This is what an unfinished "
            f"csynth looks like -- the file set is self-consistent without "
            f"it. Re-run the synthesis; do not hand-add a top.")
    order, seen = [], set()

    def visit(mod):
        fname = owner.get(mod)
        if fname is None or fname in seen:
            return
        seen.add(fname)
        for dep in sorted(_needs(files[fname], set(owner) - {mod})):
            visit(dep)
        order.append(fname)

    visit(top)
    return order


def export(prj, lanes, groups):
    """Copy the emitted RTL out of a kept csynth project and write one
    manifest per top."""
    src = os.path.join(prj, "out.prj", "solution1", "impl", "verilog")
    if not os.path.isdir(src):
        src = os.path.join(prj, "out.prj", "solution1", "syn", "verilog")
    if not os.path.isdir(src):
        raise ExportError(f"{prj} holds no emitted Verilog -- run "
                          f"reduce_csynth.py --keep first")
    dest = os.path.join(ASIC, f"reduce_{lanes}_{groups}")
    rtl = os.path.join(dest, "rtl")
    shutil.rmtree(dest, ignore_errors=True)
    os.makedirs(rtl)
    files = {}
    for name in sorted(os.listdir(src)):
        if not name.endswith((".v", ".sv")):
            continue
        with open(os.path.join(src, name), errors="replace") as fh:
            files[name] = fh.read()
        shutil.copyfile(os.path.join(src, name), os.path.join(rtl, name))
    if not files:
        raise ExportError(f"{src} contains no .v files")

    region_top = f"dot_tree_{lanes}_{groups}"
    tree_top = None
    for text in files.values():
        m = _TREE_TOP.search(text)
        if m:
            tree_top = m.group(1)
            break
    if tree_top is None:
        raise ExportError(
            "no module matched the tree's pipeline-module pattern; the work "
            "loop's emitted name has changed and the tree-only top would be "
            "a module that does not exist")

    tops = {"tree": tree_top, "region": region_top}
    meta = {"lanes": lanes, "groups": groups, "tops": tops,
            "clock_ns": CLOCK_NS, "files": len(files)}
    for which, top in tops.items():
        order = cone(files, top)
        path = os.path.join(dest, f"sv2v_manifest_{which}.f")
        with open(path, "w") as fh:
            fh.write(f"# Top module: {top}\n"
                     f"# {'The adder tree ALONE' if which == 'tree' else 'The whole composed region'}"
                     f" -- {len(order)} of {len(files)} emitted files, the\n"
                     f"# instantiation cone of the top and nothing else.\n"
                     f"# Dependency order, leaves first. GENERATED by\n"
                     f"# reduce_asic.py; do not edit.\n")
            # BARE filenames: the collector resolves each entry against
            # `design_path`, which already IS the rtl/ directory, so an
            # `rtl/` prefix here becomes `rtl/rtl/` and the collector stops.
            for n in order:
                fh.write(f"{n}\n")
        meta.setdefault("manifest_files", {})[which] = len(order)
    with open(os.path.join(dest, "MANIFEST.json"), "w") as fh:
        json.dump(meta, fh, indent=2, sort_keys=True)
        fh.write("\n")
    print(f"exported {len(files)} files to {dest}")
    for which, top in tops.items():
        print(f"  {which:7s} top={top}  "
              f"{meta['manifest_files'][which]} files in its cone")
    return dest, meta


CONSTRUCT = '''"""The adder tree through DC, at one top. Generated by reduce_asic.py."""

import os

from mflowgen.components import Graph, Node


def construct():
  graph = Graph()
  adk_name = 'freepdk-45nm'
  design_path = os.environ['REDUCE_RTL']
  top = os.environ['REDUCE_TOP']
  manifest = os.environ['REDUCE_MANIFEST']

  parameters = {
    'construct_path': __file__,
    'design_name': top,
    'top_module': top,
    'adk': adk_name,
    'adk_view': 'view-standard',
    'clock_period': %s,
    'clock_port': 'ap_clk',
    'topographical': True,
    'flatten_effort': 3,
    'design_path': os.path.join(design_path, 'rtl'),
    'manifest': os.path.join(design_path, manifest),
    'sv2v_include_dirs': '.',
    'normalize_rtl': False,
    'sram_mode': 'none',
  }

  # The vendored flow: allo/backend/asic/{nodes,adks} at the repository root,
  # four levels above examples/tinytpu/asic_reduce/<config>/. Set
  # ALLO_ASIC_FLOW when this file is run from a copy outside the checkout --
  # a /scratch build tree on the synthesis host, for instance.
  this_dir = os.path.dirname(os.path.abspath(__file__))
  repo = os.path.dirname(os.path.dirname(os.path.dirname(
      os.path.dirname(this_dir))))
  asic_dir = os.environ.get('ALLO_ASIC_FLOW',
                            os.path.join(repo, 'allo', 'backend', 'asic'))
  nodes_dir = os.path.join(asic_dir, 'nodes')
  if not os.path.isdir(nodes_dir):
    raise SystemExit(
      f'no node library at {nodes_dir}. Set ALLO_ASIC_FLOW to a checkout of it, '
      'or run allo/backend/asic/tools/preflight.py to see what is missing.')
  graph.sys_path.append(os.path.join(asic_dir, 'adks'))
  graph.set_adk(adk_name)
  adk = graph.get_adk_node()

  names = ['sv2v-design-collector', 'sram-collateral',
           'asic-flow-utilities', 'synopsys-dc-synthesis']
  nodes = {n: Node(os.path.join(nodes_dir, n)) for n in names}
  for node in nodes.values():
    graph.add_node(node)
  synth = nodes['synopsys-dc-synthesis']
  graph.connect_by_name(adk, synth)
  graph.connect_by_name(nodes['sram-collateral'], synth)
  graph.connect_by_name(nodes['sv2v-design-collector'], synth)
  graph.connect_by_name(nodes['asic-flow-utilities'], synth)
  graph.update_params(parameters)
  return graph
''' % CLOCK_NS


def main(argv):
    lanes, groups = 8, 2
    for a in argv:
        if ":" in a:
            lanes, groups = (int(x) for x in a.split(":"))
    if "--export" in argv:
        prj = os.path.join(HERE, f"reduce_{lanes}_{groups}.prj")
        dest, _ = export(prj, lanes, groups)
        with open(os.path.join(dest, "construct-reduce.py"), "w") as fh:
            fh.write(CONSTRUCT)
        return 0
    print(__doc__)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
