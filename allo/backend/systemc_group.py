# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""A synthesis top made of several kernels of one SystemC region.

``synth_top`` names ONE kernel module (``catapult.py``: the region's own top
also synthesizes the testbench kernels). A memory with declared ports (README
D-12) is lowered into several kernels -- the owners of its ports, and a
server -- whose union is the hardware unit to measure. ``synth_group``
(``configs={"synth_group": {"name": "rf_grp", "kernels": ["rd_a_0", ...]}}``)
adds one more ``SC_MODULE`` to the emitted file holding those instances and the
signals between them; every signal that also reaches a kernel outside the
group becomes a port of it, named as the region's signal. The region's own
``top`` is untouched, so csim runs exactly as before, and ``run.tcl`` names
the group as ``DESIGN_HIERARCHY``.

Only ``sc_signal`` links (``Wire``) may cross the group boundary: a
Connections channel crossing it is refused, naming the signal.
"""

import re

_INST = re.compile(r"^  (\w+) (u\d+);$")
_BIND = re.compile(r"^    (u\d+)\.(\w+)\((\w+)\);$")
_SIG = re.compile(r"^  sc_signal<(.+)> (\w+);$")
_PORT = re.compile(r"^  (sc_in|sc_out)<(.+)> (\w+);")


def _block(code, module):
    start = code.index(f"SC_MODULE({module}) {{")
    end = code.index("\n};", start)
    return start, end, code[start:end]


def add_synth_group(code, name, kernels, top="top"):
    """Return ``code`` with ``SC_MODULE(<name>)`` added before ``top``."""
    _, _, top_blk = _block(code, top)
    insts, binds, sigs = {}, [], {}
    for line in top_blk.splitlines():
        if (m := _INST.match(line)) and m.group(1) not in ("sc_in_clk",):
            insts[m.group(2)] = m.group(1)
        elif m := _BIND.match(line):
            binds.append(m.groups())
        elif m := _SIG.match(line):
            sigs[m.group(2)] = m.group(1).strip()
    by_type = {t: i for i, t in insts.items()}
    missing = [k for k in kernels if k not in by_type]
    assert not missing, (
        f"synth_group {name}: no kernel module {missing} in the region "
        f"(it has {sorted(by_type)})"
    )
    members = {by_type[k] for k in kernels}
    dirs = {}  # (instance, port) -> sc_in / sc_out
    for inst in members:
        _, _, blk = _block(code, insts[inst])
        for line in blk.splitlines():
            if m := _PORT.match(line):
                dirs[(inst, m.group(3))] = m.group(1)
    inside, outside = {}, set()
    for inst, port, sig in binds:
        if port in {"clk", "rst", "done"}:
            continue
        if inst in members:
            inside.setdefault(sig, []).append((inst, port))
        else:
            outside.add(sig)
    decls = []
    # boundary ports in the order the kernels outside bind them (a driver's
    # port order, then a sink's), internal signals after
    first = {}
    for inst, port, sig in binds:
        if inst not in members:
            first.setdefault(sig, len(first))
    ordered = sorted(inside, key=lambda s: (s not in outside, first.get(s, 0)))
    for sig in ordered:
        uses = inside[sig]
        assert sig in sigs, (
            f"synth_group {name}: `{sig}` crosses into the group and is not an "
            f"sc_signal (a Connections channel); only Wire links may cross"
        )
        if sig in outside:
            kinds = {dirs.get(u) for u in uses}
            d = "sc_out" if "sc_out" in kinds else "sc_in"
            decls.append(f"  {d}< {sigs[sig]} > {sig};")
        else:
            decls.append(f"  sc_signal< {sigs[sig]} > {sig};")
    order = [i for i in insts if i in members]
    lines = [
        f"SC_MODULE({name}) {{",
        "  sc_in_clk clk;",
        "  sc_in<bool> rst;",
        "  sc_out<bool> done;",
    ] + decls
    lines += [f"  {insts[i]} {i};" for i in order]
    lines += [f"  sc_signal<bool> {i}_done;" for i in order]
    inits = ", ".join(f'{i}("{i}")' for i in order)
    lines.append(f"  SC_CTOR({name}) : {inits} {{")
    for i in order:
        lines += [
            f"    {i}.clk(clk);",
            f"    {i}.rst(rst);",
            f"    {i}.done({i}_done);",
        ]
        lines += [
            f"    {i}.{p}({s});"
            for ii, p, s in binds
            if ii == i and p not in ("clk", "rst", "done")
        ]
    lines.append(
        "    SC_METHOD(_agg_done); sensitive << "
        + " << ".join(f"{i}_done" for i in order)
        + ";"
    )
    lines.append("  }")
    lines.append(
        "  void _agg_done() { done.write("
        + " && ".join(f"{i}_done.read()" for i in order)
        + "); }"
    )
    lines.append("};\n\n")
    at = code.index(f"SC_MODULE({top}) {{")
    return code[:at] + "\n".join(lines) + code[at:]
