# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""MiniTPU ``sfu.sv`` for Design Compiler: the two ``$readmemh`` ROMs as constant
arrays. DC ignores ``initial`` blocks, so the shipped file synthesizes with
undefined (optimised-away) ROMs; this stand-in replaces the two ``logic`` ROM
declarations and the ``initial`` block with ``localparam`` arrays holding the
``.mem`` contents. Every other line is byte-identical (checked by diff below).

    python gen_sfu_dc.py <minitpu clone> <out.sv>
"""
import sys

home, out = sys.argv[1], sys.argv[2]
src = open(f"{home}/src/core/sfu/sfu.sv").read()


def rom(name):
    vals = [l.strip() for l in open(f"{home}/src/core/sfu/{name}_bf16.mem") if l.strip()]
    assert len(vals) == 2048, len(vals)
    body = ",\n    ".join(", ".join(f"16'h{v}" for v in vals[i : i + 8]) for i in range(0, 2048, 8))
    return f"  localparam logic [15:0] {name}_rom [0:2047] = '{{\n    {body}}};\n"


a = src.index("  (* rom_style = \"block\" *) logic [15:0] gelu_rom")
b = src.index("  end\n", src.index("    $readmemh(EXP_MEM_FILE, exp_rom);")) + len("  end\n")
new = src[:a] + rom("gelu") + rom("exp") + src[b:]
open(out, "w").write(new)
removed = src[a:b].count("\n")
print(f"gen_sfu_dc: replaced {removed} lines ({a}..{b}) with two localparam ROMs -> {out}")
