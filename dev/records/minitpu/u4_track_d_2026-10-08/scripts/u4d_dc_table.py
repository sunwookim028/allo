# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The record's DC table: each Allo unit's Catapult RTL beside MiniTPU's own
module, same DC flow, both clocks (from u4d_dc_summary.py's TSV).

    python3 u4d_dc_summary.py <dc> > dc.tsv; python3 u4d_dc_table.py dc.tsv"""
import sys

rows = {}
for line in open(sys.argv[1]).read().splitlines()[1:]:
    c = line.split("\t")
    if len(c) > 8:
        rows[c[0]] = c
PAIRS = [  # (label, catapult run stem, minitpu run stem or None)
    ("seq_issue:locked (I1+A1)", "c_issue", None),
    ("vpu_cmd:streams (D-23, depth 2)", "c_vpucmd_d2", None),
    ("vpu_wb:locked (W1)", "c_wb_locked", None),
    ("vpu_wb:units (W1 owns vreg.w)", "c_wb_units", None),
    ("vpu_wb:selftimed (D-24 probe)", "c_wb_selftimed", None),
    ("fetch:fq (F1 queue)", "c_fq", "m_fq"),
    ("fetch:iram (F1, IRAM 256 rows as registers)", "c_fpi256", None),
    ("fetch f1_d12 amended (IRAM 256 rows, D-12 server)", "c_fd12a", None),
    ("loop_ctrl (L1)", "c_loop", "m_loop"),
    ("seq_decoder (C1)", "c_dec", "m_dec"),
    ("agu_resolve (C1)", "c_agu", "m_agu"),
    ("vpu_adapter:slots (C1)", "c_vad", "m_vad"),
    ("scalar_agu S_LAT 1", "c_sagu1", "m_sagu1"),
    ("scalar_agu S_LAT 2 (shipped)", "c_sagu2", "m_sagu2"),
    ("scalar_agu S_LAT 3", "c_sagu3", "m_sagu3"),
    ("dma_addr_gen:bits", "c_dag_bits", "m_dag"),
    ("dma_addr_gen:c1", "c_dag_c1", "m_dag"),
    ("dma_desc_adapter (A1)", "c_desc", "m_desc"),
    ("dma:bits_reset core (D1)", "c_dmab", "m_dma"),
    ("sequencer (IRAM 2 rows) / Allo issue+fq+loop+sagu2", None, "m_sequencer_iram2"),
]


def cell(r):
    if r is None:
        return "--"
    return f"{float(r[2]):,.0f} ({r[5]}/{r[6]} reset); {float(r[7]):+.2f} ns; {r[8]} MHz"


def sumcell(stems, k):
    rs = [rows.get(f"{s}_{k}") for s in stems]
    if any(r is None for r in rs):
        return "--"
    a = sum(float(r[2]) for r in rs)
    g = sum(int(r[5]) for r in rs)
    gr = sum(int(r[6]) for r in rs)
    return f"{a:,.0f} ({g}/{gr} reset); worst {min(float(r[7]) for r in rs):+.2f} ns (sum of 4)"


print(".. list-table::\n   :header-rows: 1\n   :widths: 26 10 32 32\n")
print("   * - unit / variant\n     - clock\n     - Allo: Catapult RTL -> DC (area um2 (regs/reset); slack; Fmax)\n"
      "     - MiniTPU module, same DC flow")
for label, c, m in PAIRS:
    for k, p in (("3p33", "3.33"), ("2p0", "2.0")):
        if c is None:
            ac = sumcell(["c_issue", "c_fq", "c_loop", "c_sagu2"], k)
        else:
            ac = cell(rows.get(f"{c}_{k}"))
            if c == "c_dmab" and k == "2p0" and ac != "--":
                ac += " -- the 3.33 ns RTL (Catapult refused II=1 at 2.0 ns)"
        mc = cell(rows.get(f"{m}_{k}")) if m else "--"
        if ac == "--" and mc == "--":
            continue
        print(f"   * - {label}\n     - {p}\n     - {ac}\n     - {mc}")
