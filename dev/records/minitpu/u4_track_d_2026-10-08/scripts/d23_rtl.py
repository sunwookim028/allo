# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""D-23 on Catapult RTL: ``d23_rate.py``'s three numbers from ``u4d_check``'s stamps.

    $ALLO_PYTHON d23_rtl.py <prj>...     (after u4d_check.py vpu_cmd <variant> <prj> --token-outs 16 17 18)

Per project: the issue kernel's row accepts (stall cycles: rows not accepted one
cycle after the previous), the issue cycle of every issued bundle against the
RTL's (``bundle_issued_o`` row ``t`` leaves at ``t + c`` for one ``c``), and per
slot the command token's put -> receiver-output latency (the k-th real token
of slot S belongs to the k-th row whose S valid is set; the pad tokens come
after the trace, ``vpu_cmd``'s harness loop).
"""
import os
import sys

import numpy as np

SLOTS = (("V", "v_valid_o", 0), ("X", "x_valid_o", 1), ("M", "m_valid_o", 2))


def hist(x):
    v, c = np.unique(np.asarray(x), return_counts=True)
    return dict(zip(v.tolist(), c.tolist()))


for prj in sys.argv[1:]:
    z = np.load(os.path.join(prj, "u4d_stamps.npz"))
    outs = [str(x) for x in z["outs"]]
    acc = z["acc"]
    n = len(acc)
    stall = int(acc[-1] - acc[0] - (n - 1))
    iss = z["got_bundle_issued_o"]
    t_iss = np.flatnonzero(iss)
    cyc_iss = z[f"out_{outs[0]}"]  # ISS is the issue kernel's first output (seq_issue._io order)
    d_ = cyc_iss[t_iss] - t_iss
    c0 = int(cyc_iss[0])  # the first row's own offset: the no-stall edge
    on = int((d_ == c0).sum())
    off = f"{on}/{len(t_iss)} at +{c0} (the RTL's cycle), latest +{int(d_.max())}"
    if len(set(d_.tolist())) <= 4:
        off += f" {hist(d_)}"
    print(f"D23-RTL {os.path.basename(prj.rstrip('/'))}: rows {n}, stall cycles {stall}, rows at interval 1 "
          f"{int((np.diff(acc) == 1).sum())}/{n - 1}; issues {len(t_iss)}, issue edge - RTL cycle: {off}")
    rx = outs[-3:]
    for (s, vname, j) in SLOTS:
        rows = np.flatnonzero(z[f"got_{vname}"])
        c = z[f"out_{rx[j]}"][: len(rows)]
        lat = c - acc[rows]
        gap = np.diff(c)
        print(f"    {s}: {len(rows)} commands; row accept -> receiver token {hist(lat)}; "
              f"receiver token intervals min {int(gap.min()) if len(gap) else '-'}")
