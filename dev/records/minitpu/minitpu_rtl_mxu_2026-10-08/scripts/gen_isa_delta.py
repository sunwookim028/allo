#!/usr/bin/env python3
# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prototype of the agreed manifest -> ``versions.list.<name>.deltas`` seam
(``u1_matrix.rst``, "Agreed seam with minitpu-comp, 2026-10-08"; README D-16,
D-21): write the ISA-version deltas an Allo-built MXU implies, from the
Catapult manifest (``latency.json``, provenance) and the RTL measurement
(``measured.json``, the numbers -- a manifest reports kernels, not the MXU's
contract quantities, so the contract numbers are measured on the RTL by
``tb_mxu_single_port_lat`` and the manifest is cited as provenance).

    gen_isa_delta.py --base <isa_latency.json> --manifest <latency.json> \\
        --measured <measured.json> --name allo-mxu [--allo-commit C] [--minitpu-pin P] \\
        [--write <out.json> | --check <versions.json>]

``measured.json``::

    {"dim": 4, "push_to_valid": 25, "pop_interval": 7, "pop_beats": 1,
     "switch_span": null, "output_fifo_depth": 64, "tokens_per_cycle": 1.0,
     "tb": "tb_mxu_single_port_lat", "rtl": "<path or hash>"}

``tokens_per_cycle`` below 1 (the wrapper's lag grows) voids every latency:
they go to ``unresolved`` with the early values, and only the capacities are
emitted.

Mapping (the seam): MXU ``push_to_valid`` (last push -> output_valid at the
``mxu`` boundary) -> ``matrix.result_latency.vmatpush`` **plus the stream
rows minus one** (85 = 4 + 82 - 1 at b3ba0a4d: the vmatpush's four rows,
counted from issue); ``pop_interval`` -> ``matrix.issue_interval.vmatpop``;
``pop_beats`` -> ``rtl_params.WB_W_MPOP_LAST`` = ``WB_W_MPOP_FIRST`` + beats
- 1; ``switch_span`` -> ``matrix.weight_switch.span``; the output FIFO ->
``resources.mxu_output_fifo.depth``. Overrides only: a dotted path the base
lacks is refused. A quantity measured at a DIM other than the base profile's
``num_lanes`` is NOT a delta for that profile: it goes to ``unresolved`` with
the measured value and the DIM, because the number is the MXU's at that
geometry and nothing extrapolates it. ``--check`` re-derives and compares
with an existing ``versions.list.<name>`` block, exit 1 on disagreement.
"""
import argparse, json, sys


def get(d, path):
    cur = d
    for k in path.split("."):
        if not isinstance(cur, dict) or k not in cur:
            raise KeyError(path)
        cur = cur[k]
    return cur


def deltas(base, man, meas, name, allo_commit, pin):
    lanes = base["profile"]["num_lanes"]
    sub = base["profile"]["num_sublanes"]
    src = (f"allo {allo_commit}, {man.get('tool')} {man.get('clock_period_ns')} ns, "
           f"{meas.get('tb')} on {meas.get('rtl')}, MiniTPU pin {pin}")
    out, unresolved = [], []
    units = man.get("units", {})
    bad = [k for k, u in units.items() if u.get("status") != "scheduled"]
    if bad:
        unresolved.append({"what": "manifest entries not 'scheduled' (D-10: not consumed)", "units": bad})

    def put(what, path, value, note=""):
        get(base, path)  # overrides only
        out.append({"what": what, "set": {path: value}, "source": src + (f"; {note}" if note else "")})

    dim = meas.get("dim")
    same_geom = dim == lanes
    rate = meas.get("tokens_per_cycle")
    if rate is not None and rate < 1.0:
        # a core that consumes fewer tokens than cycles has a lag that GROWS: no latency measured on it is a
        # stationary number, so nothing goes into the deltas but the capacities
        unresolved.append({"what": "every latency quantity", "tokens_per_cycle": rate,
                           "why": "the core runs below one token per cycle (lockstep violation): push->valid and the pop "
                                  "interval grow with time since reset; measured early values are in this file for the record",
                           "early_values": {k: meas.get(k) for k in ("push_to_valid", "pop_interval", "switch_span")}})
        meas = dict(meas, push_to_valid=None, pop_interval=None, switch_span=None)
    rows = sub  # a vmatpush streams NUM_SUBLANES rows
    if meas.get("push_to_valid") is not None:
        v = meas["push_to_valid"] + rows - 1
        if same_geom:
            put("vmatpush result latency: MXU push->valid + stream rows - 1", "matrix.result_latency.vmatpush", v,
                f"mxu push->valid {meas['push_to_valid']} at DIM {dim}")
        else:
            unresolved.append({"what": "matrix.result_latency.vmatpush", "measured_at_dim": dim,
                               "mxu_push_to_valid": meas["push_to_valid"], "would_be": v,
                               "why": f"measured at DIM {dim}, profile is {lanes} lanes; nothing extrapolates a token lag"})
    if meas.get("pop_interval") is not None:
        if same_geom:
            put("vmatpop issue interval: pop -> next group valid", "matrix.issue_interval.vmatpop", meas["pop_interval"])
        else:
            unresolved.append({"what": "matrix.issue_interval.vmatpop", "measured_at_dim": dim, "value": meas["pop_interval"]})
    if meas.get("pop_beats") is not None:
        first = get(base, "rtl_params.WB_W_MPOP_FIRST")
        put("vmatpop writeback beats: WB_W_MPOP_LAST = WB_W_MPOP_FIRST + beats - 1", "rtl_params.WB_W_MPOP_LAST",
            first + meas["pop_beats"] - 1)
    if meas.get("switch_span") is not None:
        if same_geom:
            put("weight-switch span", "matrix.weight_switch.span", meas["switch_span"])
        else:
            unresolved.append({"what": "matrix.weight_switch.span", "measured_at_dim": dim, "value": meas["switch_span"]})
    else:
        unresolved.append({"what": "matrix.weight_switch.span", "why": "not measured (needs tb_matrix_weight_pipelining's scenario at the mxu boundary)"})
    if meas.get("output_fifo_depth") is not None:
        put("per-lane output FIFO depth in result rows", "resources.mxu_output_fifo.depth", meas["output_fifo_depth"])
    return {"deltas": out, "unresolved": unresolved,
            "note": "generated by gen_isa_delta.py (allo dev/records/minitpu/minitpu_rtl_mxu_2026-10-08); overrides only"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True); ap.add_argument("--manifest", required=True)
    ap.add_argument("--measured", required=True); ap.add_argument("--name", default="allo-mxu")
    ap.add_argument("--allo-commit", default="?"); ap.add_argument("--minitpu-pin", default="b3ba0a4d")
    ap.add_argument("--write"); ap.add_argument("--check")
    a = ap.parse_args()
    base = json.load(open(a.base)); man = json.load(open(a.manifest)); meas = json.load(open(a.measured))
    block = {"versions": {"list": {a.name: deltas(base, man, meas, a.name, a.allo_commit, a.minitpu_pin)}}}
    if a.check:
        have = json.load(open(a.check))
        want = block["versions"]["list"][a.name]["deltas"]
        got = get(have, f"versions.list.{a.name}.deltas")
        strip = lambda ds: [{"what": d["what"], "set": d["set"]} for d in ds]
        if strip(want) != strip(got):
            print("DELTA-MISMATCH", json.dumps(strip(want)), "vs", json.dumps(strip(got)))
            return 1
        print(f"DELTA-MATCH versions.list.{a.name}: {len(got)} entries")
        return 0
    text = json.dumps(block, indent=1)
    if a.write:
        open(a.write, "w").write(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
