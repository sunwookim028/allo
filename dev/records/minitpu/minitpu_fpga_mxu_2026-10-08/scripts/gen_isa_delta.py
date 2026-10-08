#!/usr/bin/env python3
# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Write the ``versions.list.<name>`` entry an Allo-built MXU implies, for
MiniTPU's ISA file (``isa/latency.json`` on minitpu-comp master >= f1e978e;
``docs/isa_latency.json`` at the ``b3ba0a4d`` pin), from the RTL measurement.

    gen_isa_delta.py --base <latency.json> --measured <measured.json> --name allo-mxu-vitis \\
        [--manifest <vitis_manifest.json>] [--allo-commit C] [--minitpu-pin P] \\
        [--bitstream 0xB1xxxxxx=<description>] [--write <out.json> | --check <versions.json>]

Adapted from ``minitpu_rtl_mxu_2026-10-08/scripts/gen_isa_delta.py`` (the
seam prototype, ``u1_matrix.rst`` "Agreed seam with minitpu-comp"); three
changes, each from this record:

1. **Latencies by difference, not by absolute mapping.** The prototype set
   ``result_latency.vmatpush = push_to_valid + rows - 1``, which is 84 for
   ``mxu.sv`` on the measuring tb (push->valid 81 at DIM 16) against the
   published 85: the tb counts from one edge later than the ISA's issue
   cycle. Measuring ``mxu.sv`` on the SAME tb at the SAME DIM and adding the
   difference to the base cancels the convention:
   ``result_latency = base + (P_allo - P_mxu.sv)`` and
   ``issue_interval.vmatpop = base + (Q_allo - Q_mxu.sv)`` (``Q`` is
   pop -> next valid; the wrapper's pop mask makes it the lag plus one).
2. **Overrides that change something.** A quantity equal to the base is not
   a delta; it is listed under ``unchanged`` with its evidence. master's
   ``compiler/isa/versions.py`` refuses a delta that "moves nothing", so the
   prototype's ``WB_W_MPOP_LAST = 3`` entry would not resolve there.
3. **``bitstreams``.** master's schema maps board ``build_id`` -> description
   per version (``run_image.py`` refuses an image on another version's
   bitstream); ``--bitstream`` fills it.

``measured.json``::

    {"dim": 16, "push_to_valid": P, "pop_to_next_valid": Q,
     "ref": {"push_to_valid": 81, "pop_to_next_valid": 0, "rtl": "mxu.sv b3ba0a4d"},
     "pop_beats": 1, "switch_span": null | S, "output_fifo_depth": 64,
     "tokens_per_cycle": 1.0, "token_lag": L, "acc_lag": A, "tb": "...", "rtl": "..."}

A rate below one token per cycle voids every latency (they go to
``unresolved``), as in the prototype. A DIM other than the base profile's
``num_lanes`` puts every latency in ``unresolved`` (nothing extrapolates a
lag). ``--check`` re-derives and compares ``what``/``set`` with an existing
block: ``DELTA-MATCH`` or ``DELTA-MISMATCH`` (exit 1); with ``--versions-py``
(master's ``compiler/isa/versions.py``) it also inserts the block into the
base and resolves it with that reader (``RESOLVES``: each moved path with its
base and new value; ``bitstreams()`` must accept the build ids).
"""
import argparse, json, sys


def get(d, path):
    cur = d
    for k in path.split("."):
        if not isinstance(cur, dict) or k not in cur:
            raise KeyError(path)
        cur = cur[k]
    return cur


def entry(base, man, meas, allo_commit, pin, bitstreams):
    lanes = base["profile"]["num_lanes"]
    tool = f"{man.get('tool')} {man.get('clock_period_ns')} ns" if man else "?"
    src = f"allo {allo_commit}, {tool}, {meas.get('tb')} on {meas.get('rtl')}, MiniTPU pin {pin}"
    deltas, unchanged, unresolved = [], [], []
    if man:
        bad = [k for k, u in man.get("units", {}).items() if u.get("ii") != 1]
        if bad:
            unresolved.append({"what": "kernels not at II 1 (no stationary token rate)", "units": bad})

    def put(what, path, value, note):
        b = get(base, path)  # overrides only
        if b == value:
            unchanged.append({"what": what, "path": path, "value": value, "evidence": note})
        else:
            deltas.append({"what": what, "set": {path: value}, "source": f"{src}; {note}"})

    dim, rate, ref = meas["dim"], meas.get("tokens_per_cycle"), meas.get("ref", {})
    lat_ok = (rate is None or rate >= 1.0) and dim == lanes
    if rate is not None and rate < 1.0:
        unresolved.append({"what": "every latency quantity", "tokens_per_cycle": rate,
                           "why": "the core runs below one token per cycle: its lag grows, nothing is stationary"})
    elif dim != lanes:
        unresolved.append({"what": "every latency quantity", "measured_at_dim": dim,
                           "why": f"measured at DIM {dim}, the profile is {lanes} lanes; nothing extrapolates a lag"})
    if lat_ok:
        p, pr = meas["push_to_valid"], ref["push_to_valid"]
        put("vmatpush result latency: base + (Allo MXU push->valid - mxu.sv push->valid), same tb, same DIM",
            "matrix.result_latency.vmatpush", get(base, "matrix.result_latency.vmatpush") + (p - pr),
            f"push->valid {p} vs mxu.sv {pr} at DIM {dim}")
        q, qr = meas["pop_to_next_valid"], ref["pop_to_next_valid"]
        put("vmatpop issue interval: base + (pop->next valid - mxu.sv's); the wrapper masks output_valid until "
            "the post-pop token arrives (token lag + 1)",
            "matrix.issue_interval.vmatpop", get(base, "matrix.issue_interval.vmatpop") + (q - qr),
            f"pop->next valid {q} vs mxu.sv {qr} at DIM {dim}, token lag {meas.get('token_lag')}")
        if meas.get("switch_span") is not None:
            put("weight-switch span", "matrix.weight_switch.span", meas["switch_span"], meas.get("switch_span_note", ""))
        else:
            unresolved.append({"what": "matrix.weight_switch.span", "why": meas.get("switch_span_note", "not measured")})
    if meas.get("pop_beats") is not None:
        first = get(base, "rtl_params.WB_W_MPOP_FIRST")
        put("vmatpop writeback beats: WB_W_MPOP_LAST = WB_W_MPOP_FIRST + beats - 1", "rtl_params.WB_W_MPOP_LAST",
            first + meas["pop_beats"] - 1, f"one output token is a whole VREG ({meas['pop_beats']} beat)")
    if meas.get("output_fifo_depth") is not None:
        put("per-lane output FIFO depth in result rows", "resources.mxu_output_fifo.depth", meas["output_fifo_depth"],
            "the back's lane rings: ENTRIES 16 x NUM_SUBLANES 4")
    out = {"status": "experimental",
           "describes": meas.get("describes", "the Allo MXU (mxu_wide) through Vitis HLS in place of mxu.sv"),
           "deltas": deltas, "unchanged": unchanged, "unresolved": unresolved,
           "note": "generated by gen_isa_delta.py (allo dev/records/minitpu/minitpu_fpga_mxu_2026-10-08); overrides only"}
    if bitstreams:
        out["bitstreams"] = bitstreams
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True); ap.add_argument("--measured", required=True)
    ap.add_argument("--manifest"); ap.add_argument("--name", default="allo-mxu-vitis")
    ap.add_argument("--allo-commit", default="?"); ap.add_argument("--minitpu-pin", default="b3ba0a4d")
    ap.add_argument("--bitstream", action="append", default=[])
    ap.add_argument("--write"); ap.add_argument("--check")
    ap.add_argument("--versions-py", help="minitpu-comp master's compiler/isa/versions.py, to resolve the entry with")
    a = ap.parse_args()
    base = json.load(open(a.base)); meas = json.load(open(a.measured))
    man = json.load(open(a.manifest)) if a.manifest else None
    bits = dict(b.split("=", 1) for b in a.bitstream)
    block = {"versions": {"list": {a.name: entry(base, man, meas, a.allo_commit, a.minitpu_pin, bits)}}}
    if a.check:
        have = json.load(open(a.check))
        strip = lambda ds: [{"what": d["what"], "set": d["set"]} for d in ds]
        want = strip(block["versions"]["list"][a.name]["deltas"])
        got = strip(get(have, f"versions.list.{a.name}.deltas"))
        if want != got:
            print("DELTA-MISMATCH", json.dumps(want), "vs", json.dumps(got))
            return 1
        print(f"DELTA-MATCH versions.list.{a.name}: {len(got)} entries")
        if a.versions_py:
            import copy, importlib.util
            spec = importlib.util.spec_from_file_location("mtpu_versions", a.versions_py)
            vm = importlib.util.module_from_spec(spec); spec.loader.exec_module(vm)
            data = copy.deepcopy(base)
            data["versions"]["list"][a.name] = get(have, f"versions.list.{a.name}")
            for what, moved, _ in vm.deltas(data, a.name):
                print(f"RESOLVES {a.name}: {what[:60]}... " + ", ".join(f"{p} {b} -> {n}" for p, (b, n) in moved.items()))
            res = vm.resolve(data, a.name)
            ids = {hex(k): v for k, v in vm.bitstreams(data).items() if v == a.name}
            print(f"RESOLVES {a.name}: resolve() ok (version={res['version']}); bitstreams {ids or 'none'}")
        return 0
    text = json.dumps(block, indent=1)
    if a.write:
        open(a.write, "w").write(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
