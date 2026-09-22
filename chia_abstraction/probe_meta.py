# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Probe metadata, importable with NO dependencies. FROZEN.

`probes.py` imports `numpy` and `allo`, because it builds and runs designs.
The MCP tool surface runs in the CHIA environment, which has neither -- so
`list_probes` and `declare_probe` raised `ModuleNotFoundError: numpy`, the
agent could not declare a call, the probe rung was skipped, and a new
abstraction could never reach `expressive`.

That is the no-caller lesson happening to the harness itself: the rung was
proved end to end from the `allo` environment and never exercised through the
path the AGENT uses. So this module holds everything the tool surface needs and
imports nothing; `probes.py` imports it for the same constants, so the two
cannot drift.
"""

from __future__ import annotations

#: probe -> what the agent is told about it. `right_answer` is the outcome a
#: declaration of ONE write port on `buffer` must produce.
PROBES = {
    "ports_dual": {
        "design": (
            "def ports_kernel(A: int32[16], B: int32[16]):\n"
            "    buf: int32[16] = 0\n"
            "    for i in range(8):\n"
            "        buf[2 * i] = A[2 * i] + 1\n"
            "        buf[2 * i + 1] = A[2 * i + 1] + 1\n"
            "    for j in range(16):\n"
            "        B[j] = buf[j]\n"),
        "frozen_schedule": 's.pipeline("i")',
        "buffer": "buf",
        "measured_rtl": {
            "modules_for_buf": 1,
            "module_name": "ports_kernel_buf_RAM_AUTO_1R1W.v",
            "write_statements_per_instance": 2,
            "read_statements_per_instance": 1,
            "instances": 1,
            "note": "Vitis names it 1R1W and writes the array from two "
                    "always-blocks: a true dual-write-port RAM."},
        "right_answer": "refused",
    },
    "ports_banked": {
        "design": "the same kernel as ports_dual",
        "frozen_schedule": 's.pipeline("i"); s.partition(s.buf, '
                           "partition_type=2, dim=1, factor=2)   # 2 = Cyclic",
        "buffer": "buf",
        "measured_rtl": {
            "modules_for_buf": 1,
            "module_name": "ports_kernel_buf_RAM_AUTO_1R1W.v",
            "write_statements_per_instance": 1,
            "read_statements_per_instance": 1,
            "instances": 2,
            "note": "cyclic banking by 2 gives each bank exactly one writer."},
        "right_answer": "accepted",
    },
}

#: What the agent must produce for a site, and what is refused.
CALL_CONTRACT = (
    "Exactly one call expression `s.<method>(<literals>)`, where <method> is a "
    "Schedule method YOUR PATCH ADDS and every argument is a literal (str, "
    "int, float, bool, None, or a tuple/list of those). The harness applies it "
    "to the frozen design at the frozen schedule above, then emits. The "
    "outcome is `accepted`, `refused` (the call or the emission raised) or "
    "`crashed` (an MLIR assertion or abort -- that REJECTS the candidate: a "
    "crash is a defect, not a legality rule). An accepted `ports_banked` is "
    "then run through csim and must be bit-exact."
)


def describe() -> dict:
    return {name: {**info, "call_contract": CALL_CONTRACT}
            for name, info in PROBES.items()}
