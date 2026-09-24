# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cosim the program the FROZEN mapper chose for this hardware. FROZEN.

`cosim.py` is byte-identical to main and stays that way -- it is the testbench
generator, the numpy golden reference, `SHAPES` and every Vitis/TCL setting, and
nothing in the co-design loop may move it. What this file does is supply it a
different *program*: main's `cosim.py` builds each testbench's imem from
`isa_dsl.gemm_program`, and here that name is bound, in frozen code, to

    gemm_from_nest(mapspace.select(M, K, N)[0], M, K, N, relu)

so the stream under the RTL is the best nest the candidate's own hardware can
encode, picked exhaustively by the rule in `mapspace.py`. Nothing else about
the measurement changes: same testbench generator, same [-4, 4] operands from
seed 0, same golden reference computed by `cosim.py` itself with numpy, same
`mismatches = 0` requirement per shape, same csynth target.

Why the mapper's program and not the candidate's `gemm_program`: the agent
proposes hardware, and enumeration -- not the agent -- answers what that
hardware can run. If the scored program were the one the agent hand-wrote, a
hardware change would be scored together with the agent's guess at how to use
it, and "the best mapping for this hardware" would be a sample rather than a
claim. `gemm_program` is still what `bench_isa.py` and `stress_isa.py` verify,
and `codesign_gate.py` requires the two generators to agree on the canonical
nest, so the mapper's emitter is not a second, unverified code path.

    python gate_runner.py codesign_cosim
"""

from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "..")))

import mapspace  # noqa: E402
from examples.accelerator.tinytpu_vitis import cosim  # noqa: E402
from examples.accelerator.tinytpu_vitis.isa_dsl import gemm_from_nest  # noqa: E402

#: shape -> the nest the mapper chose. Resolved once, before synthesis, so the
#: same nest is used for every testbench built against the one RTL build -- and
#: so a shape the hardware can encode nothing for fails here rather than
#: half-way through a Vitis run.
CHOSEN = {}
for _shape in cosim.SHAPES:
    _nest, _found = mapspace.select(*_shape)
    CHOSEN[tuple(_shape)] = _nest
    print(f"MAPSPACE {_shape[0]}x{_shape[1]}x{_shape[2]}: CHOSEN "
          f"{mapspace.describe(_nest)} of {_found['encodable']}/"
          f"{_found['total']} encodable", flush=True)


def mapped_gemm_program(M, K, N, relu=False):
    """What `cosim.py` will assemble into each testbench's instruction memory."""
    nest = CHOSEN.get((M, K, N))
    if nest is None:                # a shape cosim.py asks for and we did not
        nest, _ = mapspace.select(M, K, N)
        CHOSEN[(M, K, N)] = nest
    return gemm_from_nest(nest, M, K, N, relu)


# The one rebinding, in frozen code, of a frozen module's program source.
cosim.gemm_program = mapped_gemm_program

sys.exit(cosim.main())
