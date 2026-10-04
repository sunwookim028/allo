# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""U3 track E prototypes: the template's composition mechanisms, outside
``allo/`` (README Design target; ``dev/records/minitpu/u3_composition_design_2026-10-04.rst``).

``engines.py``      the MAC plug-in: an ``Engine`` record (types, latency,
                    accumulate order, bodies, numpy references, directives)
``instantiate.py``  per-instance binding of a ``compose.Unit`` (names, channels,
                    parameters) by AST rename -- the C9 workaround at compose level
``mac_pe.py``       one PE source, two MAC engines; each in its own region and
                    both in one
``matrix_engine.py`` systolic chain and adder tree behind one interface that
                    declares its accumulate order; checked against the contract
                    reference evaluated with that order
``optional.py``     optional modules as deltas over an architecture, with the
                    ISA slots they bring
``legality.py``     derived-parameter legality for the MXU and the tree, held
                    to the Phase 0 measurements
``run_u3e.py``      the gate: one line per check
"""
