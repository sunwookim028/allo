# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""TinyTPU-isa as an INSTANCE of the template (README D-4, the
"TinyTPU as an instance" track).

The frozen design is ``examples/tinytpu/ip/`` (D-4: read-only, the
regression reference). This package composes the same machine from the
template's mechanisms and holds it to TinyTPU's own gates:

``geometry.py``     ``TinyTpuGeometry``: the parameter set as a D-20 record.
                    ``VW``/``AW`` DERIVE from the bound engine's widths, the
                    memory sizes from ``T`` and ``MAXDIM`` (``ip/params.py``'s
                    rules, unchanged), and the accumulator-distance relation
                    ``AR_RAW_DIST <= T`` is a legality here, not an assert in
                    the assembler alone.
``pe.py``           the processing element in the template's form: the MAC is
                    an engine slot (``MAC_IN``, ``MAC_ACC``, ``MAC_MUL``,
                    ``MAC_ADD``, ``MAC_PACK``, D-15) bound at composition to
                    ``engines.INT8_INT32``; instanced ``(T, T)`` (D-17).
``accumulator.py``  the accumulator file as an ``Option`` (D-19): ``accu`` and
                    ``dma_st``, their channels, the result matrix, and the
                    five ISA slots that exist only with it.
``instance.py``     the wiring: ``base(geometry)`` + ``ACCUMULATOR`` ->
                    ``architecture()``; ``TinyTpuInstance`` (region, schedule,
                    assembler, programs, as ``ip.tinytpu.TinyTPU``);
                    ``compare_isa`` (``isa_slots`` against ``isa_spec.json``).
``glue/``           a copy of the harness glue (``microarch_isa.py``) that
                    builds the instance under ``TPU_INSTANCE=template``, and
                    ``run_gates.py``, which mounts it in front of
                    ``examples.tinytpu`` so the frozen gates run unedited.

What is REUSED from the frozen design by import: the six units
``sequencer``, ``dma_ld``, ``spm``, ``vru``, ``wld``, ``accu``, ``dma_st``
(the template has no counterpart yet), the ISA layout (``ip/isa.py``), the
assembler and the reference programs. What is RE-EXPRESSED: the PE (engine
slots), the parameter set (a geometry record), the accumulator (an option),
the wiring (base + option).
"""
