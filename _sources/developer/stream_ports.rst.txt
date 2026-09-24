..  Copyright Allo authors. All Rights Reserved.
    SPDX-License-Identifier: Apache-2.0

..  Licensed to the Apache Software Foundation (ASF) under one
    or more contributor license agreements.  See the NOTICE file
    distributed with this work for additional information
    regarding copyright ownership.  The ASF licenses this file
    to you under the Apache License, Version 2.0 (the
    "License"); you may not use this file except in compliance
    with the License.  You may obtain a copy of the License at

..    http://www.apache.org/licenses/LICENSE-2.0

..  Unless required by applicable law or agreed to in writing,
    software distributed under the License is distributed on an
    "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
    KIND, either express or implied.  See the License for the
    specific language governing permissions and limitations
    under the License.

.. _stream-ports:

##########################################
Stream Ports: a Unit with an Interface
##########################################

``@df.unit`` in ``allo/dataflow.py``, the rules in ``allo/netlist.py``, the
instantiation in ``allo/ir/units.py``. This is gap 1 of
:ref:`extending-allo-gaps`, landed.

What it is
==========

A ``@df.kernel`` nested in a ``@df.region()`` reaches its streams by lexical
name. Its interface is therefore not a property of the kernel: it is derived,
by whole-region analysis, from the region the kernel happens to sit in. A
``@df.unit`` names its streams in its own signature instead:

.. code-block:: python

   @df.unit()
   def increment(src: Stream[int32, 4], dst: Stream[int32, 4]):
       for i in range(N):
           dst.put(src.get() + 1)

and a region instantiates it, wiring ports to streams of its own choosing:

.. code-block:: python

   @df.region()
   def two(A: int32[N], B: int32[N]):
       a: Stream[int32, 4]
       b: Stream[int32, 4]
       c: Stream[int32, 4]
       produce(dst=a, mem=A)
       first  = increment(src=a, dst=b)      # the same unit,
       second = increment(src=b, dst=c)      # twice, on different streams
       consume(src=c, mem=B)

An instance may be named (``first = ...``) or left anonymous, and its ports may
be passed positionally or by keyword. The nested ``@df.kernel`` form is
untouched and still works, including in the same region as instances; this is
an addition, not a replacement.

What it makes possible
----------------------

Three things the composition on ``design-modular`` named as impossible:

.. list-table::
   :header-rows: 1
   :widths: 56 44

   * - Was impossible
     - Now
   * - Two architectures must agree on a channel's **name**.
     - **Fixed.** A port's name is the unit's; the channel's is the region's,
       and nothing requires them to match. TinyTPU-isa's 16 channels are wired
       under 16 different names in ``units_isa.py``, and a test asserts that
       *no* port name equals the channel it is bound to.
   * - A unit **cannot be instantiated twice** against different channels.
     - **Fixed.** Each instantiation is a separate ``func.func``.
       ``test_a_unit_is_instantiated_twice_against_different_streams``.
   * - A unit cannot be **built or tested outside a composed region**.
     - **Not fixed**, and it is a separate change -- see `What it does not do`_.

How it works, and why there was no IR change
============================================

The destination was already there. ``move_stream_to_interface`` takes the
``allo.stream_construct`` ops a kernel body holds, hoists them into that
function's arguments, and records each one's direction in ``stypes``;
``_build_top`` then constructs the streams once in the region and wires the
kernels by ``call``. Allo's IR has always been a netlist over ported units.

That pass keys a kernel's stream to the **name attribute** on the construct op
it hoists. So a port is, exactly, *a construct op carrying the channel's name
bound to the unit's parameter name*. ``bind_ports`` (``allo/ir/units.py``) is
those two lines, and everything downstream is unchanged.

.. note::

   **One of the two blockers the gap analysis located did not need fixing.**
   ``Stream.__class_getitem__`` did -- a parameter annotation is *evaluated*,
   so ``Stream[int32, 4]`` had to denote a type and not merely parse
   (``allo/ir/types.py``). The second, ``ctx.get_symbol(new_name).clone(...)``
   at six sites in ``allo/ir/builder.py``, was predicated on a port becoming a
   block argument in the frontend. It does not: it becomes a construct op that
   the *existing* pass turns into a block argument, so those six lines are
   untouched and the risk they represented was never taken. The prediction was
   right about the mechanism and wrong about which end of it to change.

An instantiation is expanded into the nested kernel the frontend already
builds, at ``TypeInferer.visit_FunctionDef`` time, before any IR exists. The
unit's body is spliced with its **own** module namespace, captured at the
``@df.unit`` that defined it, so a unit resolves its names where it was
written and not where it is instantiated.

The legality rules
==================

Every rule below is decided from the signatures and the netlist alone, before
any IR is built. Each refuses and transforms nothing. Each names the site and
the repair, and each has a refused example *and* an accepted example in
``tests/dataflow/test_stream_ports.py``.

Checked at the ``@df.unit``, so a broken interface never reaches a region:

.. list-table::
   :header-rows: 1
   :widths: 26 46 28

   * - Rule
     - Legal if and only if
     - Refuses
   * - ``direction-single-valued``
     - a **scalar** port is used for ``get``/``try_get``/``empty`` only, or
       for ``put``/``try_put``/``full`` only.
     - one port both read and written.
   * - ``dangling-port``
     - every declared port is used at least once.
     - a port in the signature that the body ignores -- which would leave the
       channel with a missing endpoint and no diagnostic.
   * - ``port-arity``
     - a port declared with an array shape is subscripted, and one declared
       without is not.
     - ``ch.put(x)`` on ``Stream[T, d][N]``.
   * - ``unresolved-port``
     - an annotation naming ``Stream`` resolves to one in the unit's own
       module.
     - a port whose element type or depth is only defined where it is
       *instantiated* -- the dependency ports exist to remove.

Checked at the ``@df.region()`` that wires them:

.. list-table::
   :header-rows: 1
   :widths: 30 42 28

   * - Rule
     - Legal if and only if
     - Refuses
   * - ``wiring-arity``
     - every parameter of the unit is wired exactly once, and every name
       passed is a parameter of the unit.
     - a forgotten port; a typo'd keyword.
   * - ``wiring-type``
     - the port and the channel agree on element type, element shape, depth
       and array shape.
     - ``Stream[int32, 8]`` wired to a depth-4 channel.
   * - ``single-producer-single-consumer``
     - each **scalar** channel has exactly one writing endpoint and exactly
       one reading endpoint, counting ports and any lexically nested kernel
       that touches it.
     - a second reader, whether it is another instance or a nested
       ``@df.kernel``.
   * - ``unconnected-stream``
     - every declared stream has a writer and a reader.
     - a declaration nothing wires.
   * - ``zero-capacity-cycle``
     - no cycle of the wiring graph consists entirely of depth-0 streams.
     - a rendezvous loop that cannot hold a token. Feedback itself is legal
       and must be: it is normal in dataflow.

The decision procedure for the last one is exact and linear, not a cycle
enumeration: depths are non-negative, so a cycle has total depth 0 **iff**
every edge on it has depth 0, and it is enough to ask whether the subgraph of
depth-0 edges has a cycle at all.

The obligation, which is not a rule
-----------------------------------

**Deadlock-freedom is not decided here, and is not implied.** It needs per-unit
production and consumption rates, and a netlist carries none. Following
:ref:`s.dependence <extending-allo-dependence>`, the undecidable half ships as
a declared obligation rather than as a silence:

.. code-block:: python

   @df.region(deadlock_free_because="one token in flight, one slot each way")
   def loop(A: int32[N], B: int32[N]):
       ...

A region whose netlist carries feedback and declares no premise **warns**
(``allo.netlist.UndeclaredPremise``) at the definition, saying that the
netlist rules passed and that nothing has checked that the loops drain. The
premise is recorded as ``allo.netlist.netlist_of(region).obligation`` and is
checked by nothing. A region with no feedback leaves no obligation at all, so
the default constrains nothing and says nothing.

Two honest limits on the cycle analysis, both conservative:

* An **array channel** is a chain: a unit reads ``ch[i]`` and writes
  ``ch[i + 1]``, the index is a runtime value, and nothing static can tell the
  two streams apart. Such a port is ``io`` -- the direction the IR already
  carries for it -- it is **exempt from single-producer-single-consumer**, and
  it makes its instance look like a self-loop to the cycle analysis. So a
  design with chains will be asked for a premise even when every chain is
  acyclic. That is over-reporting, in the safe direction, and the premise
  ``units_isa.py`` declares says exactly this.
* ``zero-capacity-cycle`` inherits the same over-approximation, so it would
  refuse a depth-0 chain that is in fact acyclic. Depth-0 chains are not
  useful, and no design in this tree has one.

The demonstration
=================

Toy first, real design second.

.. code-block:: bash

   python -m pytest tests/dataflow/test_stream_ports.py -q           # 18 tests, ~3 s
   python -m pytest tests/dataflow/test_stream_ports_tinytpu.py -q   # 4 tests, ~10 s
   python examples/tinytpu/lift_units.py                             # regenerate units_isa.py

``tests/dataflow/test_stream_ports.py`` holds one unit instantiated twice
against different streams, the *same* units composed into a second topology
unedited, a port bound to a channel whose name resembles nothing in the unit,
an array-port chain replicated with ``mapping=``, a mixed region where a unit
instance and a nested ``@df.kernel`` share one stream, and one refused and one
accepted example of every rule above.

``examples/tinytpu/units_isa.py`` is the real test bed, and
it is **generated** by ``lift_units.py`` from ``microarch_isa.py`` rather than
hand-copied, so "the bodies are unchanged" is a property of the process and not
a claim:
**TinyTPU-isa's eight kernel bodies, unchanged, at module level**, each
declaring the streams it used to capture. The counts are the census on
``design-modular`` exactly -- sequencer 5, dma_ld 3, spm 4, vru 4, wld 3, pe 5,
accu 3, dma_st 2, **29 ports over 16 channels** -- and the architecture wires
every one of them under a name the design does not use. Both regions run the
same programs at 4x4x4, 8x8x8 and 16x16x16, gemm and gemm.relu, and agree
element for element.

.. important::

   ``microarch_isa.py`` is still the design. ``units_isa.py`` is generated from
   it and held against it; nothing in the shipped accelerator, its schedule or
   its cycle counts changed, and ``reproduce.sh --no-cosim`` reports
   ``ALL EXACT`` / ``STRESS OK`` as before. The ported architecture is evidence
   that the extension expresses what the design does, not a replacement for it.

What it does not do
===================

**A unit still cannot be built alone, and that is a separate change.**
``df.build`` takes a region, and a unit with unconnected ports is not a
program: to run one you need stimulus on every input port and a drain on every
output port, and how many tokens to supply is written in the unit's *body*
(its loop trip counts), not in its type. So a standalone ``df.build(unit)``
would have to either synthesize drivers from a per-port token count the caller
supplies, or infer rates -- and inferring rates is the same missing analysis
that makes deadlock-freedom an obligation. What ports *do* buy today is that a
unit's interface is inspectable and checked without a region
(``unit.__allo_unit__.ports``), and that its test harness is a three-line
region rather than a copy of the architecture.

**Value parameters are still bound by name.** A unit's memref parameters are
wired through the existing ``args=[...]`` path, which resolves region-scope
names. The parameter name itself is now the unit's own -- the conflict
assertion in ``allo/ir/infer.py`` is relaxed for a unit instance, because a
unit's parameters legitimately shadow -- but the *binding* is still a name
lookup in the region, not a port.

**A unit cannot be parametrized at the instantiation site.** ``@df.unit`` takes
``mapping=``, and the unit's sizes come from its own module. Two instances at
two different ``T`` need two modules, or the source-level composition
``design-modular`` built. Joining stream ports to
``tests/dataflow/test_hierachical.py``'s ``inner[P0, P1]`` type parameters is
the obvious next step and was not attempted here.

A fix that fell out
-------------------

``get_global_vars`` seeded a region's namespace from
``functools.wraps``-wrapper's ``__globals__``, which is ``allo/dataflow.py``'s
module dict, not the module the region was written in. Everything worked only
because the *caller's* module usually happened to have the same names in scope
-- ``bench_isa.py`` imports ``T`` and ``MAXDIM``, so building TinyTPU-isa from
it resolved ``Stream[UInt(VW), QD][T]`` correctly; building it from a pytest
fixture did not. ``allo/ir/utils.py`` now unwraps before reading
``__globals__``. This is a pre-existing defect, found by the ports work and
fixed with it.
