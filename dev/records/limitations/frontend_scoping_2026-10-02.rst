..  Copyright Allo authors. All Rights Reserved.
    SPDX-License-Identifier: Apache-2.0

######################################################
Front-end name resolution: two silent miscompiles fixed
######################################################

:Date: 2026-10-02
:Host: zhang-21
:Branch: ``core-scoping`` (review branch, based on ``origin/main`` ``06650725``)
:Found by: the MiniTPU U1 ALU composition probe, record
   ``dev/records/minitpu/u1_alu_2026-10-02.rst`` on ``u1-pilot`` (C1-C5),
   repros ``dev/records/minitpu/u1_alu/repros.py``

Python-only change: ``allo/ir/{utils,visitor,infer,builder,symbol_resolver}.py``,
``allo/customize.py``. No C++.

The bugs
========

Before the fix the front end had one name space per build. ``get_global_vars``
walks every Python function reachable from the top function and merges all
their modules' globals into one dict, first name wins, and every called
function was type-inferred and built with a copy of the *caller's* dict. The
kernel's own parameters and locals were consulted after that dict in some
places. Five consequences, the first two silent:

C3 (silent). A reused function reads the caller's globals.
   ``engine_lib.py``: ``K = 3``, ``def scale(x) -> int32: return x * K``; the
   calling module has ``K = 5``. A kernel ``c[i] = scale(a[i])`` gives
   ``[0 5 10 15]``; Python gives ``[0 3 6 9]``. Same with a closure
   (``make(3)`` returning ``scale`` reads the caller's ``K``), and with a
   callee's shape constant (``x: int32[N]`` takes the caller's ``N``: a
   "shape mismatch" error if the shapes differ, a silent re-size if they are
   only compatible). It also held for a ``@df.region`` from another module
   called from a kernel.
C4 (silent). A module-level numpy array shadows a kernel parameter.
   ``infer.py`` ``visit_assignment_val`` took the global array of the
   assigned name before the kernel's own symbols. ``a = np.full(4, 7)`` at
   module level and a parameter ``a``: ``x: int32[1] = a[1:2]`` reads the
   global: ``[7 7 7 7]``. Also for ``x: int32[4] = a``. Related, found here:
   slice bounds were evaluated in the globals too, so ``n: int32 = 0;
   t = A[n:n+2]`` with a global ``n = 2`` silently built ``A[2:4]``
   (``[12, 13]`` for ``[10, 11]``).
C1. A function passed as a value does not link: the call was named after the
   call-site name (``eng``), the function after its def name (``twice``).
C2. Two functions with one def name (``add`` from two modules) could not
   coexist: ``redefinition of symbol named 'add'``. Worse once C3 is fixed:
   built functions were cached *by name*, so ``lib1.f`` -> ``helper`` and
   ``lib2.g`` -> ``helper`` would silently share ``lib1.helper``.
C5. A call argument was not converted to the parameter type
   (``f(b ^ 0x8000)`` with ``f(x: uint16)``: ``operand type mismatch``).

The fix
=======

* ``utils.callee_global_vars(caller_vars, func, top_globals)``: a called
  Python function is inferred and built with its own module's
  ``__globals__`` and then its closure over the caller's view. The caller's
  view stays underneath only as a fallback for names the callee's scope does
  not have (front-end entries such as ``df.p0``; an enclosing function's
  locals named only in a nested function's annotations, which leave no closure
  cell). A function nested in the top function's own module keeps the
  caller's view under its closure, as before (that view already holds the
  module and the enclosing frame's locals). A generic function's type
  parameters are not overlaid; the instantiation binds them.
* Used at both call sites (``infer.visit_Call``, ``builder.build_Call``) and
  for a ``@df.region`` called from a kernel. A call through an attribute
  (``lib.scale(x)``) uses the object it resolves to, not whatever ``scale``
  is in scope.
* ``ASTContext.py_func_ops`` (shared by every copy of a context): built
  Python functions are keyed on (function object, template id). The symbol is
  the def name; another object with the same def name gets ``<name>_v<n>``.
  Fixes C1 and C2. Copying a callee's built ``func.func`` back into the caller
  no longer overwrites a caller name bound to something else.
* C5: a scalar argument whose MLIR type differs from the parameter's is cast
  with ``build_cast_op`` (only where the call was invalid before).
* C4: ``visit_assignment_val`` consults ``ctx.is_local`` (parameters and
  locals; a ``meta_for`` index is a placeholder, not a local) before a global
  array, and does not treat ``G[i]`` as a constant slice when ``i`` is local.
  ``ASTResolver.resolve_constant(..., locals_shadow=True)`` returns "not a
  constant" for an expression that reads a local; used for array slice
  bounds (``resolve_slice`` and the builder's two slice-bound sites).

Not changed, and why: ``get_global_vars`` still merges modules for the *top*
function (the top function's own names win, so a valid program is unaffected;
an invalid one that names something only a callee's module defines still
builds, as before). ``ConstExpr`` right-hand sides and stream indices are
still evaluated in ``global_vars`` (a local there is not a constant anyway; not
probed). ``@df.unit``'s decorator-time snapshot is C9, not touched.

After the fix C4's own repro (``a[1:2]`` of a *parameter* into an
``int32[1]``) crashes LLVM in the simulator and fails to parse on the CPU
backend: a rank-0 ``memref.subview`` then an ``affine.load`` of an ``i32``.
That is a separate, pre-existing bug -- origin/main crashes identically on
the same kernel with no global named ``a``. The regression tests use the
whole-array and the 2-D row (``b[1]``) forms. With a loop index
(``a[i:i+1]``) the result is now the same "Cannot broadcast (4,) to (1,)" as
with no global (dynamic slices are unsupported), not a ``NameError``.

Tests
=====

``tests/test_scoping.py`` (14, CPU backend) and
``tests/dataflow/test_df_scoping.py`` (3, simulator): C3 from a global and
from a caller local; two-level reuse ``outer -> inner`` with the caller owning
another ``inner``; ``lib.scale`` vs a local ``scale``; a closure; a nested
helper that must still see enclosing locals (body and annotation); a callee's
constant as a shape; C1; C2; C5; C4 whole-array and sliced; a global array
still a constant without a shadowing parameter; a local slice bound refused;
``df.kernel`` in ``df.region`` calling a reused function; C4 in a kernel; a
``@df.region`` from another module called from a kernel. On origin/main 15 of
the 17 fail (the two that pass are the must-not-regress cases).

Impact vs main
==============

Baseline: a detached worktree at origin/main ``06650725``, same bindings
(``mlir/build`` symlink), same host, same env.

.. list-table::
   :header-rows: 1

   * - check
     - origin/main
     - core-scoping
   * - TinyTPU emitted Vitis (sha256 of ``str(s.build("vhls"))``)
     - ``6bc774bc...b2ef95``
     - ``6bc774bc...b2ef95`` (identical)
   * - TinyTPU emitted Catapult
     - ``ade1ab5d...88e038``
     - ``ade1ab5d...88e038`` (identical)
   * - ``gen_isa.py --check``
     - ISA OK
     - ISA OK
   * - ``lift_units.py --check``
     - UNITS OK
     - UNITS OK
   * - ``bench_isa.py``
     - ALL EXACT
     - ALL EXACT
   * - ``stress_isa.py``
     - STRESS OK 640/640
     - STRESS OK 640/640
   * - ``act_compile.py --gate``
     - ACT GATE OK 12/12
     - ACT GATE OK 12/12
   * - ``examples/minitpu/run.py --quick``
     - PASS, EXACT vs model
     - PASS, EXACT vs model
   * - ``pytest tests/dataflow --ignore=tests/dataflow/aie``
     - 119 pass, 22 fail, 20 skip/xfail
     - 122 pass (+3 new), 22 fail, 20 skip/xfail
   * - ``pytest tests/act``
     - 193 pass, 2 fail, 4 skip
     - 193 pass, 2 fail, 4 skip
   * - ``pytest tests/test_*.py``
     - 435 pass, 37 fail, 10 skip
     - 449 pass (+14 new), 37 fail, 10 skip
   * - ``pytest examples/machsuite tests/ip tests/ip_integration tests/systemc tests/limits``
     - 65 pass, 6 fail, 2 skip/xfail
     - 65 pass, 6 fail (same tests), 2 skip/xfail

No pre-existing test changed outcome (compared per test case from the JUnit
XML). The failures are the same set on both sides and environmental: Vitis
HLS csim/csynth project builds, ``GLIBCXX_3.4.32`` missing for compiled
wrappers, ``past.verify`` absent, a ``/tmp/allo_test_pynq_prj`` owned by
another user, and ``tests/act/test_bindings.py`` asserting that ``allo`` and
its extension come from one checkout (the worktrees symlink the main
checkout's build). Two are not environmental and are pre-existing on main:
``tests/test_stateful.py::test_nested_stateful_collision`` and
``::test_multiple_nested_same_name`` (``Stateful`` globals named by variable
name collide across functions -- the same name-keyed-symbol class as C2, not
addressed here).

Did anything rely on the old resolution?
----------------------------------------

No. ``callee_global_vars`` was instrumented to log, for every called
function, each name it reads whose value differs between the old (caller)
view and the new one. TinyTPU's build calls no Python function through this
path (its units are composed by ``compose``), so it could not have relied on
C3, and its emitted code is byte-identical. Over the whole test run above
(825 callee builds, 167 distinct functions, including ``allo.library``,
MachSuite, ``tests/ip`` and ``tests/systemc``) the only names that resolve
to a different value are the ones in the new scoping tests. An earlier instrumented run showed the generic-function
case: ``allo.library.nn`` / ``systolic.PE_kernel`` read their type parameters
(``Ty``, ``L``, ``D``, ``K``...) as closure cells; overlaying those
``TypeVar`` cells was harmless (the instantiation rebinds them) but they are
now skipped. The MiniTPU U1 ALU (``u1-pilot``, where the bugs were found) was
not re-run: that branch carries C++ emitter changes this build does not have.

Upstream
========

``cornell-zhang/allo`` ``main`` at ``3f2ea5d4`` (2026-09-30) has the same code:
``get_global_vars`` (``allo/ir/utils.py``) merges first-name-wins;
``infer.visit_Call`` and ``builder.build_Call`` take
``func = ctx.global_vars[obj_name]`` and build it in ``ctx.copy()``;
``visit_assignment_val`` checks ``ctx.global_vars`` for a numpy array before
the kernel's symbols; built functions are cached by call-site name. Upstream
also lacks the fork's ``inspect.unwrap`` in ``_get_global_vars``. Related
upstream issue: #169 ("Can't resolve variables with the same name inside
callFunc", open since 2024-08; its repro already passes on the fork's main,
with and without this fix). Draft below, not filed.

Draft upstream issue
--------------------

    **[Bug][Frontend] A called function resolves free names in the caller's
    globals; a global numpy array shadows a kernel parameter (silent wrong
    results)**

    *1. A function reused from another module reads the caller's global of the
    same name.* ``get_global_vars`` merges the globals of every reachable
    function's module into one dict (first name wins), and ``visit_Call`` /
    ``build_Call`` infer and build the callee with a copy of the caller's
    dict::

        # engine_lib.py
        from allo.ir.types import int32
        K = 3
        def scale(x: int32) -> int32:
            return x * K

        # main.py
        import numpy as np, allo
        from allo.ir.types import int32
        from engine_lib import scale
        K = 5
        def kernel(A: int32[4], C: int32[4]):
            for i in range(4):
                C[i] = scale(A[i])
        mod = allo.customize(kernel).build()
        C = np.zeros(4, np.int32); mod(np.arange(4, dtype=np.int32), C)
        print(C)   # [0 5 10 15]; Python: [0 3 6 9]

    The same happens with a closure (``make(3)`` returning ``scale``), with a
    callee's shape constant, and for a ``df.region`` from another module called
    from a kernel. Built callees are also cached by call-site name, so two
    functions called ``helper`` from two modules share one ``func.func``, a
    function passed as a value (``eng = twice; eng(x)``) emits ``call @eng`` to
    a function named ``twice``, and two functions with one def name give
    ``redefinition of symbol``.

    *2. A module-level numpy array shadows a kernel parameter.*
    ``TypeInferer.visit_assignment_val`` checks ``ctx.global_vars`` for an
    ``np.ndarray`` of the assigned name before the kernel's symbols::

        a = np.full(4, 7, np.int32)
        def kernel(a: int32[4], C: int32[4]):
            x: int32[4] = a        # reads the global: C == [7 7 7 7]
            ...

    Slice bounds are evaluated in the globals too (``A[n:n+2]`` with a local
    ``n`` and a global ``n``).

    *Expected:* Python scoping -- a callee's free names resolve in its own
    ``__globals__`` and closure; parameters and locals shadow globals.

    *Fix* (sunwookim028/allo ``core-scoping``): build each called Python
    function with its own ``__globals__`` + closure over the caller's view;
    key built functions on the function object; check local symbols before
    global arrays and in slice-bound constant folding; cast scalar call
    arguments to the parameter type. Byte-identical output on the fork's
    designs; no test changes outcome.
