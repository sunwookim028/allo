# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Typing stubs for the lazy schedule frontend."""

from collections.abc import Sequence, Iterable
from typing import Any, Generic, Literal, ParamSpec, TypeVar, overload

from ..backend.cpu import CPU
from ..backend.vitis.core import Vitis
from ..backend.rtl.core import RTL
from .._mlir.ir import Module
from .model import Ref, LoopRef, OpRef, BufferRef
from .query import Query

P = ParamSpec("P")
R = TypeVar("R")

class Schedule(Generic[P, R]):
    """Lazy schedule: primitives accumulate a transform program plus a predicted
    snapshot, and ``apply()`` runs the program once on the real IR. Loop/op/buffer
    handles and queries resolve lazily off the prediction, so primitives can be
    chained before any IR is materialized."""

    query: Query

    # partition kinds
    Complete: int
    Block: int
    Cyclic: int
    Skew: int

    # bind_storage enums
    BRAM = ...
    LUTRAM = ...
    URAM = ...
    SRL = ...

    RAM_1P = ...
    RAM_1WNR = ...
    RAM_2P = ...
    RAM_S2P = ...
    RAM_T2P = ...
    ROM_1P = ...
    ROM_2P = ...
    ROM_NP = ...
    FIFO = ...

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Build a schedule over a kernel or MLIR ``module`` (usually obtained
        from ``Kernel.schedule()``). An optional ``primary`` selects which
        function the primitives target by default."""

    def __str__(self) -> str:
        """Return the payload IR as a string (auto-applies any pending transforms)."""

    def __call__(
        self,
        backend: Literal["cpu", "vitis"] = "cpu",
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> R:
        """Call the scheduled kernel with ``*args``/``**kwargs`` (auto-applies any
        pending transforms) on specified ``backend`` (default: CPU JIT)."""

    @classmethod
    def from_module(cls, module: Any, context: Any = ...) -> Schedule[Any, Any]:
        """Build a schedule from an in-memory MLIR ``module``."""

    @classmethod
    def from_string(cls, text: str) -> Schedule[Any, Any]:
        """Build a schedule by parsing MLIR assembly ``text``."""

    @classmethod
    def from_file(cls, path: str) -> Schedule[Any, Any]:
        """Build a schedule by parsing the MLIR file at ``path``."""
    # --- export to backend (typed by backend literal) -------------------
    @overload
    def export(self, backend: Literal["vitis"], **kwargs: Any) -> Vitis[P, R]:
        """Apply pending transforms and hand the scheduled kernel to the Vitis
        HLS backend, returning a ``Vitis`` handle (csim/synth/emulation).
        ``kwargs`` are forwarded to the backend (``part``/``device``,
        ``freq_mhz``, ...)."""

    @overload
    def export(self, backend: Literal["cpu"], **kwargs: Any) -> CPU[P, R]:
        """Apply pending transforms and hand the scheduled kernel to the CPU
        (LLVM JIT) backend, returning a ``CPU`` handle."""

    @overload
    def export(self, backend: Literal["rtl"], **kwargs: Any) -> RTL[P, R]:
        """Apply pending transforms and hand the scheduled kernel to the RTL
        backend (open-source, cocotb-first), returning an ``RTL`` handle
        (``csim`` / ``cosim`` / ``synth``). ``kwargs`` are forwarded
        (``device``/``library``, ``freq_mhz``, ``simulator``, ...)."""

    def export_cpu(self, **kwargs: Any) -> CPU[P, R]:
        """Shorthand for ``export("cpu", **kwargs)``."""

    def export_vitis(self, **kwargs: Any) -> Vitis[P, R]:
        """Shorthand for ``export("vitis", **kwargs)``."""
    # --- gated real-IR access -------------------------------------------
    @property
    def payload(self) -> Module:
        """The materialized MLIR module; auto-applies any pending transforms."""

    @property
    def module(self) -> Module:
        """Alias for payload"""

    @property
    def snapshot(self) -> Any:
        """The schedule snapshot of the last applied IR. Raises while transforms
        are still pending (call ``apply()`` first)."""
    # --- chainable transforms (return self for fluent pipelines) --------
    def cse(
        self, targets: Iterable[Ref | str] | Ref | str | None = ...
    ) -> Schedule[P, R]:
        """Common-subexpression elimination on ``targets`` (default: the primary
        function)."""

    def dce(
        self, targets: Iterable[Ref | str] | Ref | str | None = ...
    ) -> Schedule[P, R]:
        """Dead-code elimination on ``targets`` (default: the primary function)."""

    def licm(
        self, targets: Iterable[Ref | str] | Ref | str | None = ...
    ) -> Schedule[P, R]:
        """Loop-invariant code motion on ``targets`` (default: the primary
        function)."""

    def canonicalize(
        self, targets: Iterable[Ref | str] | Ref | str | None = ...
    ) -> Schedule[P, R]:
        """Run canonicalization patterns on ``targets`` (default: the primary
        function)."""

    def pipeline(
        self, targets: Iterable[Ref | str] | Ref | str | None = ..., *, ii: int = 1
    ) -> Schedule[P, R]:
        """Pipeline the target loops with initiation interval ``ii``, emitting
        ``#pragma HLS pipeline II=ii``. ``ii`` must be positive."""

    def unroll(
        self,
        targets: Iterable[Ref | str] | Ref | str | None = ...,
        *,
        factor: int = 0,
        tag_only: bool = False,
    ) -> Schedule[P, R]:
        """Unroll the target loops by ``factor`` (0 = full unroll), mapping to
        ``transform.allo.unroll``. With ``tag_only`` it only annotates the loop
        (``#pragma HLS unroll``) instead of structurally unrolling it."""

    def partition(
        self,
        targets: Iterable[Ref | str] | Ref | str | None,
        *,
        dim: int = 0,
        kind=Complete,
        factor: int = 0,
    ) -> Schedule[P, R]:
        """Array-partition the target buffers (``transform.allo.partition``).
        ``kind`` is ``Complete``/``Block``/``Cyclic``/``Skew``; ``dim`` selects
        the dimension (0 = all dims, but a ``Skew`` must name its distribution
        dimension); ``factor`` is the bank count (must stay 0 for
        ``Complete``)."""

    def bind_storage(
        self,
        targets: Iterable[Ref | str] | Ref | str,
        *,
        impl,
        mem_type,
    ) -> Schedule[P, R]:
        """Bind the target buffers to a memory resource
        (``transform.allo.bind_storage`` -> ``#pragma HLS bind_storage``).
        ``impl`` is the resource (``s.BRAM``/``s.URAM``/``s.LUTRAM``/...) and
        ``mem_type`` the port configuration (``s.RAM_2P``/``s.ROM_1P``/...).

        Vitis-only scheduling primitive; other backends ignore it."""

    def streamline(
        self,
        producer: str | Iterable[str],
        consumer: str | Iterable[str],
        *,
        producer_ids: Iterable[int] | int | None = None,
        consumer_ids: Iterable[int] | int | None = None,
        lanes: int = 1,
        depth: int = 2,
    ) -> Schedule[P, R]:
        """Convert the DRAM memory boundary between two fused stage kernels into
        on-chip stream hand-offs (``transform.allo.streamline`` / ``to_stream``).
        ``producer``/``consumer`` are the stage kernel names; ``*_id`` selects a
        specific repeat copy. Each shared memref the producer only writes and the
        consumer only reads becomes a FIFO; unconvertible boundaries are skipped."""

    def dataflow(
        self, targets: Iterable[Ref | str] | Ref | str | None = ...
    ) -> Schedule[P, R]:
        """Tag ``targets`` (default: the primary function) for task-level
        parallelism, emitting ``#pragma HLS dataflow`` so top-level statements run
        as a concurrent dataflow network."""

    def apply(self) -> Schedule[P, R]:
        """Run the accumulated transform program on the real IR (a no-op when
        nothing is pending) and reconcile the predicted snapshot with the result."""

    def compose(self, *callees: Schedule[Any, Any], id: Any = ...) -> Schedule[P, R]:
        """Apply each ``callee``'s whole schedule to the specialized copy of that
        kernel inside this one. Every callee must be a kernel ``self`` calls
        directly; ``id`` selects a specific repeat/specialized copy."""

    def cleanup_schedule_ids(self) -> Schedule[P, R]:
        """Strip the internal schedule-id annotations from the payload IR."""

    def debug_dump(self, *, include_values: bool = ...) -> Schedule[P, R]:
        """Print the schedule state (dirty flag, op/value counts, last-applied
        tree, and transform script) for debugging."""
    # --- loop transforms ------------------------------------------------
    def tile(
        self,
        targets: Iterable[LoopRef | str],
        *,
        factors: Iterable[int],
    ) -> tuple[list[LoopRef], list[LoopRef]]:
        """Tile the given perfectly-nested loop band by ``factors``
        (``transform.allo.tile``). Returns ``(tile_loops, point_loops)`` in depth
        order."""

    def reorder(self, targets: Iterable[LoopRef | str]) -> tuple[LoopRef, ...]:
        """Reorder the selected loops within their perfect affine band
        (``transform.allo.reorder``); unselected loops keep their positions.
        Returns the loop handles in the requested order."""

    def split(
        self, target: LoopRef | str, *, factor: int = 1
    ) -> tuple[LoopRef, LoopRef]:
        """Split ``target`` loop into ``(outer, inner)`` with tiling ``factor``
        (``transform.allo.split``). ``factor`` must be positive."""

    def flatten(self, targets: Iterable[LoopRef | str]) -> LoopRef:
        """Flatten the selected perfectly-nested affine band into a single loop
        (``transform.allo.flatten``); requires normalized loops. Returns the
        flattened loop."""

    def compute_at(self, producer: OpRef | str, axis: LoopRef | str) -> LoopRef:
        """Move the ``producer`` loop nest under the consumer loop ``axis``,
        fusing producer-consumer computation (``transform.allo.compute_at``).
        Returns the consumer ``axis`` loop."""

    def buffer_at(self, buffer: BufferRef | str, axis: LoopRef | str) -> BufferRef:
        """Create a temporary buffer for ``buffer`` at loop ``axis`` and rewrite
        its accesses under that stage to use it (``transform.allo.buffer_at``);
        the original buffer is left unchanged. Returns the new buffer handle."""

    def reuse_at(
        self, buffer: BufferRef | str, axis: LoopRef | str, *, ring: bool = False
    ) -> BufferRef:
        """Create a reuse (line/window) buffer for ``buffer`` at loop ``axis`` and
        rewrite eligible loads to read from it (``transform.allo.reuse_at``).
        ``ring`` uses a ring buffer. Returns the new buffer handle."""

    def outline(
        self,
        target: Ref | str,
        *,
        func_name: str,
        mapping: Sequence[int] | int | None = None,
    ) -> tuple[OpRef, OpRef]:
        """Outline ``target`` into a new callable named ``func_name`` plus a
        callsite (``transform.allo.outline``). With ``mapping`` the callable is an
        ``allo.kernel``/``allo.invoke`` carrying that dataflow mapping, otherwise a
        plain ``func.func``/``func.call``. Returns ``(callable, callsite)``."""
    # --- queries --------------------------------------------------------
    def op(
        self,
        name: str,
        *,
        under: OpRef | str | None = None,
        kind: str | None = None,
        path: str | None = None,
    ) -> OpRef:
        """Resolve exactly one op handle by ``name``, optionally scoped by
        ``under``, filtered by op ``kind``, or addressed directly by ``path``."""

    def loop(
        self,
        name: str,
        *,
        under: OpRef | str | None = None,
        path: str | None = None,
    ) -> LoopRef:
        """Resolve exactly one loop handle by ``name``, optionally scoped by
        ``under`` or addressed directly by ``path``."""

    def loops(
        self,
        *names: str,
        under: OpRef | str | None = None,
        path: str | None = None,
    ) -> tuple[LoopRef, ...]:
        """Resolve several loop handles by ``names`` (or every loop in scope when
        no names are given), optionally scoped by ``under``/``path``."""

    def buffer(
        self,
        name: str | None = None,
        *,
        under: OpRef | str | None = None,
        path: str | None = None,
    ) -> BufferRef:
        """Resolve exactly one buffer handle by ``name``, optionally scoped by
        ``under`` or addressed directly by ``path``."""
    # --- introspection --------------------------------------------------
    def format_tree(self, *, include_values: bool = ...) -> str:
        """Return the last-applied schedule tree as text."""

    def dump_tree(self, *, include_values: bool = ...) -> str:
        """Print and return the last-applied schedule tree."""

    def dump_transform_script(self) -> str:
        """Return the accumulated transform-dialect script as text."""

    def __getattr__(self, name: str) -> Any: ...
