# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import functools
import copy
from collections.abc import Iterable, Sequence
from typing import Literal, Generic, TypeVar, ParamSpec
from enum import Enum

from ..lang.kernel import Kernel
from .errors import (
    InvalidScheduleArgumentError,
    ScheduleError,
    ScheduleLookupError,
    ScheduleStateError,
    ScheduleTransformError,
    ScheduleTypeError,
)
from .keys import (
    annotate_schedule_keys,
    derived_key,
    read_schedule_keys,
)
from .model import (
    BufferRef,
    LoopRef,
    OpRef,
    PredictedOp,
    PredictedSnapshot,
    Ref,
    ScheduleSnapshot,
    SingleTarget,
    Targets,
)
from .query import Query
from .script import TransformScript
from .._mlir.ir import Context, Module, Value, IntegerAttr, IntegerType
from .._mlir import schedule as schedule_d
from .._mlir.schedule import (
    ScheduleOpTrait,
    PIPELINE_II_ATTR_NAME,
    UNROLL_FACTOR_ATTR_NAME,
    DATAFLOW_ATTR_NAME,
)
from .._mlir.dialects import allo as allo_d
from .._mlir.dialects import transform as t
from .._mlir._mlir_libs._allo import ir_ext

from .._mlir.dialects.transform import allo as ta
from .._mlir.dialects.transform import interpreter
from ..logging import log_debug, text_tail


def _within_context(method):
    """Run a schedule primitive under ``with self.context`` so upstream ODS attr
    builders (StrArrayAttr/I64Attr/…) can resolve the MLIR context."""

    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        with self.context:
            return method(self, *args, **kwargs)

    return wrapper


P = ParamSpec("P")
R = TypeVar("R")


class BindStorageImpl(Enum):
    BRAM = "bram"
    LUTRAM = "lutram"
    URAM = "uram"
    SRL = "srl"


class BindStorageType(Enum):
    RAM_1P = "ram_1p"
    RAM_1WNR = "ram_1wnr"
    RAM_2P = "ram_2p"
    RAM_S2P = "ram_s2p"
    RAM_T2P = "ram_t2p"
    ROM_1P = "rom_1p"
    ROM_2P = "rom_2p"
    ROM_NP = "rom_np"
    FIFO = "fifo"


class Schedule(Generic[P, R]):
    """Lazy schedule frontend: primitives accumulate a reusable transform program
    (``@sched(%root)``) and a predicted snapshot; ``apply()`` runs the program once.

    ``s.payload`` / ``s.snapshot`` expose the real IR and require ``apply()`` first
    (they raise while transforms are pending). Handles and queries work lazily off
    the predicted snapshot.
    """

    # --- partition enums ---
    Complete = 0
    Block = 1
    Cyclic = 2

    # --- bind_storage enums ---
    BRAM = BindStorageImpl.BRAM
    LUTRAM = BindStorageImpl.LUTRAM
    URAM = BindStorageImpl.URAM
    SRL = BindStorageImpl.SRL

    RAM_1P = BindStorageType.RAM_1P
    RAM_1WNR = BindStorageType.RAM_1WNR
    RAM_2P = BindStorageType.RAM_2P
    RAM_S2P = BindStorageType.RAM_S2P
    RAM_T2P = BindStorageType.RAM_T2P
    ROM_1P = BindStorageType.ROM_1P
    ROM_2P = BindStorageType.ROM_2P
    ROM_NP = BindStorageType.ROM_NP
    FIFO = BindStorageType.FIFO

    def __init__(
        self,
        module: Module | None = None,
        context: Context | None = None,
        *,
        kernel: Kernel[P, R] | None = None,
        primary: str | None = None,
    ):
        assert not (kernel and module), "cannot specify both kernel and module"
        if kernel is not None:
            module = kernel.module
        assert module is not None
        self.kernel = kernel
        self.context = context if context is not None else module.context
        allo_d.register_extensions(self.context)
        self.dirty = False

        schedule_d.annotate_schedule_ids(module)
        bootstrap = ScheduleSnapshot.from_raw(
            schedule_d.collect_schedule_snapshot(module)
        )
        self._primary_name, self._primary_path = self._detect_primary(
            bootstrap, primary
        )
        annotate_schedule_keys(module, bootstrap.relkey_by_id)
        # `_payload` is the (keyed) module; apply() never mutates it structurally — it
        # parses `str(_payload)` into a fresh working copy, runs the delta there, and
        # rebinds `_payload` to it. The snapshot is collected from the named module so
        # value names (e.g. buffer "B") are available before the first apply.
        self._payload: Module = ir_ext.clone_module(module)
        self._real = ScheduleSnapshot.from_raw(
            schedule_d.collect_schedule_snapshot(module),
            primary_path=self._primary_path,
        )
        self.predicted = PredictedSnapshot.from_real(self._real)
        self.script = TransformScript(self, self._primary_name)
        self.query = Query(self)

    def __str__(self) -> str:
        return self.payload.__str__()

    def __call__(self, backend: str = "cpu", *args: P.args, **kwargs: P.kwargs) -> R:
        if self.kernel is None:
            raise ScheduleError("Cannot call a schedule without a source kernel")
        if backend == "vitis":
            return self.export_vitis()(*args, **kwargs)
        if backend == "cpu":
            return self.export_cpu()(*args, **kwargs)
        raise ScheduleError(f"unsupported backend '{backend}' for execution")

    @staticmethod
    def _detect_primary(snap: ScheduleSnapshot, primary: str | None) -> tuple[str, str]:
        """Return (name, path) of the function the schedule operates on: the named
        kernel when given, else the single function child of the module root (the
        one whose name has no ``.`` — nested-callee copies are ``parent.callee``)."""
        funcs = [
            node
            for node in snap.ops
            if node.parent_id == snap.root_id
            and node.has_trait(ScheduleOpTrait.FUNCTION_LIKE)
        ]
        if primary is not None:
            for node in funcs:
                if node.name == primary:
                    return primary, node.path
            raise InvalidScheduleArgumentError(
                f"primary function '{primary}' not found in module"
            )
        own = [node for node in funcs if node.name is None or "." not in node.name]
        candidates = own or funcs
        assert candidates, "module has no function to schedule"
        node = candidates[0]
        return (node.name or snap.relkey_of(node.id)), node.path

    @classmethod
    def from_module(cls, module: Module, context: Context | None = None) -> Schedule:
        return cls(module, context)

    @classmethod
    def from_string(cls, text: str) -> Schedule:
        context = Context()
        allo_d.register_dialect(context)
        module = Module.parse(text, context)
        return cls(module, context)

    @classmethod
    def from_file(cls, path: str) -> Schedule:
        with open(path, "r", encoding="utf-8") as handle:
            return cls.from_string(handle.read())

    # --- export to backend ----------------------------------------------
    def export(self, backend: Literal["cpu", "vitis"], **kwargs):
        if not self.kernel:
            raise ScheduleError("Cannot export to backends without a source kernel")
        self.apply()
        # Simplify the redundant IR that reuse_at leaves behind (merge per-access
        # affine.ifs, CSE, drop dead index math). The pass is a no-op when the
        # module contains no reuse_at output.
        from ..backend.base import run_pipeline

        with self.context:
            run_pipeline(self._payload, "builtin.module(reuse-cleanup)")
        # shallow copy, not modifying the original kernel
        kernel = copy.copy(self.kernel)
        kernel.module = self._payload

        if backend == "cpu":
            from ..backend import CPU

            return CPU(kernel, **kwargs)
        elif backend == "vitis":
            from ..backend.vitis import Vitis

            return Vitis(kernel, **kwargs)

        raise ScheduleError(f"unsupported backend '{backend}' for export()")

    def export_cpu(self, **kwargs):
        return self.export("cpu", **kwargs)

    def export_vitis(self, **kwargs):
        return self.export("vitis", **kwargs)

    # --- gated real-IR access --------------------------------------------

    @property
    def payload(self) -> Module:
        self.apply()  # auto-apply pending transforms for user convenience
        return self._payload

    @property
    def snapshot(self) -> ScheduleSnapshot:
        self._require_materialized("snapshot")
        return self._real

    def _require_materialized(self, what: str) -> None:
        if self.dirty:
            raise ScheduleStateError(
                f"{what} has pending transforms; call .apply() to materialize them"
            )

    # --- query aliases ----------------------------------------------------

    def op(
        self,
        name: str | None = None,
        *,
        under: OpRef | str | None = None,
        kind: str | None = None,
        path: str | None = None,
    ) -> OpRef:
        ref = self.query.op(name, under=under, kind=kind, path=path).one()
        assert isinstance(ref, OpRef)
        return ref

    def loop(
        self,
        name: str | None = None,
        *,
        under: OpRef | str | None = None,
        path: str | None = None,
    ) -> LoopRef:
        ref = self.query.loop(name, under=under, path=path).one()
        assert isinstance(ref, LoopRef)
        return ref

    def loops(
        self,
        *names: str,
        under: OpRef | str | None = None,
        path: str | None = None,
    ) -> tuple[LoopRef, ...]:
        selection = self.query.loop(under=under, path=path)
        result = selection.names(*names) if names else tuple(selection.all())
        return tuple(r for r in result if isinstance(r, LoopRef))

    def buffer(
        self,
        name: str | None = None,
        *,
        under: OpRef | str | None = None,
        path: str | None = None,
    ) -> BufferRef:
        ref = self.query.buffer(name, under=under, path=path).one()
        assert isinstance(ref, BufferRef)
        return ref

    def live(self, ref: Ref) -> Ref:
        self.predicted.require_live(ref)
        return ref

    # --- pattern / cleanup primitives (approximate) ----------------------

    @_within_context
    def cse(self, targets: Targets = None) -> Schedule:
        return self._apply_op_pass(
            targets, "cse", t.ApplyCommonSubexpressionEliminationOp
        )

    @_within_context
    def dce(self, targets: Targets = None) -> Schedule:
        return self._apply_op_pass(targets, "dce", t.ApplyDeadCodeEliminationOp)

    @_within_context
    def licm(self, targets: Targets = None) -> Schedule:
        return self._apply_op_pass(targets, "licm", t.ApplyLoopInvariantCodeMotionOp)

    def _apply_op_pass(self, targets: Targets, desc: str, op_cls) -> Schedule:
        ops = self._resolve_op_targets(targets, desc)
        self.script.set_callsite_loc()
        for op in ops:
            op_cls(self._op_handle(op), **self.script.kw)
            self._predicted_mark_approx(op)
        self._mark_dirty()
        return self

    @_within_context
    def apply_patterns(
        self, patterns: str | Iterable[str], targets: Targets = None
    ) -> Schedule:
        pattern_names = [patterns] if isinstance(patterns, str) else list(patterns)
        if not pattern_names:
            raise InvalidScheduleArgumentError("apply_patterns requires a pattern")
        supported = {"canonicalize": t.ApplyCanonicalizationPatternsOp}
        pattern_ops = []
        for pattern in pattern_names:
            op = supported.get(pattern)
            if op is None:
                raise InvalidScheduleArgumentError(
                    f"unsupported pattern '{pattern}' in apply_patterns"
                )
            pattern_ops.append(op)

        ops = self._resolve_op_targets(targets, "apply_patterns")
        self.script.set_callsite_loc()
        for op in ops:
            apply_op = t.ApplyPatternsOp(self._op_handle(op), **self.script.kw)
            region = apply_op.regions[0]
            body = region.blocks[0] if len(region.blocks) else region.blocks.append()
            ip = self.script.builder.save_insertion_point()
            self.script.builder.set_insertion_point_to_end(body)
            for pattern_op in pattern_ops:
                pattern_op(**self.script.kw)
            self.script.builder.restore_insertion_point(ip)
            self._predicted_mark_approx(op)
        self._mark_dirty()
        return self

    def canonicalize(self, targets: Targets = None) -> Schedule:
        return self.apply_patterns("canonicalize", targets)

    # --- tag primitives (no structural change) ---------------------------

    @_within_context
    def pipeline(self, targets: Targets = None, *, ii: int = 1) -> Schedule:
        self._require_int("pipeline ii", ii)
        if ii < -1:
            raise InvalidScheduleArgumentError(
                f"pipeline ii must be -1 (disable) or non-negative (0 = auto), got {ii}"
            )
        loops = self._resolve_loop_targets(targets, "pipeline")
        self.script.set_callsite_loc()
        ii_attr = IntegerAttr.get(IntegerType.get_signless(64), ii)
        for loop in loops:
            self.script.annotate_attr(
                self.script.match(loop.key), PIPELINE_II_ATTR_NAME, ii_attr
            )
        self._mark_dirty()
        return self

    @_within_context
    def dataflow(self, targets: Targets = None) -> Schedule:
        """Tag a function for task-level parallelism (``#pragma HLS dataflow``).

        Defaults to the primary function. The Vitis HLS emitter turns the
        ``dataflow`` attribute into the pragma, so the function's top-level
        statements (e.g. the PE-grid invokes of a systolic array) run as a
        concurrent dataflow network instead of sequentially.
        """
        ops = self._resolve_op_targets(targets, "dataflow")
        self.script.set_callsite_loc()
        for op in ops:
            t.AnnotateOp(self._op_handle(op), DATAFLOW_ATTR_NAME, **self.script.kw)
        self._mark_dirty()
        return self

    @_within_context
    def unroll(
        self, targets: Targets = None, *, factor: int = 0, tag_only: bool = False
    ) -> Schedule:
        self._require_int("unroll factor", factor)
        if factor < 0:
            raise InvalidScheduleArgumentError(
                f"unroll factor must be non-negative, got {factor}"
            )
        loops = self._resolve_loop_targets(targets, "unroll")
        self.script.set_callsite_loc()
        factor_attr = IntegerAttr.get(IntegerType.get_signless(64), factor)
        for loop in loops:
            if tag_only:
                self.script.annotate_attr(
                    self.script.match(loop.key), UNROLL_FACTOR_ATTR_NAME, factor_attr
                )
                continue
            ta.AlloLoopUnrollOp(self.script.match(loop.key), factor, **self.script.kw)
            self.predicted.mark_approx(self._pred(loop))
            # add cleanups
            self.canonicalize(targets=loop)
            self.cse(targets=loop)
        self._mark_dirty()
        return self

    @_within_context
    def partition(
        self,
        targets: Targets,
        *,
        dim: int = 0,
        kind=Complete,
        factor: int = 0,
    ) -> Schedule:
        self._require_int("partition dim", dim)
        self._require_int("partition factor", factor)
        if dim < 0:
            raise InvalidScheduleArgumentError(
                f"partition dim must be non-negative, got {dim}"
            )
        if kind not in (self.Complete, self.Block, self.Cyclic):
            raise InvalidScheduleArgumentError(
                "partition kind must be Schedule.Complete, Schedule.Block, "
                "or Schedule.Cyclic"
            )
        if kind == self.Complete:
            if factor != 0:
                raise InvalidScheduleArgumentError(
                    "complete partition cannot have non-zero factor"
                )
        elif factor <= 1:
            raise InvalidScheduleArgumentError(
                f"{self._partition_kind_name(kind)} partition factor must be "
                f"greater than 1, got {factor}"
            )

        buffers = self._resolve_buffer_targets(targets, "partition")
        axis = allo_d.PartitionAxisAttr.get(kind, factor, dim, self.context)
        part = allo_d.PartitionAttr.get([axis], self.context)
        self.script.set_callsite_loc()
        for buf in buffers:
            handle = self.script.match_value(buf.owner_key, buf.number, buf.source)
            ta.PartitionOp(handle, part, **self.script.kw)
        self._mark_dirty()
        return self

    @_within_context
    def bind_storage(
        self,
        targets: Targets,
        *,
        impl: BindStorageImpl,
        mem_type: BindStorageType,
    ) -> Schedule:
        if not isinstance(impl, BindStorageImpl):
            raise InvalidScheduleArgumentError(
                f"bind_storage impl must be one of {', '.join(e.name for e in BindStorageImpl)}, "
            )
        if not isinstance(mem_type, BindStorageType):
            raise InvalidScheduleArgumentError(
                f"bind_storage mem_type must be one of {', '.join(e.name for e in BindStorageType)}, "
            )
        buffers = self._resolve_buffer_targets(targets, "bind_storage")
        self.script.set_callsite_loc()
        for buf in buffers:
            handle = self.script.match_value(buf.owner_key, buf.number, buf.source)
            ta.BindStorageOp(handle, mem_type.value, impl.value, **self.script.kw)
        self._mark_dirty()
        return self

    # --- structural primitives -------------------------------------------

    @_within_context
    def affine(self, targets: Targets = None) -> list[LoopRef]:
        loops = self._resolve_loop_targets(targets, "affine")
        self.script.set_callsite_loc()
        out: list[LoopRef] = []
        for loop in loops:
            raised = ta.RaiseToAffineOp(
                self.script.any_op_type,
                self.script.match(loop.key),
                **self.script.kw,
            ).result
            self.script.annotate_key(raised, loop.key)
            if loop.name is not None:
                self.script.annotate_name(raised, loop.name)
            node = self.predicted.flip_kind(self._pred(loop), "affine.for")
            out.append(self.predicted.make_loop_ref(node))
        self._mark_dirty()
        return out

    @_within_context
    def split(
        self, target: SingleTarget | None = None, *, factor: int = 1
    ) -> tuple[LoopRef, LoopRef]:
        self._require_int("split factor", factor)
        if factor <= 0:
            raise InvalidScheduleArgumentError(
                f"split factor must be positive, got {factor}"
            )
        loop = self._resolve_single_loop_target(target, "split")
        okey, ikey = derived_key(loop.key, "outer"), derived_key(loop.key, "inner")

        self.script.set_callsite_loc()
        out_h, in_h = ta.LoopSplitOp(
            self.script.any_op_type,
            self.script.any_op_type,
            self.script.match(loop.key),
            factor,
            **self.script.kw,
        ).results
        self.script.annotate_key(out_h, okey)
        self.script.annotate_key(in_h, ikey)

        outer, inner = self.predicted.split(self._pred(loop), okey, ikey)
        self._mark_dirty()
        return self.predicted.make_loop_ref(outer), self.predicted.make_loop_ref(inner)

    @_within_context
    def reorder(self, targets: Targets) -> tuple[LoopRef, ...]:
        desired = [
            self._require_affine(loop)
            for loop in self._resolve_loop_targets(targets, "reorder")
        ]
        if len(desired) < 2:
            raise InvalidScheduleArgumentError("reorder requires at least two loops")
        desired_keys = [(loop.scope, loop.key) for loop in desired]
        if len(set(desired_keys)) != len(desired_keys):
            raise InvalidScheduleArgumentError("reorder targets must be unique")

        desired_pred = [self._pred(loop) for loop in desired]
        current = sorted(desired_pred, key=lambda op: self.predicted.depth(op))
        current_keys = [(op.scope, op.key) for op in current]
        permutation = [current_keys.index((op.scope, op.key)) for op in desired_pred]

        self.script.set_callsite_loc()
        handles = [self.script.match(op.key) for op in current]
        merged = t.MergeHandlesOp(handles, deduplicate=False, **self.script.kw).result
        ta.LoopReorderOp(merged, permutation, **self.script.kw)

        self.predicted.reorder(desired_pred)
        self._mark_dirty()
        return tuple(self.predicted.make_loop_ref(self._pred(loop)) for loop in desired)

    @_within_context
    def tile(
        self, targets: Targets = None, *, factors: int | Iterable[int] = 1
    ) -> tuple[list[LoopRef], list[LoopRef]]:
        loops = self._resolve_loop_targets(targets, "tile")
        factor_list = self._normalize_tile_factors(factors, len(loops))
        band = sorted((self._pred(loop) for loop in loops), key=self.predicted.depth)
        tile_keys = [derived_key(op.key, "tile") for op in band]
        point_keys = [derived_key(op.key, "point") for op in band]

        self.script.set_callsite_loc()
        handles = [self.script.match(loop.key) for loop in loops]
        merged = t.MergeHandlesOp(handles, deduplicate=True, **self.script.kw).result
        tiled = ta.LoopTileOp(
            self.script.any_op_type,
            self.script.any_op_type,
            merged,
            factor_list,
            **self.script.kw,
        )
        self._split_and_annotate(tiled.results[0], tile_keys)
        self._split_and_annotate(tiled.results[1], point_keys)

        tiles, points = self.predicted.tile(band, tile_keys, point_keys)
        self._mark_dirty()
        return (
            [self.predicted.make_loop_ref(op) for op in tiles],
            [self.predicted.make_loop_ref(op) for op in points],
        )

    @_within_context
    def flatten(self, targets: Targets) -> LoopRef:
        loops = self._resolve_loop_targets(targets, "flatten")
        if len(loops) < 2:
            raise InvalidScheduleArgumentError(
                "flatten requires at least two loop targets"
            )
        band = sorted((self._pred(loop) for loop in loops), key=self.predicted.depth)
        flat_key = derived_key(band[0].key, "flat")

        self.script.set_callsite_loc()
        handles = [self.script.match(loop.key) for loop in loops]
        merged = t.MergeHandlesOp(handles, deduplicate=True, **self.script.kw).result
        flattened = ta.LoopFlattenOp(
            self.script.any_op_type, merged, **self.script.kw
        ).result
        self.script.annotate_key(flattened, flat_key)

        flat = self.predicted.flatten(band, flat_key)
        self._mark_dirty()
        return self.predicted.make_loop_ref(flat)

    @_within_context
    def compute_at(self, target: SingleTarget, axis: SingleTarget) -> LoopRef:
        producer = self._resolve_single_op_target(target, "compute_at target")
        axis_loop = self._require_affine(
            self._resolve_single_loop_target(axis, "compute_at axis")
        )
        self.script.set_callsite_loc()
        ta.ComputeAtOp(
            self.script.match(producer.key),
            self.script.match(axis_loop.key),
            **self.script.kw,
        )
        # compute_at fuses the whole producer loop nest into the consumer and
        # erases the producer's outer loops, so the precise post-structure cannot
        # be predicted: mark the entire producer nest approximate (reconcile
        # rebuilds it from the real IR) instead of just reparenting the target.
        self.predicted.mark_approx(self._loop_nest_root(self._pred(producer)))
        self._mark_dirty()
        return self.predicted.make_loop_ref(self._pred(axis_loop))

    def _loop_nest_root(self, node: PredictedOp) -> PredictedOp:
        """Outermost loop-like ancestor of ``node`` within its function."""
        root = node
        while root.parent is not None:
            parent = self.predicted.op(*root.parent)
            if parent is None or not parent.has_trait(ScheduleOpTrait.LOOP_LIKE):
                break
            root = parent
        return root

    @_within_context
    def buffer_at(self, target: SingleTarget, axis: SingleTarget) -> BufferRef:
        buf = self._resolve_single_buffer_target(target, "buffer_at target")
        axis_loop = self._require_affine(
            self._resolve_single_loop_target(axis, "buffer_at axis")
        )
        base = buf.name or buf.owner_key
        local_key = derived_key(base, "local")

        self.script.set_callsite_loc()
        local = ta.BufferAtOp(
            self.script.any_value_type,
            self.script.match_value(buf.owner_key, buf.number, buf.source),
            self.script.match(axis_loop.key),
            **self.script.kw,
        ).result
        alloc = self.script.defining_op_handle(local)
        self.script.annotate_key(alloc, local_key)

        alloc_op = self.predicted.add_alloc(
            buf.scope, local_key, "memref.alloc", axis_loop.skey
        )
        value = self.predicted.add_value(alloc_op.skey, 0, "res")
        self._mark_dirty()
        return self.predicted.make_buffer_ref(value)

    @_within_context
    def reuse_at(
        self, target: SingleTarget, axis: SingleTarget, *, ring: bool = False
    ) -> BufferRef:
        buf = self._resolve_single_buffer_target(target, "reuse_at target")
        axis_loop = self._require_affine(
            self._resolve_single_loop_target(axis, "reuse_at axis")
        )
        base = buf.name or buf.owner_key
        reuse_key = derived_key(base, "reuse")

        self.script.set_callsite_loc()
        reuse = ta.ReuseAtOp(
            self.script.any_value_type,
            self.script.match_value(buf.owner_key, buf.number, buf.source),
            self.script.match(axis_loop.key),
            use_ring_buffer=ring,
            **self.script.kw,
        ).result
        alloc = self.script.defining_op_handle(reuse)
        self.script.annotate_key(alloc, reuse_key)

        alloc_op = self.predicted.add_alloc(
            buf.scope, reuse_key, "memref.alloc", axis_loop.skey
        )
        value = self.predicted.add_value(alloc_op.skey, 0, "res")
        self._mark_dirty()
        return self.predicted.make_buffer_ref(value)

    @_within_context
    def outline(
        self,
        target: SingleTarget,
        *,
        func_name: str,
        mapping: Sequence[int] | int | None = None,
    ) -> tuple[OpRef, OpRef]:
        if not isinstance(func_name, str) or not func_name:
            raise InvalidScheduleArgumentError("outline requires a non-empty func_name")
        source = self._resolve_single_op_target(target, "outline target")
        if self._pred(source).parent is None:
            raise InvalidScheduleArgumentError("outline cannot target the payload root")
        mapping_values = self._normalize_mapping(mapping, "outline mapping")
        call_key = derived_key(source.key, "call")

        self.script.set_callsite_loc()
        any_op = self.script.any_op_type
        kwargs = dict(self.script.kw)
        if mapping_values is not None:
            kwargs["mapping"] = mapping_values
        outlined = ta.OutlineOp(
            any_op, any_op, self.script.match(source.key), func_name, **kwargs
        )
        self.script.annotate_key(outlined.results[0], func_name)
        self.script.annotate_key(outlined.results[1], call_key)

        kernel_op = self.predicted.add_function(
            self.predicted.root_scope, func_name, None
        )
        call_op = self.predicted._add_op(
            source.scope,
            call_key,
            "func.call",
            None,
            self._pred(source).parent,
            [],
            ScheduleOpTrait(0),
            exact=False,
        )
        self._mark_dirty()
        return (
            self.predicted.make_op_ref(kernel_op),
            self.predicted.make_op_ref(call_op),
        )

    # --- application ------------------------------------------------------

    def apply(self) -> Schedule:
        if not self.dirty:
            return self
        delta = self.script.pending()
        if delta:
            entry = self.script.build_entry(delta)
            if not self.script.module.operation.verify():
                self.script.discard_entry(entry)
                raise ScheduleTransformError(
                    "transform script verification failed",
                    notes=self._transform_error_notes(),
                )
            # Run the unapplied tail on a clone of the current payload (already-applied
            # transforms are not re-run); keep `_payload` as last-good on failure.
            work = ir_ext.clone_module(self._payload)
            try:
                interpreter.apply_named_sequence(
                    work.operation,
                    entry.operation,
                    self.script.module.operation,
                )
            except Exception as exc:  # interpreter raises (no failed/err tuple)
                self.script.discard_entry(entry)
                raise ScheduleTransformError(
                    "failed to apply transform script",
                    notes=self._transform_error_notes(str(exc)),
                ) from exc
            self.script.discard_entry(entry)
            if not work.operation.verify():
                raise ScheduleTransformError(
                    "payload module verification failed after scheduling",
                    notes=self._transform_error_notes(str(work)),
                )

            schedule_d.annotate_schedule_ids(work)
            raw = schedule_d.collect_schedule_snapshot(work)
            stamped = read_schedule_keys(work)
            real = ScheduleSnapshot.from_raw(
                raw, stamped, primary_path=self._primary_path
            )
            self.predicted.reconcile(real)
            self._payload = work
            self._real = real
        self.script.commit()
        self.dirty = False
        self.query = Query(self)
        return self

    materialize = apply

    def _copy_symbol(self, name: str, id=None) -> str:
        """Callee-copy symbol for a stage kernel: ``{primary}.{name}[.{id}]``
        (the same scheme ``compose`` uses for repeat copies)."""
        sym = f"{self._primary_name}.{name}"
        return sym if id is None else f"{sym}.{id}"

    @_within_context
    def streamline(
        self,
        producer,
        consumer,
        *,
        producer_ids=None,
        consumer_ids=None,
        lanes: int = 1,
        depth: int = 2,
    ) -> Schedule:
        """Convert the DRAM memory boundaries between stage kernels into on-chip
        stream hand-offs (the ``to_stream`` fusion).

        ``producer`` and ``consumer`` are stage kernel names (as composed), each a
        single name or a list. One producer with several consumers fans the output
        out through a generated ``tee`` (residual / skip connections); several
        producers with one consumer fan in through a generated ``merge`` (each
        producer must fill a disjoint contiguous row-major block). A ``*_ids`` list
        (matching the names) selects specific repeat copies. Every memref the
        producers only write and the consumers only read (a DRAM intermediate)
        becomes a FIFO; un-convertible boundaries are skipped with a diagnostic.

        ``lanes`` widens each boundary to ``L`` parallel FIFOs moving ``L``
        elements/cycle (the bandwidth lever, for boundaries whose contiguous dim
        ``L`` divides). ``lanes=1`` (default) uses a scalar FIFO.

        ``depth`` sets the FIFO depth (default 2). On a reconvergent fork/join
        (e.g. a residual) the short branch's FIFO must hold the latency skew or
        the dataflow deadlocks; streamline warns and names the depth to set.
        """
        if not isinstance(lanes, int) or lanes <= 0:
            raise InvalidScheduleArgumentError(
                "streamline lanes must be a positive int"
            )
        if not isinstance(depth, int) or depth <= 0:
            raise InvalidScheduleArgumentError(
                "streamline depth must be a positive int"
            )

        def _handles(names, ids, which):
            names = [names] if isinstance(names, str) else list(names)
            ids = ids if isinstance(ids, (list, tuple)) else [ids] * len(names)
            if len(ids) != len(names):
                raise InvalidScheduleArgumentError(
                    f"streamline {which}_ids must match the number of {which}s"
                )
            syms = [self._copy_symbol(n, i) for n, i in zip(names, ids)]
            nodes = [self._resolve_copy(s) for s in syms]
            return syms, nodes

        p_syms, p_nodes = _handles(producer, producer_ids, "producer")
        c_syms, c_nodes = _handles(consumer, consumer_ids, "consumer")
        self.script.set_callsite_loc()
        p_handles = [self.script.match_invoke_by_callee(s) for s in p_syms]
        c_handles = [self.script.match_invoke_by_callee(s) for s in c_syms]
        ta.StreamlineOp(
            p_handles, c_handles, lanes=lanes, depth=depth, **self.script.kw
        )
        # streamline rewrites the callee signatures and the parent wiring, so the
        # precise post-structure can't be predicted: mark them approximate and
        # let reconcile rebuild from the real IR after apply().
        for n in (*p_nodes, *c_nodes):
            self.predicted.mark_approx(n)
        self.predicted.mark_approx(self._primary_pred())
        self._mark_dirty()
        return self

    def compose(self, *callees: Schedule, id=None) -> Schedule:
        """Apply each ``callee``'s whole schedule to the specialized copy of that kernel
        inside this kernel. Pass several direct callees to compose them in one call:
        ``s.compose(a, b)`` is exactly ``s.compose(a); s.compose(b)``, so every callee
        must be a kernel ``self`` calls directly (a non-direct callee has no
        ``"{primary}.{callee_primary}"`` copy and raises).

        The copy is the symbol ``"{primary}.{callee_primary}"`` (with an optional
        ``.{id}`` suffix for a specific specialized/repeat copy). Generic callees are
        concrete by construction (templates are bound before ``Kernel.schedule()``), so
        compose needs no instantiation.

        A ``callee`` may itself have composed sub-kernels: its include plan lists every
        body it realizes, keyed by the callee-relative copy symbol. Re-prefixing those
        keys onto this copy maps them to the transitive copies the compiler emits
        (e.g. ``mid.inner`` -> ``top.mid.inner``), and every body is imported, so the
        callee's full schedule runs verbatim on the matching copies."""
        if not callees:
            raise InvalidScheduleArgumentError("compose requires at least one callee")
        for callee in callees:
            self._compose(callee, id)
        return self

    def _compose(self, callee: Schedule, id) -> None:
        copy_key = self._copy_symbol(callee._primary_name, id)
        # Resolve the top-level copy up front so a missing callee always reports, even
        # when the callee schedule is empty (no includes to iterate below).
        self._resolve_copy(copy_key)

        callee_primary = callee._primary_name
        body_map = self.script.import_bodies(callee.script)
        for match_key, body_sym in callee.script.includes:
            assert match_key == callee_primary or match_key.startswith(
                callee_primary + "."
            ), f"unexpected callee include key '{match_key}'"
            new_key = copy_key + match_key[len(callee_primary) :]
            copy_node = self._resolve_copy(new_key)
            self.script.compose_include(new_key, body_map[body_sym])
            self.predicted.mark_approx(copy_node)
        self._mark_dirty()

    def _resolve_copy(self, copy_key: str) -> PredictedOp:
        node = self.predicted.op(self.predicted.root_scope, copy_key)
        if node is not None and node.has_trait(ScheduleOpTrait.FUNCTION_LIKE):
            return node
        available = sorted(
            op.key
            for op in self.predicted.ops
            if op.scope == self.predicted.root_scope
            and op.has_trait(ScheduleOpTrait.FUNCTION_LIKE)
            and op.key != self._primary_name
        )
        raise ScheduleLookupError(
            f"compose: callee copy '{copy_key}' not found in '{self._primary_name}'",
            notes=[f"available callee symbols: {available}"] if available else [],
        )

    def cleanup_schedule_ids(self) -> Schedule:
        schedule_d.cleanup_schedule_ids(self._payload)
        log_debug("removed schedule ids from payload IR")
        return self

    # --- introspection ----------------------------------------------------

    def format_tree(self, *, include_values: bool = True) -> str:
        return self._real.format_tree(include_values=include_values)

    def dump_tree(self, *, include_values: bool = True) -> str:
        text = self.format_tree(include_values=include_values)
        print(text)
        return text

    def dump_transform_script(self) -> str:
        return self.script.dump_text()

    def debug_dump(self, *, include_values: bool = True) -> Schedule:
        print("=== Schedule State ===")
        print(f"dirty={self.dirty}")
        print(f"ops={len(self._real.ops)}")
        print(f"values={len(self._real.values)}")
        print("--- last applied tree ---")
        print(self.format_tree(include_values=include_values))
        print("--- transform_script ---")
        print(self.dump_transform_script())
        return self

    # --- target resolution helpers ---------------------------------------

    def _resolve_op_targets(self, targets: Targets, desc: str) -> list[OpRef]:
        if targets is None:
            return [self.predicted.make_op_ref(self._primary_pred())]
        return [
            self.query.resolve_op(target, desc=desc)
            for target in self._targets(targets, desc)
        ]

    def _resolve_loop_targets(self, targets: Targets, desc: str) -> list[LoopRef]:
        if targets is None:
            ref = self.query.loop().one()
            assert isinstance(ref, LoopRef)
            return [ref]
        return [
            self.query.resolve_loop(target) for target in self._targets(targets, desc)
        ]

    def _resolve_buffer_targets(self, targets: Targets, desc: str) -> list[BufferRef]:
        if targets is None:
            raise InvalidScheduleArgumentError(f"{desc} requires a buffer target")
        return [
            self.query.resolve_buffer(target) for target in self._targets(targets, desc)
        ]

    def _resolve_single_op_target(
        self, target: SingleTarget | None, desc: str
    ) -> OpRef:
        if target is None:
            raise InvalidScheduleArgumentError(f"{desc} requires a target")
        ops = self._resolve_op_targets(target, desc)
        if len(ops) != 1:
            raise InvalidScheduleArgumentError(f"{desc} requires exactly one target")
        return ops[0]

    def _resolve_single_loop_target(
        self, target: SingleTarget | None, desc: str
    ) -> LoopRef:
        if target is None:
            raise InvalidScheduleArgumentError(f"{desc} requires a target")
        loops = self._resolve_loop_targets(target, desc)
        if len(loops) != 1:
            raise InvalidScheduleArgumentError(f"{desc} requires exactly one loop")
        return loops[0]

    def _resolve_single_buffer_target(
        self, target: SingleTarget | None, desc: str
    ) -> BufferRef:
        if target is None:
            raise InvalidScheduleArgumentError(f"{desc} requires a target")
        buffers = self._resolve_buffer_targets(target, desc)
        if len(buffers) != 1:
            raise InvalidScheduleArgumentError(f"{desc} requires exactly one buffer")
        return buffers[0]

    def _require_affine(self, loop: LoopRef) -> LoopRef:
        node = self._pred(loop)
        if not node.has_trait(ScheduleOpTrait.AFFINE_FOR):
            raise ScheduleTypeError(
                f"{loop.describe()} must be an affine.for loop, got kind '{loop.kind}'"
            )
        return loop

    def _targets(self, targets: Targets, desc: str) -> list[SingleTarget]:
        if isinstance(targets, (Ref, str)):
            return [targets]
        if not isinstance(targets, Iterable):
            raise InvalidScheduleArgumentError(
                f"{desc} target must be a ref, name, or iterable of refs/names, "
                f"got {type(targets).__name__}"
            )
        out = list(targets)
        if not out:
            raise InvalidScheduleArgumentError(f"{desc} requires at least one target")
        for target in out:
            if not isinstance(target, (Ref, str)):
                raise InvalidScheduleArgumentError(
                    f"{desc} target must be a ref or name, got {type(target).__name__}"
                )
        return out

    # --- predicted helpers ------------------------------------------------

    def _pred(self, ref: Ref) -> PredictedOp:
        node = self.predicted.op(ref.scope, ref.key)
        assert (
            node is not None
        ), f"{ref.describe()} is not live in the predicted snapshot"
        return node

    def _primary_pred(self) -> PredictedOp:
        node = self.predicted.op(self.predicted.root_scope, self._primary_name)
        assert node is not None, "primary function missing from predicted snapshot"
        return node

    def _op_handle(self, ref: OpRef) -> Value:
        """Transform handle for an op target. The primary function is the body's
        ``%func`` root; anything else is matched by its bare key within ``%func``."""
        if ref.scope == self.predicted.root_scope and ref.key == self._primary_name:
            return self.script.root
        return self.script.match(ref.key)

    def _predicted_mark_approx(self, ref: OpRef) -> None:
        node = self.predicted.op(ref.scope, ref.key)
        if node is not None:
            self.predicted.mark_approx(node)

    def _split_and_annotate(self, handle: Value, keys: list[str]) -> None:
        split = t.SplitHandleOp(
            [self.script.any_op_type] * len(keys), handle, **self.script.kw
        )
        for idx, key in enumerate(keys):
            self.script.annotate_key(split.results[idx], key)

    # --- misc helpers -----------------------------------------------------

    def _mark_dirty(self) -> None:
        self.dirty = True

    def _require_int(self, desc: str, value: int) -> None:
        if type(value) is not int:
            raise InvalidScheduleArgumentError(
                f"{desc} must be an int, got {type(value).__name__}"
            )

    def _normalize_tile_factors(
        self, factors: int | Iterable[int], expected: int
    ) -> list[int]:
        if type(factors) is int:
            out = [factors] * expected
        elif isinstance(factors, Iterable):
            out = list(factors)
        else:
            raise InvalidScheduleArgumentError(
                f"tile factors must be an int or iterable of ints, got "
                f"{type(factors).__name__}"
            )
        if len(out) != expected:
            raise InvalidScheduleArgumentError(
                f"tile expects {expected} factors, got {len(out)}"
            )
        for factor in out:
            self._require_int("tile factor", factor)
            if factor <= 0:
                raise InvalidScheduleArgumentError(
                    f"tile factors must be positive, got {factor}"
                )
        return out

    def _normalize_mapping(
        self, mapping: Sequence[int] | int | None, desc: str
    ) -> list[int] | None:
        if mapping is None:
            return None
        if type(mapping) is int:
            out = [mapping]
        elif isinstance(mapping, Sequence) and not isinstance(mapping, (str, bytes)):
            out = list(mapping)
        else:
            raise InvalidScheduleArgumentError(
                f"{desc} must be an int, sequence of ints, or None, got "
                f"{type(mapping).__name__}"
            )
        for value in out:
            self._require_int(desc, value)
            if value <= 0:
                raise InvalidScheduleArgumentError(
                    f"{desc} values must be positive, got {value}"
                )
        return out

    def _transform_error_notes(self, detail: str = "") -> list[str]:
        notes = []
        detail = detail.strip()
        if detail:
            notes.append(text_tail(detail, 40))
        notes.append("transform script:\n" + text_tail(str(self.script.module), 120))
        return notes

    def _partition_kind_name(self, kind) -> str:
        return {
            self.Complete: "complete",
            self.Block: "block",
            self.Cyclic: "cyclic",
        }.get(kind, str(kind))
