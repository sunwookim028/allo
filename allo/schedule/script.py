# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Protocol

from .._mlir.ir import (
    Context,
    Value,
    StringAttr,
    Module,
    Location,
    FlatSymbolRefAttr,
    UnitAttr,
    InsertionPoint,
)
from .._mlir.dialects import transform as t
from .._mlir.dialects.transform import structured as ts
from .._mlir.dialects.transform import allo as ta

from .errors import capture_schedule_location
from .keys import SCHEDULE_KEY_ATTR_NAME
from .._mlir.schedule import SCHEDULE_NAME_ATTR_NAME
from ..compiler.builder import AlloOpBuilder


class TransformHost(Protocol):
    context: Context


# pylint: disable-next=too-many-instance-attributes
class TransformScript:
    """Builds the schedule's reusable transform program as a set of leaf *body*
    sequences plus an ordered include *plan*.

    Primitives append to the current per-apply **batch** body (``@__body_{n}``), matching
    their targets by *bare* ``allo.schedule.key`` rooted at the body's ``%func`` argument,
    so each body is a self-contained function of one function and can be re-run verbatim
    on a renamed copy (this is what makes ``.compose()`` work). The ``_plan`` is the
    ordered list of ``(match_key, body_sym)`` includes the program realizes; ``apply()``
    runs only the tail past ``_applied`` (incremental), while ``.compose()`` re-prefixes
    the whole plan. No entry sequence is kept resident — ``apply()`` builds a throwaway
    one for the unapplied tail and erases it.
    """

    def __init__(self, host: TransformHost, primary_key: str):
        self.context = host.context
        self.builder = AlloOpBuilder(self.context)
        self.builder.set_unknown_loc()
        self._primary_key = primary_key
        self._body_counter = 0
        # Ordered (match_key, body_sym) includes; the full transitive plan a parent
        # re-prefixes when it composes this schedule.
        self._plan: list[tuple[str, str]] = []
        self._applied = 0
        self._current_batch = None
        self._current_root: Value | None = None

        with self.context, Location.unknown(self.context):
            self.module = Module.create()
            self.module.operation.attributes["transform.with_named_sequence"] = (
                UnitAttr.get(self.context)
            )
            self.any_op_type = t.AnyOpType.get()
            self.any_value_type = t.AnyValueType.get()
            self.any_param_type = t.AnyParamType.get()
            # Sequences only match/transform payload under their argument; they never
            # consume the function/module handle, so it is readonly (required by
            # `transform.include` to declare an effect on the callee's operand).
            self._readonly = [{"transform.readonly": UnitAttr.get(self.context)}]

    @property
    def kw(self) -> dict:
        """``ip``/``loc`` kwargs for body op construction (before the batch's yield)."""
        return {"ip": self.builder._ip, "loc": self.builder._loc}

    @property
    def includes(self) -> list[tuple[str, str]]:
        """The full (match_key, body_sym) plan, in emission order."""
        return list(self._plan)

    @property
    def root(self) -> Value:
        """Transform handle for the function the current batch operates on; opens a new
        batch body on first use after construction/apply."""
        return self._open_batch()

    def _open_batch(self) -> Value:
        if self._current_batch is None:
            saved_loc = self.builder._loc
            unknown = Location.unknown(self.context)
            with self.context, unknown:
                self._body_counter += 1
                sym = f"__body_{self._body_counter}"
                # the MLIR op builders take an extension __init__ that pylint cannot see
                # pylint: disable-next=too-many-function-args
                batch = t.NamedSequenceOp(
                    sym,
                    [self.any_op_type],
                    [],
                    sym_visibility="private",
                    arg_attrs=self._readonly,
                    ip=InsertionPoint(self.module.body),
                    loc=unknown,
                )
                yield_op = t.YieldOp([], ip=InsertionPoint(batch.body), loc=unknown)
            self._current_batch = batch
            self._current_root = batch.bodyTarget
            self._plan.append((self._primary_key, sym))
            self.builder.restore_insertion_point(InsertionPoint(yield_op.operation))
            self.builder.set_loc(saved_loc)
        assert self._current_root is not None
        return self._current_root

    def set_callsite_loc(self) -> None:
        loc = capture_schedule_location()
        if loc is None:
            self.builder.set_unknown_loc()
            return
        self.builder.set_loc(
            Location.file(loc.file_name, loc.line, loc.col + 1, self.context)
        )

    # --- body handle matching (rooted at %func, fresh per use) ------------

    def match(self, key: str) -> Value:
        return ts.MatchOp(
            self.any_op_type,
            self.root,
            op_attrs={SCHEDULE_KEY_ATTR_NAME: StringAttr.get(key, self.context)},
            **self.kw,
        ).results[0]

    def match_invoke_by_callee(self, callee_symbol: str) -> Value:
        """Match the unique ``allo.invoke`` under the primary whose callee is
        ``callee_symbol`` (a kernel copy ``{primary}.{name}[.{id}]``)."""
        return ts.MatchOp(
            self.any_op_type,
            self.root,
            ops=["allo.invoke"],
            op_attrs={"callee": FlatSymbolRefAttr.get(callee_symbol, self.context)},
            **self.kw,
        ).results[0]

    def match_value(self, owner_key: str, number: int, source: str) -> Value:
        # The primary function is the body's %func root; `structured.match` only
        # sees ops *nested under* it, so the function's own block arguments are
        # reached from the root handle, not by matching its key.
        owner_handle = (
            self.root if owner_key == self._primary_key else self.match(owner_key)
        )
        source_kind = {"arg": 1, "res": 2}[source]
        return ta.MatchValueOp(
            self.any_value_type,
            owner_handle,
            number,
            source_kind=source_kind,
            **self.kw,
        ).result

    def defining_op_handle(self, handle: Value) -> Value:
        return t.GetDefiningOp(self.any_op_type, handle, **self.kw).result

    # --- annotation -------------------------------------------------------

    def annotate_attr(self, handle: Value, name: str, attr) -> None:
        # Upstream `transform.annotate` only attaches a param's value, so wrap the
        # static attribute in a `transform.param.constant` first.
        param = t.ParamConstantOp(self.any_param_type, attr, **self.kw).param
        t.AnnotateOp(handle, name, param=param, **self.kw)

    def _annotate(self, handle: Value, name: str, value: str) -> None:
        self.annotate_attr(handle, name, StringAttr.get(value, self.context))

    def annotate_key(self, handle: Value, key: str) -> None:
        self._annotate(handle, SCHEDULE_KEY_ATTR_NAME, key)

    def annotate_name(self, handle: Value, name: str) -> None:
        self._annotate(handle, SCHEDULE_NAME_ATTR_NAME, name)

    # --- compose ----------------------------------------------------------

    def import_bodies(self, other: TransformScript) -> dict[str, str]:
        """Copy every body sequence of ``other`` into this transform module under fresh
        private symbols, returning ``{old_sym: new_sym}``. Round-trips through text so it
        works even when the two schedules live in different MLIR contexts."""
        parsed = Module.parse(str(other.module), self.context)
        mapping: dict[str, str] = {}
        for op in parsed.body.operations:
            if op.operation.name != "transform.named_sequence":
                continue
            old = StringAttr(op.operation.attributes["sym_name"]).value
            if not old.startswith("__body"):
                continue
            cloned = op.operation.clone(InsertionPoint(self.module.body))
            self._body_counter += 1
            fresh = f"__body_{self._body_counter}"
            cloned.attributes["sym_name"] = StringAttr.get(fresh, self.context)
            mapping[old] = fresh
        return mapping

    def compose_include(self, copy_key: str, body_sym: str) -> None:
        self._plan.append((copy_key, body_sym))

    # --- incremental application ------------------------------------------

    def pending(self) -> list[tuple[str, str]]:
        return self._plan[self._applied :]

    def commit(self) -> None:
        self._applied = len(self._plan)
        self._current_batch = None

    def build_entry(self, plan_slice: list[tuple[str, str]], name: str = "__apply"):
        """Build a throwaway ``@name(%module)`` entry in this module that matches each
        plan entry's function and includes its body. Caller runs then erases it."""
        unknown = Location.unknown(self.context)
        with self.context, unknown:
            # pylint: disable-next=too-many-function-args
            entry = t.NamedSequenceOp(
                name,
                [self.any_op_type],
                [],
                arg_attrs=self._readonly,
                ip=InsertionPoint(self.module.body),
                loc=unknown,
            )
            root = entry.bodyTarget
            yield_op = t.YieldOp([], ip=InsertionPoint(entry.body), loc=unknown)
            ip = InsertionPoint(yield_op.operation)
            for match_key, body_sym in plan_slice:
                handle = ts.MatchOp(
                    self.any_op_type,
                    root,
                    op_attrs={
                        SCHEDULE_KEY_ATTR_NAME: StringAttr.get(match_key, self.context)
                    },
                    ip=ip,
                    loc=unknown,
                ).results[0]
                t.IncludeOp(
                    [],
                    FlatSymbolRefAttr.get(body_sym, self.context),
                    t.FailurePropagationMode.Propagate,
                    [handle],
                    ip=ip,
                    loc=unknown,
                )
        return entry

    @staticmethod
    def discard_entry(entry) -> None:
        entry.operation.erase()

    def dump_text(self) -> str:
        """Full program text (all bodies + a representative entry over the whole plan)."""
        entry = self.build_entry(self._plan, name="__transform_main")
        text = str(self.module)
        self.discard_entry(entry)
        return text
