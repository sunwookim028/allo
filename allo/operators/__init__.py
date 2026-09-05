# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# pylint: disable=unused-argument

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, overload

from .spmw import *

# `min`, `max` and `pow` are the DSL's own names (see allo/operators/arith.py).
# pylint: disable-next=redefined-builtin
from .arith import *

from .._mlir.dialects import allo as _allo_ops
from ..lang.core import (
    AlloValue as _AlloValue,
    ShapedType as _ShapedType,
    unwrap_if_constexpr as _unwrap,
)
from ..lang.operator import operator as _operator
from ..compiler.builder import AlloOpBuilder as _AlloOpBuilder
from .utils import operator_body_unreachable as _unreachable

# `assume.nodep` enum encodings (mirror allo::AssumeDepTypeEnum / AssumeDepDirEnum).
_DEP_TYPE_VALUES = {"inter": 0, "intra": 1}
_DEP_DIR_VALUES = {"raw": 0, "war": 1, "waw": 2}

if TYPE_CHECKING:
    # Overloads exist only for editor/LSP hinting; the runtime `assume` below is
    # a single operator that dispatches on the number of positional arguments.

    @overload
    def assume(condition: bool, /) -> None:
        """Assume a boolean predicate holds (an ``llvm.assume``-style hint)."""

    @overload
    def assume(
        array: Any,
        iv: int,
        /,
        *,
        # `type=` is the documented keyword of the `assume.nodep` hint.
        # pylint: disable-next=redefined-builtin
        type: Literal["inter", "intra"] = "inter",
        direction: Literal["raw", "war", "waw"] | None = None,
        distance: int | None = None,
        dependent: bool = False,
    ) -> None:
        """Assume (no) memory dependence on ``array`` in the loop of ``iv``
        (a ``#pragma HLS dependence`` analogue)."""

    def assume(*args: Any, **kwargs: Any) -> None: ...  # overload implementation

else:

    @_operator
    def assume(*args, **kwargs):
        """Scheduler hint. Two forms, dispatched on argument count:

        - ``assume(condition)`` -- assume a boolean predicate holds
          (``allo.assume.ssa``; like ``llvm.assume``).
        - ``assume(array, iv, type=..., direction=..., distance=..., dependent=...)``
          -- assume the presence/absence of a memory dependence on ``array``,
          scoped to the loop of induction variable ``iv`` (``allo.assume.nodep``;
          the ``#pragma HLS dependence`` analogue). ``dependent=False`` (default)
          prunes a conservative dependence to recover II.

        Unchecked (undefined behavior if false); carries no hardware.
        """
        _unreachable()

    @assume.build
    def _(builder: _AlloOpBuilder, *args, **kwargs):
        # assume.ssa: a single boolean predicate.
        if len(args) == 1:
            if kwargs:
                return builder.compile_error(
                    "allo.assume(condition) takes no keyword arguments"
                )
            (cond,) = args
            if not isinstance(cond, _AlloValue):
                return builder.compile_error(
                    "allo.assume(condition) expects a runtime boolean expression"
                )
            _allo_ops.AssumeSSAOp(
                cond.handle, ip=builder.save_insertion_point(), loc=builder.get_loc()
            )
            return None

        # assume.nodep: (array, iv) + dependence attributes.
        if len(args) == 2:
            array, iv = args
            if not (
                isinstance(array, _AlloValue) and isinstance(array.type, _ShapedType)
            ):
                return builder.compile_error(
                    "allo.assume(array, iv, ...): the first argument must be an array"
                )
            if not isinstance(iv, _AlloValue):
                return builder.compile_error(
                    "allo.assume(array, iv, ...): the second argument must be a loop "
                    "induction variable"
                )
            opts = {k: _unwrap(v) for k, v in kwargs.items()}
            dep_type = opts.pop("type", "inter")
            direction = opts.pop("direction", None)
            distance = opts.pop("distance", None)
            dependent = opts.pop("dependent", False)
            if opts:
                return builder.compile_error(
                    f"allo.assume: unexpected keyword argument(s) {sorted(opts)}"
                )
            ctx = builder.context
            if dep_type not in _DEP_TYPE_VALUES:
                return builder.compile_error(
                    f"allo.assume: type must be 'inter' or 'intra', got '{dep_type}'"
                )
            dep_type_attr = _allo_ops.AssumeDepTypeAttr.get(
                _DEP_TYPE_VALUES[dep_type], ctx
            )
            dir_attr = None
            if direction is not None:
                if direction not in _DEP_DIR_VALUES:
                    return builder.compile_error(
                        "allo.assume: direction must be 'raw', 'war' or 'waw', got "
                        f"'{direction}'"
                    )
                dir_attr = _allo_ops.AssumeDepDirAttr.get(
                    _DEP_DIR_VALUES[direction], ctx
                )
            _allo_ops.AssumeNoDepOp(
                array.handle,
                iv.handle,
                dep_type_attr,
                dependent=bool(dependent),
                direction=dir_attr,
                distance=None if distance is None else int(distance),
                ip=builder.save_insertion_point(),
                loc=builder.get_loc(),
            )
            return None

        return builder.compile_error(
            "allo.assume expects assume(condition) or "
            "assume(array, iv, type=..., ...)"
        )
