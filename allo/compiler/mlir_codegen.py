# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import ast

import builtins
import copy
import operator

from contextlib import contextmanager
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Type, cast
from types import ModuleType

import numpy as np

from .._mlir import ir
from .._mlir.ir import (
    Context,
    Module as ModuleOp,
    Location,
    Value,
    FunctionType,
    Block,
    TypeAttr,
    StringAttr,
    DenseI32ArrayAttr,
    InsertionPoint,
    OpResult,
    UnitAttr,
    MLIRError,  # type: ignore
)
from .._mlir import schedule as schedule_d
from .._mlir.passmanager import PassManager
from .._mlir._mlir_libs._allo import ir_ext
from .._mlir.dialects.allo import (
    ReturnOp,
    InvokeOp,
    KernelOp,
    SIGNED_ATTR_NAME,
    LAZY_ATTR_NAME,
    register_dialect,
)
from .._mlir.dialects.cf import BranchOp, CondBranchOp
from .._mlir.dialects.scf import (
    IfOp,
    IndexSwitchOp,
    YieldOp as SCFYieldOp,
    WhileOp,
    ConditionOp,
    ParallelOp,
    ReduceOp,
)
from .._mlir.dialects.affine import AffineIfOp, AffineYieldOp
from .._mlir.dialects.arith import SelectOp
from .._mlir.dialects.ub import PoisonOp
from .builder import AlloOpBuilder
from ..lang.kernel import ConstevalFunction, Kernel, KernelOptions
from ..lang.kernel import kernel as kernel_decorator
from ..lang.core import (
    ConstexprValue,
    AlloValue,
    ValueBase,
    TypeBase,
    DType,
    ShapedType,
    BufferType,
    StreamType,
    StatefulType,
    StatefulValue,
    Range,
    Grid,
    ConstexprType,
    constexpr,
    unwrap_if_constexpr,
    index,
    bool as AlloBool,
)
from ..lang.operator import Operator, BoundOperator, NO_FOLD
from ..lang.module import Module as AlloModule
from ..operators import arith as arith_ops, memory as mem_ops
from ..operators.utils import BitSlice
from .errors import CompilationError, StaticAssertionError, InternalCompilerError


def generate_function_type(
    context: Context, arg_types: Sequence[TypeBase], res_types: Sequence[TypeBase]
) -> FunctionType:
    mlir_arg_types = []
    for ty in arg_types:
        if isinstance(ty, ConstexprType):
            continue
        mlir_arg_types.append(ty.materialize(context))
    mlir_res_types = []
    for ty in res_types:
        if isinstance(ty, StreamType):
            raise TypeError("Stream is not allowed as a kernel return type.")
        if isinstance(ty, ConstexprType):
            continue
        mlir_res_types.append(ty.materialize(context))
    return FunctionType.get(mlir_arg_types, mlir_res_types, context)


def generate_signedness_marker(
    arg_types: Sequence[TypeBase], res_types: Sequence[TypeBase]
) -> str:
    """Build the ``allo.signed`` marker: one char per MLIR func operand then
    result, in order. 's' = signed integer, 'u' = unsigned integer, 'x' =
    non-integer. The filtering mirrors ``generate_function_type`` so the marker
    length equals the function's operand + result count."""

    def sign_char(ty: TypeBase) -> str:
        if isinstance(ty, ShapedType):
            ty = ty.dtype
        if isinstance(ty, DType):
            if ty.is_int():
                return "s"
            if ty.is_uint():
                return "u"
        if isinstance(ty, StreamType):
            return sign_char(ty.base_type)
        return "x"

    chars = [sign_char(ty) for ty in arg_types if not isinstance(ty, ConstexprType)]
    chars += [
        sign_char(ty)
        for ty in res_types
        if not isinstance(ty, (ConstexprType, StreamType))
    ]
    return "".join(chars)


def _global_symbol(func_name: str, var_id: str, kind: str, node: ast.AST) -> str:
    """Canonical name for a compiler-emitted module global or helper kernel,
    shared by stateful variables (``kind="stateful"``), list/NumPy-initialized
    constants (``kind="const"``) and bufferize copy kernels (``kind="bufferize"``).
    Keyed on the source declaration -- enclosing kernel, variable, line and column
    -- so the name is stable and unique: repeated kernel instantiations resolve to
    one symbol, while distinct declarations never collide. The C++ emitter
    sanitizes it into a valid identifier."""
    return f"_allo_{kind}_{func_name}_{var_id}_l{node.lineno}c{node.col_offset}"


class ReturnPlacementChecker(ast.NodeVisitor):
    def __init__(self, src: str, file_name: str, begin_line: int):
        self.src = src
        self.file_name = file_name
        self.begin_line = begin_line
        self.function_depth = 0
        self.loop_depth = 0
        self.if_depth = 0

    def visit_FunctionDef(self, node: ast.FunctionDef):
        if self.function_depth > 0:
            return
        self.function_depth += 1
        for stmt in node.body:
            self.visit(stmt)
        self.function_depth -= 1

    def visit_Return(self, node: ast.Return):
        if self.loop_depth > 0:
            raise CompilationError(
                self.src,
                "'return' is not supported inside loops (for/grid/while).",
                node,
                file_name=self.file_name,
                begin_line=self.begin_line,
            )
        if self.if_depth > 1:
            raise CompilationError(
                self.src,
                "'return' is not supported inside nested 'if' statements.",
                node,
                file_name=self.file_name,
                begin_line=self.begin_line,
            )

    def visit_For(self, node: ast.For):
        self.loop_depth += 1
        self.generic_visit(node)
        self.loop_depth -= 1

    def visit_While(self, node: ast.While):
        self.loop_depth += 1
        self.generic_visit(node)
        self.loop_depth -= 1

    def visit_If(self, node: ast.If):
        self.if_depth += 1
        self.generic_visit(node)
        self.if_depth -= 1


@dataclass(frozen=True)
class NestedKernelSymbol:
    name: str
    node: ast.FunctionDef
    owner_func_name: str
    mapping: tuple[int, ...]


class _AffineOperands:
    """Accumulates the dim (affine IV) and symbol operands of an ``AffineMap``,
    preserving MLIR's dims-before-symbols operand order and deduplicating by SSA
    handle. ``dim``/``symbol`` return the operand's position."""

    def __init__(self):
        self.dims: list = []
        self.symbols: list = []

    @staticmethod
    def _position(store: list, value) -> int:
        for pos, existing in enumerate(store):
            if existing.handle == value.handle:
                return pos
        store.append(value)
        return len(store) - 1

    def dim(self, value) -> int:
        return self._position(self.dims, value)

    def symbol(self, value) -> int:
        return self._position(self.symbols, value)


# Sentinel for "name not found" in the scope-lookup chain.
_ABSENT = object()


class MLIRCodeGenerator(ast.NodeVisitor):
    def __init__(
        self,
        context: Context,
        module: ModuleOp,
        builder: AlloOpBuilder,
        kernel: Kernel,
        arg_types: Sequence[TypeBase],
        res_types: Sequence[TypeBase],
        func_name: str,
        file_name: str,
        begin_line: int,
        options: KernelOptions,
        gscope: dict,
        callee_context: dict[str, object] | None = None,
        fscope: dict[str, object] | None = None,
        closure_scope: dict[str, object] | None = None,
        forbidden_closure_scope: dict[str, object] | None = None,
        active_kernel_calls: list[str] | None = None,
        is_top: bool = False,
    ):
        # setup basic info
        self.context = context
        self.module = module
        self.builder = builder
        self.func_name = func_name
        self.file_name = file_name
        self.begin_line = begin_line
        self.options = options
        self.kernel = kernel
        self.mapping = tuple(kernel.mapping)
        self.arg_types = arg_types
        self.res_types = res_types
        self.is_top = is_top

        # trackers
        self.gscope = gscope
        self.lscope: dict[str, object] = (
            {} if callee_context is None else callee_context.copy()
        )
        self.fscope: dict[str, object] = {} if fscope is None else fscope.copy()
        self.closure_scope = {} if closure_scope is None else closure_scope.copy()
        self.forbidden_closure_scope = (
            {} if forbidden_closure_scope is None else forbidden_closure_scope.copy()
        )
        self._active_kernel_calls = (
            [] if active_kernel_calls is None else active_kernel_calls
        )
        self._kernel_base_names: dict[str, int] = {}
        self._entry_function_visited = False
        self.generated_func = None
        self.name_loc_prefix = None
        self.visiting_consteval_fn = False
        self.visiting_default_args = False
        self.dry_run_loop_analysis = False
        # AlloValues that are induction variables of enclosing affine loops
        # (affine.for / affine.parallel). Only these are valid affine dimensions,
        # so a memory access is affine iff every index atom is one of them, a
        # compile-time constant, or a top-level symbol. Snapshotted/restored by
        # EnterSubRegion.
        self._affine_ivs: list[AlloValue] = []
        # Kernel entry block + cache of hoisted index_cast'd symbol operands, so a
        # symbol stays a valid (top-level) affine symbol at any nesting depth.
        self._entry_block: Block | None = None
        self._affine_symbol_casts: list[tuple[AlloValue, AlloValue]] = []
        self.block_terminated = False
        self.has_explicit_return_annotation = False

        self.compile_error = self.builder.compile_error

    builtin_namespace = {
        "range": Range,
        "max": arith_ops.max,
        "min": arith_ops.min,
    }

    def _local_lookup(self, name: str, absent):
        val = self.lscope.get(name, absent)
        if val is not absent:
            return val
        return self.fscope.get(name, absent)

    def _closure_lookup(self, name: str, absent):
        val = self.closure_scope.get(name, absent)
        if val is not absent:
            return val
        val = self.fscope.get(name, absent)
        if val is not absent:
            return val
        if name in self.forbidden_closure_scope:
            captured = self.forbidden_closure_scope[name]
            captured_ty = (
                captured.type
                if isinstance(captured, ValueBase)
                else type(captured).__name__
            )
            return self.compile_error(
                f"Invalid closure capture '{name}' in kernel '{self.func_name}'. "
                "Only constexpr values, kernels, types, consteval functions, operators, and modules can be captured from outer scope, "
                f"but got '{captured_ty}'."
            )
        return absent

    def _global_lookup(self, name: str, absent):
        val = self.gscope.get(name, absent)
        if self._is_allowed_global_var(name, val, absent):
            if self._is_python_scalar_const(val):
                return ConstexprValue(val)
            return val
        return absent

    def lookup(self, name: str):
        for lookup_fn in (
            self._local_lookup,
            self._closure_lookup,
            self._global_lookup,
            self.builtin_namespace.get,
        ):
            val = lookup_fn(name, _ABSENT)
            if val is not _ABSENT:
                return val
        return self.compile_error(f"Name '{name}' is not defined in the current scope")

    def _is_global_constexpr(self, name: str):
        marker = object()
        val = self.gscope.get(name, marker)
        if val is marker:
            return False
        return isinstance(val, ConstexprValue)

    @staticmethod
    def _is_python_scalar_const(val: object):
        return isinstance(val, (builtins.int, builtins.float, builtins.bool))

    def _is_allowed_static_value(self, name: str, val: object):
        return (
            name in self.builtin_namespace
            or isinstance(val, ModuleType)
            or isinstance(val, Kernel)
            or isinstance(val, NestedKernelSymbol)
            or isinstance(val, (Operator, BoundOperator))
            or val is Range
            or val is Grid
            or isinstance(val, TypeBase)
            or isinstance(val, ConstexprValue)
            or isinstance(val, ConstevalFunction)
            # A captured NumPy array is a compile-time constant array initializer.
            or isinstance(val, np.ndarray)
        )

    def _is_allowed_global_var(self, name: str, val: object, absent):
        if val is absent:
            return False
        if name in self.builtin_namespace:
            return True
        if self.visiting_consteval_fn or self.visiting_default_args:
            # allow all global names when visiting default argument values, since we don't have good way to track the usage of default argument values and enforce the restriction only on used ones. This is a bit unsound but should be fine in practice since default argument values are usually simple and unlikely to have side effects.
            return True

        return (
            self._is_allowed_static_value(name, val)
            or self._is_global_constexpr(name)
            or self._is_python_scalar_const(val)
        )

    @contextmanager
    def _name_loc_prefix(self, prefix):
        previous = self.name_loc_prefix
        self.name_loc_prefix = prefix
        try:
            yield
        finally:
            self.name_loc_prefix = previous

    def _set_value(self, name: str, value: object):
        self.lscope[name] = value

    def _maybe_set_loc_to_name(self, name, value):
        # Attach a NameLoc to the defining op so the source name survives into the
        # IR (used by name-prefixed printing and the schedule snapshot's value
        # naming). Block arguments (kernel params, induction vars) already carry
        # their own NameLocs and have no defining op to retag.
        if isinstance(value, ValueBase):
            if value.handle is None:
                return
            handle = value.handle
            if isinstance(handle, OpResult):
                op = handle.owner
                op.location = Location.name(name, op.location)
            return
        assert isinstance(value, Value), "invalid call to _maybe_set_loc_to_name"

    def visit(self, node: ast.AST):
        if node is None:
            return

        last_node = self.builder.curr_node
        last_loc = self.builder.get_loc()
        last_src = self.builder.src
        last_file_name = self.builder.file_name
        last_begin_line = self.builder.begin_line

        # recursive visit
        self.builder.src = self.kernel.src
        self.builder.file_name = self.file_name
        self.builder.begin_line = self.begin_line
        self.builder.curr_node = node
        if hasattr(node, "lineno") and hasattr(node, "col_offset"):
            loc = Location.file(
                self.file_name,
                node.lineno + self.begin_line - 1,  # type: ignore
                node.col_offset,  # type: ignore
                self.context,
            )
            if self.name_loc_prefix is not None:
                loc = Location.name(self.name_loc_prefix, loc)
            self.builder.set_loc(loc)
        try:
            return super().visit(node)
        finally:
            # restore the builder state
            self.builder.curr_node = last_node
            self.builder.src = last_src
            self.builder.file_name = last_file_name
            self.builder.begin_line = last_begin_line
            self.builder.set_loc(last_loc)

    def generic_visit(self, node: ast.AST):
        return self.compile_error(f"Unsupported syntax: {ast.unparse(node)}")

    def visit_compound_stmts(self, stmts, allow_nested_kernel_def: bool = False):
        if not isinstance(stmts, list):
            stmts = [stmts]
        for stmt in stmts:
            if self.block_terminated:
                break
            if isinstance(stmt, ast.FunctionDef):
                if not allow_nested_kernel_def:
                    return self.compile_error(
                        "Nested kernel definitions are only supported at the top level of a kernel body."
                    )
                self.visit(stmt)
                continue
            self.visit(stmt)

    def visit_Module(self, node: ast.Module):
        ast.NodeVisitor.generic_visit(self, node)

    def visit_Pass(self, node):
        pass

    def visit_FunctionDef(self, node: ast.FunctionDef):
        if not self._entry_function_visited:
            self._entry_function_visited = True
            return self._visit_entry_function_def(node)
        return self._register_nested_kernel_def(node)

    def _visit_entry_function_def(self, node: ast.FunctionDef):
        self._precheck_return_placement(node)
        self.block_terminated = False
        self.has_explicit_return_annotation = node.returns is not None

        arg_names, _ = self.visit(node.args)
        for i, default in enumerate(node.args.defaults[::-1]):
            arg_node = node.args.args[-i - 1]
            annotation = arg_node.annotation
            name = arg_node.arg
            if name in self.lscope:
                continue
            # construct a fake assignment node to visit the default argument value
            target = ast.Name(id=name, ctx=ast.Store())
            if annotation is None:
                return self.compile_error(
                    "Default arguments must have type annotations"
                )
            init_node = ast.AnnAssign(
                target=target,
                annotation=annotation,
                value=default,
                simple=1,
            )
            try:
                self.visiting_default_args = True
                self.visit(init_node)
            finally:
                self.visiting_default_args = False

        fn_ty: FunctionType = generate_function_type(
            self.context, self.arg_types, self.res_types
        )
        visibility = "public" if self.is_top else "private"
        fn_op = KernelOp(
            self.func_name,
            TypeAttr.get(fn_ty),
            DenseI32ArrayAttr.get(list(self.mapping)),  # type: ignore
            sym_visibility=visibility,
            ip=self.builder._ip,
            loc=self.builder._loc,
        )
        self.generated_func = fn_op
        fn_op.operation.attributes[SIGNED_ATTR_NAME] = StringAttr.get(
            generate_signedness_marker(self.arg_types, self.res_types), self.context
        )
        if self.kernel._is_lazy_consteval:
            fn_op.operation.attributes[LAZY_ATTR_NAME] = UnitAttr.get(self.context)

        # Build the entry block with NameLoc-tagged arguments so the printed IR
        # shows the source parameter names (e.g. %buf) when name-loc prefixing is
        # enabled, matching the legacy behaviour.
        non_constexpr_names = [
            nm
            for nm, ty in zip(arg_names, self.arg_types)
            if not isinstance(ty, ConstexprType)
        ]
        arg_locs = [
            Location.name(nm, Location.unknown(self.context))
            for nm in non_constexpr_names
        ]
        entry_block = fn_op.regions[0].blocks.append(*fn_ty.inputs, arg_locs=arg_locs)
        self._entry_block = entry_block
        arg_handles = list(entry_block.arguments)

        arg_idx = 0
        for name, ty in zip(arg_names, self.arg_types):
            if isinstance(ty, ConstexprType):
                if not isinstance(self.lscope.get(name), ConstexprValue):
                    return self.compile_error(
                        f"Missing constexpr argument binding for parameter '{name}' in function '{self.func_name}'."
                    )
                continue
            assert arg_idx < len(arg_handles)
            handle = arg_handles[arg_idx]
            arg_idx += 1
            proxy = AlloValue(handle, ty)
            self._set_value_with_loc(name, proxy)
        assert arg_idx == len(arg_handles)

        # visit the function body
        self.builder.set_insertion_point_to_start(entry_block)
        self.visit_compound_stmts(node.body, allow_nested_kernel_def=True)

        # restore the function context
        if not self.block_terminated:
            if len(self.res_types) > 0:
                return self.compile_error(
                    "Missing return statement for non-void function. Please add a top-level return statement matching the declared return type."
                )
            ip, _ = self.builder.get_insertion_point_and_loc()
            self.builder.set_insertion_point_to_end(ip.block)
            ReturnOp([], ip=self.builder._ip, loc=self.builder._loc)
        self.builder.set_insertion_point_after(fn_op.operation)

    def _resolve_kernel_decorator(self, decorator: ast.AST):
        if isinstance(decorator, ast.Name):
            return self.gscope.get(decorator.id, self.closure_scope.get(decorator.id))
        if isinstance(decorator, ast.Attribute):
            base = unwrap_if_constexpr(self.visit(decorator.value))
            return getattr(base, decorator.attr)
        return None

    def _parse_nested_kernel_mapping(
        self, node: ast.AST | None, kernel_name: str
    ) -> tuple[int, ...]:
        if node is None:
            return ()
        values = unwrap_if_constexpr(self.visit(node))
        if isinstance(values, int):
            values = (values,)
        if not isinstance(values, tuple):
            return self.compile_error(
                f"Nested kernel '{kernel_name}' mapping must be a sequence of constant ints."
            )
        mapping = []
        for value in values:
            if not isinstance(value, int):
                return self.compile_error(
                    f"Nested kernel '{kernel_name}' mapping must be a sequence of constant ints."
                )
            mapping.append(value)
        return tuple(mapping)

    def _register_nested_kernel_def(self, node: ast.FunctionDef):
        if len(node.decorator_list) != 1:
            return self.compile_error(
                f"Nested function '{node.name}' must use exactly one '@kernel' decorator."
            )

        decorator = node.decorator_list[0]
        mapping_node = None
        if isinstance(decorator, ast.Call):
            for kw in decorator.keywords:
                if kw.arg != "mapping":
                    return self.compile_error(
                        f"Nested kernel '{node.name}' does not support decorator keyword argument '{kw.arg}'."
                    )
                mapping_node = kw.value
            decorator = decorator.func

        if self._resolve_kernel_decorator(decorator) is not kernel_decorator:
            return self.compile_error(
                f"Nested function '{node.name}' is not allowed. Only allo kernels are supported for nested definitions."
            )

        if node.name in self.lscope or node.name in self.fscope:
            return self.compile_error(
                f"Nested kernel name '{node.name}' conflicts with an existing local symbol."
            )
        self.fscope[node.name] = NestedKernelSymbol(
            name=node.name,
            node=node,
            owner_func_name=self.func_name,
            mapping=self._parse_nested_kernel_mapping(mapping_node, node.name),
        )

    def _precheck_return_placement(self, node: ast.FunctionDef):
        ReturnPlacementChecker(self.kernel.src, self.file_name, self.begin_line).visit(
            node
        )

    def visit_arguments(self, node: ast.arguments):
        args_names = [self.visit(arg) for arg in node.args]
        kwargs_names = self.visit(node.kwarg)  # type: ignore
        return args_names, kwargs_names

    def visit_arg(self, node: ast.arg):
        return node.arg

    def visit_keyword(self, node: ast.keyword):
        return node.arg, self.visit(node.value)

    def visit_Constant(self, node: ast.Constant):
        return ConstexprValue(node.value)

    def visit_Expr(self, node: ast.Expr):
        ast.NodeVisitor.generic_visit(self, node)

    def visit_Slice(self, node: ast.Slice):
        # Slices only appear as integer bit slices ``x[lo:hi]`` in this frontend.
        # The width ``hi - lo`` must be statically known (the offset may be
        # dynamic), so infer it affinely from the AST before lowering the bounds.
        if node.step is not None:
            return self.compile_error("Bit slice does not support a step.")
        lo = self.visit(node.lower) if node.lower is not None else None
        hi = self.visit(node.upper) if node.upper is not None else None
        width = self._infer_bit_slice_width(node.lower, node.upper)
        return BitSlice(lo, hi, width)

    def _infer_bit_slice_width(self, lower, upper):
        """Infer the constant bit width ``hi - lo`` of a slice, or ``None`` when
        the difference is not a compile-time constant. The offset may still be
        dynamic: ``x[i:i+2]`` has width 2 because the ``i`` terms cancel.

        Reuses the affine decomposer with a *symbolic* atom resolver: any
        non-constant sub-expression becomes an opaque placeholder dimension keyed
        by its source text, so identical offsets cancel under MLIR's affine
        simplification and the residual difference is read off as a constant."""
        if lower is None or upper is None:
            return None
        placeholders: dict[str, ir.AffineExpr] = {}

        def symbolic_atom(node):
            key = ast.unparse(node)
            expr = placeholders.get(key)
            if expr is None:
                expr = ir.AffineExpr.get_dim(len(placeholders))
                placeholders[key] = expr
            return expr

        lo = self._build_affine_expr(lower, symbolic_atom)
        hi = self._build_affine_expr(upper, symbolic_atom)
        if lo is None or hi is None:
            return None
        return self._affine_constant_value(hi - lo)

    @staticmethod
    def _affine_constant_value(expr: ir.AffineExpr):
        """Return the integer value of a (possibly simplified) constant
        ``AffineExpr``, or ``None`` if it is not constant."""
        try:
            return ir.AffineConstantExpr(expr).value
        except ValueError:
            return None

    def _try_constexpr_int(self, node):
        if isinstance(node, ast.Constant):
            return node.value if isinstance(node.value, int) else None
        if isinstance(node, ast.Name):
            val = unwrap_if_constexpr(self.lookup(node.id))
            return val if isinstance(val, int) else None
        return None

    # ----------------------------------------------------------------------
    # Affine analysis: recover an AffineExpr/AffineMap from an index/bound AST.
    # `_build_affine_expr` is a pure recursive decomposer (`+ - *(const) // / %`)
    # parameterised by an *atom* resolver for its leaves, shared by the affine
    # map builder (dims = affine IVs, symbols = top-level params) and bit-slice
    # width inference (opaque text placeholders). No `self.visit`; the only side
    # effect is hoisting symbol casts, deferred until a map is known affine.
    # ----------------------------------------------------------------------
    def _build_affine_expr(self, node: ast.AST, atom):
        """Lower an integer expression into an ``AffineExpr`` using ``atom(node)``
        for non-constant leaves (returning an ``AffineExpr`` or ``None``), or
        ``None`` if the expression is not integer-affine."""
        const = self._try_constexpr_int(node)
        if const is not None:
            return ir.AffineExpr.get_constant(const)
        if isinstance(node, ast.UnaryOp):
            if isinstance(node.op, ast.UAdd):
                return self._build_affine_expr(node.operand, atom)
            if isinstance(node.op, ast.USub):
                sub = self._build_affine_expr(node.operand, atom)
                return None if sub is None else sub * (-1)
            return None
        if isinstance(node, ast.BinOp):
            return self._build_affine_binop(node, atom)
        return atom(node)

    def _build_affine_binop(self, node: ast.BinOp, atom):
        op = node.op
        if isinstance(op, (ast.Add, ast.Sub)):
            lhs = self._build_affine_expr(node.left, atom)
            rhs = self._build_affine_expr(node.right, atom)
            if lhs is None or rhs is None:
                return None
            return lhs + rhs if isinstance(op, ast.Add) else lhs - rhs
        if isinstance(op, ast.Mult):
            # Affine multiplication requires a compile-time constant factor.
            lc = self._try_constexpr_int(node.left)
            if lc is not None:
                rhs = self._build_affine_expr(node.right, atom)
                return None if rhs is None else rhs * lc
            rc = self._try_constexpr_int(node.right)
            if rc is not None:
                lhs = self._build_affine_expr(node.left, atom)
                return None if lhs is None else lhs * rc
            return None
        if isinstance(op, (ast.FloorDiv, ast.Div, ast.Mod)):
            # The divisor/modulus must be a positive compile-time constant.
            rc = self._try_constexpr_int(node.right)
            if rc is None or rc <= 0:
                return None
            lhs = self._build_affine_expr(node.left, atom)
            if lhs is None:
                return None
            if isinstance(op, ast.Mod):
                return lhs % rc
            return ir.AffineExpr.get_floor_div(lhs, rc)
        return None

    def _affine_symbol_value(self, value):
        """A value is a valid affine symbol if it is a loop-invariant top-level
        value of the kernel — here, an integer/index kernel parameter. Returns the
        value when eligible (pure: the index cast, if any, is materialized later),
        else ``None``."""
        if not isinstance(value, AlloValue):
            return None
        owner = value.handle.owner  # Block for a block arg, Operation for a result
        if not isinstance(owner, Block) or owner != self._entry_block:
            return None
        ty = value.type
        if ty == index:
            return value
        if isinstance(ty, DType) and (ty.is_int() or ty.is_uint()):
            return value
        return None

    def _materialize_affine_symbol(self, value: AlloValue) -> AlloValue:
        """Return an ``index``-typed top-level operand for a symbol, hoisting an
        ``index_cast`` to the start of the kernel entry block (cached) so it
        remains a valid affine symbol regardless of loop nesting depth."""
        if value.type == index:
            return value
        for src, casted in self._affine_symbol_casts:
            if src.handle == value.handle:
                return casted
        assert self._entry_block is not None
        saved_ip = self.builder.save_insertion_point()
        self.builder.set_insertion_point_to_start(self._entry_block)
        casted = self.builder.cast(value, index)
        self.builder.restore_insertion_point(saved_ip)
        self._affine_symbol_casts.append((value, casted))
        return casted

    def _affine_map_atom(self, acc: "_AffineOperands"):
        """Atom resolver for real affine maps: a ``Name`` bound to an enclosing
        affine IV becomes a dim, a top-level integer/index parameter a symbol;
        anything else fails (``None``)."""

        def atom(node):
            if not isinstance(node, ast.Name):
                return None
            value = self.lookup(node.id)
            if not isinstance(value, AlloValue):
                return None
            if any(iv.handle == value.handle for iv in self._affine_ivs):
                return ir.AffineExpr.get_dim(acc.dim(value))
            if self._affine_symbol_value(value) is not None:
                return ir.AffineExpr.get_symbol(acc.symbol(value))
            return None

        return atom

    def _build_affine_value_map(self, nodes):
        """Build ``(AffineMap, operands)`` for index/bound expressions sharing one
        operand list (dims then symbols), or ``None`` if any is non-affine.
        Symbol casts are materialized only once the whole map is known affine."""
        acc = _AffineOperands()
        atom = self._affine_map_atom(acc)
        exprs = []
        for node in nodes:
            expr = self._build_affine_expr(node, atom)
            if expr is None:
                return None
            exprs.append(expr)
        operands = list(acc.dims) + [
            self._materialize_affine_symbol(sym) for sym in acc.symbols
        ]
        return ir.AffineMap.get(len(acc.dims), len(acc.symbols), exprs), operands

    def _build_single_bound(self, node):
        """Build ``(AffineMap, operands)`` for one loop bound (``None`` node means
        the constant 0), or ``None`` if the bound is not affine."""
        if node is None:
            return ir.AffineMap.get(0, 0, [ir.AffineExpr.get_constant(0)]), []
        return self._build_affine_value_map([node])

    def _affine_index_nodes(self, node: ast.Subscript):
        """Per-dimension index AST nodes of a subscript, or ``None`` if it is a
        slice/partial access (handled by the eager subview path instead)."""
        sl = node.slice
        if isinstance(sl, ast.Slice):
            return None
        if isinstance(sl, ast.Tuple):
            if any(isinstance(e, ast.Slice) for e in sl.elts):
                return None
            return list(sl.elts)
        return [sl]

    def _build_affine_access(self, buffer, node: ast.Subscript):
        """Try to express a full-rank ``buffer[idx...]`` access affinely. Returns
        ``(affine_map, operands)`` or ``None`` to signal an eager fallback."""
        if not (isinstance(buffer, AlloValue) and isinstance(buffer.type, BufferType)):
            return None
        index_nodes = self._affine_index_nodes(node)
        if index_nodes is None or len(index_nodes) != buffer.type.rank:
            return None
        return self._build_affine_value_map(index_nodes)

    @staticmethod
    def _range_bound_nodes(call: ast.Call):
        """Map a ``range(...)`` call's AST to ``(lb_node, ub_node)``; a ``None``
        node means the default lower bound 0."""
        args = call.args
        kw = {k.arg: k.value for k in call.keywords}
        start = args[0] if len(args) >= 1 else kw.get("start")
        stop = args[1] if len(args) >= 2 else kw.get("stop")
        if stop is None:  # range(stop): lower bound defaults to 0
            return None, start
        return start, stop

    @staticmethod
    def _grid_bound_nodes(call: ast.Call):
        """Per-dimension ``(lb_node, ub_node)`` for a ``grid(...)`` call's AST, or
        ``None`` if a spec is malformed. A ``None`` lb node means default 0."""
        specs = []
        for arg in call.args:
            if isinstance(arg, ast.Tuple):
                elts = arg.elts
                if len(elts) == 1:
                    specs.append((None, elts[0]))
                elif len(elts) in (2, 3):
                    specs.append((elts[0], elts[1]))
                else:
                    return None
            else:
                specs.append((None, arg))
        return specs

    # Comparison operators expressible as a single affine integer-set constraint,
    # mapped to ``(make_residual, is_equality)`` where the residual is constrained
    # ``>= 0`` (inequality) or ``== 0`` (equality). ``!=`` is a disjunction with no
    # single-constraint form and is intentionally absent (so it forces scf.if).
    _AFFINE_CONSTRAINT_OPS: dict[Type[ast.cmpop], tuple] = {
        ast.Eq: (lambda lhs, rhs: lhs - rhs, True),
        ast.LtE: (lambda lhs, rhs: rhs - lhs, False),
        ast.Lt: (lambda lhs, rhs: rhs - lhs - 1, False),
        ast.GtE: (lambda lhs, rhs: lhs - rhs, False),
        ast.Gt: (lambda lhs, rhs: lhs - rhs - 1, False),
    }

    @staticmethod
    def _flatten_affine_and(node: ast.AST):
        """Flatten an ``if`` test into the comparison nodes joined by ``and`` (an
        integer set is a conjunction of constraints), or ``None`` if it contains
        anything else (``or``, a bare value, a call, ...)."""
        if isinstance(node, ast.Compare):
            return [node]
        if isinstance(node, ast.BoolOp) and isinstance(node.op, ast.And):
            out = []
            for value in node.values:
                sub = MLIRCodeGenerator._flatten_affine_and(value)
                if sub is None:
                    return None
                out.extend(sub)
            return out
        return None

    def _build_affine_constraint(self, node: ast.Compare, atom):
        """Lower a single comparison into an integer-set constraint
        ``(residual_expr, is_equality)``, or ``None`` if its operator or operands
        are not affine."""
        if len(node.ops) != 1 or len(node.comparators) != 1:
            return None
        spec = self._AFFINE_CONSTRAINT_OPS.get(type(node.ops[0]))
        if spec is None:
            return None
        lhs = self._build_affine_expr(node.left, atom)
        rhs = self._build_affine_expr(node.comparators[0], atom)
        if lhs is None or rhs is None:
            return None
        make_residual, is_equality = spec
        return make_residual(lhs, rhs), is_equality

    def _build_affine_condition(self, test: ast.AST):
        """Try to express an ``if`` test as ``(IntegerSet, operands)`` so it can
        lower to ``affine.if``. Returns ``None`` (falling back to ``scf.if``) when
        the test is not a conjunction of affine comparisons, or has no runtime
        operand -- a constant condition must keep the constexpr branch semantics
        (compile-time branch selection)."""
        comparisons = self._flatten_affine_and(test)
        if comparisons is None:
            return None
        acc = _AffineOperands()
        atom = self._affine_map_atom(acc)
        exprs = []
        eq_flags = []
        for comparison in comparisons:
            built = self._build_affine_constraint(comparison, atom)
            if built is None:
                return None
            residual, is_equality = built
            exprs.append(residual)
            eq_flags.append(is_equality)
        if len(acc.dims) + len(acc.symbols) == 0:
            return None
        operands = list(acc.dims) + [
            self._materialize_affine_symbol(sym) for sym in acc.symbols
        ]
        integer_set = ir.IntegerSet.get(
            len(acc.dims), len(acc.symbols), exprs, eq_flags
        )
        return integer_set, operands

    def visit_Compare(self, node: ast.Compare):
        if not (len(node.ops) == 1 and len(node.comparators) == 1):
            return self.compile_error(
                "simultaneous multi-way comparisons are not supported"
            )
        lhs = self.visit(node.left)
        rhs = self.visit(node.comparators[0])
        library_op = self._available_comparison_methods.get(type(node.ops[0]), None)
        if library_op is None:
            return self.compile_error(
                f"Unsupported comparison operator '{type(node.ops[0]).__name__}' in allo kernel functions",
            )
        return self.call_operator(library_op, [lhs, rhs])

    _available_comparison_methods: dict[Type[ast.cmpop], Operator] = {
        ast.Eq: arith_ops.eq,
        ast.NotEq: arith_ops.ne,
        ast.Lt: arith_ops.lt,
        ast.LtE: arith_ops.le,
        ast.Gt: arith_ops.gt,
        ast.GtE: arith_ops.ge,
    }

    def _ast_expr_may_be_float(self, node: ast.AST) -> bool:
        if isinstance(node, ast.Constant):
            return isinstance(node.value, float)

        if isinstance(node, ast.Name):
            val = unwrap_if_constexpr(self.lookup(node.id))
            if isinstance(val, AlloValue):
                return isinstance(val.dtype, DType) and val.dtype.is_float()
            return isinstance(val, float)

        if isinstance(node, ast.Subscript):
            return self._ast_expr_may_be_float(node.value)

        if isinstance(node, ast.UnaryOp):
            return self._ast_expr_may_be_float(node.operand)

        if isinstance(node, ast.BinOp):
            return self._ast_expr_may_be_float(
                node.left
            ) or self._ast_expr_may_be_float(node.right)

        if isinstance(node, ast.Call):
            return True

        return False

    def _materialize_constexpr_pair(self, lhs, rhs):
        if isinstance(lhs, ConstexprValue) and isinstance(rhs, ConstexprValue):
            return lhs, rhs
        if isinstance(lhs, ConstexprValue):
            assert isinstance(rhs, AlloValue)
            lhs = self.builder.cast(lhs, rhs.dtype)
        if isinstance(rhs, ConstexprValue):
            assert isinstance(lhs, AlloValue)
            rhs = self.builder.cast(rhs, lhs.dtype)
        return lhs, rhs

    def _prepare_binary_operands(
        self, lhs: AlloValue, rhs: AlloValue, op_name: str
    ) -> tuple[AlloValue, AlloValue]:
        assert isinstance(lhs, AlloValue) and isinstance(rhs, AlloValue)
        term_signs = [1, -1] if op_name == "sub" else None
        dst_ty = self.builder.get_promoted_dtype_nary(
            op_name, [lhs.dtype, rhs.dtype], term_signs=term_signs
        )
        lhs = self.builder.cast_to_dtype(lhs, dst_ty)
        rhs = self.builder.cast_to_dtype(rhs, dst_ty)
        return self.builder.broadcast_pair(lhs, rhs)

    # Direct (HLS) lowering for +/-/*: constexpr-fold function and the library
    # operator used for shaped operands. Scalar operands dispatch to
    # ``builder.create_{op_name}``.
    _DIRECT_BINARY_OPS = {
        "add": (operator.add, arith_ops.add),
        "sub": (operator.sub, arith_ops.sub),
        "mul": (operator.mul, arith_ops.mul),
    }

    def _lower_direct_binary(self, op_name: str, lhs, rhs):
        fold, library_op = self._DIRECT_BINARY_OPS[op_name]

        if isinstance(lhs, ConstexprValue) and isinstance(rhs, ConstexprValue):
            return ConstexprValue(fold(lhs.value, rhs.value))

        lhs, rhs = self._materialize_constexpr_pair(lhs, rhs)
        if not (isinstance(lhs, AlloValue) and isinstance(rhs, AlloValue)):
            return self.compile_error(
                f"Binary operator '{op_name}' expects runtime values to be AlloValues"
            )

        if isinstance(lhs.type, ShapedType) or isinstance(rhs.type, ShapedType):
            return self.call_operator(library_op, [lhs, rhs])

        lhs, rhs = self._prepare_binary_operands(lhs, rhs, op_name)
        assert isinstance(lhs.dtype, DType)
        return getattr(self.builder, f"create_{op_name}")(
            lhs, rhs, floating=lhs.dtype.is_float()
        )

    def _lower_binary_values(self, op: ast.operator, lhs, rhs):
        if isinstance(op, ast.Add):
            return self._lower_direct_binary("add", lhs, rhs)
        if isinstance(op, ast.Sub):
            return self._lower_direct_binary("sub", lhs, rhs)
        if isinstance(op, ast.Mult):
            return self._lower_direct_binary("mul", lhs, rhs)

        library_op = self._available_binary_methods.get(type(op), None)
        if library_op is None:
            return self.compile_error(
                f"Unsupported binary operator '{type(op).__name__}' in allo kernel functions",
            )
        return self.call_operator(library_op, [lhs, rhs])

    def _lower_binop_tree(self, node: ast.BinOp):
        def lower_expr(expr):
            if isinstance(expr, ast.BinOp):
                lhs = lower_expr(expr.left)
                rhs = lower_expr(expr.right)
                return self._lower_binary_values(expr.op, lhs, rhs)
            return self.visit(expr)

        return lower_expr(node)

    def _collect_add_sub_terms(self, node: ast.AST, sign: int, out):
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            self._collect_add_sub_terms(node.left, sign, out)
            self._collect_add_sub_terms(node.right, sign, out)
            return
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Sub):
            self._collect_add_sub_terms(node.left, sign, out)
            self._collect_add_sub_terms(node.right, -sign, out)
            return
        out.append((self.visit(node), sign))

    def _collect_mul_terms(self, node: ast.AST, out):
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
            self._collect_mul_terms(node.left, out)
            self._collect_mul_terms(node.right, out)
            return
        out.append(self.visit(node))

    def _materialize_constexpr_terms(self, terms):
        anchor = None
        for term in terms:
            if isinstance(term, AlloValue):
                anchor = term.dtype
                break
        if anchor is None:
            return terms

        materialized = []
        for term in terms:
            if isinstance(term, ConstexprValue):
                materialized.append(self.builder.cast(term, anchor))
            else:
                materialized.append(term)
        return materialized

    def _lower_nary_add_sub(self, node: ast.BinOp):
        signed_terms = []
        self._collect_add_sub_terms(node, sign=1, out=signed_terms)
        values = [value for value, _ in signed_terms]
        signs = [sign for _, sign in signed_terms]

        if all(isinstance(value, ConstexprValue) for value in values):
            total = 0
            for value, sign in zip(values, signs):
                total += sign * value.value
            return ConstexprValue(total)

        values = self._materialize_constexpr_terms(values)
        if not all(isinstance(value, AlloValue) for value in values):
            return self.compile_error(
                "n-ary add/sub lowering expects runtime values to be AlloValues"
            )

        if all(isinstance(value.type, DType) for value in values):
            dtypes = [value.dtype for value in values]
            op_name = "sub" if any(sign < 0 for sign in signs) else "add"
            dst_ty = self.builder.get_promoted_dtype_nary(
                op_name, dtypes, term_signs=signs
            )
            casted = [self.builder.cast_to_dtype(value, dst_ty) for value in values]
            floating = dst_ty.is_float()
            if any(sign < 0 for sign in signs):
                return self.builder.create_sub_nary(casted, signs, floating=floating)
            return self.builder.create_add_nary(casted, floating=floating)

        if all(sign > 0 for sign in signs):
            return self.builder.reduce_balanced(
                values,
                lambda lhs, rhs: self.call_operator(arith_ops.add, [lhs, rhs]),
            )

        result = None
        for value, sign in zip(values, signs):
            if result is None:
                result = (
                    value
                    if sign > 0
                    else self.call_operator(arith_ops.sub, [ConstexprValue(0), value])
                )
            elif sign > 0:
                result = self.call_operator(arith_ops.add, [result, value])
            else:
                result = self.call_operator(arith_ops.sub, [result, value])
        assert result is not None
        return result

    def _lower_nary_mul(self, node: ast.BinOp):
        terms = []
        self._collect_mul_terms(node, terms)

        if all(isinstance(term, ConstexprValue) for term in terms):
            product = 1
            for term in terms:
                product *= term.value
            return ConstexprValue(product)

        terms = self._materialize_constexpr_terms(terms)
        if not all(isinstance(term, AlloValue) for term in terms):
            return self.compile_error(
                "n-ary mul lowering expects runtime values to be AlloValues"
            )

        if all(isinstance(term.type, DType) for term in terms):
            dtypes = [term.dtype for term in terms]
            dst_ty = self.builder.get_promoted_dtype_nary("mul", dtypes)
            casted = [self.builder.cast_to_dtype(term, dst_ty) for term in terms]
            return self.builder.create_mul_nary(casted, floating=dst_ty.is_float())

        return self.builder.reduce_balanced(
            terms,
            lambda lhs, rhs: self.call_operator(arith_ops.mul, [lhs, rhs]),
        )

    def visit_BinOp(self, node):
        if self.builder.typing_style == "hls":
            if (
                not self.options.fast_math
                and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult))
                and self._ast_expr_may_be_float(node)
            ):
                return self._lower_binop_tree(node)
            if isinstance(node.op, (ast.Add, ast.Sub)):
                return self._lower_nary_add_sub(node)
            if isinstance(node.op, ast.Mult):
                return self._lower_nary_mul(node)

        lhs = self.visit(node.left)
        rhs = self.visit(node.right)
        library_op = self._available_binary_methods.get(type(node.op), None)
        if library_op is None:
            return self.compile_error(
                f"Unsupported binary operator '{type(node.op).__name__}' in allo kernel functions",
            )
        return self.call_operator(library_op, [lhs, rhs])

    _available_binary_methods: dict[Type[ast.operator], Operator] = {
        ast.Add: arith_ops.add,
        ast.Sub: arith_ops.sub,
        ast.Mult: arith_ops.mul,
        ast.Div: arith_ops.div,
        ast.FloorDiv: arith_ops.floordiv,
        ast.Mod: arith_ops.mod,
        ast.Pow: arith_ops.pow,
        ast.LShift: arith_ops.lshift,
        ast.RShift: arith_ops.rshift,
        ast.BitAnd: arith_ops.bitwise_and,
        ast.BitOr: arith_ops.bitwise_or,
        ast.BitXor: arith_ops.bitwise_xor,
    }

    def visit_UnaryOp(self, node):
        operand = self.visit(node.operand)
        fn = self._available_unary_methods.get(type(node.op), None)
        if fn is None:
            return self.compile_error(
                f"Unsupported unary operator '{type(node.op).__name__}' in allo kernel functions",
            )
        return self.call_operator(fn, [operand])

    _available_unary_methods: dict[Type[ast.unaryop], Operator] = {
        ast.UAdd: arith_ops.pos,
        ast.USub: arith_ops.neg,
        ast.Not: arith_ops.logical_not,
        ast.Invert: arith_ops.invert,
    }

    def visit_BoolOp(self, node):
        library_op = self._available_boolop_methods.get(type(node.op), None)
        if library_op is None:
            return self.compile_error(
                f"Unsupported boolean operator '{type(node.op).__name__}' in allo kernel functions",
            )
        nontrivial_values = []

        for subnode in node.values:
            value = self.visit(subnode)
            if isinstance(value, ConstexprValue):
                # constant folding
                bv = bool(unwrap_if_constexpr(value))
                if (bv is False) and (library_op is arith_ops.logical_and):
                    return ConstexprValue(False)
                if (bv is True) and (library_op is arith_ops.logical_or):
                    return ConstexprValue(True)
                # otherwise constexpr has no effect, so can be skipped
            elif isinstance(value, AlloValue) and isinstance(value.type, ShapedType):
                return self.compile_error(
                    "non-scalar values are not supported in boolean operations"
                )
            else:
                nontrivial_values.append(value)

        if len(nontrivial_values) == 0:
            # all values are constant folded
            if library_op == arith_ops.logical_and:
                return ConstexprValue(True)
            else:
                return ConstexprValue(False)

        while len(nontrivial_values) >= 2:
            # reduce from left to right
            rhs = nontrivial_values.pop()
            lhs = nontrivial_values.pop()
            res = self.call_operator(library_op, [lhs, rhs])
            nontrivial_values.append(res)

        assert len(nontrivial_values) == 1
        return nontrivial_values[0]

    _available_boolop_methods: dict[Type[ast.boolop], Operator] = {
        ast.And: arith_ops.logical_and,
        ast.Or: arith_ops.logical_or,
    }

    def visit_Break(self, node):
        return self.compile_error(
            "'break' statement is not supported in allo kernel functions"
        )

    def visit_Continue(self, node):
        return self.compile_error(
            "'continue' statement is not supported in allo kernel functions"
        )

    def visit_Return(self, node: ast.Return):
        if node.value is None or (
            isinstance(node.value, ast.Constant) and node.value.value is None
        ):
            return_vals = []
        elif isinstance(node.value, ast.Tuple):
            return_vals = [self.visit(elt) for elt in node.value.elts]
        else:
            return_vals = [self.visit(node.value)]

        if len(return_vals) > 0 and not self.has_explicit_return_annotation:
            return self.compile_error(
                "Return values require an explicit return annotation."
            )
        if len(return_vals) != len(self.res_types):
            return self.compile_error(
                f"Return value count mismatch: expected {len(self.res_types)}, got {len(return_vals)}."
            )

        coerced = []
        for value, dst_type in zip(return_vals, self.res_types):
            if not isinstance(value, (AlloValue, ConstexprValue)):
                return self.compile_error(
                    f"Unsupported return value '{value}' of type '{type(value).__name__}'."
                )
            coerced.append(self.builder.cast(value, dst_type))

        ReturnOp(
            [value.handle for value in coerced],
            ip=self.builder._ip,
            loc=self.builder._loc,
        )
        self.block_terminated = True

    def _visit_if_with_return_impl(
        self, cond: AlloValue, node: ast.If, then_has_return, else_has_return
    ):
        continue_vals = None
        end_if = None
        if_terminated = False
        with EnterSubRegion(self):
            ip, last_loc = self.builder.get_insertion_point_and_loc()
            parent_region = ip.block.region
            then_block = self.builder.create_block(parent_region)
            else_block = self.builder.create_block(parent_region)
            end_if = self.builder.create_block(parent_region)

            # branch out from current block to then/else
            self.builder.set_insertion_point_and_loc(ip, last_loc)
            CondBranchOp(
                cond.handle,
                [],
                [],
                then_block,
                else_block,
                ip=self.builder._ip,
                loc=self.builder._loc,
            )

            liveins = self.lscope.copy()

            # then branch
            self.lscope = liveins.copy()
            self.builder.set_insertion_point_to_start(then_block)
            self.block_terminated = False
            self.visit_compound_stmts(node.body)
            then_vals = self.lscope.copy()
            then_terminated = self.block_terminated

            # else branch
            self.lscope = liveins.copy()
            self.builder.set_insertion_point_to_start(else_block)
            self.block_terminated = False
            if node.orelse:
                self.visit_compound_stmts(node.orelse)
                else_vals = self.lscope.copy()
            else:
                else_vals = liveins.copy()
            else_terminated = self.block_terminated

            # if both branches return, there is no fallthrough path
            if then_terminated and else_terminated:
                continue_vals = liveins
                ir_ext.erase_block(end_if)
                if_terminated = True

            # if exactly one branch returns, continue with the non-returning branch.
            elif then_terminated and not else_terminated:
                self.builder.set_insertion_point_to_end(else_block)
                BranchOp([], end_if, ip=self.builder._ip, loc=self.builder._loc)
                continue_vals = else_vals

            elif not then_terminated and else_terminated:
                self.builder.set_insertion_point_to_end(then_block)
                BranchOp([], end_if, ip=self.builder._ip, loc=self.builder._loc)
                continue_vals = then_vals

            else:
                assert not (then_has_return or else_has_return)

        assert end_if is not None and continue_vals is not None
        self.block_terminated = if_terminated
        if if_terminated:
            self.lscope = continue_vals.copy()
            return
        self.builder.set_insertion_point_to_start(end_if)
        self.lscope = continue_vals.copy()

    def visit_if_impl(self, cond, node: ast.If, affine_cond=None):
        """Lower a non-returning ``if`` to ``scf.if`` (runtime ``cond`` value) or,
        when ``affine_cond`` is an ``(IntegerSet, operands)`` pair, to
        ``affine.if``. Branch visiting and phi handling are identical; only the op
        and its yield terminator differ."""
        with EnterSubRegion(self):
            ip, last_loc = self.builder.get_insertion_point_and_loc()

            parent_region = ip.block.region
            then_block = self.builder.create_block(parent_region)
            else_block = self.builder.create_block(parent_region)

            # compute phi arguments
            phi_names, phi_types, then_handles, else_handles = (
                self._visit_then_else_block(node, then_block, else_block)
            )

            # create if op
            self.builder.set_insertion_point_and_loc(ip, last_loc)
            # if we have phi arguments, we must create else region
            has_else = len(node.orelse) > 0 or len(phi_names) > 0
            phi_ir_types = [ty.materialize(self.context) for ty in phi_types]
            if affine_cond is not None:
                integer_set, operands = affine_cond
                if_op = AffineIfOp(
                    integer_set,
                    phi_ir_types,
                    cond_operands=[v.handle for v in operands],
                    has_else=has_else,
                    ip=self.builder._ip,
                    loc=self.builder._loc,
                )
                yield_cls = AffineYieldOp
            else:
                if_op = IfOp(
                    cond.handle,
                    phi_ir_types,
                    has_else=has_else,
                    ip=self.builder._ip,
                    loc=self.builder._loc,
                )
                yield_cls = SCFYieldOp
            ir_ext.merge_block_before(then_block, if_op.then_block)
            then_block = if_op.then_block
            assert then_block is not None
            self.builder.set_insertion_point_to_end(then_block)
            yield_cls(then_handles, ip=self.builder._ip, loc=self.builder._loc)
            if has_else:
                ir_ext.merge_block_before(else_block, if_op.else_block)
                else_block = if_op.else_block
                assert else_block is not None
                self.builder.set_insertion_point_to_end(else_block)
                yield_cls(else_handles, ip=self.builder._ip, loc=self.builder._loc)
            else:
                ir_ext.erase_block(else_block)

        # update lscope with phi results
        res_handles = list(if_op.results)
        phi_proxies = [
            AlloValue(handle, ty) for handle, ty in zip(res_handles, phi_types)
        ]
        for name, proxy in zip(phi_names, phi_proxies):
            self._set_value_with_loc(name, proxy)

    def _visit_then_else_block(
        self, node: ast.If, then_block: Block, else_block: Block
    ):
        # get a copy of current live-ins
        liveins = self.lscope.copy()
        # visit then block
        self.lscope = liveins.copy()
        self.builder.set_insertion_point_to_start(then_block)
        self.visit_compound_stmts(node.body)
        then_vals = self.lscope.copy()  # capture live-ins in then block
        # restore lscope for else visiting
        self.lscope = liveins.copy()
        # visit else block
        self.builder.set_insertion_point_to_start(else_block)
        if node.orelse:
            self.visit_compound_stmts(node.orelse)
            else_vals = self.lscope.copy()  # capture live-ins in else block
        else:
            else_vals = liveins.copy()

        # compute phi arguments
        phi_names = []
        phi_types: list[TypeBase] = []
        then_handles = []
        else_handles = []
        for name, value in liveins.items():
            then_proxy = then_vals.get(name, value)
            else_proxy = else_vals.get(name, value)
            if not isinstance(then_proxy, AlloValue) or not isinstance(
                else_proxy, AlloValue
            ):
                continue
            then_handle = then_proxy.handle
            else_handle = else_proxy.handle
            if then_handle == else_handle:
                continue  # value is not redefined in either block, no need for phi
            # type check
            if isinstance(value, ConstexprValue):
                return self.compile_error(
                    f"Variable '{name}' is defined as a constexpr in the outer scope, but is assigned to non-constexpr values in the then vs else branches."
                )
            assert isinstance(value, AlloValue)
            outer_ty = value.type
            then_ty = then_proxy.type
            else_ty = else_proxy.type
            if then_ty != else_ty or then_ty != outer_ty:
                return self.compile_error(
                    f"Variable '{name}' has incompatible types in outer scope vs then vs else branches: {outer_ty} vs {then_ty} vs {else_ty}."
                )
            phi_types.append(then_proxy.type)
            phi_names.append(name)
            then_handles.append(then_handle)
            else_handles.append(else_handle)
        return phi_names, phi_types, then_handles, else_handles

    def visit_IfExp(self, node: ast.IfExp):
        cond = self.visit(node.test)
        if isinstance(cond, AlloValue):
            cond = self.builder.to_condition(cond)
            # if exp cannot define new variables
            ip, last_loc = self.builder.get_insertion_point_and_loc()

            then_val = self.visit(node.body)
            else_val = self.visit(node.orelse)

            # type check
            # Case 1: both branches are constexprs
            then_is_constexpr = isinstance(then_val, ConstexprValue)
            else_is_constexpr = isinstance(else_val, ConstexprValue)
            if then_is_constexpr and else_is_constexpr:
                return self.compile_error(
                    "Cannot deduce the type of a ternary whose branches are both "
                    "compile-time constants, as the bit-width would have to be guessed. "
                    "Annotate one branch with its intended type (e.g. `cast(<value>, "
                    "int32)`) or assign through a typed variable."
                )
            # Case 2: both branches are AlloValues:
            if not then_is_constexpr and not else_is_constexpr:
                if then_val.type != else_val.type:
                    return self.compile_error(
                        f"Type mismatch between then vs else branches of ternary expression: {then_val.type} vs {else_val.type}."
                    )
            # Case 3: exactly one branch is a constexpr, use the other branch's type as the result type
            res_type = then_val.type if not then_is_constexpr else else_val.type
            if then_is_constexpr:
                then_val = self.builder.cast(then_val, res_type)
            if else_is_constexpr:
                else_val = self.builder.cast(else_val, res_type)

            # create select op
            self.builder.set_insertion_point_and_loc(ip, last_loc)
            sel_op = SelectOp(
                cond.handle,
                then_val.handle,
                else_val.handle,
                ip=self.builder._ip,
                loc=self.builder._loc,
            )
            return AlloValue(sel_op.result, res_type)
        else:
            # constexpr path
            cond = self._unwrap_constexpr_condition(cond, "Ternary expression")
            selected = node.body if cond else node.orelse
            return self.visit(selected)

    _condition_types = {
        bool,
        int,
        type(None),
    }

    def _unwrap_constexpr_condition(self, cond, context: str):
        """Unwrap a constexpr branch condition to a Python bool/int/None, or raise
        a compile error naming the accepted condition types."""
        assert isinstance(cond, ConstexprValue)
        value = unwrap_if_constexpr(cond)
        if type(value) not in self._condition_types:
            allowed = ", ".join(t.__name__ for t in self._condition_types)
            return self.compile_error(
                f"{context} conditionals can only accept values of type "
                f"{{{allowed}}}, not objects of type {type(value).__name__}."
            )
        return value

    def _branch_has_return(self, stmts):
        # TODO: maybe a better checking
        return any(isinstance(stmt, ast.Return) for stmt in stmts)

    def visit_If(self, node: ast.If):
        then_has_return = self._branch_has_return(node.body)
        else_has_return = self._branch_has_return(node.orelse)
        # A runtime condition that is a conjunction of affine relations over
        # enclosing loop IVs / top-level symbols lowers to affine.if, keeping the
        # branch bodies in the affine domain. Returning branches need the cf-based
        # path, so they are excluded.
        if not (then_has_return or else_has_return):
            affine_cond = self._build_affine_condition(node.test)
            if affine_cond is not None:
                return self.visit_if_impl(None, node, affine_cond=affine_cond)
        cond = self.visit(node.test)
        if isinstance(cond, AlloValue):
            cond = self.builder.to_condition(cond)
            if then_has_return or else_has_return:
                self._visit_if_with_return_impl(
                    cond, node, then_has_return, else_has_return
                )
            else:
                self.visit_if_impl(cond, node)
        else:
            # constexpr path
            cond = self._unwrap_constexpr_condition(cond, "`if`")
            selected = node.body if cond else node.orelse
            self.visit_compound_stmts(selected)

    @staticmethod
    def _is_wildcard_pattern(pattern) -> bool:
        # `case _:` parses to a capture-less, name-less MatchAs.
        return (
            isinstance(pattern, ast.MatchAs)
            and pattern.pattern is None
            and pattern.name is None
        )

    def _match_case_value(self, pattern) -> int:
        """Fold an integer-literal `case <int>:` pattern to a Python int."""
        if not isinstance(pattern, ast.MatchValue):
            return self.compile_error(
                "Only integer-literal patterns (`case <int>:`) and the wildcard "
                "(`case _:`) are supported in match statements."
            )
        value = unwrap_if_constexpr(self.visit(pattern.value))
        if isinstance(value, bool) or not isinstance(value, int):
            return self.compile_error(
                "match case patterns must be compile-time integer constants."
            )
        return value

    def visit_Match(self, node: ast.Match):
        subject = self.visit(node.subject)
        if not isinstance(subject, AlloValue) or not isinstance(subject.type, DType):
            return self.compile_error("match subject must be a runtime value.")
        if not (subject.type.is_int_signless() or subject.type.is_index()):
            return self.compile_error("match is only supported on integer subjects.")
        # scf.index_switch requires an `index`-typed argument.
        arg = self.builder.scalar_cast(subject, index)

        case_values: list[int] = []
        case_bodies: list = []
        default_body: list | None = None
        for case in node.cases:
            if case.guard is not None:
                return self.compile_error(
                    "guards (`case ... if ...:`) are not supported in match statements."
                )
            if self._branch_has_return(case.body):
                return self.compile_error(
                    "'return' is not supported inside match cases."
                )
            if self._is_wildcard_pattern(case.pattern):
                if default_body is not None:
                    return self.compile_error(
                        "match can only have a single wildcard `case _:`."
                    )
                default_body = case.body
            else:
                value = self._match_case_value(case.pattern)
                if value in case_values:
                    return self.compile_error(f"duplicate match case value {value}.")
                case_values.append(value)
                case_bodies.append(case.body)

        # Visit each region body once in a temporary block (default first, then
        # the case regions aligned with case_values). Scalar live-ins reassigned
        # in any region are threaded out as scf.index_switch results (phi); the
        # result types must be known at op creation, so the op is built only
        # after visiting the bodies and each temp block is then spliced in.
        region_bodies = [default_body or []] + case_bodies
        saved_terminated = self.block_terminated
        with EnterSubRegion(self):
            ip, last_loc = self.builder.get_insertion_point_and_loc()
            parent_region = ip.block.region
            liveins = self.lscope.copy()

            temp_blocks = []
            region_scopes = []
            for body in region_bodies:
                self.lscope = liveins.copy()
                block = self.builder.create_block(parent_region)
                self.builder.set_insertion_point_to_start(block)
                self.block_terminated = False
                self.visit_compound_stmts(body)
                temp_blocks.append(block)
                region_scopes.append(self.lscope.copy())

            phi_names, phi_types, region_handles = self._compute_match_phi(
                liveins, region_scopes
            )

            self.builder.set_insertion_point_and_loc(ip, last_loc)
            phi_ir_types = [ty.materialize(self.context) for ty in phi_types]
            # The wrapper creates one (empty) block per region: regions[0] is the
            # default region, the rest are the case regions aligned with cases.
            switch_op = IndexSwitchOp(
                phi_ir_types,
                arg.handle,
                case_values,
                ip=self.builder._ip,
                loc=self.builder._loc,
            )
            region_blocks = [switch_op.default_block] + [
                switch_op.case_block(i) for i in range(len(case_values))
            ]
            for temp_block, region_block, handles in zip(
                temp_blocks, region_blocks, region_handles
            ):
                ir_ext.merge_block_before(temp_block, region_block)
                self.builder.set_insertion_point_to_end(region_block)
                SCFYieldOp(handles, ip=self.builder._ip, loc=self.builder._loc)
        self.block_terminated = saved_terminated

        # bind phi results in the enclosing scope
        for name, handle, ty in zip(phi_names, switch_op.results, phi_types):
            self._set_value_with_loc(name, AlloValue(handle, ty))

    def _compute_match_phi(self, liveins, region_scopes):
        """Find scalar live-ins reassigned in any region and the value each
        region carries for them, mirroring the then/else phi logic of ``if``. A
        region that does not redefine a name yields the (dominating) live-in."""
        phi_names: list[str] = []
        phi_types: list[TypeBase] = []
        region_handles: list[list] = [[] for _ in region_scopes]
        for name, value in liveins.items():
            if not isinstance(value, AlloValue):
                continue
            proxies = [scope.get(name, value) for scope in region_scopes]
            if any(not isinstance(p, AlloValue) for p in proxies):
                continue
            if all(p.handle == value.handle for p in proxies):
                continue  # not redefined in any region
            for proxy in proxies:
                if proxy.type != value.type:
                    return self.compile_error(
                        f"Variable '{name}' has incompatible types across match "
                        f"cases: {value.type} vs {proxy.type}."
                    )
            phi_names.append(name)
            phi_types.append(value.type)
            for handles, proxy in zip(region_handles, proxies):
                handles.append(proxy.handle)
        return phi_names, phi_types, region_handles

    def visit_Attribute(self, node):
        lhs = self.visit(node.value)
        try:
            attr = getattr(lhs, node.attr)
        except AttributeError:
            if isinstance(lhs, ConstexprValue):
                try:
                    attr = getattr(lhs.value, node.attr)
                except AttributeError:
                    return self.compile_error(
                        f"constexpr value '{lhs.value}' has no attribute '{node.attr}'."
                    )
            else:
                lhs_type = (
                    lhs.type if isinstance(lhs, ValueBase) else type(lhs).__name__
                )
                return self.compile_error(
                    f"Object of type '{lhs_type}' has no attribute '{node.attr}'."
                )

        if isinstance(attr, BoundOperator):
            return attr
        if isinstance(attr, Operator):
            if isinstance(lhs, ValueBase):
                return BoundOperator(attr, lhs)
            return attr
        return attr

    def visit_Subscript(self, node: ast.Subscript):
        return self.visit_Subscript_Load(node)

    @staticmethod
    def _as_index_tuple(slices):
        """Normalize parsed subscript slices to a tuple: a single scalar index
        becomes a 1-tuple, an existing tuple is returned unchanged."""
        if isinstance(slices, (AlloValue, ConstexprValue)):
            return (slices,)
        return slices

    def visit_Subscript_Store(self, node, value):
        assert isinstance(node.ctx, ast.Store)
        lhs = self.visit(node.value)
        built = self._build_affine_access(lhs, node)
        if built is not None:
            affine_map, operands = built
            val = self.builder.cast(value, lhs.dtype)
            self.builder.create_affine_store(val, lhs, affine_map, operands)
            return None
        slices = self._as_index_tuple(self.visit(node.slice))
        result = self.call_operator(mem_ops.store, [lhs, slices, value])
        # Bit (slice) insertion on an integer scalar produces a new SSA value
        # rather than mutating storage in place, so write it back to the source.
        if (
            isinstance(result, AlloValue)
            and isinstance(lhs, AlloValue)
            and isinstance(lhs.type, DType)
        ):
            writeback = copy.copy(node.value)
            writeback.ctx = ast.Store()
            self._do_assignment(writeback, result)
        return result

    def visit_Subscript_Load(self, node):
        assert isinstance(node.ctx, ast.Load)
        lhs = self.visit(node.value)
        built = self._build_affine_access(lhs, node)
        if built is not None:
            affine_map, operands = built
            return self.builder.create_affine_load(lhs, affine_map, operands)
        slices = self.visit(node.slice)
        if isinstance(lhs, Kernel):
            template_args = slices if isinstance(slices, tuple) else (slices,)
            return lhs[template_args]
        if isinstance(lhs, tuple) and isinstance(slices, ConstexprValue):
            return lhs[slices.value]
        return self.call_operator(mem_ops.load, [lhs, self._as_index_tuple(slices)])

    def visit_ListComp(self, node):
        if len(node.generators) != 1:
            return self.compile_error(
                "only single generator is supported in list comprehensions"
            )
        comp = node.generators[0]
        iter = self.visit(comp.iter)
        if not isinstance(iter, tuple):
            return self.compile_error(
                "only tuple iteration is supported in list comprehensions"
            )

        results = []
        for item in iter:
            if not isinstance(comp.target, ast.Name):
                return self.compile_error(
                    "only simple variable targets are supported in list comprehensions",
                )
            self._set_value(comp.target.id, item)
            results.append(self.visit(node.elt))
        return tuple(results)

    def visit_Tuple(self, node):
        return tuple(self.visit(e) for e in node.elts)

    def visit_Name(self, node):
        if type(node.ctx) is ast.Store:
            return node.id
        val = self.lookup(node.id)
        if isinstance(val, StatefulValue):
            return self._read_stateful(val)
        return val

    def visit_List(self, node):
        return tuple(self.visit(e) for e in node.elts)

    def _flatten_list_initializer(
        self, node: ast.AST
    ) -> tuple[tuple[int, ...], list[int | float]]:
        if isinstance(node, ast.List):
            values = []
            shapes = []
            for elt in node.elts:
                shape, flat_values = self._flatten_list_initializer(elt)
                shapes.append(shape)
                values.extend(flat_values)
            if len(shapes) == 0:
                return (0,), values
            first_shape = shapes[0]
            if any(shape != first_shape for shape in shapes):
                return self.compile_error(
                    f"Ragged list initializer '{ast.unparse(node)}' is not supported."
                )
            return (len(node.elts), *first_shape), values

        value = unwrap_if_constexpr(self.visit(node))
        if type(value) not in (builtins.int, builtins.float):
            return self.compile_error(
                f"List initializer elements must be compile-time int or float constants, got '{ast.unparse(node)}'."
            )
        return (), [value]  # type: ignore

    def _visit_shaped_list_initializer(self, node: ast.AnnAssign, dst_type: ShapedType):
        name = node.target.id
        shape, values = self._flatten_list_initializer(node.value)
        if tuple(shape) != tuple(dst_type.shape):
            return self.compile_error(
                f"List initializer shape mismatch for '{name}': expected {tuple(dst_type.shape)}, got {shape}."
            )
        global_name = self._global_symbol(node, name, "const")
        return self.builder.make_shaped_constant(values, dst_type, global_name)

    def _visit_numpy_array_initializer(
        self, node: ast.AnnAssign, dst_type: ShapedType, array: np.ndarray
    ):
        """Lower a captured NumPy array into a shaped constant, mirroring the list
        initializer path: the row-major elements feed ``make_shaped_constant`` and
        are coerced to ``dst_type.dtype`` there."""
        name = node.target.id
        if not (
            np.issubdtype(array.dtype, np.integer)
            or np.issubdtype(array.dtype, np.floating)
        ):
            return self.compile_error(
                f"NumPy array initializer for '{name}' must have an integer or "
                f"floating-point dtype, got '{array.dtype}'."
            )
        if tuple(array.shape) != tuple(dst_type.shape):
            return self.compile_error(
                f"NumPy array initializer shape mismatch for '{name}': "
                f"expected {tuple(dst_type.shape)}, got {tuple(array.shape)}."
            )
        global_name = self._global_symbol(node, name, "const")
        return self.builder.make_shaped_constant(
            array.reshape(-1).tolist(), dst_type, global_name
        )

    def visit_AugAssign(self, node: ast.AugAssign):
        lhs = copy.deepcopy(node.target)
        lhs.ctx = ast.Load()
        rhs = ast.copy_location(ast.BinOp(left=lhs, op=node.op, right=node.value), node)
        assign = ast.copy_location(ast.Assign(targets=[node.target], value=rhs), node)
        self.visit(assign)

    def _type_annotation_scope(self):
        scope = self.builtin_namespace.copy()
        scope.update(self.gscope)
        scope.update(self.closure_scope)
        scope.update(self.fscope)
        scope.update(self.lscope)
        for key, value in list(scope.items()):
            if self._is_python_scalar_const(value):
                scope[key] = ConstexprValue(value)
        return scope

    def _parse_annotation(self, annotation: ast.AST, name: str) -> TypeBase:
        scope = self._type_annotation_scope()
        if isinstance(annotation, ast.Constant) and annotation.value is None:
            return self.compile_error(f"Missing type annotation for '{name}'.")
        # A quoted annotation is a string literal; use its contents. Any other
        # expression is rendered back to source so it evaluates uniformly.
        if isinstance(annotation, ast.Constant) and isinstance(annotation.value, str):
            text = annotation.value
        else:
            text = ast.unparse(annotation)
        if text in {"constexpr", "Constexpr"}:
            return constexpr
        try:
            return self.kernel.parse_type_annotation(text, scope=scope)
        except Exception as e:
            return self.compile_error(
                f"Unsupported type annotation '{text}' for '{name}': {e}"
            )

    def _stateful_init_values(
        self, node: ast.AnnAssign, inner: TypeBase
    ) -> list[int | float]:
        if node.value is None:
            return self.compile_error(
                f"Stateful variable '{node.target.id}' must be initialized with a "
                "compile-time constant."
            )
        if isinstance(inner, BufferType):
            if isinstance(node.value, ast.List):
                shape, values = self._flatten_list_initializer(node.value)
                if tuple(shape) != tuple(inner.shape):
                    return self.compile_error(
                        f"Stateful array '{node.target.id}' initializer shape "
                        f"mismatch: expected {tuple(inner.shape)}, got {shape}."
                    )
                return values
            num = 1
            for dim in inner.shape:
                num *= dim
            return [self._const_init_scalar(node.value)] * num
        return [self._const_init_scalar(node.value)]

    def _const_init_scalar(self, node: ast.AST) -> int | float:
        value = unwrap_if_constexpr(self.visit(node))
        if type(value) not in (builtins.int, builtins.float):
            return self.compile_error(
                "Stateful variable initializer must be a compile-time int or float "
                f"constant, got '{ast.unparse(node)}'."
            )
        return value

    def _global_symbol(self, node: ast.AST, var_id: str, kind: str) -> str:
        """Instance-scoped wrapper over the module-level ``_global_symbol``, keyed
        on this generator's entry kernel."""
        return _global_symbol(self.kernel.func_name, var_id, kind, node)

    def _visit_stateful_decl(self, node: ast.AnnAssign, parsed_type: StatefulType):
        inner = parsed_type.inner
        values = self._stateful_init_values(node, inner)
        global_name = self._global_symbol(node, node.target.id, "stateful")
        stateful = self.builder.make_stateful(global_name, inner, values)
        self._set_value_with_loc(node.target.id, stateful)

    def _read_stateful(self, sv: StatefulValue):
        # Scalars load their current value; arrays expose the backing buffer so
        # subscripting and whole-array use go straight to persistent storage.
        if sv.is_scalar:
            return self.builder.create_load(sv.storage, [])
        return sv.storage

    def _write_stateful(self, sv: StatefulValue, value):
        if sv.is_scalar:
            self.builder.create_store(self.builder.cast(value, sv.type), sv.storage, [])
        else:
            self.builder.store_into_buffer(sv.storage, value)

    def visit_AnnAssign(self, node: ast.AnnAssign):
        if isinstance(node.target, ast.Attribute):
            return self.compile_error(
                "assignment to attributes is not supported in allo kernel functions"
            )
        if not isinstance(node.target, ast.Name):
            return self.compile_error(
                "annotated assignment only supports simple variable targets"
            )
        if node.target.id in self.lscope:
            return self.compile_error(
                f"Variable '{node.target.id}' is already defined in the current scope."
            )

        parsed_type = self._parse_annotation(node.annotation, node.target.id)
        if isinstance(parsed_type, StatefulType):
            return self._visit_stateful_decl(node, parsed_type)
        if isinstance(parsed_type, StreamType):
            if node.value is not None:
                return self.compile_error(
                    f"Stream '{node.target.id}' must be declared without an initializer."
                )
            self._set_value_with_loc(
                node.target.id, self.builder.create_stream(parsed_type)
            )
            return

        if node.value is None:
            if isinstance(parsed_type, ShapedType):
                self._set_value_with_loc(
                    node.target.id, self.builder.make_buffer(parsed_type)
                )
                return
            return self.compile_error(
                f"Annotated variable '{node.target.id}' must have an initializer."
            )

        if isinstance(parsed_type, ShapedType) and isinstance(node.value, ast.List):
            with self._name_loc_prefix(node.target.id):
                value = self._visit_shaped_list_initializer(node, parsed_type)
            self._set_value_with_loc(node.target.id, value)
            return

        with self._name_loc_prefix(node.target.id):
            value = self.visit(node.value)

        if isinstance(value, np.ndarray):
            if not isinstance(parsed_type, ShapedType):
                return self.compile_error(
                    f"NumPy array can only initialize a shaped variable, but "
                    f"'{node.target.id}' is annotated as "
                    f"'{ast.unparse(node.annotation)}'."
                )
            with self._name_loc_prefix(node.target.id):
                const = self._visit_numpy_array_initializer(node, parsed_type, value)
            self._set_value_with_loc(node.target.id, const)
            return

        if isinstance(parsed_type, ConstexprType):
            if isinstance(value, AlloValue):
                return self.compile_error(
                    f"Unsupported assignment with type annotation 'constexpr' and value of type '{value.type}'."
                )
            self._set_value(node.target.id, ConstexprValue(value))
            return

        if not isinstance(value, (AlloValue, ConstexprValue)):
            return self.compile_error(
                f"Unsupported initializer for variable '{node.target.id}' with type annotation '{ast.unparse(node.annotation)}'."
            )
        self._set_value_with_loc(node.target.id, self.builder.cast(value, parsed_type))

    def visit_Assign(self, node: ast.Assign):
        targets = node.targets
        if len(targets) != 1:
            return self.compile_error("multiple assignment targets are not supported")
        target = targets[0]
        if isinstance(target, ast.Name):
            with self._name_loc_prefix(target.id):
                value = self.visit(node.value)
        else:
            value = self.visit(node.value)
        self._do_assignment(target, value)

    def _do_assignment(self, target, value: object):
        assert isinstance(target.ctx, ast.Store)
        if isinstance(target, ast.Subscript):
            return self.visit_Subscript_Store(target, value)
        if isinstance(target, ast.Tuple):
            assert isinstance(value, tuple)
            for i, elt in enumerate(target.elts):
                self._do_assignment(elt, value[i])
            return
        if isinstance(target, ast.Attribute):
            return self.compile_error(
                "assignment to attributes is not supported in allo kernel functions"
            )
        if isinstance(target, ast.Name):
            target = self.visit(target)
            # the first time we see a variable is considered its definition site, and its type if inferred from the assigned value. subsequent assignments to the same variable must be type-compatible with the first definition.
            if target not in self.lscope:
                if isinstance(value, ConstexprValue):
                    return self.compile_error(
                        "Constexpr variables must be explicitly declared with type annotation. Please add a type annotation of 'constexpr' to this variable."
                    )
                self._set_value_with_loc(target, value)
                return
            proxy = self.lscope[target]
            if isinstance(proxy, StatefulValue):
                # Persistent storage: store into the global, keep the binding.
                return self._write_stateful(proxy, value)
            if isinstance(proxy, ConstexprValue):
                return self.compile_error(
                    f"Cannot reassign to variable '{target}' defined as a constexpr"
                )
            assert isinstance(proxy, AlloValue)
            if isinstance(value, ConstexprValue):
                ret = self.builder.materialize_literal_like(value.value, proxy)
            elif isinstance(value, AlloValue):
                ret = self.builder.cast(value, proxy.type)
            else:
                assert False, f"unsupported assignment value: {value}"
            self._set_value_with_loc(target, ret)

    def _set_value_with_loc(self, target, value):
        self._set_value(target, value)
        self._maybe_set_loc_to_name(target, value)

    def _test_loop_iter_args(self, node, liveins: dict, ignore: set[str]):
        ip, last_loc = self.builder.get_insertion_point_and_loc()
        # create dummy block
        block = self.builder.create_block(ip.block.region)
        self.builder.set_insertion_point_to_start(block)
        self.lscope = liveins.copy()
        # dry visit
        old_dry_run = self.dry_run_loop_analysis
        self.dry_run_loop_analysis = True
        try:
            self.visit_compound_stmts(node.body)
        finally:
            self.dry_run_loop_analysis = old_dry_run
        dry_run_scope = self.lscope.copy()
        # restore insertion point before analyzing dry-run live-outs. Keep the
        # dummy block alive until after the analysis because dry_run_scope can
        # still point to values created inside it.
        self.builder.set_insertion_point_and_loc(ip, last_loc)

        # compute live-outs
        init_types = []
        init_handles = []
        names = []
        error_msg = None

        for name, livein in liveins.items():
            if name in ignore:
                continue
            if isinstance(livein, ConstexprValue):
                continue
            if isinstance(livein, StatefulValue):
                # Stateful vars live in memory; they carry no SSA loop value.
                continue
            assert isinstance(livein, AlloValue)
            loop_val = dry_run_scope[name]
            assert isinstance(loop_val, AlloValue)
            if loop_val.handle == livein.handle:
                continue  # variable is not assigned in the loop body
            # type check
            if type(loop_val) != type(livein) or loop_val.type != livein.type:
                error_msg = f"Loop variable '{name}' has incompatible types in outer scope vs loop body: {livein.type} vs {loop_val.type}."
                break
            names.append(name)
            init_handles.append(livein.handle)
            init_types.append(livein.type)

        # restore lscope
        self.lscope = liveins.copy()
        ir_ext.erase_block(block)
        if error_msg is not None:
            return self.compile_error(error_msg)
        return names, init_handles, init_types

    def visit_While(self, node: ast.While):
        if node.orelse:
            return self.compile_error(
                "'while' statement with 'else' block is not supported"
            )
        with EnterSubRegion(self):
            liveins = self.lscope.copy()
            names, init_handles, init_types = self._test_loop_iter_args(
                node, liveins, ignore=set()
            )
            # create while op
            init_ir_types = [ty.materialize(self.context) for ty in init_types]
            while_op = WhileOp(
                init_ir_types, init_handles, ip=self.builder._ip, loc=self.builder._loc
            )

            # create before region
            before_block = self.builder.create_block(while_op.before, init_ir_types)
            self.builder.set_insertion_point_to_start(before_block)
            block_args = list(before_block.arguments)
            for name, arg, ty in zip(names, block_args, init_types):
                proxy = AlloValue(arg, ty)
                self._set_value_with_loc(name, proxy)

            # visit condition
            cond = self.visit(node.test)
            if isinstance(cond, AlloValue):
                cond = self.builder.to_condition(cond)
            else:
                cond = self.builder.cast(cond, AlloBool)
            self.builder.set_insertion_point_to_end(before_block)
            # create cond
            ConditionOp(
                cond.handle, block_args, ip=self.builder._ip, loc=self.builder._loc
            )

            # create after region
            after_block = self.builder.create_block(while_op.after, init_ir_types)
            self.builder.set_insertion_point_to_start(after_block)
            body_handles = list(after_block.arguments)
            for name, arg, ty in zip(names, body_handles, init_types):
                proxy = AlloValue(arg, ty)
                self._set_value_with_loc(name, proxy)

            # visit loop body
            self.visit_compound_stmts(node.body)

            # create yield
            yield_handles = [
                cast(AlloValue, self.lscope[name]).handle for name in names
            ]
            self.builder.set_insertion_point_to_end(after_block)
            SCFYieldOp(yield_handles, ip=self.builder._ip, loc=self.builder._loc)

        # update lscope with iter args
        res_handles = list(while_op.results)
        res_proxies = [
            AlloValue(handle, ty) for handle, ty in zip(res_handles, init_types)
        ]
        for name, proxy in zip(names, res_proxies):
            self._set_value_with_loc(name, proxy)

    def visit_For(self, node: ast.For):
        if node.orelse:
            return self.compile_error(
                "'for' statement with 'else' block is not supported"
            )
        if not isinstance(node.iter, ast.Call):
            return self.compile_error(
                "Only 'for' loops over 'range()/grid()' are supported"
            )

        IteratorClass = self.visit(node.iter.func)
        iter_args = [self.visit(arg) for arg in node.iter.args]
        iter_kwargs = {kw.arg: self.visit(kw.value) for kw in node.iter.keywords}

        if IteratorClass is Range:
            iterator = IteratorClass(*iter_args, **iter_kwargs)  # type: ignore
            lb = iterator.start
            ub = iterator.stop
            step = iterator.step
        elif IteratorClass is Grid:
            iterator = IteratorClass(*iter_args, **iter_kwargs)  # type: ignore
            return self.visit_Grid(node, iterator)
        else:
            return self.compile_error(
                "Only 'for' loops over 'range()' and 'grid()' are supported"
            )

        if not isinstance(node.target, ast.Name):
            return self.compile_error(
                "loop target must be a single variable in 'for' loops"
            )

        if isinstance(step, ConstexprValue) and step.value <= 0:
            return self.compile_error(
                "loop step must be a positive integer in 'for' loops"
            )

        # A loop lowers to affine.for (so its body can use affine.load/store) when
        # the step is a positive constant and both bounds are affine expressions
        # over enclosing affine IVs / top-level symbols / constants; else scf.for.
        affine_bounds = None
        if isinstance(step, ConstexprValue) and type(step.value) is builtins.int:
            lb_node, ub_node = self._range_bound_nodes(node.iter)
            lb_built = self._build_single_bound(lb_node)
            ub_built = self._build_single_bound(ub_node)
            if lb_built is not None and ub_built is not None:
                affine_bounds = (lb_built, ub_built)
        is_affine = affine_bounds is not None
        if not is_affine:
            lb, ub, step = self.builder.normalize_indices(
                (lb, ub, step), expected_len=3
            )

        with EnterSubRegion(self):
            index_ty = index.materialize(self.context)
            iv_placeholder = PoisonOp(
                index_ty, ip=self.builder._ip, loc=self.builder._loc
            )
            iv_proxy = AlloValue(iv_placeholder.result, index)
            self._set_value(node.target.id, iv_proxy)
            if is_affine:
                self._affine_ivs.append(iv_proxy)

            liveins = self.lscope.copy()  # capture live-ins before visiting loop body
            names, init_handles, init_types = self._test_loop_iter_args(
                node, liveins, ignore={node.target.id}
            )
            arg_locs = [Location.name(node.target.id, self.builder._loc)] + [
                Location.name(nm, self.builder._loc) for nm in names
            ]
            if is_affine:
                (lb_map, lb_operands), (ub_map, ub_operands) = affine_bounds
                for_op = self.builder.create_affine_for(
                    lb_map,
                    [v.handle for v in lb_operands],
                    ub_map,
                    [v.handle for v in ub_operands],
                    step.value,
                    init_handles,
                    arg_locs=arg_locs,
                )
            else:
                for_op = self.builder.create_scf_for(
                    lb.handle,
                    ub.handle,
                    step.handle,
                    init_handles,
                    arg_locs=arg_locs,
                )
            # Default the loop's schedule name to its induction variable, so an
            # unnamed `for i in range(N)` is queryable as `s.loop("i")`. An
            # explicit `range(N, name=...)` still wins.
            loop_name = iterator.name or node.target.id
            for_op.operation.attributes[schedule_d.SCHEDULE_NAME_ATTR_NAME] = (
                self.builder.get_string_attr(loop_name)
            )
            for_op_body = for_op.body
            self.builder.set_insertion_point_to_start(for_op_body)
            block_handles = [
                # skip the first argument which is the induction variable
                for_op_body.arguments[i + 1]
                for i in range(len(init_handles))
            ]
            block_args = [
                AlloValue(handle, ty) for handle, ty in zip(block_handles, init_types)
            ]
            for iter_name, proxy in zip(names, block_args):
                self._set_value_with_loc(iter_name, proxy)
            # visit loop body
            self.visit_compound_stmts(node.body)
            # create yield
            yield_handles = [
                cast(AlloValue, self.lscope[iter_name]).handle for iter_name in names
            ]
            self.builder.set_insertion_point_to_end(for_op_body)
            yield_cls = AffineYieldOp if is_affine else SCFYieldOp
            yield_cls(yield_handles, ip=self.builder._ip, loc=self.builder._loc)
            assert len(for_op.regions) == 1

            # update induction variable with the actual one
            iv = for_op.induction_variable
            iv_placeholder.result.replace_all_uses_with(iv)
            iv_placeholder.operation.erase()
            self._set_value_with_loc(node.target.id, AlloValue(iv, index))

        # update lscope with iter args
        res_handles = list(for_op.results)
        for iter_name, handle, ty in zip(names, res_handles, init_types):
            proxy = AlloValue(handle, ty)
            self._set_value_with_loc(iter_name, proxy)

    def visit_Grid(self, node: ast.For, iterator: Grid):
        if len(iterator.starts) <= 1:
            return self.compile_error(
                "Use range() for single-dimensional loops; grid() requires at least two dimensions."
            )
        if not isinstance(node.target, ast.Tuple):
            return self.compile_error(
                "loop target must be a tuple of variables in 'for' loops over 'grid()'"
            )
        if len(node.target.elts) != len(iterator.starts):
            return self.compile_error(
                f"loop target must have the same number of variables as the dimensions of the grid iterator. Expected {len(iterator.starts)} variables, but got {len(node.target.elts)}."
            )

        lbs = iterator.starts
        ubs = iterator.stops
        steps = iterator.steps

        if any(isinstance(step, ConstexprValue) and step.value <= 0 for step in steps):
            return self.compile_error(
                "loop step must be a positive integer in 'for' loops"
            )

        # A grid lowers to affine.parallel when every step is a positive constant
        # and all bounds are affine (over enclosing IVs / symbols / constants);
        # otherwise scf.parallel. Lower/upper maps share operands concatenated as
        # lower-then-upper, as affine.parallel requires.
        affine_bounds = None
        if all(
            isinstance(s, ConstexprValue) and type(s.value) is builtins.int
            for s in steps
        ):
            specs = self._grid_bound_nodes(node.iter)
            if specs is not None:
                lb_nodes = [
                    lb if lb is not None else ast.Constant(0) for lb, _ in specs
                ]
                ub_nodes = [ub for _, ub in specs]
                lower = self._build_affine_value_map(lb_nodes)
                upper = self._build_affine_value_map(ub_nodes)
                if lower is not None and upper is not None:
                    affine_bounds = (lower, upper)
        is_affine = affine_bounds is not None

        with EnterSubRegion(self):
            index_ty = index.materialize(self.context)
            iv_placeholders = [
                PoisonOp(index_ty, ip=self.builder._ip, loc=self.builder._loc)
                for _ in lbs
            ]
            targets = set()
            for i, target in enumerate(node.target.elts):
                if not isinstance(target, ast.Name):
                    return self.compile_error(
                        "loop target must be a single variable in 'for' loops over 'grid()'"
                    )
                iv_proxy = AlloValue(iv_placeholders[i].result, index)
                self._set_value(target.id, iv_proxy)
                if is_affine:
                    self._affine_ivs.append(iv_proxy)
                targets.add(target.id)

            liveins = self.lscope.copy()  # capture live-ins before visiting loop body
            names, init_handles, init_types = self._test_loop_iter_args(
                node, liveins, ignore=targets
            )
            if len(init_handles) > 0:
                return self.compile_error(
                    "Non-trivial loop-carried dependencies are not supported in "
                    "'for' loops over 'grid()' at this moment."
                )
            # create parallel op
            if is_affine:
                (lb_map, lb_operands), (ub_map, ub_operands) = affine_bounds
                par_op, par_op_body = self.builder.create_affine_parallel(
                    lb_map,
                    [v.handle for v in lb_operands],
                    ub_map,
                    [v.handle for v in ub_operands],
                    [step.value for step in steps],
                    arg_locs=[
                        Location.name(t.id, self.builder._loc) for t in node.target.elts
                    ],
                )
            else:
                par_op = ParallelOp(
                    [],
                    [lb.handle for lb in self.builder.normalize_indices(lbs)],
                    [ub.handle for ub in self.builder.normalize_indices(ubs)],
                    [step.handle for step in self.builder.normalize_indices(steps)],
                    init_handles,
                    ip=self.builder._ip,
                    loc=self.builder._loc,
                )
                # scf.parallel has no auto-created body: build a block with one
                # index induction variable per dimension and the scf.reduce
                # terminator. see: https://mlir.llvm.org/docs/Dialects/SCFDialect/#scfparallel-scfparallelop
                par_op_body = par_op.region.blocks.append(
                    *([index_ty] * len(lbs)),
                    arg_locs=[
                        Location.name(t.id, self.builder._loc) for t in node.target.elts
                    ],
                )
                with InsertionPoint(par_op_body):
                    ReduceOp([], 0)
            if iterator.name:
                assert isinstance(iterator.name, str)
                par_op.operation.attributes[schedule_d.SCHEDULE_NAME_ATTR_NAME] = (
                    self.builder.get_string_attr(iterator.name)
                )
            self.builder.set_insertion_point_to_start(par_op_body)
            # no iter args now, so no block arguments other than induction variables
            # visit loop body
            self.visit_compound_stmts(node.body)

            ivs = list(par_op_body.arguments)
            for iv, placeholder in zip(ivs, iv_placeholders):
                placeholder.result.replace_all_uses_with(iv)
                placeholder.operation.erase()
            for iv, target in zip(ivs, node.target.elts):
                proxy = AlloValue(iv, index)
                self._set_value_with_loc(target.id, proxy)  # type: ignore

        # update lscope with iter args
        res_handles = list(par_op.results)
        for name, handle, ty in zip(names, res_handles, init_types):
            proxy = AlloValue(handle, ty)
            self._set_value_with_loc(name, proxy)

    def visit_JoinedStr(self, node):
        values = list(node.values)
        for i, value in enumerate(values):
            if isinstance(value, ast.Constant):
                values[i] = str(value.value)  # type: ignore
            elif isinstance(value, ast.FormattedValue):
                conversion_code = value.conversion
                evaluated = self.visit(value.value)
                if not isinstance(evaluated, ConstexprValue):
                    return self.compile_error(
                        "Cannot evaluate f-string containing non-constexpr conversion values, found conversion of type "
                        + str(type(evaluated)),
                    )
                values[i] = (  # type: ignore
                    "{}" if conversion_code < 0 else "{!" + chr(conversion_code) + "}"
                ).format(evaluated.value)
            else:
                assert False, f"unexpected value type in JoinedStr: {type(value)}"
        return "".join(values)  # type: ignore

    def visit_Call(self, node):
        fn = unwrap_if_constexpr(self.visit(node.func))
        static_fn = self.statically_implemented_functions.get(fn, None)
        if static_fn is not None:
            return static_fn(self, node)

        self.visiting_consteval_fn = isinstance(fn, ConstevalFunction)
        try:
            # build kwargs and args
            kws = dict(self.visit(kw) for kw in node.keywords)
            args = []
            for arg in node.args:
                if isinstance(arg, ast.Starred):
                    arg = self.visit(arg.value)
                    assert isinstance(arg, tuple)
                    args.extend(arg)
                else:
                    ret = self.visit(arg)
                    args.append(ret)
        finally:
            self.visiting_consteval_fn = False
        return self.call_function(fn, args, kws)

    def call_function(self, fn, args, kws):
        """Dispatch callable targets across kernel/op/type/consteval frontends."""

        if isinstance(fn, NestedKernelSymbol):
            return self.call_nested_kernel(fn, args, kws)
        if isinstance(fn, Kernel):
            return self.call_kernel(fn, args, kws)
        if isinstance(fn, AlloModule):
            return self.call_kernel(fn.module, args, kws)
        if isinstance(fn, (Operator, BoundOperator)):
            return self.call_operator(fn, args, kws)
        if isinstance(fn, ConstevalFunction):
            if fn.lazy:
                k = Kernel(fn.fn, mapping=(), options=KernelOptions())
                k._is_lazy_consteval = True
                return self.call_kernel(k, args, kws, is_lazy=True)
            else:
                try:
                    ret = fn(*args, **kws)
                    # TODO: check if returned value is valid
                    return ConstexprValue(ret)
                except CompilationError:
                    raise
                except Exception as e:
                    return self.compile_error(
                        f"error when calling consteval function '{fn.__name__}': {e}"
                    )
        fn_mod = getattr(fn, "__module__", type(fn).__module__)
        fn_name = getattr(fn, "__name__", type(fn).__name__)
        return self.compile_error(
            f"only allo kernel functions, operations, and consteval functions can be called in allo kernel functions, but got {fn_mod}.{fn_name}"
        )

    def _next_called_kernel_name(self, fn: Kernel | NestedKernelSymbol | str) -> str:
        if isinstance(fn, Kernel):
            callee_name = fn.func_name
        elif isinstance(fn, NestedKernelSymbol):
            callee_name = fn.name
        else:
            callee_name = fn
        base_name = f"{self.func_name}.{callee_name}"
        if base_name in self._kernel_base_names:
            name = f"{base_name}.{self._kernel_base_names[base_name]}"
            self._kernel_base_names[base_name] += 1
        else:
            name = base_name
            self._kernel_base_names[base_name] = 1
        return name

    def _kernel_call_key(self, fn: Kernel) -> str:
        bindings = ",".join(
            f"{name}={value}" for name, value in sorted(fn.template_bindings.items())
        )
        return f"kernel:{fn.__module__}.{fn.__qualname__}[{bindings}]"

    def _nested_call_key(self, nested: NestedKernelSymbol) -> str:
        return f"nested:{nested.owner_func_name}.{nested.name}"

    def _check_recursive_call(self, key: str):
        if key in self._active_kernel_calls:
            chain = " -> ".join(self._active_kernel_calls + [key])
            return self.compile_error(
                f"Recursive kernel calls are not supported: {chain}"
            )

    def _build_kernel_call_operand(
        self, value: object, expected_ty: TypeBase, arg_name: str
    ):
        if isinstance(expected_ty, ConstexprType):
            assert False, "constexpr arguments do not have call operands"
        if isinstance(expected_ty, StreamType) and not isinstance(value, AlloValue):
            return self.compile_error(
                f"Kernel call argument '{arg_name}' type mismatch: expected '{expected_ty}', got '{type(value).__name__}'."
            )
        if not isinstance(value, (AlloValue, ConstexprValue)):
            value = ConstexprValue(value)
        try:
            return self.builder.cast(value, expected_ty).handle
        except CompilationError:
            value_ty = value.type if isinstance(value, AlloValue) else "constexpr"
            return self.compile_error(
                f"Kernel call argument '{arg_name}' type mismatch: expected '{expected_ty}', got '{value_ty}'."
            )

    def _prepare_kernel_call_args(
        self, callee_name: str, bound_items, arg_types: Sequence[TypeBase]
    ):
        bound_items = list(bound_items)
        if len(arg_types) != len(bound_items):
            return self.compile_error(
                f"Kernel specialization argument count mismatch for '{callee_name}': expected {len(bound_items)}, got {len(arg_types)}."
            )

        callee_context: dict[str, object] = {}
        call_operands: list[Value] = []
        for (arg_name, arg_val), expected_ty in zip(bound_items, arg_types):
            if isinstance(expected_ty, ConstexprType):
                if isinstance(arg_val, AlloValue):
                    return self.compile_error(
                        f"Kernel call argument '{arg_name}' must be constexpr, but got runtime value of type '{arg_val.type}'."
                    )
                if not isinstance(arg_val, ConstexprValue):
                    arg_val = ConstexprValue(arg_val)
                callee_context[arg_name] = arg_val
                continue
            call_operands.append(
                self._build_kernel_call_operand(arg_val, expected_ty, arg_name)
            )
        return callee_context, call_operands

    def _decode_kernel_call_results(
        self, call_op: InvokeOp, res_types: Sequence[TypeBase]
    ):
        if any(isinstance(ty, ConstexprType) for ty in res_types):
            return self.compile_error(
                "Kernel calls returning constexpr values are not supported."
            )
        if len(res_types) == 0:
            return None
        if len(call_op.results) != len(res_types):
            return self.compile_error(
                f"Kernel call result count mismatch: expected {len(res_types)}, got {len(call_op.results)}."
            )
        results = [
            AlloValue(handle, ty) for handle, ty in zip(call_op.results, res_types)
        ]
        if len(results) == 1:
            return results[0]
        return tuple(results)

    def _make_dry_run_call_results(self, res_types: Sequence[TypeBase]):
        if any(isinstance(ty, ConstexprType) for ty in res_types):
            return self.compile_error(
                "Kernel calls returning constexpr values are not supported."
            )
        if len(res_types) == 0:
            return None
        results = [
            AlloValue(
                PoisonOp(
                    ty.materialize(self.context),
                    ip=self.builder._ip,
                    loc=self.builder._loc,
                ).result,
                ty,
            )
            for ty in res_types
        ]
        if len(results) == 1:
            return results[0]
        return tuple(results)

    def _parse_return_types(self, node: ast.FunctionDef) -> list[TypeBase]:
        if node.returns is None or (
            isinstance(node.returns, ast.Constant) and node.returns.value is None
        ):
            return []
        if isinstance(node.returns, ast.Tuple):
            res_types = [
                self._parse_annotation(ret, f"return[{i}]")
                for i, ret in enumerate(node.returns.elts)
            ]
        else:
            res_types = [self._parse_annotation(node.returns, "return")]
        if any(isinstance(ty, StreamType) for ty in res_types):
            return self.compile_error("Stream is not allowed as a kernel return type.")
        if any(isinstance(ty, StatefulType) for ty in res_types):
            return self.compile_error(
                "Stateful is not allowed as a kernel return type."
            )
        return res_types

    def _bind_nested_arguments(self, nested: NestedKernelSymbol, args, kws):
        fn_args = nested.node.args
        if (
            fn_args.posonlyargs
            or fn_args.kwonlyargs
            or fn_args.vararg is not None
            or fn_args.kwarg is not None
        ):
            return self.compile_error(
                f"Nested kernel '{nested.name}' only supports regular positional/keyword arguments."
            )

        params = fn_args.args
        param_names = [param.arg for param in params]
        if len(args) > len(params):
            return self.compile_error(
                f"Invalid arguments for nested kernel '{nested.name}': expected at most {len(params)} positional arguments, got {len(args)}."
            )

        bound = {name: value for name, value in zip(param_names, args)}
        for kw_name, kw_val in kws.items():
            if kw_name not in param_names:
                return self.compile_error(
                    f"Invalid arguments for nested kernel '{nested.name}': unexpected keyword argument '{kw_name}'."
                )
            if kw_name in bound:
                return self.compile_error(
                    f"Invalid arguments for nested kernel '{nested.name}': multiple values for argument '{kw_name}'."
                )
            bound[kw_name] = kw_val

        defaults = fn_args.defaults
        first_default_idx = len(params) - len(defaults)
        for idx, param in enumerate(params):
            if param.arg in bound:
                continue
            if idx < first_default_idx:
                return self.compile_error(
                    f"Invalid arguments for nested kernel '{nested.name}': missing required argument '{param.arg}'."
                )
            try:
                self.visiting_default_args = True
                bound[param.arg] = self.visit(defaults[idx - first_default_idx])
            finally:
                self.visiting_default_args = False

        return {param.arg: bound[param.arg] for param in params}

    def _infer_nested_arg_type(self, name: str, value: object) -> TypeBase:
        if isinstance(value, AlloValue):
            return value.type
        if isinstance(value, ConstexprValue):
            return constexpr
        return self.compile_error(
            f"Cannot infer type for nested kernel argument '{name}' from value of type '{type(value).__name__}'."
        )

    def _specialize_nested_kernel(self, nested: NestedKernelSymbol, bound):
        arg_types = []
        for param in nested.node.args.args:
            value = bound[param.arg]
            if param.annotation is None:
                arg_types.append(self._infer_nested_arg_type(param.arg, value))
            else:
                ty = self._parse_annotation(param.annotation, param.arg)
                if isinstance(ty, StatefulType):
                    return self.compile_error(
                        f"Parameter '{param.arg}' of nested kernel '{nested.name}' "
                        "cannot be Stateful; declare stateful variables locally."
                    )
                arg_types.append(ty)
        return arg_types, self._parse_return_types(nested.node)

    def _build_nested_capture_scopes(self):
        closure_scope: dict[str, object] = {}
        forbidden_scope: dict[str, object] = {}
        for name, value in self.lscope.items():
            if self._is_allowed_static_value(name, value):
                closure_scope[name] = value
            else:
                forbidden_scope[name] = value
        return closure_scope, self.fscope.copy(), forbidden_scope

    def _lower_kernel_invocation(
        self,
        key,
        callee_label,
        error_src,
        sub_res_types,
        call_operands,
        build_and_visit,
    ):
        """Shared plumbing for kernel/nested-kernel calls: emit the callee into the
        module, build the ``InvokeOp`` and decode its results. ``build_and_visit``
        owns the per-callee differences (location, source, sub-generator
        construction and the visited node)."""
        if self.dry_run_loop_analysis:
            return self._make_dry_run_call_results(sub_res_types)

        ip, last_loc = self.builder.get_insertion_point_and_loc()
        sub_generator = None
        self._active_kernel_calls.append(key)
        try:
            self.builder.set_insertion_point_to_end(self.module.body)
            sub_generator = build_and_visit()
            if sub_generator.generated_func is None:
                return self.compile_error(
                    f"Internal error: failed to materialize kernel '{callee_label}'."
                )
        except CompilationError as e:
            raise CompilationError(
                e.src if e.src is not None else error_src,
                f"error when compiling kernel '{callee_label}' called from '{self.func_name}': {e.error_msg}",
                e.node,
                file_name=e.file_name,
                begin_line=e.begin_line,
            ) from e
        finally:
            self._active_kernel_calls.pop()
            self.builder.src = self.kernel.src
            self.builder.file_name = self.file_name
            self.builder.begin_line = self.begin_line
            self.builder.set_insertion_point_and_loc(ip, last_loc)

        assert sub_generator is not None and sub_generator.generated_func is not None
        call_op = InvokeOp(
            [ty.materialize(self.context) for ty in sub_res_types],
            sub_generator.func_name,
            call_operands,
            ip=self.builder._ip,
            loc=self.builder._loc,
        )
        return self._decode_kernel_call_results(call_op, sub_res_types)

    def call_nested_kernel(self, nested: NestedKernelSymbol, args, kws):
        key = self._nested_call_key(nested)
        self._check_recursive_call(key)

        bound = self._bind_nested_arguments(nested, args, kws)
        sub_arg_types, sub_res_types = self._specialize_nested_kernel(nested, bound)
        callee_context, call_operands = self._prepare_kernel_call_args(
            nested.name, bound.items(), sub_arg_types
        )
        closure_scope, closure_fscope, forbidden_scope = (
            self._build_nested_capture_scopes()
        )

        def build_and_visit():
            self.builder.set_loc(
                Location.file(
                    self.file_name,
                    self.begin_line + nested.node.lineno - 1,
                    1,
                    self.context,
                )
            )
            self.builder.src = self.kernel.src
            sub_generator = MLIRCodeGenerator(
                self.context,
                self.module,
                self.builder,
                kernel=self.kernel,
                func_name=self._next_called_kernel_name(nested),
                file_name=self.file_name,
                begin_line=self.begin_line,
                gscope=self.gscope,
                arg_types=sub_arg_types,
                res_types=sub_res_types,
                options=self.options,
                callee_context=callee_context,
                fscope=closure_fscope,
                closure_scope=closure_scope,
                forbidden_closure_scope=forbidden_scope,
                active_kernel_calls=self._active_kernel_calls,
            )
            # set the mapping for sub kernels
            sub_generator.mapping = nested.mapping
            sub_generator.visit(nested.node)
            return sub_generator

        return self._lower_kernel_invocation(
            key,
            nested.name,
            self.kernel.src,
            sub_res_types,
            call_operands,
            build_and_visit,
        )

    def call_kernel(self, fn: Kernel, args, kws, is_lazy=False):
        """Lower/call a kernel specialization and decode structured return values."""

        key = self._kernel_call_key(fn)
        self._check_recursive_call(key)

        try:
            bound = fn.signature.bind(*args, **kws)
            bound.apply_defaults()
        except TypeError as e:
            return self.compile_error(
                f"Invalid arguments for kernel '{fn.func_name}': {e}."
            )

        try:
            sub_arg_types = list(fn.parse_argument_annotations())
            sub_res_types = list(fn.parse_return_annotation())
        except Exception as e:
            return self.compile_error(
                f"Failed to specialize kernel '{fn.func_name}': {e}"
            )

        callee_context, call_operands = self._prepare_kernel_call_args(
            fn.func_name, bound.arguments.items(), sub_arg_types
        )

        def build_and_visit():
            self.builder.set_loc(
                Location.file(fn.file_name, fn.begin_line, 1, self.context)
            )
            self.builder.src = fn.src
            sub_generator = MLIRCodeGenerator(
                self.context,
                self.module,
                self.builder,
                kernel=fn,
                func_name=self._next_called_kernel_name(fn),
                file_name=fn.file_name,
                begin_line=fn.begin_line,
                gscope=fn.get_capture_scope(),
                arg_types=sub_arg_types,
                res_types=sub_res_types,
                options=fn.options,
                callee_context=callee_context,
                active_kernel_calls=self._active_kernel_calls,
                is_top=False,
            )
            sub_generator.visit(fn.parse())
            return sub_generator

        return self._lower_kernel_invocation(
            key, fn.func_name, fn.src, sub_res_types, call_operands, build_and_visit
        )

    def call_operator(self, fn: Operator | BoundOperator, args, kwargs={}):
        if isinstance(fn, BoundOperator):
            args = fn.bind_args(args)
            fn = fn.op
        ip, last_loc = self.builder.get_insertion_point_and_loc()

        # try folding first
        if fn.fold_impl is not None:
            ret = fn.fold_impl(*args, **kwargs)
            if ret is not NO_FOLD:
                return ret

        # fold failed, build IR
        if fn.build_impl is None:
            return self.compile_error(
                f"Operator '{fn.__name__}' does not define a construction implementation"
            )
        try:
            return fn.build_impl(self.builder, *args, **kwargs)
        finally:
            # restore states
            self.builder.set_insertion_point_and_loc(ip, last_loc)

    @staticmethod
    def static_executor(python_fn):
        def ret(self, node: ast.Call):
            kws = {
                name: unwrap_if_constexpr(value)
                for name, value in (self.visit(keyword) for keyword in node.keywords)
            }
            args = [unwrap_if_constexpr(self.visit(arg)) for arg in node.args]
            return ConstexprValue(python_fn(*args, **kws))

        return ret

    def execute_static_assert(self, node: ast.Call) -> None:
        arg_count = len(node.args)
        if not (0 < arg_count <= 2) or len(node.keywords):
            raise TypeError(
                "`static_assert` requires one or two positional arguments only"
            )

        passed = unwrap_if_constexpr(self.visit(node.args[0]))
        if not isinstance(passed, bool):
            raise NotImplementedError(
                "Assertion condition could not be determined at compile-time. Make sure that it depends only on `constexpr` values"
            )
        if not passed:
            if arg_count == 1:
                message = ""
            else:
                try:
                    message = self.visit(node.args[1])
                except Exception as e:
                    message = "<failed to evaluate assertion message: " + repr(e) + ">"

            raise StaticAssertionError(
                self.kernel.src,
                str(unwrap_if_constexpr(message)),
                self.builder.curr_node,  # type: ignore
                file_name=self.file_name,
                begin_line=self.begin_line,
            )
        return None

    statically_implemented_functions = {
        # dsl.static_assert: execute_static_assert,
        print: static_executor(print),
        len: static_executor(len),
    }


class EnterSubRegion:
    """Scoped helper that snapshots/restores frontend symbol state + insertion point."""

    def __init__(self, generator: MLIRCodeGenerator):
        self.generator = generator

    def __enter__(self):
        self.lscope = self.generator.lscope.copy()
        self.affine_ivs = self.generator._affine_ivs.copy()
        self.ip = self.generator.builder.save_insertion_point()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.generator.lscope = self.lscope
        self.generator._affine_ivs = self.affine_ivs
        self.generator.builder.restore_insertion_point(self.ip)


def compile(
    fn: Kernel,
    arg_types: Sequence[TypeBase | str] = [],
    res_types: Sequence[TypeBase | str] = [],
    options: KernelOptions | None = None,
    show_traceback=False,
):
    """Compile a kernel function into an MLIR module."""
    import os

    if os.environ.get("ALLO_SHOW_COMPILER_TRACEBACK", "") == "1":
        show_traceback = True
    if not isinstance(fn, Kernel):
        raise TypeError(
            "Only allo.kernel functions can be compiled with allo.compile()"
        )
    fn.check_templates_bounded()
    if not arg_types:
        arg_types = fn.parse_argument_annotations()
    else:
        arg_types = [fn.parse_type_annotation(t) for t in arg_types]
    if len(arg_types) != len(fn.signature.parameters):
        raise ValueError(
            f"The number of provided argument types ({len(arg_types)}) does not match the number of arguments in the kernel signature ({len(fn.signature.parameters)})."
        )
    if not res_types:
        res_types = fn.parse_return_annotation()
    else:
        res_types = [fn.parse_type_annotation(t) for t in res_types]
    if any(isinstance(ty, StreamType) for ty in res_types):
        raise TypeError("Stream is not allowed as a kernel return type.")
    effective_options = fn.options if options is None else options

    try:
        context = Context()
        register_dialect(context)

        # Establish a default context + location so attribute/type construction
        # (which the codegen does without an explicit context=) resolves them.
        with context, Location.unknown(context):
            # initialize builder
            builder = AlloOpBuilder(
                context, typing_style=effective_options.typing_style
            )
            builder.src = fn.src
            builder.file_name = fn.file_name
            builder.begin_line = fn.begin_line
            builder.set_loc(Location.file(fn.file_name, fn.begin_line, 1, context))
            builder.curr_node = None
            module = ModuleOp.create(builder.get_loc())
            builder.module = module
            builder.set_insertion_point_to_end(module.body)

            # start codegen
            generator = MLIRCodeGenerator(
                context,
                module,
                builder,
                kernel=fn,
                func_name=fn.func_name,
                file_name=fn.file_name,
                begin_line=fn.begin_line,
                gscope=fn.get_capture_scope(),
                arg_types=arg_types,
                res_types=res_types,
                options=effective_options,
                active_kernel_calls=[f"kernel:{fn.__module__}.{fn.__qualname__}"],
                is_top=True,
            )
            generator.visit(fn.parse())

            # verify
            try:
                module.operation.verify()
            except MLIRError as e:
                raise InternalCompilerError(
                    "Generated MLIR module failed verification. This likely indicates a bug in the compiler. Please report this to the developers with the kernel code that triggered this error.\n"
                    f"\n===Verification Error===\n"
                    f"{e}\n"
                    "\n===Internal MLIR===\n"
                    f"{module}"
                ) from e

            PassManager.parse("builtin.module(canonicalize,cse)", context).run(
                module.operation
            )
        fn.module = module
        # transfer the ownership of context to kernel
        fn.context = context
        return module
    except (StaticAssertionError, CompilationError, InternalCompilerError) as exc:
        if show_traceback:
            raise
        else:
            raise exc.with_traceback(None) from None
    except Exception as exc:
        if show_traceback:
            raise
        raise RuntimeError(
            "Internal compiler error during Allo kernel compilation.\n"
            f"Error type: {type(exc).__name__}\n"
            f"Error message: {exc}\n"
            "Re-run with show_traceback=True to see the full traceback."
        ) from None
