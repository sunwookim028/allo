# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import ast
import inspect
import textwrap
import functools
import re
from collections.abc import Sequence
from typing import Literal, ParamSpec, Generic, TypeVar, overload
from collections.abc import Callable
from dataclasses import dataclass

from ..lang.core import (
    constexpr,
    TypeBase,
    Template,
    TensorType,
    BufferType,
    DType,
    ShapedType,
    StreamType,
    Stream,
    ShapeExpr,
    StreamExpr,
    Stateful,
    StatefulExpr,
    StatefulType,
    unwrap_if_constexpr,
)
from ..logging import log_fatal

from .._mlir.ir import Module, Context

P = ParamSpec("P")
R = TypeVar("R")
T = TypeVar("T")


@dataclass
class KernelOptions:
    enable_tensor: bool = False
    typing_style: Literal["cpp", "hls"] = "hls"
    fast_math: bool = False


def _register_cmdline_source(fn: Callable) -> None:
    """Make ``python -c`` kernel sources visible to :mod:`inspect`.

    Under ``python -c "<code>"`` CPython < 3.13 compiles the code with
    ``co_filename == "<string>"`` but never stores it where ``linecache`` can
    find it, so :func:`inspect.getsourcelines` raises ``OSError``. The original
    text is still available in :data:`sys.orig_argv`; register it in
    ``linecache`` so inspect's normal lookup resolves it. (Python 3.13+ does
    this natively; notebooks rely on IPython's own ``linecache`` hooks.)
    """
    import linecache
    import sys

    filename = fn.__code__.co_filename
    if filename != "<string>" or filename in linecache.cache:
        return
    # ``sys.argv[0] == "-c"`` is CPython's reliable marker for command mode.
    if not (sys.argv and sys.argv[0] == "-c"):
        return
    # The code string sits right before the script args in ``orig_argv``; the
    # length difference locates it regardless of preceding/combined flags.
    source = sys.orig_argv[len(sys.orig_argv) - len(sys.argv)]
    lines = source.splitlines(keepends=True)
    if lines and not lines[-1].endswith("\n"):
        lines[-1] += "\n"
    # ``mtime=None`` keeps ``linecache.checkcache`` from evicting the entry.
    linecache.cache[filename] = (len(source), None, lines, filename)


class Kernel(Generic[P, R]):
    _module: Module

    def __init__(
        self,
        fn: Callable[P, R],
        *,
        mapping: Sequence[int | Template],
        options: KernelOptions,
        template: Sequence[Template] = (),
        template_bindings: dict[str, object] | None = None,
        definition_scope: dict[str, object] | None = None,
    ):
        assert all(isinstance(arg, Template) for arg in template)
        template_names = [arg.name for arg in template]
        if len(template_names) != len(set(template_names)):
            log_fatal("Template arguments must be unique")
        # verify the mappings
        if mapping and not all(isinstance(m, (int, Template)) for m in mapping):
            log_fatal(
                "Every mapping argument should be either a const int or a template variable"
            )
        self.fn = fn
        self.file_name = fn.__code__.co_filename
        self.func_name = fn.__name__
        self.signature = inspect.signature(fn)
        self.mapping = mapping
        self.options = options
        self.template = tuple(template)
        self.template_bindings = (
            {} if template_bindings is None else template_bindings.copy()
        )
        assert set(self.template_bindings).issubset(set(template_names))
        self.definition_scope = (
            {} if definition_scope is None else definition_scope.copy()
        )
        self._module: Module | None = None
        self.context: Context | None = None

        # record whether this kernel is a lazy consteval kernel
        self._is_lazy_consteval = False

        try:
            raw_src, begin_line = inspect.getsourcelines(fn)
        except OSError:
            _register_cmdline_source(fn)
            try:
                raw_src, begin_line = inspect.getsourcelines(fn)
            except OSError:
                log_fatal(
                    f"Could not retrieve source code for kernel '{fn.__name__}'. "
                    "Kernels must be defined in a source file, a notebook cell, or "
                    "via 'python -c'; sources from the interactive REPL, piped stdin, "
                    "or dynamically generated functions (e.g. exec/eval) cannot be recovered."
                )

        src = textwrap.dedent("".join(raw_src))
        match = re.search(r"^def\s+\w+\s*\(", src, re.MULTILINE)
        if match:
            start_pos = match.start()
            offset = src[:start_pos].count("\n")
            self.begin_line = begin_line + offset
            self.src = src[start_pos:]
        else:
            self.begin_line = begin_line
            self.src = src

        # save metadata
        self.__name__ = fn.__name__
        self.__doc__ = fn.__doc__
        self.__globals__ = fn.__globals__
        self.__module__ = fn.__module__
        self.__qualname__ = fn.__qualname__
        self.capture_scope = self._build_capture_scope()

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:
        """Run the kernel with CPU"""
        from ..backend import CPU

        return CPU(self).run(*args, **kwargs)

    def __str__(self) -> str:
        return str(self.compile())

    def __getitem__(self, bindings):
        if not self.template:
            raise TypeError(f"Kernel '{self.func_name}' has no template arguments")
        if self.template_bindings:
            raise TypeError(f"Kernel '{self.func_name}' is already specialized")
        if not isinstance(bindings, tuple):
            bindings = (bindings,)
        if len(bindings) != len(self.template):
            raise TypeError(
                f"Kernel '{self.func_name}' expects {len(self.template)} template arguments, "
                f"got {len(bindings)}"
            )
        assert len(bindings) == len(self.template), (
            f"Kernel {self.func_name} expects {len(self.template)} template arguments, "
            f"got {len(bindings)}"
        )

        template_bindings = {}
        for template, value in zip(self.template, bindings):
            value = unwrap_if_constexpr(value)
            if not isinstance(value, (TypeBase, int, float)):
                raise TypeError(
                    f"Unsupported template binding for '{template.name}' in kernel "
                    f"'{self.func_name}': {type(value).__name__}"
                )
            template_bindings[template.name] = value

        # apply the mappings here
        mapping = []
        for idx, m in enumerate(self.mapping):
            if isinstance(m, int):
                mapping.append(m)
            else:
                val = template_bindings.get(m.name)
                if isinstance(val, float):
                    log_fatal(
                        f"Mapping argument ({idx}) should bind to a constant int value, but got a float"
                    )

        return Kernel(
            self.fn,
            mapping=mapping,
            options=self.options,
            template=self.template,
            template_bindings=template_bindings,
            definition_scope=self.definition_scope,
        )

    @property
    def module(self) -> Module:
        return self.compile()

    @module.setter
    def module(self, value: Module):
        self._module = value

    def check_templates_bounded(self):
        if not self.template:
            return
        expected = {arg.name for arg in self.template}
        if set(self.template_bindings) != expected:
            missing = ", ".join(sorted(expected - set(self.template_bindings)))
            raise TypeError(
                f"Templated kernel '{self.func_name}' must be specialized before compilation"
                + (f": missing {missing}" if missing else "")
            )

    def parse(self):
        tree = ast.parse(self.src)
        assert isinstance(tree, ast.Module)
        assert len(tree.body) == 1
        assert isinstance(tree.body[0], ast.FunctionDef)
        return tree.body[0]

    def _build_capture_scope(self):
        fn = self.fn
        scope = self.__globals__ | self.definition_scope
        if fn.__closure__ is None:
            return scope | self.template_bindings
        nonlocals = {
            name: cell.cell_contents
            for name, cell in zip(fn.__code__.co_freevars, fn.__closure__)
        }
        return scope | nonlocals | self.template_bindings

    def get_capture_scope(self):
        return self.capture_scope

    def parse_type_annotation(
        self, annotation: object, scope: dict[str, object] | None = None
    ) -> TypeBase:
        if scope is None:
            scope = self.get_capture_scope()
        annotation = unwrap_if_constexpr(annotation)
        # Quoted or `from __future__ import annotations` annotations arrive as
        # source text; evaluate them into the same descriptors that unquoted
        # annotations (e.g. `i32[4]`, `Stream[i32][2,2]`) produce.
        if isinstance(annotation, str):
            annotation = unwrap_if_constexpr(self._eval_annotation(annotation, scope))
        if annotation is constexpr:
            return constexpr
        if isinstance(annotation, Template):
            return self._resolve_template_type(annotation, scope)
        if isinstance(annotation, ShapeExpr):
            return self._resolve_shape_expr(annotation, scope)
        if isinstance(annotation, StreamExpr):
            return self._resolve_stream_expr(annotation, scope)
        if isinstance(annotation, StatefulExpr):
            return self._resolve_stateful_expr(annotation, scope)
        if isinstance(annotation, TypeBase):
            return annotation
        raise TypeError(f"Unsupported type annotation: {annotation!r}")

    def _eval_annotation(self, text: str, scope: dict[str, object]) -> object:
        # `dtype[]` (rank-0) is spelled `dtype[()]` as a Python expression.
        text = re.sub(r"\[\s*\]", "[()]", text.strip())
        # `Stream` and `constexpr` are annotation builtins available without an
        # import; a name in the user's scope shadows them.
        eval_scope = {"Stream": Stream, "Stateful": Stateful, "constexpr": constexpr}
        eval_scope.update(
            {name: unwrap_if_constexpr(value) for name, value in scope.items()}
        )
        try:
            return eval(text, {"__builtins__": {}}, eval_scope)
        except Exception as e:
            raise TypeError(f"Unsupported type annotation '{text}': {e}") from e

    def _resolve_template_type(
        self, template: Template, scope: dict[str, object]
    ) -> TypeBase:
        if template.name not in scope:
            raise TypeError(f"Template '{template.name}' is not bound")
        bound = unwrap_if_constexpr(scope[template.name])
        if not isinstance(bound, TypeBase):
            raise TypeError(
                f"Template '{template.name}' must bind to a type in type annotations"
            )
        return bound

    def _resolve_dtype(self, dtype: object, scope: dict[str, object]) -> DType:
        if isinstance(dtype, Template):
            dtype = self._resolve_template_type(dtype, scope)
        if not isinstance(dtype, DType):
            raise TypeError(f"Expected a scalar dtype, got {dtype!r}")
        return dtype

    def _resolve_shape(self, shape: Sequence, scope: dict[str, object]) -> list[int]:
        dims = []
        for dim in shape:
            dim = unwrap_if_constexpr(dim)
            if isinstance(dim, Template):
                if dim.name not in scope:
                    raise TypeError(f"Template '{dim.name}' is not bound")
                dim = unwrap_if_constexpr(scope[dim.name])
            if type(dim) is not int:
                raise TypeError(f"Shape dimension must be a constexpr int, got {dim!r}")
            if dim < 0:
                raise TypeError(f"Shape dimensions must be non-negative, got {dim}")
            dims.append(dim)
        return dims

    def _resolve_shape_expr(
        self, expr: ShapeExpr, scope: dict[str, object]
    ) -> ShapedType:
        dtype = self._resolve_dtype(expr.dtype, scope)
        shape = self._resolve_shape(expr.shape, scope)
        if self.options.enable_tensor:
            return TensorType(shape=shape, dtype=dtype)
        return BufferType(shape=shape, dtype=dtype)

    def _resolve_stream_expr(
        self, expr: StreamExpr, scope: dict[str, object]
    ) -> StreamType:
        base = unwrap_if_constexpr(expr.base)
        if isinstance(base, ShapeExpr):
            # A stream's transmission unit is always a buffer, never a tensor.
            base_type: DType | ShapedType = BufferType(
                shape=self._resolve_shape(base.shape, scope),
                dtype=self._resolve_dtype(base.dtype, scope),
            )
        elif isinstance(base, Template):
            resolved = self._resolve_template_type(base, scope)
            if not isinstance(resolved, (DType, ShapedType)):
                raise TypeError("Stream base type must be a scalar or buffer type")
            base_type = resolved
        elif isinstance(base, (DType, ShapedType)):
            base_type = base
        else:
            raise TypeError(
                f"Stream base type must be a scalar or buffer type, got {base!r}"
            )
        shape = self._resolve_shape(expr.shape, scope)
        depth = self._resolve_shape((expr.depth,), scope)[0]
        return StreamType(base_type, depth, shape)

    def _resolve_stateful_expr(
        self, expr: StatefulExpr, scope: dict[str, object]
    ) -> StatefulType:
        # A stateful variable must be backed by mutable storage, so an array
        # state is always a buffer (never a tensor), regardless of `enable_tensor`.
        base = unwrap_if_constexpr(expr.base)
        if isinstance(base, ShapeExpr):
            inner: DType | BufferType = BufferType(
                shape=self._resolve_shape(base.shape, scope),
                dtype=self._resolve_dtype(base.dtype, scope),
            )
        else:
            if isinstance(base, Template):
                base = self._resolve_template_type(base, scope)
            if isinstance(base, DType):
                inner = base
            elif isinstance(base, ShapedType):
                inner = BufferType(shape=base.shape, dtype=base.dtype)
            else:
                raise TypeError(
                    f"Stateful base type must be a scalar or buffer type, got {base!r}"
                )
        return StatefulType(inner)

    @functools.cache
    def parse_argument_annotations(self) -> list[TypeBase]:
        arg_types = []
        scope = self.get_capture_scope()
        for param in self.signature.parameters.values():
            annotation = param.annotation
            if annotation is inspect.Parameter.empty:
                raise TypeError(
                    f"Parameter '{param.name}' is missing a type annotation. Please provide an explicit type annotation for all parameters."
                )
            ty = self.parse_type_annotation(annotation, scope=scope)
            if isinstance(ty, StatefulType):
                raise TypeError(
                    f"Parameter '{param.name}' cannot be Stateful; stateful "
                    "variables can only be declared locally within a kernel."
                )
            arg_types.append(ty)
        return arg_types

    @functools.cache
    def parse_return_annotation(self) -> list[TypeBase]:
        annotation = self.signature.return_annotation
        annotation = unwrap_if_constexpr(annotation)
        if annotation is inspect.Signature.empty or annotation is None:
            return []
        scope = self.get_capture_scope()
        # A tuple return (`-> (i32, f32)`) is a real tuple when unquoted, and
        # evaluates to one when quoted; `None`/`"None"` means a void kernel.
        if isinstance(annotation, str):
            annotation = unwrap_if_constexpr(self._eval_annotation(annotation, scope))
            if annotation is None:
                return []
        if isinstance(annotation, tuple):
            res_types = [
                self.parse_type_annotation(elt, scope=scope) for elt in annotation
            ]
        else:
            res_types = [self.parse_type_annotation(annotation, scope=scope)]
        for ty in res_types:
            if isinstance(ty, StreamType):
                raise TypeError("Stream is not allowed as a kernel return type.")
            if isinstance(ty, StatefulType):
                raise TypeError("Stateful is not allowed as a kernel return type.")
        return res_types

    def compile(self):
        if self._module is not None:
            return self._module

        from ..compiler.mlir_codegen import compile

        self.check_templates_bounded()
        arg_types = self.parse_argument_annotations()
        res_types = self.parse_return_annotation()
        return compile(self, arg_types, res_types, options=self.options)

    def schedule(self):
        from ..schedule import Schedule

        if self.template:
            expected = {arg.name for arg in self.template}
            missing = expected - set(self.template_bindings)
            if missing:
                raise TypeError(
                    f"Cannot schedule templated kernel '{self.func_name}': specialize "
                    f"it first (e.g. {self.func_name}[...]); missing "
                    f"{', '.join(sorted(missing))}"
                )
        self.compile()
        return Schedule(context=self.context, kernel=self, primary=self.func_name)


@overload
def kernel(fn: Callable[P, R]) -> Kernel[P, R]: ...


@overload
def kernel(
    *template: Template,
    mapping: Sequence = (),
    options: KernelOptions = KernelOptions(),
) -> Callable[[Callable[P, R]], Kernel[P, R]]: ...


def kernel(
    *args,
    mapping: Sequence = (),
    options: KernelOptions = KernelOptions(),
) -> Kernel[P, R] | Callable[[Callable[P, R]], Kernel[P, R]]:
    frame = inspect.currentframe()
    assert frame is not None and frame.f_back is not None
    definition_scope = frame.f_back.f_locals.copy()
    if len(args) == 1 and callable(args[0]) and not isinstance(args[0], Template):
        fn = args[0]
        template = ()
    else:
        fn = None
        template = args
        assert all(isinstance(arg, Template) for arg in template)

    def decorator(fn: Callable[P, R]) -> Kernel[P, R]:
        assert callable(
            fn
        ), "The @kernel decorator can only be applied to callable objects"
        return Kernel(
            fn,
            mapping=mapping,
            options=options,
            template=template,
            definition_scope=definition_scope,
        )

    if fn is not None:
        return decorator(fn)
    return decorator


class ConstevalFunction(Generic[P, R]):
    def __init__(self, fn: Callable[P, R], lazy: bool):
        self.fn = fn
        self.lazy = lazy
        self.__name__ = fn.__name__
        self.__doc__ = fn.__doc__
        self.__globals__ = fn.__globals__
        self.__module__ = fn.__module__
        self.__qualname__ = fn.__qualname__

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:
        from ..lang.core import unwrap_if_constexpr

        args = [unwrap_if_constexpr(arg) for arg in args]  # type: ignore
        kwargs = {k: unwrap_if_constexpr(v) for k, v in kwargs.items()}  # type: ignore
        return self.fn(*args, **kwargs)


@overload
def consteval(fn: Callable[P, R]) -> ConstevalFunction[P, R]: ...


@overload
def consteval(
    *, lazy: bool = False
) -> Callable[[Callable[P, R]], ConstevalFunction[P, R]]: ...


def consteval(
    fn=None, *, lazy: bool = False
) -> ConstevalFunction[P, R] | Callable[[Callable[P, R]], ConstevalFunction[P, R]]:
    if fn is not None:
        return ConstevalFunction(fn, lazy=lazy)

    def decorator(fn: Callable[P, R]) -> ConstevalFunction[P, R]:
        return ConstevalFunction(fn, lazy=lazy)

    return decorator
