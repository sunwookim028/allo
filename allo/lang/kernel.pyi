# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Typing stubs for the kernel decorator and its handle."""

from inspect import Signature
from typing import (
    Any,
    Callable,
    Generic,
    Literal,
    ParamSpec,
    TypeAlias,
    TypeVar,
    overload,
)

from ..schedule.core import Schedule
from .core import Template
from ..lang.core import TypeBase

P = ParamSpec("P")
R = TypeVar("R")

# The native ``_mlir`` bindings carry no stubs, so the compiled module is ``Any``.
Module: TypeAlias = Any

class KernelOptions:
    enable_tensor: bool
    typing_style: Literal["cpp", "hls"]
    fast_math: bool
    def __init__(
        self,
        enable_tensor: bool = ...,
        typing_style: Literal["cpp", "hls"] = ...,
        fast_math: bool = ...,
    ) -> None: ...

class Kernel(Generic[P, R]):
    fn: Callable[P, R]
    func_name: str
    signature: Signature
    options: KernelOptions
    module: Module
    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R: ...
    def __getitem__(self, bindings: Any) -> Kernel[P, R]: ...
    def __str__(self) -> str: ...
    def schedule(self) -> Schedule[P, R]: ...
    def compile(self) -> Module: ...
    def parse_argument_annotations(self) -> list[TypeBase]: ...
    def parse_return_annotation(self) -> list[TypeBase]: ...

@overload
def kernel(fn: Callable[P, R]) -> Kernel[P, R]: ...
@overload
def kernel(
    *template: Template,
    mapping: Any = ...,
    options: KernelOptions = ...,
) -> Callable[[Callable[P, R]], Kernel[P, R]]: ...

class ConstevalFunction(Generic[P, R]):
    fn: Callable[P, R]
    lazy: bool
    def __init__(self, fn: Callable[P, R], lazy: bool) -> None: ...
    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R: ...

@overload
def consteval(fn: Callable[P, R]) -> ConstevalFunction[P, R]: ...
@overload
def consteval(
    *, lazy: bool = ...
) -> Callable[[Callable[P, R]], ConstevalFunction[P, R]]: ...
