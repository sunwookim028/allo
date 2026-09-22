# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Instantiating a ``@df.unit`` inside a ``@df.region`` body.

A unit declares its stream ports, so a region wires them by position rather
than by hoping the unit and the region agree on a name. This module turns each
instantiation in a region body into the nested kernel the rest of the frontend
already knows how to build, and records for that kernel which region-scope
stream each of its ports is bound to.

The wiring itself is checked before anything is built: see :mod:`allo.netlist`.
"""

import ast
import copy

from ..netlist import (
    Channel,
    Instance,
    Netlist,
    REGISTRY,
    UnitSpec,
    _evaluate_annotation,
    as_stream,
    stream_uses,
)


def unit_spec(obj):
    return getattr(obj, "__allo_unit__", None)


def _decorator(node, name):
    for decorator in node.decorator_list:
        if (
            isinstance(decorator, ast.Call)
            and isinstance(decorator.func, ast.Attribute)
            and decorator.func.attr == name
        ):
            return decorator
    return None


def is_unit_instance(node):
    return hasattr(node, "port_bindings")


def is_region(node):
    return isinstance(node, ast.FunctionDef) and _decorator(node, "region") is not None


def _keyword(call, name):
    for keyword in call.keywords:
        if keyword.arg == name:
            return keyword.value
    return None


def _channels(body, global_vars):
    channels = {}
    for statement in body:
        if not (
            isinstance(statement, ast.AnnAssign)
            and isinstance(statement.target, ast.Name)
            and statement.value is None
        ):
            continue
        stream = as_stream(_evaluate_annotation(statement.annotation, global_vars))
        if stream is None:
            continue
        dtype, array_shape = stream
        channels[statement.target.id] = Channel(statement.target.id, dtype, array_shape)
    return channels


def _lexical_uses(body):
    """Stream uses inside kernels the region still nests, by direction."""
    lexical = {}
    for statement in body:
        if not (
            isinstance(statement, ast.FunctionDef) and _decorator(statement, "kernel")
        ):
            continue
        for name, (directions, _) in stream_uses(statement.body).items():
            for direction in directions:
                lexical.setdefault((name, direction), []).append(statement.name)
    return lexical


def _instantiation(statement, global_vars):
    """``(instance name, unit, call)`` if ``statement`` instantiates a unit."""
    if isinstance(statement, ast.Expr):
        call, name = statement.value, None
    elif (
        isinstance(statement, ast.Assign)
        and len(statement.targets) == 1
        and isinstance(statement.targets[0], ast.Name)
    ):
        call, name = statement.value, statement.targets[0].id
    else:
        return None
    if not (isinstance(call, ast.Call) and isinstance(call.func, ast.Name)):
        return None
    spec = unit_spec(global_vars.get(call.func.id))
    if spec is None:
        return None
    return name, spec, call


def _bindings(spec, call):
    """Which region-scope name each parameter of ``spec`` is wired to."""
    positional = [port.name for port in spec.ports] + list(spec.values)
    bindings = {}
    for index, argument in enumerate(call.args):
        if index < len(positional) and isinstance(argument, ast.Name):
            bindings[positional[index]] = argument.id
    for keyword in call.keywords:
        if keyword.arg is not None and isinstance(keyword.value, ast.Name):
            bindings[keyword.arg] = keyword.value.id
    return bindings


def _as_kernel(spec, name, bindings, mapping):
    """The nested ``@df.kernel`` this instance is, with its ports bound."""
    tree = copy.deepcopy(spec.tree)
    tree.name = name
    tree.args.args = [
        argument for argument in tree.args.args if argument.arg in spec.values
    ]
    keywords = [
        ast.keyword(
            arg="mapping",
            value=ast.List(
                elts=[ast.Constant(value=extent) for extent in mapping],
                ctx=ast.Load(),
            ),
        )
    ]
    if spec.values:
        keywords.append(
            ast.keyword(
                arg="args",
                value=ast.List(
                    elts=[
                        ast.Name(id=bindings[value], ctx=ast.Load())
                        for value in spec.values
                    ],
                    ctx=ast.Load(),
                ),
            )
        )
    tree.decorator_list = [
        ast.Call(
            func=ast.Attribute(
                value=ast.Name(id="df", ctx=ast.Load()), attr="kernel", ctx=ast.Load()
            ),
            args=[],
            keywords=keywords,
        )
    ]
    tree.port_bindings = {
        port.name: (bindings[port.name], port.array_shape) for port in spec.ports
    }
    tree.unit_globals = spec.namespace
    ast.fix_missing_locations(tree)
    return tree


def expand_region_units(node, global_vars):
    """Rewrite a region body's unit instantiations into nested kernels.

    Returns the checked :class:`allo.netlist.Netlist`, or ``None`` when the
    region instantiates no unit -- in which case nothing here has touched it.
    """
    if getattr(node, "units_expanded", False):
        return REGISTRY.get(node.name)
    node.units_expanded = True
    found = [
        (statement, _instantiation(statement, global_vars)) for statement in node.body
    ]
    if not any(instance is not None for _, instance in found):
        return None

    netlist = Netlist(
        region=node.name,
        channels=_channels(node.body, global_vars),
        lexical=_lexical_uses(node.body),
    )
    plan, counter = [], {}
    for statement, instance in found:
        if instance is None:
            plan.append(statement)
            continue
        name, spec, call = instance
        if name is None:
            index = counter.setdefault(spec.name, 0)
            counter[spec.name] = index + 1
            name = f"{spec.name}__{index}"
        bindings = _bindings(spec, call)
        netlist.instances.append(Instance(name, spec, bindings))
        plan.append((spec, name, bindings))

    region_decorator = _decorator(node, "region")
    because = None
    if region_decorator is not None:
        premise = _keyword(region_decorator, "deadlock_free_because")
        if premise is not None:
            because = _evaluate_annotation(premise, global_vars)
    netlist.obligation = netlist.check(because=because)
    REGISTRY[node.name] = netlist

    node.body = [
        (
            item
            if isinstance(item, ast.stmt)
            else _as_kernel(item[0], item[1], item[2], item[0].mapping)
        )
        for item in plan
    ]
    return netlist


def bind_ports(ctx, node):
    """Bind each declared port of a unit instance to the stream it is wired to.

    A port is a name in the unit's own scope; the stream it names lives in the
    enclosing region's. Binding one to the other is the whole of what a port
    is at build time -- everything downstream already moves a kernel's streams
    to its interface.
    """
    for port, (channel, array_shape) in getattr(node, "port_bindings", {}).items():
        if not array_shape:
            ctx.put_symbol(name=port, val=ctx.get_symbol(name=channel))
            continue
        ctx.put_symbol(name=port, val=ctx.get_symbol(name=channel))
        for index in _indices(array_shape):
            suffix = "_".join(str(i) for i in index)
            element = ctx.get_symbol(name=f"{channel}_{suffix}", allow_missing=True)
            if element is not None:
                ctx.put_symbol(name=f"{port}_{suffix}", val=element)


def _indices(shape):
    if not shape:
        return [()]
    head, rest = shape[0], _indices(tuple(shape[1:]))
    return [(index,) + tail for index in range(head) for tail in rest]


__all__ = [
    "UnitSpec",
    "bind_ports",
    "expand_region_units",
    "is_region",
    "is_unit_instance",
    "unit_spec",
]
