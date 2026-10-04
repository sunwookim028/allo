# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Instantiate a ``compose.Unit`` under its own name, channels and parameters.

``allo/compose.py`` binds a unit's free names from ONE architecture namespace
and names the kernel after the body's def name, so one ``Unit`` composes into
a region at most once and every instance of it would see the same parameters
(``ip_gaps.rst`` row 3; C9 is the ``@df.unit`` half of the same gap). The
instantiation site is where the template needs to say "this PE, that engine".

``instance(unit, name, bind)`` returns a ``Unit`` whose body text has been
rewritten: the def name is ``name`` and every free name in ``bind`` is
replaced by its binding (``{"MAC_MUL": "MAC_MUL__a", "lhs": "lhs_a"}``). It
is a textual prototype of the binding ``compose`` would do itself once a
``Unit`` had positional ports (``stream_ports.rst``), done here so that the
composition experiments can run without a change under ``allo/``. The checks
``compose`` already makes (declaration == body, one owner per channel,
legality on the parameter set) run unchanged on the result.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, replace

from allo.compose import Unit


class _Rename(ast.NodeTransformer):
    def __init__(self, bind):
        self.bind = bind

    def visit_Name(self, node):
        if node.id in self.bind:
            return ast.copy_location(ast.Name(id=self.bind[node.id], ctx=node.ctx), node)
        return node


@dataclass(frozen=True)
class Instance(Unit):
    """A ``Unit`` with its body re-emitted under another name and binding."""

    instance_name: str = ""
    bind: dict = None

    @property
    def name(self) -> str:
        return self.instance_name

    def source(self) -> str:
        tree = ast.parse(Unit.source(self))
        fn = tree.body[0]
        fn.name = self.instance_name
        fn = _Rename(self.bind or {}).visit(fn)
        ast.fix_missing_locations(fn)
        return ast.unparse(fn)


def instance(unit: Unit, name: str, bind: dict = None, **overrides) -> Instance:
    """``bind`` maps a free name of the body (channel or parameter) to the
    name the architecture declares for this instance. Names not in ``bind``
    are kept. ``overrides`` replace declaration fields (``directives``,
    ``legality``, ``instances``, ``memories``)."""
    bind = dict(bind or {})
    sub = lambda names: tuple(bind.get(n, n) for n in names)
    fields = dict(body=unit.body, instances=unit.instances, memories=unit.memories,
                  reads=sub(unit.reads), writes=sub(unit.writes),
                  parameters=sub(unit.parameters), isa=sub(unit.isa),
                  directives=unit.directives, legality=unit.legality,
                  engines=sub(unit.engines), order=unit.order)
    fields.update(overrides)
    return Instance(instance_name=name, bind=bind, **fields)


def engine_bind(slot: str, suffix: str):
    """``("MAC", "a") -> ("MAC__a", {"MAC_IN": "MAC__a_IN", ...})``: the slot
    one instance binds its engine through, and the ``bind`` map that points
    the unit's ``engines=`` names at it (``compose.Engine``, README D-15)."""
    from allo.compose import Engine  # noqa: PLC0415

    new = f"{slot}__{suffix}"
    return new, dict(zip(Engine.names(slot), Engine.names(new)))


def bound(namespace: dict, suffix: str) -> dict:
    """``{"MAC_MUL": f} -> {"MAC_MUL__a": f}``: an engine's names for one
    instance, and the ``bind`` map that points a unit at them."""
    renamed = {f"{k}__{suffix}": v for k, v in namespace.items()}
    return renamed, {k: f"{k}__{suffix}" for k in namespace}
