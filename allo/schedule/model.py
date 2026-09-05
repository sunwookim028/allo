# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Literal

from . import keys as keys_mod
from .errors import ConsumedHandleError
from .._mlir.schedule import ScheduleOpTrait

# A node's identity within a schedule: (enclosing-function scope path, bare key).
NodeKey = tuple[str, str]


@dataclass(frozen=True)
class SourceLoc:
    file: str
    line: int
    col: int

    @classmethod
    def from_raw(cls, raw: Any) -> SourceLoc | None:
        if raw is None:
            return None
        return cls(str(raw["file"]), int(raw["line"]), int(raw["col"]))

    def format(self) -> str:
        return f"{self.file}:{self.line}:{self.col}"


# ---------------------------------------------------------------------------
# Handles — symbolic, scope-relative, re-bindable. Identity is (scope, key);
# matching uses the bare key rooted at the enclosing function handle.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Ref:
    key: str
    scope: str
    kind: str
    name: str | None

    @property
    def skey(self) -> NodeKey:
        return (self.scope, self.key)

    def display_name(self) -> str:
        return self.name or self.key

    def describe(self) -> str:
        return f"{type(self).__name__}('{self.display_name()}', scope='{self.scope}')"


@dataclass(frozen=True)
class OpRef(Ref):
    pass


@dataclass(frozen=True)
class LoopRef(OpRef):
    pass


@dataclass(frozen=True)
class BufferRef(Ref):
    owner_key: str
    number: int
    source: Literal["arg", "res"]


SingleTarget = Ref | str
Targets = SingleTarget | Iterable[SingleTarget] | None


# ---------------------------------------------------------------------------
# Real snapshot — parsed verbatim from the C++ collector, plus a key layer.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OpNode:
    id: str
    kind: str
    name: str | None
    path: str
    parent_id: str | None
    children: tuple[str, ...]
    loc: SourceLoc | None
    traits: ScheduleOpTrait

    def has_trait(self, trait: ScheduleOpTrait) -> bool:
        return bool(self.traits & trait)


@dataclass(frozen=True)
class ValueNode:
    id: str
    owner_id: str
    name: str | None
    type: str
    number: int
    source: Literal["arg", "res"]
    loc: SourceLoc | None


class ScheduleSnapshot:
    """Immutable, id-keyed graph from the C++ collector, indexed by derived key.

    ``ops_by_key`` / ``values_by_key`` are scoped to ``primary_path`` (the function the
    schedule operates on) so a parent's lookups do not see nested-callee loops.
    """

    def __init__(
        self,
        *,
        root_id: str,
        ops: list[OpNode],
        values: list[ValueNode],
        stamped: dict[str, str] | None = None,
        primary_path: str | None = None,
    ):
        self.root_id = root_id
        self.ops = tuple(ops)
        self.values = tuple(values)
        self.primary_path = primary_path

        self.ops_by_id = {node.id: node for node in self.ops}
        self.values_by_id = {node.id: node for node in self.values}
        assert len(self.ops_by_id) == len(self.ops), "duplicate operation ids"
        assert self.root_id in self.ops_by_id, "root operation id missing"

        self._keys = keys_mod.assign_keys(self, stamped)

        # Indexed by bare key, scoped to the primary function (a parent's lookups must
        # not see nested-callee loops). Lookups by name go through the predicted snapshot.
        self.ops_by_key: dict[str, OpNode] = {
            self.relkey_of(node.id): node
            for node in self.ops
            if self._in_primary(self.scope_of(node.id))
        }
        self.values_by_key: dict[str, ValueNode] = {
            self._value_relkey(value): value
            for value in self.values
            if self._in_primary(self.value_scope(value))
        }

    def _in_primary(self, scope: str) -> bool:
        return self.primary_path is None or scope == self.primary_path

    def value_scope(self, value: ValueNode) -> str:
        """Enclosing-function scope a value belongs to: the function it is
        defined in. An op result lives in the op's enclosing function; a
        function's block argument belongs to the function itself (whereas
        ``scope_of`` would place the function in *its* enclosing scope)."""
        owner = self.ops_by_id[value.owner_id]
        if owner.has_trait(ScheduleOpTrait.FUNCTION_LIKE):
            return owner.path
        return self.scope_of(value.owner_id)

    @classmethod
    def from_raw(
        cls,
        raw: dict[str, Any],
        stamped: dict[str, str] | None = None,
        primary_path: str | None = None,
    ) -> ScheduleSnapshot:
        ops = [
            OpNode(
                id=str(item["id"]),
                kind=str(item["kind"]),
                name=None if item["name"] is None else str(item["name"]),
                path=str(item["path"]),
                parent_id=(
                    None if item["parent_id"] is None else str(item["parent_id"])
                ),
                children=tuple(str(child) for child in item["children"]),
                loc=SourceLoc.from_raw(item["loc"]),
                traits=ScheduleOpTrait(int(item["traits"])),
            )
            for item in raw["ops"]
        ]
        values = [
            ValueNode(
                id=str(item["id"]),
                owner_id=str(item["owner_id"]),
                name=None if item["name"] is None else str(item["name"]),
                type=str(item["type"]),
                number=int(item["number"]),
                source=item["source"],
                loc=SourceLoc.from_raw(item["loc"]),
            )
            for item in raw["values"]
        ]
        return cls(
            root_id=str(raw["root_id"]),
            ops=ops,
            values=values,
            stamped=stamped,
            primary_path=primary_path,
        )

    # --- key access -------------------------------------------------------

    def scope_of(self, op_id: str) -> str:
        return self._keys.scope_by_id[op_id]

    def relkey_of(self, op_id: str) -> str:
        return self._keys.relkey_by_id[op_id]

    @property
    def relkey_by_id(self) -> dict[str, str]:
        return self._keys.relkey_by_id

    def skey_of(self, op_id: str) -> NodeKey:
        return (self.scope_of(op_id), self.relkey_of(op_id))

    def id_of(self, scope: str, key: str) -> str | None:
        return self._keys.id_by_scope_key.get((scope, key))

    def _value_relkey(self, value: ValueNode) -> str:
        owner_relkey = self._keys.relkey_by_id[value.owner_id]
        return f"{owner_relkey}:{value.source}{value.number}"

    # --- ref construction -------------------------------------------------

    def op_ref(self, op_id: str) -> OpRef:
        node = self.ops_by_id[op_id]
        cls = LoopRef if node.has_trait(ScheduleOpTrait.LOOP_LIKE) else OpRef
        return cls(
            key=self.relkey_of(op_id),
            scope=self.scope_of(op_id),
            kind=node.kind,
            name=node.name,
        )

    def buffer_ref(self, value_id: str) -> BufferRef:
        value = self.values_by_id[value_id]
        return BufferRef(
            key=self._value_relkey(value),
            scope=self.value_scope(value),
            kind="buffer",
            name=value.name,
            owner_key=self.relkey_of(value.owner_id),
            number=value.number,
            source=value.source,
        )

    # --- structure --------------------------------------------------------

    def format_tree(self, *, include_values: bool = True) -> str:
        lines: list[str] = []

        def append_node(node: OpNode, prefix: str, is_last: bool) -> None:
            marker = "" if node.id == self.root_id else ("`- " if is_last else "|- ")
            child_prefix = prefix
            if node.id != self.root_id:
                child_prefix += "   " if is_last else "|  "
            key = self.relkey_of(node.id)
            loc = "" if node.loc is None else f" loc={node.loc.format()}"
            lines.append(
                f"{prefix}{marker}{node.name or key} kind={node.kind} key={key}{loc}"
            )
            if include_values:
                node_values = [v for v in self.values if v.owner_id == node.id]
                for idx, value in enumerate(node_values):
                    value_last = idx == len(node_values) - 1 and not node.children
                    value_marker = "`- " if value_last else "|- "
                    lines.append(
                        f"{child_prefix}{value_marker}"
                        f"{value.name or self._value_relkey(value)} "
                        f"type={value.type} key={self._value_relkey(value)}"
                    )
            for idx, child_id in enumerate(node.children):
                child = self.ops_by_id[child_id]
                append_node(child, child_prefix, idx == len(node.children) - 1)

        append_node(self.ops_by_id[self.root_id], "", True)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Predicted snapshot — mutable mirror of the sequence's END state, indexed by
# (scope, key). Tracks every function (incl. nested callee copies) so compose
# can target them; queries project to the primary function.
# ---------------------------------------------------------------------------


@dataclass
class PredictedOp:
    scope: str
    key: str
    kind: str
    name: str | None
    parent: NodeKey | None
    children: list[NodeKey]
    traits: ScheduleOpTrait
    exact: bool = True

    @property
    def skey(self) -> NodeKey:
        return (self.scope, self.key)

    def has_trait(self, trait: ScheduleOpTrait) -> bool:
        return bool(self.traits & trait)


@dataclass
class PredictedValue:
    scope: str
    key: str
    name: str | None
    owner: NodeKey
    number: int
    source: Literal["arg", "res"]

    @property
    def skey(self) -> NodeKey:
        return (self.scope, self.key)


class PredictedSnapshot:
    """In-Python forecast of the schedule's end state. Identity is exact (we mint
    every created key); structure of analysis-heavy ops is marked ``exact=False``
    and overwritten from the real snapshot at apply."""

    def __init__(self, root_scope: str, root_key: str):
        self.root_scope = root_scope
        self.root_key = root_key
        self._by_key: dict[NodeKey, PredictedOp] = {}
        self._values: dict[NodeKey, PredictedValue] = {}
        self._consumed: dict[NodeKey, str] = {}  # skey -> consuming primitive

    @classmethod
    def from_real(cls, real: ScheduleSnapshot) -> PredictedSnapshot:
        pred = cls(real.scope_of(real.root_id), real.relkey_of(real.root_id))
        for node in real.ops:
            sk = real.skey_of(node.id)
            pred._by_key[sk] = PredictedOp(
                scope=sk[0],
                key=sk[1],
                kind=node.kind,
                name=node.name,
                parent=(
                    None if node.parent_id is None else real.skey_of(node.parent_id)
                ),
                children=[real.skey_of(c) for c in node.children],
                traits=node.traits,
            )
        for value in real.values:
            scope = real.value_scope(value)
            relkey = real._value_relkey(value)
            pred._values[(scope, relkey)] = PredictedValue(
                scope=scope,
                key=relkey,
                name=value.name,
                owner=real.skey_of(value.owner_id),
                number=value.number,
                source=value.source,
            )
        return pred

    # --- queryable views (used by Query) ---------------------------------

    @property
    def ops(self) -> list[PredictedOp]:
        return list(self._by_key.values())

    @property
    def values(self) -> list[PredictedValue]:
        return list(self._values.values())

    def ops_by_name(self, name: str) -> list[PredictedOp]:
        return [op for op in self._by_key.values() if op.name == name]

    def values_by_name(self, name: str) -> list[PredictedValue]:
        return [v for v in self._values.values() if v.name == name]

    def op(self, scope: str, key: str) -> PredictedOp | None:
        return self._by_key.get((scope, key))

    def require_live(self, ref: Ref) -> None:
        if isinstance(ref, BufferRef):
            if ref.skey not in self._values:
                raise ConsumedHandleError(f"{ref.describe()} is no longer live")
            return
        if ref.skey not in self._by_key:
            consumer = self._consumed.get(ref.skey)
            note = f"; consumed by {consumer}()" if consumer else ""
            raise ConsumedHandleError(f"{ref.describe()} is no longer live{note}")

    def is_under(self, op: PredictedOp, ancestor: NodeKey) -> bool:
        current: PredictedOp | None = op
        while current is not None:
            if current.skey == ancestor:
                return True
            current = (
                None if current.parent is None else self._by_key.get(current.parent)
            )
        return False

    def depth(self, op: PredictedOp) -> int:
        depth = 0
        current = op
        while current.parent is not None:
            parent = self._by_key.get(current.parent)
            if parent is None:
                break
            depth += 1
            current = parent
        return depth

    # --- ref construction -------------------------------------------------

    def make_op_ref(self, op: PredictedOp) -> OpRef:
        cls = LoopRef if op.has_trait(ScheduleOpTrait.LOOP_LIKE) else OpRef
        return cls(key=op.key, scope=op.scope, kind=op.kind, name=op.name)

    def make_loop_ref(self, op: PredictedOp) -> LoopRef:
        return LoopRef(key=op.key, scope=op.scope, kind=op.kind, name=op.name)

    def make_buffer_ref(self, value: PredictedValue) -> BufferRef:
        return BufferRef(
            key=value.key,
            scope=value.scope,
            kind="buffer",
            name=value.name,
            owner_key=value.owner[1],
            number=value.number,
            source=value.source,
        )

    # --- mutators (transitions) ------------------------------------------

    def _add_op(
        self,
        scope: str,
        key: str,
        kind: str,
        name: str | None,
        parent: NodeKey | None,
        children: list[NodeKey],
        traits: ScheduleOpTrait,
        *,
        exact: bool,
    ) -> PredictedOp:
        op = PredictedOp(
            scope=scope,
            key=key,
            kind=kind,
            name=name,
            parent=parent,
            children=children,
            traits=traits,
            exact=exact,
        )
        self._by_key[op.skey] = op
        return op

    def _replace_in_parent(
        self, parent: NodeKey | None, old: NodeKey, new: NodeKey
    ) -> None:
        if parent is None:
            return
        node = self._by_key.get(parent)
        if node is not None:
            node.children = [new if c == old else c for c in node.children]

    def split(
        self, loop: PredictedOp, outer_key: str, inner_key: str
    ) -> tuple[PredictedOp, PredictedOp]:
        outer_sk = (loop.scope, outer_key)
        inner_sk = (loop.scope, inner_key)
        body = list(loop.children)
        self._replace_in_parent(loop.parent, loop.skey, outer_sk)
        outer = self._add_op(
            loop.scope,
            outer_key,
            loop.kind,
            None,
            loop.parent,
            [inner_sk],
            loop.traits,
            exact=True,
        )
        inner = self._add_op(
            loop.scope,
            inner_key,
            loop.kind,
            None,
            outer_sk,
            body,
            loop.traits,
            exact=True,
        )
        for child in body:
            node = self._by_key.get(child)
            if node is not None:
                node.parent = inner_sk
        del self._by_key[loop.skey]
        self._consumed[loop.skey] = "split"
        return outer, inner

    def flip_kind(self, loop: PredictedOp, new_kind: str) -> PredictedOp:
        loop.kind = new_kind
        loop.traits = (
            loop.traits | ScheduleOpTrait.AFFINE_FOR | ScheduleOpTrait.AFFINE_LOOP
        )
        return loop

    def flatten(self, band: list[PredictedOp], flat_key: str) -> PredictedOp:
        band = sorted(band, key=self.depth)
        outermost, innermost = band[0], band[-1]
        body = list(innermost.children)
        flat_sk = (outermost.scope, flat_key)
        self._replace_in_parent(outermost.parent, outermost.skey, flat_sk)
        for op in band:
            self._by_key.pop(op.skey, None)
            self._consumed[op.skey] = "flatten"
        flat = self._add_op(
            outermost.scope,
            flat_key,
            outermost.kind,
            None,
            outermost.parent,
            body,
            outermost.traits,
            exact=True,
        )
        for child in body:
            node = self._by_key.get(child)
            if node is not None:
                node.parent = flat.skey
        return flat

    def tile(
        self, band: list[PredictedOp], tile_keys: list[str], point_keys: list[str]
    ) -> tuple[list[PredictedOp], list[PredictedOp]]:
        band = sorted(band, key=self.depth)
        outermost, innermost = band[0], band[-1]
        body = list(innermost.children)
        scope, traits = outermost.scope, outermost.traits
        chain = [(scope, k) for k in tile_keys + point_keys]
        self._replace_in_parent(outermost.parent, outermost.skey, chain[0])
        for op in band:
            self._by_key.pop(op.skey, None)
            self._consumed[op.skey] = "tile"
        new_ops: list[PredictedOp] = []
        for idx, key in enumerate(tile_keys + point_keys):
            parent = outermost.parent if idx == 0 else chain[idx - 1]
            children = [chain[idx + 1]] if idx + 1 < len(chain) else list(body)
            new_ops.append(
                self._add_op(
                    scope,
                    key,
                    outermost.kind,
                    None,
                    parent,
                    children,
                    traits,
                    exact=True,
                )
            )
        for child in body:
            node = self._by_key.get(child)
            if node is not None:
                node.parent = chain[-1]
        n = len(tile_keys)
        return new_ops[:n], new_ops[n:]

    def reorder(self, order: list[PredictedOp]) -> None:
        # `order` lists the loops outermost-first in the desired nesting. They form a
        # perfect band today; only their relative nesting changes.
        band = {op.skey for op in order}
        cur_outer = min(order, key=self.depth)
        cur_inner = max(order, key=self.depth)
        top_parent = cur_outer.parent
        body = [c for c in cur_inner.children if c not in band]
        self._replace_in_parent(top_parent, cur_outer.skey, order[0].skey)
        for idx, op in enumerate(order):
            op.parent = top_parent if idx == 0 else order[idx - 1].skey
            op.children = [order[idx + 1].skey] if idx + 1 < len(order) else list(body)
        for child in body:
            node = self._by_key.get(child)
            if node is not None:
                node.parent = order[-1].skey

    def reparent_approx(self, op: PredictedOp, new_parent: NodeKey) -> None:
        if op.parent is not None:
            old = self._by_key.get(op.parent)
            if old is not None and op.skey in old.children:
                old.children.remove(op.skey)
        op.parent = new_parent
        parent = self._by_key.get(new_parent)
        if parent is not None:
            parent.children.append(op.skey)
            parent.exact = False
        op.exact = False

    def mark_approx(self, op: PredictedOp) -> None:
        op.exact = False
        for child in list(op.children):
            node = self._by_key.get(child)
            if node is not None:
                self.mark_approx(node)

    def add_alloc(
        self, scope: str, key: str, kind: str, parent: NodeKey | None
    ) -> PredictedOp:
        op = self._add_op(
            scope,
            key,
            kind,
            None,
            parent,
            [],
            ScheduleOpTrait.MEMORY_ALLOCATE,
            exact=False,
        )
        if parent is not None:
            node = self._by_key.get(parent)
            if node is not None:
                node.children.append(op.skey)
        return op

    def add_value(
        self, owner: NodeKey, number: int, source: Literal["arg", "res"]
    ) -> PredictedValue:
        scope = owner[0]
        key = f"{owner[1]}:{source}{number}"
        value = PredictedValue(
            scope=scope,
            key=key,
            name=None,
            owner=owner,
            number=number,
            source=source,
        )
        self._values[(scope, key)] = value
        return value

    def add_function(self, scope: str, key: str, parent: NodeKey | None) -> PredictedOp:
        return self._add_op(
            scope,
            key,
            "func.func",
            key,
            parent,
            [],
            ScheduleOpTrait.FUNCTION_LIKE | ScheduleOpTrait.SYMBOL,
            exact=False,
        )

    # --- reconciliation ---------------------------------------------------

    def reconcile(self, real: ScheduleSnapshot) -> None:
        """Assert every exact loop prediction holds in the real snapshot (the safety
        net), then adopt the real snapshot as the new ground truth. Only loops are
        asserted: they carry stable stamped keys, whereas body ops (loads/stores) are
        rewritten/relocated by transforms and re-keyed structurally — they are never
        referenced by the schedule and are rebuilt from ``real`` below."""
        for op in self._by_key.values():
            if not op.exact or not op.has_trait(ScheduleOpTrait.LOOP_LIKE):
                continue
            real_id = real.id_of(op.scope, op.key)
            assert (
                real_id is not None
            ), f"predicted loop key '{op.key}' (scope '{op.scope}') missing after apply"
            real_kind = real.ops_by_id[real_id].kind
            assert (
                real_kind == op.kind
            ), f"predicted kind for '{op.key}' was {op.kind}, payload has {real_kind}"
        fresh = PredictedSnapshot.from_real(real)
        self._by_key = fresh._by_key
        self._values = fresh._values
        self._consumed = {}
        self.root_scope = fresh.root_scope
        self.root_key = fresh.root_key
