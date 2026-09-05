# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Callable, Sequence

from .core import (
    APFloat,
    APInt,
    DType,
    IndexType,
    bool as AlloBool,
    f32,
    f64,
    index,
)


class _SignedInt:
    pass


class _UnsignedInt:
    pass


RuleFn = Callable[..., DType]


def _int_kind(dtype: DType) -> type:
    if dtype.is_int():
        return _SignedInt
    if dtype.is_uint():
        return _UnsignedInt
    return type(dtype)


class TypingRule:
    def __init__(
        self, *rule_dicts: dict[tuple[type, ...], RuleFn], commutative: bool = False
    ):
        self.rules: dict[tuple[type, ...], RuleFn] = {}
        for rule_dict in rule_dicts:
            self.rules.update(rule_dict)
        self.commutative = commutative

    def call_unary(self, dtype: DType) -> DType | None:
        fn = self.rules.get((_int_kind(dtype),))
        return None if fn is None else fn(dtype)

    def call_binary(self, lhs: DType, rhs: DType) -> DType | None:
        lhs_kind = _int_kind(lhs)
        rhs_kind = _int_kind(rhs)
        fn = self.rules.get((lhs_kind, rhs_kind))
        if fn is not None:
            return fn(lhs, rhs)
        if self.commutative:
            fn = self.rules.get((rhs_kind, lhs_kind))
            if fn is not None:
                return fn(rhs, lhs)
        return None


class TypeRuleTable:
    def __init__(self, style: str, nary_promoter: Callable | None = None):
        self.style = style
        self.rules: dict[str, TypingRule] = {}
        self.nary_promoter = nary_promoter

    def register(self, op_keys: str | Sequence[str], rule: TypingRule):
        if isinstance(op_keys, str):
            op_keys = (op_keys,)
        for key in op_keys:
            self.rules[key] = rule

    def lookup(self, op_key: str, *dtypes: DType) -> DType | None:
        rule = self.rules.get(op_key)
        if rule is None:
            return None
        if len(dtypes) == 1:
            return rule.call_unary(dtypes[0])
        assert len(dtypes) == 2
        return rule.call_binary(dtypes[0], dtypes[1])

    def promote(
        self,
        op_key: str,
        dtypes: Sequence[DType],
        *,
        term_signs: Sequence[int] | None = None,
    ) -> DType | None:
        assert len(dtypes) > 0
        if self.nary_promoter is not None:
            promoted = self.nary_promoter(op_key, dtypes, term_signs)
            if promoted is not None:
                return promoted

        if len(dtypes) == 1:
            return self.lookup(op_key, dtypes[0])

        ret = dtypes[0]
        for dtype in dtypes[1:]:
            ret = self.lookup(op_key, ret, dtype)
            if ret is None:
                return None
        return ret


def select_cpp_common_int_type(lhs: DType, rhs: DType) -> APInt:
    assert lhs.is_int_signless() and rhs.is_int_signless()

    lhs_width = lhs.primitive_width
    rhs_width = rhs.primitive_width
    lhs_signed = lhs.is_int()
    rhs_signed = rhs.is_int()

    if lhs_signed == rhs_signed:
        return APInt(max(lhs_width, rhs_width), signed=lhs_signed)

    signed_width = lhs_width if lhs_signed else rhs_width
    unsigned_width = rhs_width if lhs_signed else lhs_width
    if unsigned_width >= signed_width:
        return APInt(unsigned_width, signed=False)
    return APInt(signed_width, signed=True)


def select_hls_common_int_type(lhs: DType, rhs: DType) -> APInt:
    # A mixed-sign pair promotes to a signed type wide enough for both ranges,
    # so results follow the operand values rather than C's unsigned-domain
    # reinterpretation.
    assert lhs.is_int_signless() and rhs.is_int_signless()
    if lhs.is_int() == rhs.is_int():
        return APInt(max(lhs.primitive_width, rhs.primitive_width), signed=lhs.is_int())
    signed_width = lhs.primitive_width if lhs.is_int() else rhs.primitive_width
    unsigned_width = rhs.primitive_width if lhs.is_int() else lhs.primitive_width
    return APInt(max(signed_width, unsigned_width + 1), signed=True)


def _ceil_log2(value: int) -> int:
    assert value >= 1
    return (value - 1).bit_length()


def _wider_float(lhs: APFloat, rhs: APFloat) -> APFloat:
    return lhs if lhs.primitive_width >= rhs.primitive_width else rhs


def _select_float(dtype: DType) -> APFloat:
    return f32 if dtype.primitive_width <= f32.primitive_width else f64


def _hls_nary_int_promotion(
    op_key: str,
    dtypes: Sequence[DType],
    term_signs: Sequence[int] | None = None,
) -> DType | None:
    if term_signs is not None:
        assert op_key in {"add", "sub"}
        assert len(term_signs) == len(dtypes)

    if not all(dtype.is_int_signless() for dtype in dtypes):
        return None

    if op_key in {"add", "sub"}:
        signs = [1] * len(dtypes) if term_signs is None else list(term_signs)

        signed = any(sign < 0 for sign in signs) or any(
            dtype.is_int() for dtype in dtypes
        )
        widths = []
        for dtype in dtypes:
            width = dtype.primitive_width
            if signed and dtype.is_uint():
                width += 1
            widths.append(width)
        return APInt(max(widths) + _ceil_log2(len(dtypes)), signed=signed)

    if op_key == "mul":
        width = sum(dtype.primitive_width for dtype in dtypes)
        signed = any(dtype.is_int() for dtype in dtypes)
        return APInt(width, signed=signed)

    return None


def _common_numeric_rule(
    common_int: Callable[[DType, DType], DType],
    *,
    include_index: bool = True,
    commutative: bool = False,
) -> TypingRule:
    int_rules = {
        (_SignedInt, _SignedInt): common_int,
        (_SignedInt, _UnsignedInt): common_int,
        (_SignedInt, APFloat): lambda lhs, rhs: rhs,
    }
    uint_rules = {
        (_UnsignedInt, _UnsignedInt): common_int,
        (_UnsignedInt, _SignedInt): common_int,
        (_UnsignedInt, APFloat): lambda lhs, rhs: rhs,
    }
    index_rules = (
        {
            (IndexType, IndexType): lambda lhs, rhs: index,
            (IndexType, _SignedInt): lambda lhs, rhs: index,
            (IndexType, _UnsignedInt): lambda lhs, rhs: index,
            (_SignedInt, IndexType): lambda lhs, rhs: index,
            (_UnsignedInt, IndexType): lambda lhs, rhs: index,
        }
        if include_index
        else {}
    )
    float_rules = {
        (APFloat, APFloat): _wider_float,
        (APFloat, _SignedInt): lambda lhs, rhs: lhs,
        (APFloat, _UnsignedInt): lambda lhs, rhs: lhs,
    }
    return TypingRule(
        int_rules, uint_rules, index_rules, float_rules, commutative=commutative
    )


def _hls_add_sub_rule() -> TypingRule:
    int_rules = {
        (_SignedInt, _SignedInt): lambda lhs, rhs: APInt(
            max(lhs.primitive_width, rhs.primitive_width) + 1, signed=True
        ),
        (_SignedInt, _UnsignedInt): lambda lhs, rhs: APInt(
            max(lhs.primitive_width, rhs.primitive_width + 1) + 1, signed=True
        ),
        (_SignedInt, APFloat): lambda lhs, rhs: rhs,
    }
    uint_rules = {
        (_UnsignedInt, _UnsignedInt): lambda lhs, rhs: APInt(
            max(lhs.primitive_width, rhs.primitive_width) + 1, signed=False
        ),
        (_UnsignedInt, _SignedInt): lambda lhs, rhs: APInt(
            max(lhs.primitive_width + 1, rhs.primitive_width) + 1, signed=True
        ),
        (_UnsignedInt, APFloat): lambda lhs, rhs: rhs,
    }
    index_rules = {
        (IndexType, IndexType): lambda lhs, rhs: index,
        (IndexType, _SignedInt): lambda lhs, rhs: index,
        (IndexType, _UnsignedInt): lambda lhs, rhs: index,
        (_SignedInt, IndexType): lambda lhs, rhs: index,
        (_UnsignedInt, IndexType): lambda lhs, rhs: index,
    }
    float_rules = {
        (APFloat, APFloat): _wider_float,
        (APFloat, _SignedInt): lambda lhs, rhs: lhs,
        (APFloat, _UnsignedInt): lambda lhs, rhs: lhs,
    }
    return TypingRule(int_rules, uint_rules, index_rules, float_rules)


def _hls_mul_rule() -> TypingRule:
    int_rules = {
        (_SignedInt, _SignedInt): lambda lhs, rhs: APInt(
            lhs.primitive_width + rhs.primitive_width, signed=True
        ),
        (_SignedInt, _UnsignedInt): lambda lhs, rhs: APInt(
            lhs.primitive_width + rhs.primitive_width, signed=True
        ),
        (_SignedInt, APFloat): lambda lhs, rhs: rhs,
    }
    uint_rules = {
        (_UnsignedInt, _UnsignedInt): lambda lhs, rhs: APInt(
            lhs.primitive_width + rhs.primitive_width, signed=False
        ),
        (_UnsignedInt, APFloat): lambda lhs, rhs: rhs,
    }
    index_rules = {
        (IndexType, IndexType): lambda lhs, rhs: index,
        (IndexType, _SignedInt): lambda lhs, rhs: index,
        (IndexType, _UnsignedInt): lambda lhs, rhs: index,
        (_SignedInt, IndexType): lambda lhs, rhs: index,
        (_UnsignedInt, IndexType): lambda lhs, rhs: index,
    }
    float_rules = {(APFloat, APFloat): _wider_float}
    return TypingRule(int_rules, uint_rules, index_rules, float_rules, commutative=True)


def _hls_pow_rule() -> TypingRule:
    return _common_numeric_rule(
        select_hls_common_int_type, include_index=False, commutative=False
    )


def _shift_rule() -> TypingRule:
    int_rules = {
        (_SignedInt, _SignedInt): lambda lhs, rhs: lhs,
        (_SignedInt, _UnsignedInt): lambda lhs, rhs: lhs,
        (_SignedInt, IndexType): lambda lhs, rhs: lhs,
    }
    uint_rules = {
        (_UnsignedInt, _UnsignedInt): lambda lhs, rhs: lhs,
        (_UnsignedInt, _SignedInt): lambda lhs, rhs: lhs,
        (_UnsignedInt, IndexType): lambda lhs, rhs: lhs,
    }
    return TypingRule(int_rules, uint_rules)


def _bitwise_rule(common_int: Callable[[DType, DType], DType]) -> TypingRule:
    int_rules = {
        (_SignedInt, _SignedInt): common_int,
        (_SignedInt, _UnsignedInt): common_int,
    }
    uint_rules = {
        (_UnsignedInt, _UnsignedInt): common_int,
        (_UnsignedInt, _SignedInt): common_int,
    }
    index_rules = {(IndexType, IndexType): lambda lhs, rhs: index}
    return TypingRule(int_rules, uint_rules, index_rules, commutative=True)


def _unary_invert_rule() -> TypingRule:
    int_rules = {(_SignedInt,): lambda dtype: dtype}
    uint_rules = {(_UnsignedInt,): lambda dtype: dtype}
    return TypingRule(int_rules, uint_rules)


def _hls_unary_neg_rule() -> TypingRule:
    int_rules = {
        (_SignedInt,): lambda dtype: APInt(dtype.primitive_width + 1, signed=True)
    }
    uint_rules = {
        (_UnsignedInt,): lambda dtype: APInt(dtype.primitive_width + 1, signed=True)
    }
    float_rules = {(APFloat,): lambda dtype: dtype}
    return TypingRule(int_rules, uint_rules, float_rules)


def _cpp_unary_neg_rule() -> TypingRule:
    int_rules = {(_SignedInt,): lambda dtype: dtype}
    uint_rules = {(_UnsignedInt,): lambda dtype: dtype}
    index_rules = {(IndexType,): lambda dtype: dtype}
    float_rules = {(APFloat,): lambda dtype: dtype}
    return TypingRule(int_rules, uint_rules, index_rules, float_rules)


def _logical_binary_rule(include_index: bool = False) -> TypingRule:
    int_rules = {
        (_SignedInt, _SignedInt): lambda lhs, rhs: AlloBool,
        (_SignedInt, _UnsignedInt): lambda lhs, rhs: AlloBool,
        (_SignedInt, APFloat): lambda lhs, rhs: AlloBool,
    }
    uint_rules = {
        (_UnsignedInt, _UnsignedInt): lambda lhs, rhs: AlloBool,
        (_UnsignedInt, _SignedInt): lambda lhs, rhs: AlloBool,
        (_UnsignedInt, APFloat): lambda lhs, rhs: AlloBool,
    }
    index_rules = (
        {(IndexType, IndexType): lambda lhs, rhs: AlloBool} if include_index else {}
    )
    float_rules = {
        (APFloat, APFloat): lambda lhs, rhs: AlloBool,
        (APFloat, _SignedInt): lambda lhs, rhs: AlloBool,
        (APFloat, _UnsignedInt): lambda lhs, rhs: AlloBool,
    }
    return TypingRule(int_rules, uint_rules, index_rules, float_rules, commutative=True)


def _logical_not_rule() -> TypingRule:
    int_rules = {(_SignedInt,): lambda dtype: AlloBool}
    uint_rules = {(_UnsignedInt,): lambda dtype: AlloBool}
    index_rules = {(IndexType,): lambda dtype: AlloBool}
    float_rules = {(APFloat,): lambda dtype: AlloBool}
    return TypingRule(int_rules, uint_rules, index_rules, float_rules)


def _special_function_rule(include_index: bool = False) -> TypingRule:
    int_rules = {(_SignedInt,): _select_float}
    uint_rules = {(_UnsignedInt,): _select_float}
    index_rules = {(IndexType,): _select_float} if include_index else {}
    float_rules = {(APFloat,): lambda dtype: dtype}
    return TypingRule(int_rules, uint_rules, index_rules, float_rules)


def _abs_rule() -> TypingRule:
    int_rules = {(_SignedInt,): lambda dtype: dtype}
    uint_rules = {(_UnsignedInt,): lambda dtype: dtype}
    float_rules = {(APFloat,): lambda dtype: dtype}
    return TypingRule(int_rules, uint_rules, float_rules)


_SPECIAL_FUNCTIONS = (
    "sin",
    "cos",
    "tan",
    "sinh",
    "cosh",
    "tanh",
    "exp",
    "exp2",
    "log",
    "log2",
    "sqrt",
    "rsqrt",
    "reciprocal",
    "square",
    "floor",
    "ceil",
    "erf",
)


def _make_hls_type_rules() -> TypeRuleTable:
    table = TypeRuleTable("hls", nary_promoter=_hls_nary_int_promotion)
    table.register(("add", "sub"), _hls_add_sub_rule())
    table.register("mul", _hls_mul_rule())
    table.register(
        ("div", "floordiv"), _common_numeric_rule(select_hls_common_int_type)
    )
    table.register("mod", _common_numeric_rule(select_hls_common_int_type))
    table.register("pow", _hls_pow_rule())
    table.register(
        ("eq", "ne", "lt", "le", "gt", "ge"),
        _common_numeric_rule(select_hls_common_int_type),
    )
    table.register(("lshift", "rshift"), _shift_rule())
    table.register(
        ("bitwise_and", "bitwise_or", "bitwise_xor"),
        _bitwise_rule(select_hls_common_int_type),
    )
    table.register("neg", _hls_unary_neg_rule())
    table.register("invert", _unary_invert_rule())
    table.register(("logical_and", "logical_or"), _logical_binary_rule(True))
    table.register("logical_not", _logical_not_rule())
    table.register(_SPECIAL_FUNCTIONS, _special_function_rule(False))
    table.register("abs", _abs_rule())
    table.register(
        ("max", "min"),
        _common_numeric_rule(select_hls_common_int_type, commutative=True),
    )
    return table


def _make_cpp_type_rules() -> TypeRuleTable:
    table = TypeRuleTable("cpp")
    table.register(
        ("add", "sub", "mul", "div", "floordiv", "mod"),
        _common_numeric_rule(select_cpp_common_int_type, commutative=True),
    )
    table.register("pow", _common_numeric_rule(select_cpp_common_int_type))
    table.register(
        ("eq", "ne", "lt", "le", "gt", "ge"),
        _common_numeric_rule(select_cpp_common_int_type, commutative=True),
    )
    table.register(("lshift", "rshift"), _shift_rule())
    table.register(
        ("bitwise_and", "bitwise_or", "bitwise_xor"),
        _bitwise_rule(select_cpp_common_int_type),
    )
    table.register("neg", _cpp_unary_neg_rule())
    table.register("invert", _unary_invert_rule())
    table.register(("logical_and", "logical_or"), _logical_binary_rule(True))
    table.register("logical_not", _logical_not_rule())
    table.register(_SPECIAL_FUNCTIONS, _special_function_rule(True))
    table.register("abs", _abs_rule())
    table.register(
        ("max", "min"),
        _common_numeric_rule(select_cpp_common_int_type, commutative=True),
    )
    return table


HLS_TYPE_RULES = _make_hls_type_rules()
CPP_TYPE_RULES = _make_cpp_type_rules()


def get_type_rules(style: str) -> TypeRuleTable:
    if style == "hls":
        return HLS_TYPE_RULES
    if style == "cpp":
        return CPP_TYPE_RULES
    assert False, f"Unsupported typing style: {style}"
