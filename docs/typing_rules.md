<!--- Copyright Allo authors. All Rights Reserved. -->
<!--- SPDX-License-Identifier: Apache-2.0  -->

# Typing Rules

Allo uses explicit type-promotion tables when lowering frontend expressions.
The active table is selected by `KernelOptions(typing_style=...)`:

```python
from allo.lang import KernelOptions, kernel

@kernel(options=KernelOptions(typing_style="hls"))  # default
def hls_kernel(...):
    ...

@kernel(options=KernelOptions(typing_style="cpp"))
def cpp_kernel(...):
    ...
```

The two supported styles are:

- `hls`: hardware-oriented promotion. Integer `+`, `-`, and `*` are widened to
  preserve full intermediate precision and are lowered as balanced trees.
- `cpp`: C++-like promotion. Integer binary operations select a common integer
  type and lower according to the expression tree.

If no rule exists for an operator/type combination, compilation fails with a
diagnostic such as `No hls type promotion rule for operator ...`.

## Type Categories

Typing rules operate on frontend `DType` categories:

| Category         | Examples                              | Notes                                                |
| :--------------- | :------------------------------------ | :--------------------------------------------------- |
| Signed integer   | `i8`, `i32`, `apint(23, signed=True)` | Arbitrary-width signed `APInt`.                      |
| Unsigned integer | `u1`, `u32`, `apint(17)`              | Arbitrary-width unsigned `APInt`.                    |
| Floating point   | `f16`, `bf16`, `f32`, `f64`           | `APFloat` values supported by the frontend.          |
| Index            | `index`                               | Opaque index type used for loop bounds and indexing. |
| Boolean          | `bool`, `u1`                          | `bool` is the frontend alias of `u1`.                |

The rules below describe element types. For shaped values, the same element
promotion is used inside elementwise or linalg operations; shape compatibility is
checked separately by the operator.

`Stream` is not a numeric type and does not participate in promotion. A stream's
payload type is checked at `put(value)`: the value is cast to the stream
`base_type`, and `get()` returns that exact payload type.

## Common Helpers

Several rules reuse a common integer type selector. The two styles differ on
a mixed-sign pair: `cpp` follows C++'s usual arithmetic conversions, while
`hls` selects a signed type wide enough for both value ranges, so
comparisons, divisions, and bitwise operations compute on the operand values.

For two integer types with widths `L` and `R`:

| Inputs                                                         | `cpp` common result       | `hls` common result                     |
| :------------------------------------------------------------- | :------------------------ | :-------------------------------------- |
| Both signed                                                    | signed `max(L, R)`        | signed `max(L, R)`                      |
| Both unsigned                                                  | unsigned `max(L, R)`      | unsigned `max(L, R)`                    |
| One signed, one unsigned, and `unsigned_width >= signed_width` | unsigned `unsigned_width` | signed `max(signed_width, unsigned_width + 1)` |
| One signed, one unsigned, and `unsigned_width < signed_width`  | signed `signed_width`     | signed `max(signed_width, unsigned_width + 1)` |

Examples:

| Expression types | `cpp` common | `hls` common |
| :--------------- | :----------- | :----------- |
| `i16`, `i32`     | `i32`        | `i32`        |
| `u8`, `u32`      | `u32`        | `u32`        |
| `i32`, `u32`     | `u32`        | `apint(33)`  |
| `i32`, `u16`     | `i32`        | `i32`        |

Floating-point common rules are shared by both styles:

- `float op float` promotes to the wider float.
- `float op integer` keeps the float type.
- Special math functions convert integers to `f32` if their bit width is at
  most 32, otherwise to `f64`; floating inputs keep their type.

`index` has dedicated rules rather than behaving like an integer of a fixed
width.

## HLS Rules

HLS is the default typing style. It is designed to make arithmetic bit growth
explicit for hardware generation.

### HLS Integer Add/Sub

For integer-only `+` and `-`, HLS uses an n-ary promotion rule over the whole
add/sub expression, not just pairwise promotion.

For an expression with `N` terms:

1. The result is signed if any operand is signed or any term is subtracted.
2. If the result is signed, every unsigned operand contributes `width + 1` to
   account for sign conversion.
3. The result width is `max(adjusted_term_widths) + ceil_log2(N)`.
4. All terms are cast to that result type.
5. Subtracted terms are negated.
6. The final sum is lowered as a balanced addition tree.

Examples:

| Expression types    | HLS result |
| :------------------ | :--------- |
| `i32 + i32`         | `i33`      |
| `u32 + u32`         | `u33`      |
| `u8 + i8`           | `i10`      |
| `i32 + i32 - i32`   | `i34`      |
| `u8 + u8 + u8 + u8` | `u10`      |

The balanced-tree behavior matters for generated hardware. For:

```python
out[0] = a + b + c + d
```

HLS lowering forms a pairwise tree similar to:

```text
t0 = a + b
t1 = c + d
out = t0 + t1
```

For a mixed add/sub expression:

```python
out[0] = a + b - c + d
```

All terms are first cast to the n-ary result type, `c` is negated, and the
normalized terms are reduced by the same balanced addition tree.

Floating-point add/sub is not reassociated by default. When `fast_math=False`,
any expression that may be floating-point is lowered according to the original
expression tree. With `fast_math=True`, floating-point expressions may use the
same n-ary lowering path.

### HLS Integer Mul

For integer-only multiplication, HLS also uses an n-ary promotion rule:

1. The result is signed if any operand is signed.
2. The result width is the sum of all operand widths.
3. All operands are cast to the result type.
4. The multiply is lowered as a balanced multiplication tree.

Examples:

| Expression types  | HLS result |
| :---------------- | :--------- |
| `i32 * i32`       | `i64`      |
| `u16 * u16`       | `u32`      |
| `i32 * i32 * i32` | `i96`      |
| `u8 * i8 * u4`    | `i20`      |

### Natural Width vs Built Width

These rules give an expression its **natural** width, computed from the leaves
up, so an expression never silently loses precision. They do not say how wide
the operator the RTL backend builds is. Assigning to a narrower declared type
appends a truncation, and the `narrow-demanded-bits` prepass then sinks that
truncation onto the leaves, so each operator is built at the width its consumer
actually reads:

```python
a: i48 = b * c        # b, c : i32
```

types `b * c` as `i64` and truncates, but the multiplier that reaches hardware
is 48 bits wide, with the extends folded into its operands and no truncation
left. The rewrite is bit-exact: it moves a truncation the program already
performed. Division, remainder, right shift and comparison read the high bits,
so their operands keep the natural width.

### HLS Other Numeric Operators

For `div`, `floordiv`, `mod`, `pow`, comparisons, bitwise operators, and
`max`/`min`, HLS uses the common numeric rules unless a more specific rule is
listed below.

| Operator group                             | HLS promotion                                                     |
| :----------------------------------------- | :---------------------------------------------------------------- |
| `div`, `floordiv`, `mod`                   | Common numeric type, including `index` rules.                     |
| `pow`                                      | Common numeric type, but `index` is not accepted.                 |
| `eq`, `ne`, `lt`, `le`, `gt`, `ge`         | Operands use common numeric type; result is `bool`/`u1`.          |
| `max`, `min`                               | Common numeric type; operation result has that type.              |
| `bitwise_and`, `bitwise_or`, `bitwise_xor` | Common integer type for integer pairs; `index` only with `index`. |

### HLS Shift Operators

Shift operators keep the left-hand type.

| Inputs                                                         | Result                  |
| :------------------------------------------------------------- | :---------------------- |
| signed integer shifted by signed/unsigned integer or `index`   | left-hand signed type   |
| unsigned integer shifted by signed/unsigned integer or `index` | left-hand unsigned type |
| `index` shifted by `index`                                     | `index`                 |

`index` shifted by a plain integer is not currently covered by the rule table.

### HLS Unary and Logical Operators

| Operator                             | HLS rule                                                                       |
| :----------------------------------- | :----------------------------------------------------------------------------- |
| Unary `-` on signed/unsigned integer | signed integer with `width + 1`                                                |
| Unary `-` on float                   | same float type                                                                |
| Unary `~`                            | same signed/unsigned/index type                                                |
| `logical_and`, `logical_or`          | accepts integer/float numeric pairs and `index`/`index`; result is `bool`/`u1` |
| `logical_not`                        | accepts integer, float, and `index`; result is `bool`/`u1`                     |
| `abs`                                | same signed/unsigned/float type; `index` is not accepted                       |

Special math functions in HLS accept signed integers, unsigned integers, and
floats. They do not accept `index`.

## C++ Rules

C++ typing style uses the common numeric rules more uniformly. It does not use
the HLS integer bit-growth rules for `+`, `-`, or `*`.

### C++ Arithmetic Operators

| Operator group                                | C++ promotion                                        |
| :-------------------------------------------- | :--------------------------------------------------- |
| `add`, `sub`, `mul`, `div`, `floordiv`, `mod` | Common numeric type, including `index` rules.        |
| `pow`                                         | Common numeric type, including `index` rules.        |
| `max`, `min`                                  | Common numeric type; operation result has that type. |

Examples:

| Expression types | C++ result |
| :--------------- | :--------- |
| `i32 + i32`      | `i32`      |
| `u32 + u32`      | `u32`      |
| `i32 + u32`      | `u32`      |
| `i16 * i32`      | `i32`      |
| `f32 + i32`      | `f32`      |
| `f32 + f64`      | `f64`      |

Promotion is pairwise when an expression contains more than two operands. For
`a + b + c`, the frontend promotes `a + b` first, then promotes that result with
`c`.

### C++ Comparisons, Bitwise, and Shifts

| Operator group                             | C++ promotion                                                                   |
| :----------------------------------------- | :------------------------------------------------------------------------------ |
| `eq`, `ne`, `lt`, `le`, `gt`, `ge`         | Operands use common numeric type; result is `bool`/`u1`.                        |
| `bitwise_and`, `bitwise_or`, `bitwise_xor` | Common integer type for integer pairs; `index` only with `index`.               |
| `lshift`, `rshift`                         | Same as HLS: result is the left-hand type, with the same supported input pairs. |

### C++ Unary, Logical, and Math Operators

| Operator                    | C++ rule                                                                       |
| :-------------------------- | :----------------------------------------------------------------------------- |
| Unary `-`                   | same signed/unsigned/index/float type                                          |
| Unary `~`                   | same signed/unsigned/index type                                                |
| `logical_and`, `logical_or` | accepts integer/float numeric pairs and `index`/`index`; result is `bool`/`u1` |
| `logical_not`               | accepts integer, float, and `index`; result is `bool`/`u1`                     |
| `abs`                       | same signed/unsigned/float type; `index` is not accepted                       |
| Special math functions      | integer/index inputs convert to `f32` or `f64`; float inputs keep their type   |

Because `index` has a very large opaque primitive width internally, special math
functions on `index` promote to `f64` under the current C++ rule table.

## Operator Coverage Summary

| Operator key                               | HLS                                                       | C++                                    |
| :----------------------------------------- | :-------------------------------------------------------- | :------------------------------------- |
| `add`, `sub`                               | Integer n-ary bit growth; otherwise numeric rules         | Common numeric                         |
| `mul`                                      | Integer n-ary full-width product; otherwise numeric rules | Common numeric                         |
| `div`, `floordiv`, `mod`                   | Common numeric                                            | Common numeric                         |
| `pow`                                      | Common numeric, no `index`                                | Common numeric                         |
| `eq`, `ne`, `lt`, `le`, `gt`, `ge`         | Common numeric operands, `bool` result                    | Common numeric operands, `bool` result |
| `lshift`, `rshift`                         | Left-hand type                                            | Left-hand type                         |
| `bitwise_and`, `bitwise_or`, `bitwise_xor` | Common integer or `index`/`index`                         | Common integer or `index`/`index`      |
| `neg`                                      | Integer grows to signed `width + 1`; float unchanged      | Type unchanged                         |
| `invert`                                   | Type unchanged for integer/index                          | Type unchanged for integer/index       |
| `logical_and`, `logical_or`, `logical_not` | `bool` result                                             | `bool` result                          |
| Special math functions                     | Integers/floats only                                      | Integers, `index`, and floats          |
| `abs`                                      | Type unchanged for integer/float                          | Type unchanged for integer/float       |

## Notes for Implementers

- `TypeRuleTable.promote()` first asks the style-specific n-ary promoter. Only
  HLS provides one, and it only handles integer `add`, `sub`, and `mul`.
- If the n-ary promoter returns `None`, promotion falls back to pairwise lookup
  from left to right.
- Comparisons and logical operations use the promoted type to materialize or
  cast operands, but their final value type is `bool`/`u1`.
- Shift operations do not compute a common type. The result follows the
  left-hand operand.
- Unary plus is a no-op in lowering and does not use a promotion-table entry.
