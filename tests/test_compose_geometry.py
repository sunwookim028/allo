# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""README D-20: a derived parameter is a property, and every relation it
rests on is a legality condition.

``Architecture(parameters=<geometry record>)`` binds the record's
``namespace()``: derived numbers are properties, so they bind as parameters
without being typed in, and each unit's ``legality`` runs on them. The
records are the template's (``examples/minitpu/template/legality.py``).
"""

from __future__ import annotations

import dataclasses

import pytest

from allo.compose import Architecture, Memory, unit
from allo.ir.types import int32  # noqa: F401  pylint: disable=unused-import
from examples.minitpu.template.engines import BF16_ACC24
from examples.minitpu.template.legality import MxuGeometry, mxu_legality


@unit(memories=("X",), parameters=("DIM", "PUSH_TO_VALID"), legality=mxu_legality)
def mxu_stub(x: int32[DIM]):
    for i in range(DIM):
        x[i] = PUSH_TO_VALID


def _arch(params):
    return Architecture(name="mxu_geom", parameters=params,
                        memories=(Memory("X", "int32[DIM]"),), channels=(),
                        units=(mxu_stub,))


@dataclasses.dataclass(frozen=True)
class StaleMxu(MxuGeometry):
    """PUSH_TO_VALID typed in beside the adder it depends on: right for a
    3-cycle adder (82), stale once the engine declares 5."""

    PUSH_TO_VALID: int = 82

    def namespace(self) -> dict:
        return super().namespace() | {"PUSH_TO_VALID": self.PUSH_TO_VALID}


@dataclasses.dataclass(frozen=True)
class WithProps(MxuGeometry):
    @property
    def PUSH_TO_VALID(self) -> int:  # pylint: disable=invalid-name
        return self.push_to_valid


def test_geometry_record_binds_its_derived_numbers():
    arch = _arch(MxuGeometry())
    assert arch.geometry == MxuGeometry()
    assert arch.parameters["PUSH_TO_VALID"] == 82 and arch.parameters["PE_LATENCY"] == 4
    slow = dataclasses.replace(BF16_ACC24, latency={"mul": 0, "add": 5})
    assert _arch(MxuGeometry(mac=slow)).parameters["PUSH_TO_VALID"] == 2 + 16 * 7


def test_declared_derived_number_refused():
    slow = dataclasses.replace(BF16_ACC24, latency={"mul": 0, "add": 5})
    _arch(StaleMxu())  # 82 is right at the 3-cycle adder
    with pytest.raises(AssertionError, match=r"PUSH_TO_VALID=82 must be 2 \+ DIM\*\(PE_LATENCY\+1\) = 114"):
        _arch(StaleMxu(mac=slow))


def test_namespace_key_disagreeing_with_its_property_refused():
    class Bad(WithProps):
        def namespace(self) -> dict:
            return super().namespace() | {"PUSH_TO_VALID": 85}

    with pytest.raises(AssertionError, match=r"declares PUSH_TO_VALID=85, but its derived PUSH_TO_VALID is 82"):
        _arch(Bad())


def test_parameters_must_be_dict_or_record():
    with pytest.raises(AssertionError, match="neither a dict nor a geometry record"):
        _arch((("DIM", 4),))
