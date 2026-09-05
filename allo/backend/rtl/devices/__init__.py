# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The device library"""

from __future__ import annotations

from ..device import Device
from . import alveo, kria, series7, versal
from .alveo import u55c, u280, u250
from .kria import kv260
from .series7 import pynqz2
from .versal import vck190
from .spec import Grade, Part

_DEVICES = {
    d.name: d for module in (alveo, kria, series7, versal) for d in module.DEVICES
}

#: What ``RTL(device=None)`` resolves to. Always an alias of a registered part,
#: never a device of its own.
default_device = u55c


def get(name: str) -> Device:
    """The device registered under ``name``, which is also the symbol its
    injected ``dcp.device`` carries."""
    device = _DEVICES.get(name)
    if device is None:
        raise ValueError(
            f"unknown device {name!r}; the library holds {sorted(_DEVICES)}"
        )
    return device


def names() -> list[str]:
    """Every registered device name."""
    return sorted(_DEVICES)


def parts() -> dict[str, str]:
    """Every registered device name mapped to its full vendor part number."""
    return {name: d.part for name, d in _DEVICES.items()}


__all__ = [
    "Device",
    "Grade",
    "Part",
    "default_device",
    "get",
    "names",
    "parts",
    "u55c",
    "u280",
    "u250",
    "kv260",
    "pynqz2",
    "vck190",
]
