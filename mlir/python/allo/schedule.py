# SPDX-License-Identifier: Apache-2.0
# Copyright Allo authors. All Rights Reserved.

"""Schedule analysis on top of the upstream MLIR Python bindings.

The heavy IR introspection lives in the `_allo` CAPI extension (it needs C++
interface/effect queries). This module only wraps the JSON snapshot it returns
and mirrors the trait flags so callers get a plain Python dict.
"""

import json
from enum import IntFlag

from ._mlir_libs._allo import schedule as _schedule

SCHEDULE_NAME_ATTR_NAME = _schedule.SCHEDULE_NAME_ATTR_NAME
SCHEDULE_ID_ATTR_NAME = _schedule.SCHEDULE_ID_ATTR_NAME
PIPELINE_II_ATTR_NAME = _schedule.PIPELINE_II_ATTR_NAME
PIPELINE_REWIND_ATTR_NAME = _schedule.PIPELINE_REWIND_ATTR_NAME
DATAFLOW_ATTR_NAME = _schedule.DATAFLOW_ATTR_NAME
UNROLL_FACTOR_ATTR_NAME = _schedule.UNROLL_FACTOR_ATTR_NAME
PARTITION_ATTR_NAME = _schedule.PARTITION_ATTR_NAME


class ScheduleOpTrait(IntFlag):
    """Mirror of the C++ ScheduleOpTrait bit flags (see lib/CAPI/Schedule.cpp)."""

    LOOP_LIKE = 1 << 0
    AFFINE_LOOP = 1 << 1
    SCF_LOOP = 1 << 2
    REGION_BRANCH = 1 << 3
    FUNCTION_LIKE = 1 << 4
    SYMBOL = 1 << 5
    MEMORY_ALLOCATE = 1 << 6
    MEMORY_FREE = 1 << 7
    MEMORY_READ = 1 << 8
    MEMORY_WRITE = 1 << 9
    AFFINE_FOR = 1 << 10


def annotate_schedule_ids(module):
    _schedule.annotate_schedule_ids(module)


def cleanup_schedule_ids(module):
    _schedule.cleanup_schedule_ids(module)


def collect_schedule_snapshot(module):
    """Return the schedule snapshot as a nested dict (ops/values/root_id)."""
    return json.loads(_schedule.collect_schedule_snapshot_json(module))
