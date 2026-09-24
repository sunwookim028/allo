"""Tests for adaptive welltap row-cut planning."""

import importlib.util
from pathlib import Path


PATH = Path(__file__).parent / "scripts" / "plan-welltap-cuts.py"
SPEC = importlib.util.spec_from_file_location("plan_welltap_cuts", PATH)
PLAN = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PLAN)


def test_coalesces_adjacent_rows_but_not_distinct_columns():
    boxes = [
        (422.37, 128.94, 436.24, 129.639),
        (422.37, 129.64, 436.24, 130.34),
        (516.04, 128.94, 529.909, 129.639),
        (516.04, 129.64, 529.909, 130.34),
    ]
    assert PLAN.coalesce(boxes) == [
        (422.37, 128.94, 436.24, 130.34),
        (516.04, 128.94, 529.909, 130.34),
    ]
