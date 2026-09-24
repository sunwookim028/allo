"""Tests for parameter-only Allo preflight checks."""

import pytest

from preflight import validate_clock_periods


def test_macro_clock_must_be_at_least_as_fast_as_chip():
    validate_clock_periods(10.0, 8.0)
    validate_clock_periods(10.0, 10.0)
    with pytest.raises(RuntimeError, match="greater than or equal"):
        validate_clock_periods(8.0, 10.0)
    with pytest.raises(RuntimeError, match="positive"):
        validate_clock_periods(10.0, 0.0)
