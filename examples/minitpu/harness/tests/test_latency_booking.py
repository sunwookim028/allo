# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""``latency.check_booking`` against committed Catapult manifests (D-20).

The manifests are real; the bookings come from the geometry records. No
manifest of the tree or the MXU exists yet, so the unit names are borrowed from
kernels that measure the same number: this tests the comparison, not the units.
"""
import os
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), *[".."] * 4))
sys.path.insert(0, REPO)
from examples.minitpu.harness import latency  # noqa: E402
from examples.minitpu.template.legality import MxuGeometry, TreeGeometry  # noqa: E402

REC = os.path.join(REPO, "dev/records/minitpu")
SCHEDULED = os.path.join(REC, "u2_d12_prototype_2026-10-04/catapult/cat_server/latency.json")
UNRELIABLE = os.path.join(REC, "u2_word_array_2026-10-02/catapult/wire_n64_3p33/latency.json")


def test_match(capsys):
    booked = TreeGeometry(N=1).latency  # LEAF_STAGES + 0 levels = 1
    r = latency.check_booking(SCHEDULED, {"rd_a_0": booked}, clock=3.33)
    assert r["rd_a_0"] == ("MATCH", 1, 1)
    assert "BOOKING-MATCH rd_a_0 booked=1 manifest=1" in capsys.readouterr().out


def test_deliberate_mismatch(capsys):
    r = latency.check_booking(SCHEDULED, {"rd_b_0": MxuGeometry().PE_LATENCY})
    assert r["rd_b_0"][0] == "MISMATCH"
    assert "BOOKING-MISMATCH rd_b_0 booked=%d manifest=1" % MxuGeometry().PE_LATENCY in capsys.readouterr().out


def test_unchecked(capsys):
    r = latency.check_booking(UNRELIABLE, {"wa_0": 2})
    assert r["wa_0"][0] == "UNCHECKED"
    # another clock, and a unit the manifest does not have, are not passes either
    assert latency.check_booking(SCHEDULED, {"rd_a_0": 1}, clock=2.0)["rd_a_0"][0] == "UNCHECKED"
    assert latency.check_booking(SCHEDULED, {"nope": 1})["nope"][0] == "UNCHECKED"
    assert "BOOKING-UNCHECKED wa_0 (status=unreliable)" in capsys.readouterr().out


def test_cli(capsys):
    assert latency.main([SCHEDULED, "--bookings", '{"rd_a_0": 1}']) == 0
    assert latency.main([SCHEDULED, "--bookings", '{"rd_a_0": 9}']) == 1
