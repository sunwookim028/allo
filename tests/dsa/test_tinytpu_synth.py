# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The synthesis-derived latency table, checked against a recorded report.

These run without Vitis: they parse a csynth report shaped like the one
TinyTPU actually produces and check that the derived ``(ii, depth)`` reproduces
the latency the tool measured.
"""

import textwrap

import pytest

from allo.backend.vitis.report import parse_report
from examples.accelerator.tinytpu.synth import UNIT_ORDER, derive_latency_table


def _module(name, latency, loops, lut=0):
    """One <Module> entry; ``latency`` is None for a data-dependent call."""
    lat = "undef" if latency is None else str(latency)
    loop_xml = "".join(
        textwrap.dedent(
            f"""
            <L{index}>
              <Name>{loop["name"]}</Name>
              <TripCount>{loop["trip"]}</TripCount>
              <Latency>{loop["latency"]}</Latency>
              <PipelineII>{loop["ii"]}</PipelineII>
              <PipelineDepth>{loop["depth"]}</PipelineDepth>
              <PipelineType>yes</PipelineType>
            </L{index}>
            """
        )
        for index, loop in enumerate(loops)
    )
    return f"""
    <Module>
      <Name>{name}</Name>
      <PerformanceEstimates>
        <SummaryOfOverallLatency>
          <Best-caseLatency>{lat}</Best-caseLatency>
          <Average-caseLatency>{lat}</Average-caseLatency>
          <Worst-caseLatency>{lat}</Worst-caseLatency>
          <PipelineType>no</PipelineType>
        </SummaryOfOverallLatency>
        <SummaryOfLoopLatency>{loop_xml}</SummaryOfLoopLatency>
      </PerformanceEstimates>
      <AreaEstimates><Resources><LUT>{lut}</LUT><FF>0</FF><DSP>0</DSP>
        <BRAM_18K>0</BRAM_18K><URAM>0</URAM></Resources></AreaEstimates>
    </Module>
    """


@pytest.fixture(name="report_path")
def _report_path(tmp_path):
    """A report with the two shapes TinyTPU units take: statically bounded
    (``vpu``, ``mxu``) and data-dependent (``dma_load``)."""
    loop = lambda name, trip, latency, depth: {  # noqa: E731
        "name": name,
        "trip": trip,
        "latency": latency,
        "ii": 1,
        "depth": depth,
    }
    modules = (
        _module("tinytpu_vpu", 19, [loop("VL_119", 8, 17, 11)])
        # mxu's loops were outlined into their own modules, as Vitis does.
        + _module("tinytpu_mxu", 72, [])
        + _module("tinytpu_mxu_Pipeline_A", None, [loop("VL_211", 16, 17, 3)])
        + _module("tinytpu_mxu_Pipeline_B", None, [loop("VL_220", 16, 48, 34)])
        + _module("tinytpu_dma_load", None, [])
        + _module(
            "tinytpu_dma_load_Pipeline_C", None, [loop("VL_176", "undef", "undef", 3)]
        )
        + _module("tinytpu", None, [])
    )
    xml = f"""<?xml version="1.0"?>
    <profile>
      <ReportVersion><Version>2023.2</Version></ReportVersion>
      <UserAssignments>
        <Part>xcu55c-fsvh2892-2L-e</Part>
        <TopModelName>tinytpu</TopModelName>
        <TargetClockPeriod>3.33</TargetClockPeriod>
      </UserAssignments>
      <PerformanceEstimates>
        <SummaryOfTimingAnalysis><EstimatedClockPeriod>2.433</EstimatedClockPeriod>
        </SummaryOfTimingAnalysis>
      </PerformanceEstimates>
      <AreaEstimates><Resources><LUT>8515</LUT><FF>9111</FF><DSP>20</DSP>
        <BRAM_18K>26</BRAM_18K><URAM>0</URAM></Resources></AreaEstimates>
      <ModuleInformation>{modules}</ModuleInformation>
    </profile>
    """
    path = tmp_path / "csynth.xml"
    path.write_text(xml, encoding="utf-8")
    return path


def test_loop_level_report_is_parsed(report_path):
    report = parse_report(report_path)
    vpu_loops = report.modules["tinytpu_vpu"].loops
    assert [(l.trip_count, l.ii, l.depth) for l in vpu_loops] == [(8, 1, 11)]
    # A data-dependent bound survives as None without losing ii/depth.
    dma = report.modules["tinytpu_dma_load_Pipeline_C"].loops[0]
    assert dma.trip_count is None and dma.ii == 1 and dma.depth == 3


def test_derived_table_reproduces_measured_call_latency(report_path):
    table = derive_latency_table(parse_report(report_path), top="tinytpu")

    # A statically bounded unit: depth + ii * trips is exactly what Vitis measured.
    vpu = table["vpu"]
    assert (vpu.ii, vpu.trips, vpu.call_cycles) == (1, 8, 19)
    assert vpu.depth + vpu.ii * vpu.trips == 19

    # Two sequential passes over the same trip range issue at ii=2, and the
    # loops outlined into _Pipeline_ modules are still found.
    mxu = table["mxu"]
    assert (mxu.ii, mxu.trips, mxu.call_cycles) == (2, 16, 72)
    assert mxu.depth + mxu.ii * mxu.trips == 72


def test_data_dependent_unit_falls_back_to_loop_depth(report_path):
    table = derive_latency_table(parse_report(report_path), top="tinytpu")
    dma = table["dma_load"]
    # No measured call latency, so depth is the loop's pipeline depth plus the
    # per-call overhead measured from the bounded units in the same report.
    assert dma.call_cycles is None and dma.trips is None
    assert dma.ii == 1
    assert dma.depth == 3 + (19 - 17)


def test_table_covers_only_units_the_report_carries(report_path):
    table = derive_latency_table(parse_report(report_path), top="tinytpu")
    assert set(table) == {"vpu", "mxu", "dma_load"}
    assert [name for name in UNIT_ORDER if name in table] == ["dma_load", "vpu", "mxu"]
