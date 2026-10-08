# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""A per-kernel manifest from a Vitis HLS build: ``python3 vhls_manifest.py <prj> <out.json>``.
Per dataflow process: its iteration loop's II and iteration latency, and the HLS slack estimate, read from
``csynth.xml``/the per-module ``*_csynth.xml``; tool and clock. The Catapult flow's ``latency.json`` has the same
role (provenance for the ISA delta); this is its Vitis counterpart, read from Vitis's own reports."""
import glob, json, os, re, sys
import xml.etree.ElementTree as ET

prj, out = sys.argv[1], sys.argv[2]
rep = os.path.join(prj, "out.prj/solution1/syn/report")
top = ET.parse(os.path.join(rep, "csynth.xml")).getroot()
units = {}
for f in sorted(glob.glob(os.path.join(rep, "*_csynth.xml"))):
    r = ET.parse(f).getroot()
    name = r.findtext("RTLDesignHierarchy/TopModule/ModuleName") or os.path.basename(f)[:-11]
    for loop in r.findall(".//SummaryOfLoopLatency/*"):
        if loop.tag.startswith("l_S_t_0_t"):
            ii = loop.findtext("PipelineII")
            il = loop.findtext("PipelineDepth") or loop.findtext("IterationLatency")
            proc = re.sub(r"_Pipeline_.*$", "", name)
            units[proc] = {"loop": loop.tag, "ii": int(ii) if ii and ii.isdigit() else ii,
                           "iteration_latency": int(il) if il and il.isdigit() else il,
                           "trip_count": loop.findtext("TripCount")}
man = {"tool": "vitis_hls " + (top.findtext("ReportVersion/Version") or "?"),
       "part": top.findtext("UserAssignments/Part"),
       "clock_period_ns": float(top.findtext("UserAssignments/TargetClockPeriod")),
       "estimated_clock_ns": float(top.findtext("PerformanceEstimates/SummaryOfTimingAnalysis/EstimatedClockPeriod")),
       "top": top.findtext("UserAssignments/TopModelName"), "units": units}
json.dump(man, open(out, "w"), indent=1)
print(f"manifest {man['top']}: {len(units)} processes, II {sorted(set(u['ii'] for u in units.values()), key=str)}")
