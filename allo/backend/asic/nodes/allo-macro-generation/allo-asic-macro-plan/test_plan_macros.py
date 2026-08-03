"""Dependency-free tests for macro planning and port-name adaptation."""

import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


SCRIPT = Path(__file__).with_name("plan_macros.py")
SPEC = importlib.util.spec_from_file_location("plan_macros", SCRIPT)
PLAN = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PLAN
SPEC.loader.exec_module(PLAN)


class MacroPlanTest(unittest.TestCase):
    def test_extracts_canonical_and_emits_alias(self):
        rtl = """
module chip(input clk);
  pe_a u0(.clk(clk), .a(8'd0), .y());
  pe_b u1(.clk(clk), .b(8'd0), .z());
endmodule
module child(input [7:0] a, output [7:0] y);
  assign y = a;
endmodule
module pe_a (clk, a, y);
  input clk;
  input [7:0] a;
  output wire [7:0] y;
  child u_child(.a(a), .y(y));
endmodule
module pe_b (clk, b, z);
  input clk;
  input [7:0] b;
  output reg [7:0] z;
  always @(*) z = b;
endmodule
"""
        manifest = {
            "schema_version": 2,
            "top": "chip",
            "summary": {
                "unmatched_or_ambiguous": 0,
                "unjoined_post_hls_records": 0,
            },
            "pe_instances": [
                {
                    "semantic_id": "chip/pe/pid=0",
                    "kernel": "pe",
                    "pid": [0],
                    "ports": [],
                    "post_hls_records": [],
                },
                {
                    "semantic_id": "chip/pe/pid=1",
                    "kernel": "pe",
                    "pid": [1],
                    "ports": [],
                    "post_hls_records": [],
                },
            ],
            "channels": [],
            "macro_groups": [
                {
                    "macro_class_id": "macro_alpha_test",
                    "representative": "chip/pe/pid=0",
                    "member_count": 2,
                    "members": [
                        {
                            "semantic_id": "chip/pe/pid=0",
                            "rtl_module": "pe_a",
                            "orientation": "unassigned",
                        },
                        {
                            "semantic_id": "chip/pe/pid=1",
                            "rtl_module": "pe_b",
                            "orientation": "unassigned",
                        },
                    ],
                    "proof": {
                        "status": "proven",
                        "rtl_hash": "abc123",
                    },
                }
            ],
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            inputs = root / "inputs"
            inputs.mkdir()
            (inputs / "design.v").write_text(rtl)
            (inputs / "asic-manifest-final.json").write_text(json.dumps(manifest))
            (inputs / "asic-manifest-final.tcl").write_text("")
            (inputs / "build-metadata.json").write_text("{}")
            previous = Path.cwd()
            try:
                os.chdir(root)
                with mock.patch.dict(
                    os.environ,
                    {"min_macro_reuse": "2", "macro_clock_period": "5"},
                ):
                    PLAN.main()
            finally:
                os.chdir(previous)

            index = json.loads((root / "outputs/macro-batch/index.json").read_text())
            self.assertEqual(index["selected_class_count"], 1)
            entry = index["entries"][0]
            self.assertEqual(entry["dependencies"], ["child"])
            self.assertEqual(
                entry["port_maps"]["pe_b"],
                [
                    {"canonical": "clk", "member": "clk"},
                    {"canonical": "a", "member": "b"},
                    {"canonical": "y", "member": "z"},
                ],
            )
            macro_rtl = (
                root / "outputs/macro-batch/entries/macro_alpha_test/rtl/design.v"
            ).read_text()
            self.assertIn("module pe_a", macro_rtl)
            self.assertIn("module child", macro_rtl)
            residual = (root / "outputs/residual-design.v").read_text()
            self.assertNotIn("module pe_a", residual)
            self.assertIn("module pe_b", residual)
            self.assertIn("pe_a canonical_macro", residual)
            self.assertIn(".a(b)", residual)
            self.assertIn("output wire [7:0] z;", residual)
            pin_intent = json.loads(
                (
                    root
                    / "outputs/macro-batch/entries/macro_alpha_test/pin-intent.json"
                ).read_text()
            )
            self.assertEqual(pin_intent["module"], "pe_a")
            self.assertEqual(pin_intent["stream_bundles"], [])

    def test_manifest_graph_maps_vitis_bundles_to_compass_sides(self):
        block = PLAN.module_blocks(
            """
module pe (ap_clk, west_dout, west_empty_n, west_read,
           south_din, south_full_n, south_write);
 input ap_clk;
 input [31:0] west_dout;
 input west_empty_n;
 output west_read;
 output [31:0] south_din;
 input south_full_n;
 output south_write;
endmodule
"""
        )["pe"]
        manifest = {
            "pe_instances": [
                {
                    "semantic_id": "r/k/pid=0,0",
                    "kernel": "k",
                    "pid": [0, 0],
                    "ports": [
                        {"ordinal": 0, "channel_id": "h0", "stream": "horizontal_0_0", "direction": "in"},
                        {"ordinal": 1, "channel_id": "v1", "stream": "vertical_1_0", "direction": "out"},
                    ],
                    "post_hls_records": [],
                },
                {"semantic_id": "r/k/pid=0,-1", "kernel": "k", "pid": [0, -1], "ports": []},
                {"semantic_id": "r/k/pid=1,0", "kernel": "k", "pid": [1, 0], "ports": []},
            ],
            "channels": [
                {"stream": "horizontal_0_0", "endpoints": [
                    {"pe": "r/k/pid=0,-1", "direction": "out", "accesses": []},
                    {"pe": "r/k/pid=0,0", "direction": "in", "accesses": [{"port_ordinal": 0}]},
                ]},
                {"stream": "vertical_1_0", "endpoints": [
                    {"pe": "r/k/pid=0,0", "direction": "out", "accesses": [{"port_ordinal": 1}]},
                    {"pe": "r/k/pid=1,0", "direction": "in", "accesses": []},
                ]},
            ],
        }
        intent = PLAN.build_pin_intent(manifest, "r/k/pid=0,0", block)
        self.assertEqual(
            [item["side"] for item in intent["stream_bundles"]], ["W", "S"]
        )
        self.assertEqual(intent["control_pins"], ["ap_clk"])
        # Control pins are movable collateral, so they go to the least-loaded
        # non-stream edge instead of consuming a semantic stream edge.
        self.assertEqual(intent["control_side"], "N")
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "pin-intent.tcl"
            PLAN.write_pin_intent_tcl(output, intent)
            emitted = output.read_text()
            self.assertIn("allo_asic_signal_pin_groups_W", emitted)
            self.assertIn(
                "[list {west_dout} {west_empty_n} {west_read}]", emitted
            )
            self.assertIn(
                "allo_asic_signal_pin_groups_N [list [list {ap_clk}]]", emitted
            )
        rotated = json.loads(json.dumps(intent))
        rotated["stream_bundles"][0]["side"] = "S"
        rotated["stream_bundles"][1]["side"] = "E"
        identity_map = [
            {"canonical": port, "member": port}
            for bundle in intent["stream_bundles"]
            for port in bundle["rtl_ports"]
        ]
        self.assertEqual(
            PLAN.select_member_orientation(intent, rotated, identity_map), "R90"
        )


if __name__ == "__main__":
    unittest.main()
