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
    def test_fifo_wrapper_signature_ignores_generated_root_aliases(self):
        first = """
module wrapper_a(clk, v10_dat);
  input clk;
  output [7:0] v10_dat;
endmodule
"""
        second = first.replace("wrapper_a", "wrapper_b").replace("v10", "v99")
        first_folding = {
            "owned_fifos": [{"fifo_module": "ccs_pipe_v6"}],
            "signature": [
                {
                    "root": "v10",
                    "module": "ccs_pipe_v6",
                    "parameters": {"width": "32'sd8", "fifo_sz": "32'sd8"},
                }
            ],
        }
        second_folding = {
            **first_folding,
            "signature": [
                {**first_folding["signature"][0], "root": "v99"}
            ],
        }
        first_signature = PLAN.wrapper_interface_signature(
            first, "wrapper_a", first_folding
        )
        self.assertEqual(
            first_signature,
            PLAN.wrapper_interface_signature(second, "wrapper_b", second_folding),
        )
        second_folding["signature"][0]["parameters"] = {
            "width": "32'sd8",
            "fifo_sz": "32'sd16",
        }
        self.assertNotEqual(
            first_signature,
            PLAN.wrapper_interface_signature(second, "wrapper_b", second_folding),
        )

    def test_fifo_wrapper_signature_ignores_port_and_fifo_order(self):
        first = """
module wrapper_a(clk, west_data, north_data);
  input clk;
  input [7:0] west_data;
  output [15:0] north_data;
endmodule
"""
        second = """
module wrapper_b(n99_data, clk, w42_data);
  output [15:0] n99_data;
  input clk;
  input [7:0] w42_data;
endmodule
"""
        fifo_a = {
            "module": "ccs_pipe_v6",
            "parameters": {"width": "32'sd8", "fifo_sz": "32'sd2"},
        }
        fifo_b = {
            "module": "ccs_pipe_v6",
            "parameters": {"width": "32'sd16", "fifo_sz": "32'sd4"},
        }
        first_folding = {
            "signature": [
                {"root": "west", **fifo_a},
                {"root": "north", **fifo_b},
            ]
        }
        second_folding = {
            "signature": [
                {"root": "n99", **fifo_b},
                {"root": "w42", **fifo_a},
            ]
        }
        self.assertEqual(
            PLAN.wrapper_interface_signature(first, "wrapper_a", first_folding),
            PLAN.wrapper_interface_signature(second, "wrapper_b", second_folding),
        )

    def test_fifo_wrapper_signature_keeps_vitis_fifo_parameters(self):
        wrapper = """
module wrapper(clk, data);
  input clk;
  output [7:0] data;
endmodule
"""
        first = {
            "owned_fifos": [
                {
                    "fifo_module": "fifo_w8",
                    "fifo_parameters": {"depth": 2, "width": 8},
                }
            ]
        }
        second = {
            "owned_fifos": [
                {
                    "fifo_module": "fifo_w8",
                    "fifo_parameters": {"depth": 4, "width": 8},
                }
            ]
        }
        self.assertNotEqual(
            PLAN.wrapper_interface_signature(wrapper, "wrapper", first),
            PLAN.wrapper_interface_signature(wrapper, "wrapper", second),
        )

    def test_prefers_explicit_rtl_root_module(self):
        pe = {
            "semantic_id": "top/pe/pid=0",
            "post_hls_records": [
                {
                    "rtl_modules": [{"name": "pe_core"}],
                    "rtl_root_module": "pe_wrapper",
                    "rtl_instances": [],
                }
            ],
        }
        self.assertEqual(
            PLAN.outer_module_for_member(pe, "pe_core"), "pe_wrapper"
        )

    def test_writes_vitis_clock_constraints(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "constraints.tcl"
            PLAN.write_constraints(path, 8.0, "ap_clk")
            constraints = path.read_text()

        self.assertIn(
            "create_clock -name clk -period 8 [get_ports ap_clk]", constraints
        )
        self.assertIn(
            "[remove_from_collection [all_inputs] [get_ports ap_clk]]",
            constraints,
        )

    def test_writes_catapult_clock_constraints(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "constraints.tcl"
            PLAN.write_constraints(path, 8.0, "clk")
            constraints = path.read_text()

        self.assertIn(
            "create_clock -name clk -period 8 [get_ports clk]", constraints
        )
        self.assertIn(
            "[remove_from_collection [all_inputs] [get_ports clk]]", constraints
        )

    def test_maps_catapult_ready_valid_bundles(self):
        block = PLAN.module_blocks(
            """
module pe(clk, rst, v0_rsc_dat, v0_rsc_vld, v0_rsc_rdy,
                   v1_rsc_dat, v1_rsc_vld, v1_rsc_rdy);
 input clk; input rst;
 output [25:0] v0_rsc_dat; output v0_rsc_vld; input v0_rsc_rdy;
 input [15:0] v1_rsc_dat; input v1_rsc_vld; output v1_rsc_rdy;
endmodule
"""
        )["pe"]
        bundles = PLAN.stream_bundles(block)
        self.assertEqual(
            [(item["root"], item["direction"], item["protocol"]) for item in bundles],
            [
                ("v0", "out", "catapult_ready_valid"),
                ("v1", "in", "catapult_ready_valid"),
            ],
        )
        self.assertEqual(PLAN.rtl_stream_width(block, bundles[0]), 26)
        self.assertEqual(PLAN.rtl_stream_width(block, bundles[1]), 16)

    def test_generates_producer_owned_fifo_wrapper(self):
        rtl = """
module top(ap_clk, reset);
  input ap_clk; input reset;
  pe pe_0_U0(
    .ap_clk(ap_clk), .ap_rst(reset),
    .stream_din(producer_data),
    .stream_num_data_valid(count), .stream_fifo_cap(capacity),
    .stream_full_n(full_n), .stream_write(producer_write));
  fifo fifo_0_U(
    .clk(ap_clk), .reset(reset), .if_read_ce(1'b1), .if_write_ce(1'b1),
    .if_din(producer_data), .if_full_n(full_n), .if_write(producer_write),
    .if_dout(consumer_data), .if_num_data_valid(count), .if_fifo_cap(capacity),
    .if_empty_n(empty_n), .if_read(consumer_read));
endmodule
module pe(ap_clk, ap_rst, stream_din, stream_num_data_valid,
          stream_fifo_cap, stream_full_n, stream_write);
  input ap_clk; input ap_rst;
  output [31:0] stream_din;
  input [1:0] stream_num_data_valid; input [1:0] stream_fifo_cap;
  input stream_full_n; output stream_write;
endmodule
module fifo(clk, reset, if_read_ce, if_write_ce, if_din, if_full_n,
            if_write, if_dout, if_num_data_valid, if_fifo_cap,
            if_empty_n, if_read);
  input clk; input reset; input if_read_ce; input if_write_ce;
  input [31:0] if_din; output if_full_n; input if_write;
  output [31:0] if_dout; output [1:0] if_num_data_valid;
  output [1:0] if_fifo_cap; output if_empty_n; input if_read;
endmodule
"""
        blocks = PLAN.module_blocks(rtl)
        instances = PLAN.module_instances(blocks["top"], set(blocks))
        pe_instance = next(item for item in instances if item.module == "pe")
        manifest = {
            "top": "top",
            "pe_instances": [
                {
                    "semantic_id": "top/compute/pid=0,0",
                    "kernel": "compute",
                    "pid": [0, 0],
                    "ports": [{
                        "ordinal": 0, "channel_id": "c0", "stream": "horizontal_0_1",
                        "direction": "out", "type": "!allo.stream<i32, 1>",
                    }],
                    "post_hls_records": [
                        {"rtl_equivalence_hash": "representative-rtl"}
                    ],
                },
                {
                    "semantic_id": "top/compute/pid=0,1",
                    "kernel": "compute", "pid": [0, 1], "ports": [],
                },
            ],
            "channels": [{
                "channel_id": "c0", "stream": "horizontal_0_1",
                "endpoints": [
                    {"pe": "top/compute/pid=0,0", "direction": "out",
                     "accesses": [{"port_ordinal": 0}]},
                    {"pe": "top/compute/pid=0,1", "direction": "in", "accesses": []},
                ],
            }],
        }
        wrapper, folding, connections = PLAN.generate_fifo_wrapper(
            "canonical_wrapper",
            manifest["pe_instances"][0],
            blocks["pe"],
            pe_instance,
            instances,
            manifest,
            blocks,
        )
        self.assertIn("module canonical_wrapper", wrapper)
        self.assertIn("pe producer_kernel", wrapper)
        self.assertIn("fifo folded_fifo_0_U", wrapper)
        self.assertNotIn("stream_din,\n", wrapper.split(");", 1)[0])
        self.assertIn("stream_dout", wrapper.split(");", 1)[0])
        self.assertEqual(folding["owned_fifos"][0]["fifo_instance"], "fifo_0_U")
        self.assertEqual(connections["stream_dout"], "consumer_data")

    def test_classifies_repeated_children_of_one_semantic_pe(self):
        repeated = [
            {"semantic_id": "top/load/pid=0", "rtl_module": "pipe_0"},
            {"semantic_id": "top/load/pid=0", "rtl_module": "pipe_1"},
        ]
        semantic = [
            {"semantic_id": "top/compute/pid=0", "rtl_module": "pe_0"},
            {"semantic_id": "top/compute/pid=1", "rtl_module": "pe_1"},
        ]
        self.assertEqual(PLAN.classify_macro_candidate(repeated), "repeated_hls_submodule")
        self.assertEqual(PLAN.classify_macro_candidate(semantic), "semantic_pe")

    def test_discovers_and_binds_parameterized_catapult_pipe(self):
        rtl = """
module top(input clk, input rst);
  ccs_pipe_v6 #(.rscid(32'sd7), .width(32'sd26), .fifo_sz(32'sd8)) fifo0 (
    .clk(clk), .en(1'b0), .arst(1'b1), .srst(rst),
    .din_rdy(p_rdy), .din_vld(p_vld), .din(p_dat),
    .dout_rdy(c_rdy), .dout_vld(c_vld), .dout(c_dat),
    .sz(), .sz_req(1'b0), .is_idle());
endmodule
module ccs_pipe_v6(); endmodule
"""
        top = PLAN.module_blocks(rtl)["top"]
        pipes = PLAN.catapult_pipe_instances(top)
        pe = PLAN.InstanceBlock(
            "pe", "pe0", 0, 0,
            {"v0_rsc_dat": "p_dat", "v0_rsc_vld": "p_vld", "v0_rsc_rdy": "p_rdy"},
        )
        fifo, parameters = PLAN.find_owned_catapult_fifo("v0", pe, pipes)
        self.assertEqual(fifo.name, "fifo0")
        self.assertEqual(parameters["fifo_sz"], "32'sd8")

    def test_catapult_fifo_wrapper_connects_consumer_side_at_boundary(self):
        rtl = """
module top(clk, rst);
  input clk; input rst;
  pe pe0(.clk(clk), .rst(rst),
    .v0_rsc_dat(producer_data), .v0_rsc_vld(producer_vld),
    .v0_rsc_rdy(producer_rdy));
  ccs_pipe_v6 #(.rscid(32'sd7), .width(32'sd26), .fifo_sz(32'sd8)) fifo0 (
    .clk(clk), .en(1'b0), .arst(1'b1), .srst(rst),
    .din_rdy(producer_rdy), .din_vld(producer_vld), .din(producer_data),
    .dout_rdy(consumer_rdy), .dout_vld(consumer_vld), .dout(consumer_data),
    .sz(), .sz_req(1'b0), .is_idle());
endmodule
module pe(clk, rst, v0_rsc_dat, v0_rsc_vld, v0_rsc_rdy);
  input clk; input rst;
  output [25:0] v0_rsc_dat; output v0_rsc_vld; input v0_rsc_rdy;
endmodule
module ccs_pipe_v6(); endmodule
"""
        blocks = PLAN.module_blocks(rtl)
        pe_instance = next(
            item
            for item in PLAN.module_instances(blocks["top"], set(blocks))
            if item.module == "pe"
        )
        semantic_id = "top/producer/pid=0"
        manifest = {
            "top": "top",
            "pe_instances": [{
                "semantic_id": semantic_id,
                "kernel": "producer",
                "pid": [0],
                "ports": [{
                    "ordinal": 0,
                    "channel_id": "c0",
                    "stream": "stream0",
                    "direction": "out",
                    "type": "!allo.stream<i26, 8>",
                }],
            }, {
                "semantic_id": "top/consumer/pid=0",
                "kernel": "consumer",
                "pid": [0],
                "ports": [],
            }],
            "channels": [{
                "channel_id": "c0",
                "stream": "stream0",
                "endpoints": [
                    {"pe": semantic_id, "direction": "out",
                     "accesses": [{"port_ordinal": 0}]},
                    {"pe": "top/consumer/pid=0", "direction": "in", "accesses": []},
                ],
            }],
        }

        wrapper, folding, connections = PLAN.generate_catapult_fifo_wrapper(
            "canonical_wrapper",
            manifest["pe_instances"][0],
            blocks["pe"],
            pe_instance,
            blocks["top"],
            manifest,
            blocks,
        )

        self.assertIn("ccs_pipe_v6", wrapper)
        self.assertEqual(folding["folded_fifo_count"], 1)
        self.assertEqual(connections["v0_rsc_dat"], "consumer_data")
        self.assertEqual(connections["v0_rsc_vld"], "consumer_vld")
        self.assertEqual(connections["v0_rsc_rdy"], "consumer_rdy")
        self.assertNotEqual(connections["v0_rsc_dat"], "producer_data")

    def test_systemc_fifo_wrapper_connects_dequeue_side_at_boundary(self):
        rtl = """
module top(clk, rst);
  input clk; input rst;
  pe pe0(.clk(clk), .rst(rst), .v0_dat(p_dat), .v0_vld(p_vld), .v0_rdy(p_rdy));
  Connections_Fifo_ac_int_26_false_8U_Connections_SYN_PORT fifo0 (
    .clk(clk), .rst(rst), .enq_vld(p_vld), .enq_rdy(p_rdy), .enq_dat(p_dat),
    .deq_vld(c_vld), .deq_rdy(c_rdy), .deq_dat(c_dat));
endmodule
module pe(clk, rst, v0_dat, v0_vld, v0_rdy);
  input clk; input rst; output [25:0] v0_dat; output v0_vld; input v0_rdy;
endmodule
module Connections_Fifo_ac_int_26_false_8U_Connections_SYN_PORT(); endmodule
"""
        blocks = PLAN.module_blocks(rtl)
        pe_instance = next(
            item for item in PLAN.module_instances(blocks["top"], set(blocks))
            if item.module == "pe"
        )
        semantic_id = "top/producer/pid=0"
        manifest = {
            "top": "top",
            "pe_instances": [{
                "semantic_id": semantic_id, "kernel": "producer", "pid": [0],
                "ports": [{
                    "ordinal": 0, "channel_id": "c0", "stream": "stream0",
                    "direction": "out", "type": "!allo.stream<i26, 8>",
                }],
            }, {
                "semantic_id": "top/consumer/pid=0", "kernel": "consumer",
                "pid": [0], "ports": [],
            }],
            "channels": [{
                "channel_id": "c0", "stream": "stream0",
                "endpoints": [
                    {"pe": semantic_id, "direction": "out",
                     "accesses": [{"port_ordinal": 0}]},
                    {"pe": "top/consumer/pid=0", "direction": "in", "accesses": []},
                ],
            }],
        }

        wrapper, folding, connections = PLAN.generate_catapult_fifo_wrapper(
            "canonical_wrapper", manifest["pe_instances"][0], blocks["pe"],
            pe_instance, blocks["top"], manifest, blocks, "systemc",
        )

        self.assertIn("Connections_Fifo_ac_int_26_false_8U_Connections_SYN_PORT", wrapper)
        self.assertEqual(folding["protocol"], "systemc_matchlib_ready_valid")
        self.assertEqual(folding["folded_fifo_count"], 1)
        self.assertEqual(connections["v0_dat"], "c_dat")
        self.assertEqual(connections["v0_vld"], "c_vld")
        self.assertEqual(connections["v0_rdy"], "c_rdy")

    def test_explicit_bypass_emits_empty_batch_and_unchanged_rtl(self):
        rtl = "module chip(input clk); endmodule\n"
        manifest = {
            "schema_version": 2,
            "top": "chip",
            "summary": {
                "unmatched_or_ambiguous": 0,
                "unjoined_post_hls_records": 0,
            },
            "macro_groups": [],
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            inputs = root / "inputs"
            inputs.mkdir()
            (inputs / "design.v").write_text(rtl)
            (inputs / "asic-manifest-final.json").write_text(json.dumps(manifest))
            (inputs / "build-metadata.json").write_text("{}")
            previous = Path.cwd()
            try:
                os.chdir(root)
                with mock.patch.dict(
                    os.environ,
                    {
                        "min_macro_reuse": "2",
                        "macro_clock_period": "5",
                        "bypass_macro_generation": "True",
                    },
                ):
                    PLAN.main()
            finally:
                os.chdir(previous)

            index = json.loads((root / "outputs/macro-batch/index.json").read_text())
            self.assertTrue(index["bypass_macro_generation"])
            self.assertEqual(index["implementation_style"], "flat")
            self.assertEqual(index["entries"], [])
            self.assertEqual((root / "outputs/residual-design.v").read_text(), rtl)

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
                        "method": "specialized_mlir_emitted_hls_contract",
                        "implementation_contract_hash": "hls-contract-123",
                    },
                    "rtl_audit": {
                        "authority": False,
                        "status": "generated_rtl_diverged",
                        "distinct_hashes": ["abc123", "different-rtl"],
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
            self.assertEqual(entry["candidate_kind"], "semantic_pe")
            self.assertEqual(entry["implementation_contract_hash"], "hls-contract-123")
            self.assertEqual(
                entry["equivalence_method"],
                "specialized_mlir_emitted_hls_contract",
            )
            self.assertEqual(entry["rtl_audit_status"], "generated_rtl_diverged")
            self.assertEqual(
                entry["rtl_audit_hashes"], ["abc123", "different-rtl"]
            )
            self.assertEqual(entry["rtl_hash"], "representative-rtl")
            self.assertEqual(entry["owning_kernels"], ["chip/pe"])
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
        self.assertEqual(intent["clock_pins"], ["ap_clk"])
        self.assertEqual(intent["control_pins"], [])
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
                "set allo_asic_clock_pins [list {ap_clk}]", emitted
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

    def test_balances_non_neighbor_streams_without_location_prediction(self):
        representative = "r/compute/pid=3,4"
        ports = [
            {
                "ordinal": ordinal,
                "channel_id": f"external_{ordinal}",
                "stream": f"external_{ordinal}",
                "direction": "out",
                "type": "!allo.stream<i32, 1>",
            }
            for ordinal in range(4)
        ]
        manifest = {
            "pe_instances": [{
                "semantic_id": representative,
                "kernel": "compute",
                "pid": [3, 4],
                "ports": ports,
            }],
            "channels": [{
                "stream": port["stream"],
                "endpoints": [{
                    "pe": representative,
                    "direction": "out",
                    "accesses": [{"port_ordinal": port["ordinal"]}],
                }],
            } for port in ports],
        }

        decisions = PLAN.graph_pin_sides(manifest)
        self.assertEqual(
            {decision["method"] for decision in decisions.values()},
            {"non_neighbor_load_balance"},
        )
        self.assertEqual(
            {decision["side"] for decision in decisions.values()},
            {"N", "S", "E", "W"},
        )

        relocated = json.loads(json.dumps(manifest))
        relocated_id = "r/compute/pid=99,100"
        relocated["pe_instances"][0]["semantic_id"] = relocated_id
        relocated["pe_instances"][0]["pid"] = [99, 100]
        for channel in relocated["channels"]:
            channel["endpoints"][0]["pe"] = relocated_id
        relocated_decisions = PLAN.graph_pin_sides(relocated)
        self.assertEqual(
            [decisions[(representative, ordinal)]["side"] for ordinal in range(4)],
            [relocated_decisions[(relocated_id, ordinal)]["side"] for ordinal in range(4)],
        )

        intent = {
            "stream_bundles": [{
                "method": "non_neighbor_load_balance",
                "side": "E",
                "rtl_ports": ["result_dout", "result_empty_n", "result_read"],
                "rtl_port_widths": {
                    "result_dout": 37,
                    "result_empty_n": 1,
                    "result_read": 1,
                },
            }],
            "clock_pins": [],
            "clock_side": "N",
            "control_pins": [],
            "control_side": "N",
            "auxiliary_pins": [],
            "auxiliary_pin_sides": {},
        }
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "pin-intent.tcl"
            PLAN.write_pin_intent_tcl(output, intent)
            emitted = output.read_text()
            self.assertIn(
                "[list {result_dout} 37 [list {result_empty_n} {result_read}]]",
                emitted,
            )
            self.assertNotIn(
                "allo_asic_signal_pins_E [list {result_dout}", emitted
            )

    def test_keeps_immediate_neighbor_direction_ahead_of_balancing(self):
        manifest = {
            "pe_instances": [
                {
                    "semantic_id": "r/k/pid=0,0",
                    "kernel": "k",
                    "pid": [0, 0],
                    "ports": [{
                        "ordinal": 0,
                        "channel_id": "local",
                        "stream": "local",
                        "direction": "out",
                        "type": "!allo.stream<i32, 1>",
                    }],
                },
                {
                    "semantic_id": "r/k/pid=0,1",
                    "kernel": "k",
                    "pid": [0, 1],
                    "ports": [],
                },
            ],
            "channels": [{
                "stream": "local",
                "endpoints": [
                    {
                        "pe": "r/k/pid=0,0",
                        "direction": "out",
                        "accesses": [{"port_ordinal": 0}],
                    },
                    {
                        "pe": "r/k/pid=0,1",
                        "direction": "in",
                        "accesses": [],
                    },
                ],
            }],
        }
        decision = PLAN.graph_pin_sides(manifest)[("r/k/pid=0,0", 0)]
        self.assertEqual(decision, {"side": "E", "method": "same_kernel_neighbor"})

    def test_non_neighbor_provisional_sides_do_not_constrain_reuse_orientation(self):
        canonical = {
            "stream_bundles": [
                {
                    "method": "same_kernel_neighbor",
                    "side": "N",
                    "rtl_ports": ["local_dout"],
                },
                {
                    "method": "non_neighbor_load_balance",
                    "side": "E",
                    "rtl_ports": ["result_dout"],
                },
            ]
        }
        member = {
            "stream_bundles": [
                {
                    "method": "same_kernel_neighbor",
                    "side": "N",
                    "rtl_ports": ["member_local_dout"],
                },
                {
                    "method": "non_neighbor_load_balance",
                    "side": "S",
                    "rtl_ports": ["member_result_dout"],
                },
            ]
        }
        mapping = [
            {"canonical": "local_dout", "member": "member_local_dout"},
            {"canonical": "result_dout", "member": "member_result_dout"},
        ]
        self.assertEqual(
            PLAN.select_member_orientation(canonical, member, mapping), "R0"
        )

    def test_maps_mixed_direction_pipeline_after_control_stream_is_removed(self):
        block = PLAN.module_blocks(
            """
module pe_pipeline (
  ap_clk,
  command_dout, command_empty_n, command_read,
  weight_in_dout, weight_in_empty_n, weight_in_read,
  weight_out_din, weight_out_full_n, weight_out_write,
  input_dout, input_empty_n, input_read,
  partial_in_dout, partial_in_empty_n, partial_in_read,
  input_out_din, input_out_full_n, input_out_write,
  partial_out_din, partial_out_full_n, partial_out_write
);
 input ap_clk;
 input [127:0] command_dout;
 input command_empty_n;
 output command_read;
 input [31:0] weight_in_dout;
 input weight_in_empty_n;
 output weight_in_read;
 output [31:0] weight_out_din;
 input weight_out_full_n;
 output weight_out_write;
 input [31:0] input_dout;
 input input_empty_n;
 output input_read;
 input [31:0] partial_in_dout;
 input partial_in_empty_n;
 output partial_in_read;
 output [31:0] input_out_din;
 input input_out_full_n;
 output input_out_write;
 output [31:0] partial_out_din;
 input partial_out_full_n;
 output partial_out_write;
endmodule
"""
        )["pe_pipeline"]
        directions = ["in", "in", "in", "out", "in", "in", "out", "out"]
        types = [
            "!allo.stream<i32, 1>",
            "!allo.stream<memref<4xi32>, 16>",
            "!allo.stream<i32, 16>",
            "!allo.stream<i32, 16>",
            "!allo.stream<i32, 16>",
            "!allo.stream<i32, 16>",
            "!allo.stream<i32, 16>",
            "!allo.stream<i32, 16>",
        ]
        ports = [
            {
                "ordinal": ordinal,
                "channel_id": f"channel_{ordinal}",
                "stream": f"stream_{ordinal}",
                "direction": direction,
                "type": type_text,
            }
            for ordinal, (direction, type_text) in enumerate(zip(directions, types))
        ]
        representative = "r/compute/pid=2,1"
        manifest = {
            "pe_instances": [
                {
                    "semantic_id": representative,
                    "kernel": "compute",
                    "pid": [2, 1],
                    "ports": ports,
                    "post_hls_records": [],
                }
            ],
            "channels": [
                {
                    "stream": port["stream"],
                    "endpoints": [
                        {
                            "pe": representative,
                            "direction": port["direction"],
                            "accesses": [{"port_ordinal": port["ordinal"]}],
                        }
                    ],
                }
                for port in ports
            ],
        }

        intent = PLAN.build_pin_intent(manifest, representative, block)

        self.assertEqual(
            [bundle["ordinal"] for bundle in intent["stream_bundles"]],
            [1, 2, 3, 4, 5, 6, 7],
        )
        self.assertEqual(
            intent["semantic_to_rtl_method"],
            "ordered_direction_width_subset",
        )

    def test_maps_reordered_split_process_through_parent_bundle_identity(self):
        blocks = PLAN.module_blocks(
            """
module pe_parent (
  ap_clk,
  v_mode_dout, v_mode_empty_n, v_mode_read,
  v_command_dout, v_command_empty_n, v_command_read,
  v_result_dout, v_result_empty_n, v_result_read,
  v_result_id_dout, v_result_id_empty_n, v_result_id_read,
  v_store_din, v_store_full_n, v_store_write
);
 input ap_clk;
 input [31:0] v_mode_dout; input v_mode_empty_n; output v_mode_read;
 input [127:0] v_command_dout; input v_command_empty_n; output v_command_read;
 input [31:0] v_result_dout; input v_result_empty_n; output v_result_read;
 input [31:0] v_result_id_dout; input v_result_id_empty_n; output v_result_id_read;
 output [31:0] v_store_din; input v_store_full_n; output v_store_write;
endmodule
module pe_pipeline (
  ap_clk,
  v_store_din, v_store_full_n, v_store_write,
  v_result_dout, v_result_empty_n, v_result_read,
  v_result_id_dout, v_result_id_empty_n, v_result_id_read,
  v_command_dout, v_command_empty_n, v_command_read
);
 input ap_clk;
 output [31:0] v_store_din; input v_store_full_n; output v_store_write;
 input [31:0] v_result_dout; input v_result_empty_n; output v_result_read;
 input [31:0] v_result_id_dout; input v_result_id_empty_n; output v_result_id_read;
 input [127:0] v_command_dout; input v_command_empty_n; output v_command_read;
endmodule
"""
        )
        representative = "r/vpu_lane/pid=0"
        directions = ["in", "in", "in", "in", "out"]
        widths = [32, 128, 32, 32, 32]
        ports = [
            {
                "ordinal": ordinal,
                "channel_id": f"channel_{ordinal}",
                "stream": f"stream_{ordinal}",
                "direction": direction,
                "type": f"!allo.stream<i{width}, 1>",
            }
            for ordinal, (direction, width) in enumerate(zip(directions, widths))
        ]
        manifest = {
            "pe_instances": [{
                "semantic_id": representative,
                "kernel": "vpu_lane",
                "pid": [0],
                "ports": ports,
                "post_hls_records": [{
                    "rtl_modules": [{"name": "pe_pipeline"}],
                    "rtl_instances": [{"parent_file": "/tmp/pe_parent.v"}],
                }],
            }],
            "channels": [{
                "stream": port["stream"],
                "endpoints": [{
                    "pe": representative,
                    "direction": port["direction"],
                    "accesses": [{"port_ordinal": port["ordinal"]}],
                }],
            } for port in ports],
        }

        intent = PLAN.build_pin_intent(
            manifest, representative, blocks["pe_pipeline"], blocks
        )

        self.assertEqual(
            [bundle["ordinal"] for bundle in intent["stream_bundles"]],
            [4, 2, 3, 1],
        )
        self.assertEqual(
            intent["semantic_to_rtl_method"],
            "parent_wrapper_bundle_identity",
        )


if __name__ == "__main__":
    unittest.main()
