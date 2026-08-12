import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).with_name("generate_testbench.py")
SPEC = importlib.util.spec_from_file_location("allo_testbench_generator", SCRIPT)
GENERATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GENERATOR)


def test_vitis_artifact_parsers():
    kernel = """
    void top(int32_t A[64], int32_t B[64], int32_t C[64]) {
      #pragma HLS interface m_axi port=A offset=slave bundle=gmem0
      #pragma HLS interface m_axi port=B offset=slave bundle=gmem1
      #pragma HLS interface m_axi port=C offset=slave bundle=gmem2
    }
    """
    assert GENERATOR._parse_cpp_arguments(kernel, "top") == ["A", "B", "C"]
    assert GENERATOR._parse_m_axi_pragmas(kernel) == {
        "A": "gmem0",
        "B": "gmem1",
        "C": "gmem2",
    }


def test_vitis_width_expressions_and_pointer_registers():
    rtl = """
    parameter C_M_AXI_GMEM0_DATA_WIDTH = 32;
    parameter C_M_AXI_GMEM0_WSTRB_WIDTH = (C_M_AXI_GMEM0_DATA_WIDTH / 8);
    output [C_M_AXI_GMEM0_DATA_WIDTH - 1:0] m_axi_gmem0_WDATA;
    output [C_M_AXI_GMEM0_WSTRB_WIDTH - 1:0] m_axi_gmem0_WSTRB;
    input ap_clk;
    """
    ports, parameters = GENERATOR._parse_ports(rtl)
    assert parameters["C_M_AXI_GMEM0_WSTRB_WIDTH"] == 4
    assert ports["m_axi_gmem0_WDATA"]["width"] == 32
    assert ports["m_axi_gmem0_WSTRB"]["width"] == 4

    control = """
    ADDR_A_DATA_0 = 6'h10,
    ADDR_A_DATA_1 = 6'h14,
    """
    assert GENERATOR._parse_pointer_registers(control, ["A"]) == {
        "A": {"data_0": 0x10, "data_1": 0x14}
    }


def test_vitis_integral_axi_packing_ratios():
    assert GENERATOR._packing_ratio(32, 32) == 1
    assert GENERATOR._packing_ratio(16, 32) == 2
    assert GENERATOR._packing_ratio(8, 64) == 8


def test_vitis_rejects_nonintegral_axi_packing_ratios():
    try:
        GENERATOR._packing_ratio(24, 32)
    except RuntimeError as error:
        assert "integral AXI packing ratio" in str(error)
    else:
        raise AssertionError("non-integral AXI packing ratio was accepted")


def test_vitis_simulation_templates_use_zero_based_memory_and_explicit_vcd():
    template_dir = SCRIPT.parent / "templates" / "vitis"
    memory_bfm = (template_dir / "vitis-axi-memory-bfm.sv").read_text()
    testbench = (template_dir / "testbench.sv.tpl").read_text()

    assert "BASE_ADDR = 'h0" in memory_bfm
    assert "ELEMENTS_PER_WORD = DATA_WIDTH / ELEMENT_WIDTH" in memory_bfm
    assert "packed_lane_index*ELEMENT_WIDTH +: ELEMENT_WIDTH" in memory_bfm
    assert '$fscanf(file_handle, "%h", element_value)' in memory_bfm
    assert '$test$plusargs("ALLO_DUMP_VCD")' in testbench
    assert '$dumpfile("outputs/run.vcd")' in testbench
    assert "$dumpvars(0, allo_generated_testbench)" in testbench


def test_vitis_bfms_drive_away_from_propagated_clock_edge():
    template_dir = SCRIPT.parent / "templates" / "vitis"
    memory_bfm = (template_dir / "vitis-axi-memory-bfm.sv").read_text()

    axil_bfm = (template_dir / "vitis-axilite-master-bfm.sv").read_text()
    testbench = (template_dir / "testbench.sv.tpl").read_text()

    assert 'ALLO_BAGL_BFM_DRIVE_DELAY_NS=%f' in memory_bfm
    assert "awready <= #(drive_delay_ns)" in memory_bfm
    assert "rvalid <= #(drive_delay_ns)" in memory_bfm
    assert 'ALLO_BAGL_INPUT_DELAY_NS=%f' in axil_bfm
    assert "@(negedge clk);" in axil_bfm
    assert 'ALLO_BAGL_CLK_INS_SRC_LAT_NS=%f' in testbench
    assert "#(allo_bagl_clock_compensation_ns + allo_bagl_output_delay_ns);" in testbench
