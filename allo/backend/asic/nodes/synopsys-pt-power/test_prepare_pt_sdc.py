from prepare_pt_sdc import prepare_sdc


def test_marks_input_delays_without_changing_other_constraints():
    source = """\
create_clock [get_ports clk] -period 10
set_input_delay -add_delay 0 -clock [get_clocks clk] [get_ports data]
set_output_delay 0 -clock [get_clocks clk] [get_ports result]
"""
    assert prepare_sdc(source) == """\
create_clock [get_ports clk] -period 10
set_input_delay -source_latency_included -add_delay 0 -clock [get_clocks clk] [get_ports data]
set_output_delay 0 -clock [get_clocks clk] [get_ports result]
"""


def test_is_idempotent_and_preserves_indentation():
    source = (
        "  set_input_delay -source_latency_included -min 0.2 "
        "-clock clk [get_ports data]\n"
    )
    assert prepare_sdc(prepare_sdc(source)) == source


def test_does_not_rewrite_comments():
    source = "# set_input_delay 0 -clock clk [get_ports data]\n"
    assert prepare_sdc(source) == source
