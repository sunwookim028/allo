#=============================================================================
# Makefile for Example 7: HLS Stream of Blocks
#=============================================================================

PROJECT = ex07.prj
VHLS = /opt/xilinx/Vitis_HLS/2023.2/bin/vitis_hls

# Try to find Vitis HLS include path
CFLAGS = -std=c++11
ifneq ($(VITIS_INC),)
    CFLAGS += -I$(VITIS_INC)
endif

.PHONY: all csim clean test_a test_b test_c test_d test_e test_f synth view

all: test_a test_b test_c test_d test_e test_f

test_a:
	@echo "========================================="
	@echo "Testing Version A (static structure)..."
	@echo "========================================="
	@sed -i 's/set_top stream_blocks_[a-f]/set_top stream_blocks_a/' run.tcl
	$(VHLS) -f run.tcl 2>&1 | grep -A 20 "Version A\|Input\|Output\|value"

test_b:
	@echo "========================================="
	@echo "Testing Version B (array of structures)..."
	@echo "========================================="
	@sed -i 's/set_top stream_blocks_[a-f]/set_top stream_blocks_b/' run.tcl
	$(VHLS) -f run.tcl 2>&1 | grep -A 20 "Version B\|Call\|out\|expected"

test_c:
	@echo "========================================="
	@echo "Testing Version C (accumulator)..."
	@echo "========================================="
	@sed -i 's/set_top stream_blocks_[a-f]/set_top stream_blocks_c/' run.tcl
	$(VHLS) -f run.tcl 2>&1 | grep -A 20 "Version C\|Accumulator\|value\|expected"

test_d:
	@echo "========================================="
	@echo "Testing Version D (HLS Stream interface)..."
	@echo "========================================="
	@sed -i 's/set_top stream_blocks_[a-f]/set_top stream_blocks_d/' run.tcl
	$(VHLS) -f run.tcl 2>&1 | grep -A 30 "Version D\|Baseline\|Streaming\|Advantages\|FIFO\|Pipeline"

test_e:
	@echo "========================================="
	@echo "Testing Version E (complex structure)..."
	@echo "========================================="
	@sed -i 's/set_top stream_blocks_[a-f]/set_top stream_blocks_e/' run.tcl
	$(VHLS) -f run.tcl 2>&1 | grep -A 20 "Version E\|Input\|output"

test_f:
	@echo "========================================="
	@echo "Testing Version F (with reset)..."
	@echo "========================================="
	@sed -i 's/set_top stream_blocks_[a-f]/set_top stream_blocks_f/' run.tcl
	$(VHLS) -f run.tcl 2>&1 | grep -A 20 "Version F\|State\|value\|reset"

csim: all

synth:
	@echo "Running synthesis to inspect resource usage..."
	@sed -i 's/csim_design/csim_design\ncsynth_design/' run.tcl
	$(VHLS) -f run.tcl
	@sed -i 's/csim_design\ncsynth_design/csim_design/' run.tcl

view:
	@if [ -d "$(PROJECT)/solution1/syn/report" ]; then \
		echo "=== Resource Usage for Structures ==="; \
		grep -A 10 "BRAM\|FF\|LUT" $(PROJECT)/solution1/syn/report/top_csynth.rpt | head -30; \
	else \
		echo "No synthesis report. Run 'make synth' first."; \
	fi

clean:
	rm -rf $(PROJECT) *.log

help:
	@echo "Available targets:"
	@echo "  make test_a  - Test version A (static structure)"
	@echo "  make test_b  - Test version B (array of structures)"
	@echo "  make test_c  - Test version C (accumulator)"
	@echo "  make test_d  - Test version D (HLS Stream - demonstrates efficiency)"
	@echo "  make test_e  - Test version E (complex structure)"
	@echo "  make test_f  - Test version F (with reset)"
	@echo "  make synth   - Run synthesis"
	@echo "  make view     - View synthesis report"
	@echo "  make clean    - Clean generated files"
