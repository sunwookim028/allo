#=============================================================================
# Makefile for Example 5: History Buffer
#=============================================================================

PROJECT = ex05.prj
VHLS = vitis_hls

.PHONY: all csim clean test_a test_b test_c synth view

all: test_a test_b test_c

test_a:
	@echo "========================================="
	@echo "Testing Version A (configurable delay)..."
	@echo "========================================="
	@sed -i 's/set_top history_buffer_[abc]/set_top history_buffer_a/' run.tcl
	$(VHLS) -f run.tcl 2>&1 | grep -A 30 "Version A\|Call\|out\|expected"

test_b:
	@echo "========================================="
	@echo "Testing Version B (pipelined)..."
	@echo "========================================="
	@sed -i 's/set_top history_buffer_[abc]/set_top history_buffer_b/' run.tcl
	$(VHLS) -f run.tcl 2>&1 | grep -A 30 "Version B\|Call\|out\|expected"

test_c:
	@echo "========================================="
	@echo "Testing Version C (fixed delay)..."
	@echo "========================================="
	@sed -i 's/set_top history_buffer_[abc]/set_top history_buffer_c/' run.tcl
	$(VHLS) -f run.tcl 2>&1 | grep -A 30 "Version C\|Call\|out\|expected"

csim: all

synth:
	@echo "Running synthesis to inspect shift register implementation..."
	@sed -i 's/csim_design/csim_design\ncsynth_design/' run.tcl
	$(VHLS) -f run.tcl
	@sed -i 's/csim_design\ncsynth_design/csim_design/' run.tcl

view:
	@if [ -d "$(PROJECT)/solution1/syn/report" ]; then \
		echo "=== Shift Register Implementation ==="; \
		grep -A 10 "shift_reg\|Register\|FF" $(PROJECT)/solution1/syn/report/top_csynth.rpt | head -30; \
	else \
		echo "No synthesis report. Run 'make synth' first."; \
	fi

clean:
	rm -rf $(PROJECT) *.log

help:
	@echo "Available targets:"
	@echo "  make test_a  - Test version A (configurable)"
	@echo "  make test_b  - Test version B (pipelined)"
	@echo "  make test_c  - Test version C (fixed delay)"
	@echo "  make synth   - Run synthesis"
	@echo "  make view    - View synthesis report"
	@echo "  make clean   - Clean generated files"

