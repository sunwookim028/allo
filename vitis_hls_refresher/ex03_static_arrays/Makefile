#=============================================================================
# Makefile for Example 3: Static Arrays
#=============================================================================

PROJECT = ex03.prj
VHLS = vitis_hls

.PHONY: all csim clean test_a test_b test_c synth view

all: test_a test_b test_c

test_a:
	@echo "========================================="
	@echo "Testing Version A (non-static array)..."
	@echo "========================================="
	@sed -i 's/set_top top_[abc]/set_top top_a/' run.tcl
	$(VHLS) -f run.tcl 2>&1 | grep -A 20 "Version A\|Result\|expected"

test_b:
	@echo "========================================="
	@echo "Testing Version B (static large array)..."
	@echo "========================================="
	@sed -i 's/set_top top_[abc]/set_top top_b/' run.tcl
	$(VHLS) -f run.tcl 2>&1 | grep -A 20 "Version B\|Result\|expected"

test_c:
	@echo "========================================="
	@echo "Testing Version C (static with init)..."
	@echo "========================================="
	@sed -i 's/set_top top_[abc]/set_top top_c/' run.tcl
	$(VHLS) -f run.tcl 2>&1 | grep -A 20 "Version C\|Result\|expected"

csim: all

synth:
	@echo "Running synthesis to inspect resource usage..."
	@sed -i 's/csim_design/csim_design\ncsynth_design/' run.tcl
	$(VHLS) -f run.tcl
	@sed -i 's/csim_design\ncsynth_design/csim_design/' run.tcl

view:
	@if [ -d "$(PROJECT)/solution1/syn/report" ]; then \
		echo "=== Resource Usage ==="; \
		grep -A 10 "BRAM\|URAM\|FF\|LUT" $(PROJECT)/solution1/syn/report/top_csynth.rpt | head -30; \
	else \
		echo "No synthesis report. Run 'make synth' first."; \
	fi

clean:
	rm -rf $(PROJECT) *.log

help:
	@echo "Available targets:"
	@echo "  make test_a  - Test version A"
	@echo "  make test_b  - Test version B"
	@echo "  make test_c  - Test version C"
	@echo "  make synth   - Run synthesis to see resource usage"
	@echo "  make view     - View resource usage report"
	@echo "  make clean    - Clean generated files"

