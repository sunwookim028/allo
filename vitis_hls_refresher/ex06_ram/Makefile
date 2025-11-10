#=============================================================================
# Makefile for Example 6: RAM Module
#=============================================================================

PROJECT = ex06.prj
VHLS = vitis_hls

.PHONY: all csim clean test_a test_b test_c test_d test_e synth view

all: test_a test_b test_c test_d test_e

test_a:
	@echo "========================================="
	@echo "Testing Version A (basic RAM)..."
	@echo "========================================="
	@sed -i 's/set_top ram_[a-e]/set_top ram_a/' run.tcl
	$(VHLS) -f run.tcl 2>&1 | grep -A 30 "Version A\|Write\|Read\|out\|expected"

test_b:
	@echo "========================================="
	@echo "Testing Version B (dual-port)..."
	@echo "========================================="
	@sed -i 's/set_top ram_[a-e]/set_top ram_b/' run.tcl
	$(VHLS) -f run.tcl 2>&1 | grep -A 30 "Version B\|Write\|Read\|out\|expected"

test_c:
	@echo "========================================="
	@echo "Testing Version C (with initialization)..."
	@echo "========================================="
	@sed -i 's/set_top ram_[a-e]/set_top ram_c/' run.tcl
	$(VHLS) -f run.tcl 2>&1 | grep -A 30 "Version C\|Write\|Read\|out\|expected"

test_d:
	@echo "========================================="
	@echo "Testing Version D (undefined tracking)..."
	@echo "========================================="
	@sed -i 's/set_top ram_[a-e]/set_top ram_d/' run.tcl
	$(VHLS) -f run.tcl 2>&1 | grep -A 30 "Version D\|Write\|Read\|out\|expected"

test_e:
	@echo "========================================="
	@echo "Testing Version E (single-port)..."
	@echo "========================================="
	@sed -i 's/set_top ram_[a-e]/set_top ram_e/' run.tcl
	$(VHLS) -f run.tcl 2>&1 | grep -A 30 "Version E\|Write\|Read\|out\|expected"

csim: all

synth:
	@echo "Running synthesis to inspect BRAM usage..."
	@sed -i 's/csim_design/csim_design\ncsynth_design/' run.tcl
	$(VHLS) -f run.tcl
	@sed -i 's/csim_design\ncsynth_design/csim_design/' run.tcl

view:
	@if [ -d "$(PROJECT)/solution1/syn/report" ]; then \
		echo "=== BRAM Resource Usage ==="; \
		grep -A 10 "BRAM\|RAM\|memory" $(PROJECT)/solution1/syn/report/top_csynth.rpt | head -30; \
	else \
		echo "No synthesis report. Run 'make synth' first."; \
	fi

clean:
	rm -rf $(PROJECT) *.log

help:
	@echo "Available targets:"
	@echo "  make test_a  - Test version A (basic)"
	@echo "  make test_b  - Test version B (dual-port)"
	@echo "  make test_c  - Test version C (initialized)"
	@echo "  make test_d  - Test version D (undefined tracking)"
	@echo "  make test_e  - Test version E (single-port)"
	@echo "  make synth   - Run synthesis"
	@echo "  make view    - View synthesis report"
	@echo "  make clean   - Clean generated files"

