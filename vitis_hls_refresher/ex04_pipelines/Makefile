#=============================================================================
# Makefile for Example 4: Static Variables in Pipelines
#=============================================================================

PROJECT = ex04.prj
VHLS = vitis_hls

.PHONY: all csim clean test_a test_b test_c test_d synth

all: test_a test_b test_c test_d

test_a:
	@echo "========================================="
	@echo "Testing Version A (non-static pipeline)..."
	@echo "========================================="
	@sed -i 's/set_top top_[abcd]/set_top top_a/' run.tcl
	$(VHLS) -f run.tcl 2>&1 | grep -A 20 "Version A\|Result\|II"

test_b:
	@echo "========================================="
	@echo "Testing Version B (static accumulator)..."
	@echo "========================================="
	@sed -i 's/set_top top_[abcd]/set_top top_b/' run.tcl
	$(VHLS) -f run.tcl 2>&1 | grep -A 20 "Version B\|Result\|II"

test_c:
	@echo "========================================="
	@echo "Testing Version C (shift register)..."
	@echo "========================================="
	@sed -i 's/set_top top_[abcd]/set_top top_c/' run.tcl
	$(VHLS) -f run.tcl 2>&1 | grep -A 20 "Version C\|Result\|II"

test_d:
	@echo "========================================="
	@echo "Testing Version D (state machine)..."
	@echo "========================================="
	@sed -i 's/set_top top_[abcd]/set_top top_d/' run.tcl
	$(VHLS) -f run.tcl 2>&1 | grep -A 20 "Version D\|Result\|state"

csim: all

synth:
	@echo "Running synthesis to inspect II..."
	@sed -i 's/csim_design/csim_design\ncsynth_design/' run.tcl
	$(VHLS) -f run.tcl
	@sed -i 's/csim_design\ncsynth_design/csim_design/' run.tcl

clean:
	rm -rf $(PROJECT) *.log

help:
	@echo "Available targets:"
	@echo "  make test_a  - Test version A"
	@echo "  make test_b  - Test version B"
	@echo "  make test_c  - Test version C"
	@echo "  make test_d  - Test version D"
	@echo "  make synth   - Run synthesis to see II"
	@echo "  make clean   - Clean generated files"

