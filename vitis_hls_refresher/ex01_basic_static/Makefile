#=============================================================================
# Makefile for Example 1: Basic Static Variable
#=============================================================================

PROJECT = ex01.prj
VHLS = vitis_hls

.PHONY: all csim clean view

all: csim

# Run C simulation (fastest iteration)
csim:
	@echo "========================================="
	@echo "Running C Simulation..."
	@echo "========================================="
	$(VHLS) -f run.tcl 2>&1 | grep -A 50 "Call\|Result\|Expected\|Key"

# Run synthesis to see RTL (slower)
synth:
	@echo "========================================="
	@echo "Running Synthesis..."
	@echo "========================================="
	@sed -i 's/# csynth_design/csynth_design/' run.tcl
	$(VHLS) -f run.tcl
	@sed -i 's/csynth_design/# csynth_design/' run.tcl

# View synthesis report
view:
	@if [ -d "$(PROJECT)/solution1/syn/report" ]; then \
		cat $(PROJECT)/solution1/syn/report/top_csynth.rpt | head -100; \
	else \
		echo "No synthesis report found. Run 'make synth' first."; \
	fi

clean:
	rm -rf $(PROJECT) *.log
	rm -f run.tcl.bak

help:
	@echo "Available targets:"
	@echo "  make csim   - Run C simulation (fast)"
	@echo "  make synth  - Run synthesis (slower, generates RTL)"
	@echo "  make view   - View synthesis report"
	@echo "  make clean  - Clean generated files"

