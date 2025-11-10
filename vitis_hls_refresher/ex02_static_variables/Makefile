#=============================================================================
# Makefile for Example 2: Static Variables in Loops
#=============================================================================

PROJECT = ex02.prj
VHLS = vitis_hls

.PHONY: all csim clean test_a test_b test_c

all: test_a test_b test_c

# Test Version A
test_a:
	@echo "========================================="
	@echo "Testing Version A (non-static)..."
	@echo "========================================="
	@sed -i 's/set_top top_[abc]/set_top top_a/' run.tcl
	$(VHLS) -f run.tcl 2>&1 | grep -A 30 "Version A\|Result\|expected"

# Test Version B
test_b:
	@echo "========================================="
	@echo "Testing Version B (static)..."
	@echo "========================================="
	@sed -i 's/set_top top_[abc]/set_top top_b/' run.tcl
	$(VHLS) -f run.tcl 2>&1 | grep -A 30 "Version B\|Result\|expected"

# Test Version C
test_c:
	@echo "========================================="
	@echo "Testing Version C (static with reset)..."
	@echo "========================================="
	@sed -i 's/set_top top_[abc]/set_top top_c/' run.tcl
	$(VHLS) -f run.tcl 2>&1 | grep -A 30 "Version C\|Result\|expected"

csim: test_a test_b test_c

clean:
	rm -rf $(PROJECT) *.log

help:
	@echo "Available targets:"
	@echo "  make test_a  - Test version A (non-static)"
	@echo "  make test_b  - Test version B (static)"
	@echo "  make test_c  - Test version C (static with reset)"
	@echo "  make all     - Test all versions"
	@echo "  make clean   - Clean generated files"

