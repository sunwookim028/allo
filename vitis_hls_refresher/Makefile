#=============================================================================
# Master Makefile - Run all examples
#=============================================================================

.PHONY: all clean help ex01 ex02 ex03 ex04 ex05 ex06 ex07

all: ex01 ex02 ex03 ex04 ex05 ex06 ex07 ex08

ex01:
	@echo "========================================="
	@echo "Example 1: Basic Static Variables"
	@echo "========================================="
	@cd ex01_basic_static && make csim

ex02:
	@echo "========================================="
	@echo "Example 2: Static Variables in Loops"
	@echo "========================================="
	@cd ex02_static_variables && make all

ex03:
	@echo "========================================="
	@echo "Example 3: Static Arrays"
	@echo "========================================="
	@cd ex03_static_arrays && make all

ex04:
	@echo "========================================="
	@echo "Example 4: Static in Pipelines"
	@echo "========================================="
	@cd ex04_pipelines && make all

ex05:
	@echo "========================================="
	@echo "Example 5: History Buffer"
	@echo "========================================="
	@cd ex05_history_buffer && make all

ex06:
	@echo "========================================="
	@echo "Example 6: RAM Module"
	@echo "========================================="
	@cd ex06_ram && make all

ex07:

ex08:
	@echo "========================================="
	@echo "Example 8: Shared Static Memory"
	@echo "========================================="
	@cd ex08_shared_memory && make all
	@echo "========================================="
	@echo "Example 7: Stream of Blocks"
	@echo "========================================="
	@cd ex07_stream_blocks && make all

clean:
	@cd ex01_basic_static && make clean
	@cd ex02_static_variables && make clean
	@cd ex03_static_arrays && make clean
	@cd ex04_pipelines && make clean
	@cd ex05_history_buffer && make clean
	@cd ex06_ram && make clean
	@cd ex07_stream_blocks && make clean
	@cd ex08_shared_memory && make clean

help:
	@echo "Vitis HLS Refresher - Master Makefile"
	@echo ""
	@echo "Targets:"
	@echo "  make all   - Run all examples"
	@echo "  make ex01  - Run example 1 (basic static)"
	@echo "  make ex02  - Run example 2 (static variables)"
	@echo "  make ex03  - Run example 3 (static arrays)"
	@echo "  make ex04  - Run example 4 (pipelines)"
	@echo "  make ex05  - Run example 5 (history buffer)"
	@echo "  make ex06  - Run example 6 (RAM module)"
	@echo "  make ex07  - Run example 7 (stream of blocks)"
	@echo "  make ex08  - Run example 8 (shared memory)"
	@echo "  make clean - Clean all generated files"
	@echo ""
	@echo "For detailed usage, see README.md"

ex08:
	@echo "========================================="
	@echo "Example 8: Shared Static Memory"
	@echo "========================================="
	@cd ex08_shared_memory && make all

all: ex01 ex02 ex03 ex04 ex05 ex06 ex07 ex08

clean:
	@cd ex01_basic_static && make clean
	@cd ex02_static_variables && make clean
	@cd ex03_static_arrays && make clean
	@cd ex04_pipelines && make clean
	@cd ex05_history_buffer && make clean
	@cd ex06_ram && make clean
	@cd ex07_stream_blocks && make clean

help:
	@echo "Vitis HLS Refresher - Master Makefile"
	@echo ""
	@echo "Targets:"
	@echo "  make all   - Run all examples"
	@echo "  make ex01  - Run example 1 (basic static)"
	@echo "  make ex02  - Run example 2 (static variables)"
	@echo "  make ex03  - Run example 3 (static arrays)"
	@echo "  make ex04  - Run example 4 (pipelines)"
	@echo "  make ex05  - Run example 5 (history buffer)"
	@echo "  make ex06  - Run example 6 (RAM module)"
	@echo "  make ex07  - Run example 7 (stream of blocks)"
	@echo "  make clean - Clean all generated files"
	@echo ""
	@echo "For detailed usage, see README.md"
