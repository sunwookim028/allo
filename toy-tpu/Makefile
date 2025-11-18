PROJECT_ROOT := $(abspath .)
BUILD_DIR := $(PROJECT_ROOT)/build
INCLUDE_DIR := $(PROJECT_ROOT)/include
SRC_DIR := $(PROJECT_ROOT)/src
HOST_DIR := $(PROJECT_ROOT)/host
SCRIPTS_DIR := $(PROJECT_ROOT)/scripts

CXX ?= g++
CXXFLAGS ?= -std=c++17 -Wall -Wextra -Werror -I$(INCLUDE_DIR)
LDFLAGS ?=

HOST_SOURCES := $(HOST_DIR)/host_test.cpp $(SRC_DIR)/top_module.cpp
HOST_BINARY := $(BUILD_DIR)/host_test

TRIALS ?= 10
SEED ?= 305419896

.PHONY: all setup build_host csim test hls clean

all: csim

setup:
	@$(SCRIPTS_DIR)/setup_env.sh

$(BUILD_DIR):
	@mkdir -p $(BUILD_DIR)

build_host: $(HOST_BINARY)

$(HOST_BINARY): $(BUILD_DIR) $(HOST_SOURCES)
	$(CXX) $(CXXFLAGS) $(HOST_SOURCES) -o $@ $(LDFLAGS)

csim: build_host
	$(HOST_BINARY) --trials $(TRIALS) --seed $(SEED)

test: csim

hls:
	@cd $(SCRIPTS_DIR) && vitis_hls -f run_hls.tcl

clean:
	rm -rf $(BUILD_DIR) miniTPU_HLS miniTPU_HLS.prj vitis_hls.log


