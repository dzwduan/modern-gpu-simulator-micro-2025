# GoogleTest unit tests for the remodeling primitives.
#
# The tests link the object files emitted by the simulator build, so they
# exercise the production translation units themselves; no production source is
# recompiled here. Build the simulator first, then:
#
#   make -C tests/cpp test
#
# This build is out of tree: it is not reachable from the simulator Makefiles
# and adds nothing to them.

THIS_DIR := $(patsubst %/,%,$(dir $(abspath $(lastword $(MAKEFILE_LIST)))))
REPO_ROOT := $(abspath $(THIS_DIR)/../..)
SIM_ROOT := $(REPO_ROOT)/simulator-remodeled
REMODEL_SRC_DIR := $(SIM_ROOT)/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling
# Use the configuration the simulator build itself is using: setup_environment
# exports GPGPUSIM_CONFIG (for example gcc-/cuda-12060/release), which names the
# build tree exactly. Without it, discover the trees but refuse to guess when
# more than one exists, since picking the wrong one links stale objects.
ifdef GPGPUSIM_CONFIG
REMODEL_OBJ_DIR := $(SIM_ROOT)/gpu-simulator/gpgpu-sim/build/$(GPGPUSIM_CONFIG)/remodeling
else
REMODEL_OBJ_CANDIDATES := $(wildcard $(SIM_ROOT)/gpu-simulator/gpgpu-sim/build/*/*/*/remodeling)
REMODEL_OBJ_DIR := $(if $(filter 1,$(words $(REMODEL_OBJ_CANDIDATES))),$(REMODEL_OBJ_CANDIDATES),)
endif
TRACES_OBJ_DIR := $(SIM_ROOT)/util/traces_enhanced/obj
BUILD_DIR := $(THIS_DIR)/build

CXX ?= g++
TEST_CXXFLAGS := -std=c++17 -O0 -g -Wall
# Emit and consume header dependencies so a change to a production header
# recompiles the tests instead of silently reusing a stale object.
DEPFLAGS := -MMD -MP
INCLUDES := -I$(REMODEL_SRC_DIR)
GTEST_LIBS := -lgtest_main -lgtest -pthread

# Production objects under test, plus the trace-operand objects they call into.
PRODUCTION_OBJS := \
  $(REMODEL_OBJ_DIR)/register_encoding.o \
  $(REMODEL_OBJ_DIR)/access_queue.o \
  $(TRACES_OBJ_DIR)/traced_operand.o \
  $(TRACES_OBJ_DIR)/string_utilities.o

TEST_SRCS := test_register_encoding.cc test_access_queue.cc
TEST_OBJS := $(patsubst %.cc,$(BUILD_DIR)/%.o,$(TEST_SRCS))
BIN := $(BUILD_DIR)/remodeling_primitive_tests

.PHONY: all test clean require-production-objs

all: $(BIN)

# Relinking on every invocation keeps the binary in sync with the simulator
# objects, which this Makefile deliberately does not know how to rebuild.
require-production-objs:
	@if [ -z "$(REMODEL_OBJ_DIR)" ]; then \
	  echo "error: could not determine the simulator object directory; source gpu-simulator/setup_environment_no_git.sh (sets GPGPUSIM_CONFIG) and build the simulator first" >&2; \
	  exit 1; \
	fi
	@for obj in $(PRODUCTION_OBJS); do \
	  if [ ! -f "$$obj" ]; then \
	    echo "error: missing production object $$obj; build the simulator first" >&2; \
	    exit 1; \
	  fi; \
	done

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/%.o: $(THIS_DIR)/%.cc | $(BUILD_DIR)
	$(CXX) $(TEST_CXXFLAGS) $(DEPFLAGS) $(INCLUDES) -c -o $@ $<

$(BIN): require-production-objs $(TEST_OBJS) | $(BUILD_DIR)
	$(CXX) $(TEST_CXXFLAGS) -o $@ $(TEST_OBJS) $(PRODUCTION_OBJS) $(GTEST_LIBS)

test: $(BIN)
	$(BIN)

clean:
	rm -rf $(BUILD_DIR)

-include $(TEST_OBJS:.o=.d)
