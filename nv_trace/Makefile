# NV Trace - Standalone GPU Trace Extraction Tool
# Supports NVIDIA GPUs: Volta, Turing, Ampere, Ada Lovelace (RTX 4090), Hopper

all:
	$(MAKE) -C tracer_tool
	$(MAKE) all -C tracer_tool/traces-processing

others: spinlock_tool bbv_tool occupancy_calc_tool silicon_checkpoint_tool

spinlock_tool:
	$(MAKE) -C others/spinlock_tool

bbv_tool:
	$(MAKE) -C others/bbv_tool

occupancy_calc_tool:
	$(MAKE) -C others/occupancy_calc_tool

silicon_checkpoint_tool:
	$(MAKE) -C others/silicon_checkpoint_tool

clean:
	$(MAKE) clean -C tracer_tool
	$(MAKE) clean -C tracer_tool/traces-processing
	-$(MAKE) clean -C others/spinlock_tool
	-$(MAKE) clean -C others/bbv_tool
	-$(MAKE) clean -C others/occupancy_calc_tool
	-$(MAKE) clean -C others/silicon_checkpoint_tool

NVBIT_VERSION ?= 1.7.7.1

install_nvbit:
	NVBIT_VERSION=$(NVBIT_VERSION) bash install_nvbit.sh

.PHONY: all others clean install_nvbit spinlock_tool bbv_tool occupancy_calc_tool silicon_checkpoint_tool
