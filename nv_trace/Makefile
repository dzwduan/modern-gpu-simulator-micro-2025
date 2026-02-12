# NV Trace - Standalone GPU Trace Extraction Tool
# Supports NVIDIA GPUs: Volta, Turing, Ampere, Ada Lovelace (RTX 4090), Hopper

all:
	$(MAKE) -C tracer_tool
	$(MAKE) all -C tracer_tool/traces-processing

clean:
	$(MAKE) clean -C tracer_tool
	$(MAKE) clean -C tracer_tool/traces-processing

install_nvbit:
	bash install_nvbit.sh

.PHONY: all clean install_nvbit
