ROOT := $(abspath ../../..)
CONDA ?= conda
ENV ?= allo
RUN = cd $(ROOT) && SKBUILD_EDITABLE_SKIP=$(ROOT)/build PYTHONPATH=$(ROOT) $(CONDA) run -n $(ENV)

.PHONY: oracle compiler cpu hls rtl synth ppa

oracle:
	$(RUN) python -m examples.accelerator.tinytpu.oracle

compiler:
	$(RUN) python -m examples.accelerator.tinytpu.verify

cpu:
	$(RUN) python -m examples.accelerator.tinytpu.microarch --backend cpu --out tinytpu_cpu_prj

hls:
	$(RUN) python -m examples.accelerator.tinytpu.verify --hls

rtl:
	$(RUN) python -m examples.accelerator.tinytpu.microarch --backend rtl --out tinytpu_rtl_prj

# Real Vitis HLS C-synthesis. Needs Vitis on PATH (source its settings64.sh).
synth:
	$(RUN) python -m examples.accelerator.tinytpu.synth

# The co-design objective: synthesize, re-measure the latency table, score cycles.
ppa:
	$(RUN) python -m examples.accelerator.tinytpu.ppa
