ROOT := $(abspath ../../..)
CONDA ?= conda
ENV ?= allo
RUN = cd $(ROOT) && SKBUILD_EDITABLE_SKIP=$(ROOT)/build PYTHONPATH=$(ROOT) $(CONDA) run -n $(ENV)

.PHONY: oracle compiler cpu hls rtl

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
