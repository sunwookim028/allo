ROOT := $(abspath ../../..)
CONDA ?= conda
ENV ?= allo
RUN = cd $(ROOT) && SKBUILD_EDITABLE_SKIP=$(ROOT)/build PYTHONPATH=$(ROOT) $(CONDA) run -n $(ENV)

.PHONY: oracle compiler hls

oracle:
	$(RUN) python -m examples.accelerator.tinytpu.oracle

compiler:
	$(RUN) python -m examples.accelerator.tinytpu.verify

hls:
	$(RUN) python -m examples.accelerator.tinytpu.verify --hls
