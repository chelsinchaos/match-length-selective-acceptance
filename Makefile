# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LogNovations Holdings LLC
#
# Run from research/selfref_harness/.
#
#   make venv            create .venv with the pinned requirements
#   make fetch-selcert   clone the selcert companion at the pinned commit
#   make test            the test suite (open reference backend)
#   make bench           throughput of the reference backend and open compressors

VENV   ?= .venv
PYTHON ?= $(VENV)/bin/python

.PHONY: venv fetch-selcert test bench

# The harness has its own environment so that it neither depends on nor changes
# whatever Python the surrounding repository uses.
venv:
	@if command -v uv >/dev/null 2>&1; then \
		uv venv $(VENV) --python 3.11 && uv pip install --python $(PYTHON) -r requirements.txt; \
	else \
		python3.11 -m venv $(VENV) && $(PYTHON) -m pip install -r requirements.txt; \
	fi
	$(PYTHON) -c "import nltk; nltk.download('punkt_tab', download_dir='$(VENV)/nltk_data', quiet=True)"

fetch-selcert:
	$(PYTHON) scripts/fetch_selcert.py

test:
	$(PYTHON) -m pytest -q

bench:
	$(PYTHON) bench/throughput.py --out results/throughput.json

# Optional private add-on targets. Absent from a public release; ignored if missing.
-include fastcodec_backend/private.mk
