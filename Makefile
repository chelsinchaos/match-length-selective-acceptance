# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LogNovations Holdings LLC
#
# Run from research/selfref_harness/.
#
#   make venv            create .venv with the pinned requirements
#   make fetch-selcert   clone the selcert companion at the pinned commit
#   make test            the test suite (open reference backend)
#   make bench           throughput of the reference backend and open compressors
#   make smoke           a small synthetic end-to-end experiment (what CI runs)
#   make reproduce       rebuild every number, table and figure from the raw data;
#                        run it in a fresh clone (see scripts/reproduce.py)
#   make reproduce SCORES=dir   the same, reusing the neural baselines' score files
#   make compare REF=rev        compare the records with those at a git revision
#   make report          results.json, the tables and figures, REPORT.md
#   make paper           the paper: paper/paper.tex, its figures and paper/paper.pdf (needs pdflatex)
#   make docker          build the container and run the tests and the smoke experiment in it

VENV   ?= .venv
PYTHON ?= $(VENV)/bin/python

.PHONY: venv fetch-selcert test bench smoke reproduce compare report paper docker

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

smoke:
	$(PYTHON) scripts/smoke.py

reproduce:
	$(PYTHON) scripts/reproduce.py --commit $(if $(SCORES),--scores-from $(SCORES),)

compare:
	$(PYTHON) scripts/reproduce.py --compare $(REF)

report:
	$(PYTHON) scripts/collect_results.py
	$(PYTHON) scripts/make_report.py

# The paper reads the same results.json as the report. It does not collect it again.
paper:
	$(PYTHON) scripts/make_paper.py --pdf

docker:
	docker build -t selfref .
	docker run --rm selfref

# Optional private add-on targets. Absent from a public release; ignored if missing.
-include fastcodec_backend/private.mk
