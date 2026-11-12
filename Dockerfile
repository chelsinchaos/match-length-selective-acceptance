# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LogNovations Holdings LLC
#
# The harness's locked environment in a container. Build from this directory:
#
#   docker build -t selfref .
#   docker run --rm selfref                      # the test suite and the smoke experiment
#   docker run --rm -it -v "$PWD/data:/work/data" selfref bash
#
# The image holds the code and the locked packages. It holds no dataset and no
# model weights: `make reproduce` downloads the raw files and checks them
# against the SHA-256 values in datasets.manifest.json, and the two neural
# baselines download their pinned checkpoints on first use.
#
# torch is installed from the CPU wheel index. The numbers in the report were
# computed on Apple silicon with the neural baselines on the MPS backend; on
# another backend their scores can differ in the last decimals (see REPORT.md,
# reproducibility).
FROM python:3.11-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends git make \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /work
COPY requirements.lock .
RUN pip install --no-cache-dir -r requirements.lock \
        --extra-index-url https://download.pytorch.org/whl/cpu
RUN python -c "import nltk; nltk.download('punkt_tab', download_dir='/usr/local/share/nltk_data', quiet=True)"

COPY . .
RUN python scripts/fetch_selcert.py

ENV PYTHON=python
CMD ["sh", "-c", "python -m pytest -q && python scripts/smoke.py"]
