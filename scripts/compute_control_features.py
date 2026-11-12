# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LogNovations Holdings LLC
"""Label-free score tables that two of the pre-registered controls need (RAGTruth).

    python scripts/compute_control_features.py tokens   --splits TRAIN DEV_SELECT DEV_CAL DEV_TEST CAL TEST
    python scripts/compute_control_features.py shuffled --splits DEV_TEST TEST

    tokens     the SOURCE and SELF profile features over lower-cased word tokens
               instead of bytes (the byte-versus-token stress analysis). Writes
               data/derived/ragtruth/features_tokens_<SPLIT>.npz.
    shuffled   the 16 SOURCE profile features with each output paired with the
               source of another item of the same task type (the shuffled-source
               negative control). Sources are paired by a fixed derangement of
               the split's sources within task type, so all six responses of a
               source move to the same other source. Writes
               data/derived/ragtruth/features_shuffled_source_<SPLIT>.npz.

Like ``compute_features.py``, it reads the item and source tables only and
never opens a label file.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HARNESS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HARNESS / "src"))

from selfref.controls import derangement  # noqa: E402
from selfref.extract import FeatureInput, extract_features  # noqa: E402
from selfref.tokens import token_profile_features  # noqa: E402

DERIVED = HARNESS / "data" / "derived" / "ragtruth"
DERANGEMENT_SEED = 20261141


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("kind", choices=["tokens", "shuffled"])
    parser.add_argument("--splits", nargs="+", required=True)
    args = parser.parse_args(argv)

    sources = {}
    for line in open(DERIVED / "sources.jsonl", encoding="utf-8"):
        record = json.loads(line)
        sources[record["source_id"]] = record["reference"].encode("utf-8")
    items = [json.loads(line) for line in open(DERIVED / "items.jsonl", encoding="utf-8")]

    for split in args.splits:
        rows = [item for item in items if item["split"] == split]
        start = time.perf_counter()
        if args.kind == "shuffled":
            other = derangement([item["source_id"] for item in rows],
                                [item["task_type"] for item in rows], seed=DERANGEMENT_SEED)
        names, matrix = None, []
        for item in rows:
            output = item["response"].encode("utf-8")
            if args.kind == "tokens":
                scores = token_profile_features(output, sources[item["source_id"]])
            else:
                scores = extract_features(FeatureInput(output=output,
                                                       source=sources[other[item["source_id"]]]))
                scores = {name: value for name, value in scores.items()
                          if name.startswith("source.") and ".ncd_" not in name}
            if names is None:
                names = sorted(scores)
            matrix.append([scores[name] for name in names])
        stem = "tokens" if args.kind == "tokens" else "shuffled_source"
        np.savez_compressed(DERIVED / f"features_{stem}_{split}.npz",
                            ids=np.array([item["id"] for item in rows]),
                            names=np.array(names), X=np.array(matrix, dtype=np.float64))
        print(f"{args.kind} {split}: {len(rows)} items, {len(names)} columns, "
              f"{time.perf_counter() - start:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
