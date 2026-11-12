# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LogNovations Holdings LLC
"""Compute the label-free score table for the given splits.

    python scripts/compute_features.py ragtruth --splits TRAIN DEV_SELECT DEV_CAL DEV_TEST

For every item it computes, from the output and its SOURCE reference only:

    self.*, source.*   the match-length profile features under test, plus the
                       gzip and zstd NCD (which count as baselines); for
                       WikiBio the fixed-reference frame is resample.*
    base.*             the lexical baselines
    base.random        a fixed pseudo-random score, as a control

and writes data/derived/<dataset>/features_<SPLIT>.npz with the item
identifiers, the column names and a float matrix. It never opens a label file.
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

from selfref.baselines import lexical_baselines  # noqa: E402
from selfref.extract import FeatureInput, extract_features  # noqa: E402
from selfref.splits import unit_interval  # noqa: E402

RANDOM_SCORE_SALT = "selfref-prereg-v1/random-score"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("dataset", choices=["ragtruth", "wikibio"])
    parser.add_argument("--splits", nargs="+", required=True)
    parser.add_argument("--derived", type=Path, default=None,
                        help="another derived directory with the same layout "
                             "(default: data/derived/<dataset>)")
    args = parser.parse_args(argv)

    derived = args.derived or HARNESS / "data" / "derived" / args.dataset
    sources = {}
    for line in open(derived / "sources.jsonl", encoding="utf-8"):
        record = json.loads(line)
        if args.dataset == "ragtruth":
            sources[record["source_id"]] = record["reference"].encode("utf-8")
        else:
            sources[record["source_id"]] = tuple(s.encode("utf-8") for s in record["resamples"])
    items = [json.loads(line) for line in open(derived / "items.jsonl", encoding="utf-8")]

    for split in args.splits:
        rows = [item for item in items if item["split"] == split]
        start = time.perf_counter()
        names, matrix, megabytes = None, [], 0.0
        for item in rows:
            if args.dataset == "ragtruth":
                output = item["response"].encode("utf-8")
                source = sources[item["source_id"]]
                megabytes += (len(output) + len(source)) / 1e6
                scores = extract_features(FeatureInput(output=output, source=source))
                scores.update(lexical_baselines(output, source))
            else:
                # WikiBio: the output is one sentence and the references are the
                # other samples drawn for the same prompt (the RESAMPLE frame).
                output = item["sentence"].encode("utf-8")
                resamples = sources[item["group"]]
                megabytes += (len(output) + sum(len(r) for r in resamples)) / 1e6
                scores = extract_features(FeatureInput(output=output, resamples=resamples))
                scores.update(lexical_baselines(output, b"\n".join(resamples)))
                scores["base.position"] = float(item["position"])
                scores["base.n_sentences"] = float(item["n_sentences"])
            scores["base.random"] = unit_interval(item["id"], RANDOM_SCORE_SALT)
            if names is None:
                names = sorted(scores)
            matrix.append([scores[name] for name in names])
        np.savez_compressed(derived / f"features_{split}.npz",
                            ids=np.array([item["id"] for item in rows]),
                            names=np.array(names), X=np.array(matrix, dtype=np.float64))
        elapsed = time.perf_counter() - start
        print(f"{split}: {len(rows)} items, {len(names)} columns, {megabytes:.1f} MB, "
              f"{elapsed:.0f}s ({elapsed / megabytes:.2f} s/MB)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
