# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LogNovations Holdings LLC
"""Build the study tables from the raw files.

    python scripts/build_tables.py ragtruth

Verifies the raw files against the manifest, then writes under
data/derived/<dataset>/ (git-ignored):

    items.jsonl        label-free item records, every split
    sources.jsonl      source identifier and SOURCE reference text
    labels_open.json   loss and annotation summaries, open splits only
    build_report.json  counts

The labels of the sealed splits are not written anywhere.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

HARNESS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HARNESS / "src"))

from selfref.datasets.build import build_ragtruth, build_wikibio  # noqa: E402


def verify_raw(dataset: str) -> dict[str, Path]:
    manifest = json.loads((HARNESS / "datasets.manifest.json").read_text())
    paths = {}
    for entry in manifest["files"]:
        if entry["dataset"] != dataset:
            continue
        path = HARNESS / entry["local"]
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != entry["sha256"]:
            raise SystemExit(f"{entry['local']}: SHA-256 differs from the manifest")
        paths[path.name] = path
    return paths


def write(out: Path, built) -> None:
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "items.jsonl", "w", encoding="utf-8") as handle:
        for item in built.items:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    with open(out / "sources.jsonl", "w", encoding="utf-8") as handle:
        for source_id, reference in built.sources.items():
            if isinstance(reference, bytes):                       # one SOURCE reference
                record = {"source_id": source_id, "reference": reference.decode("utf-8")}
            else:                                                  # a list of resamples
                record = {"source_id": source_id,
                          "resamples": [sample.decode("utf-8") for sample in reference]}
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    (out / "labels_open.json").write_text(json.dumps(built.labels, indent=0) + "\n")
    (out / "build_report.json").write_text(json.dumps(built.report, indent=2) + "\n")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("dataset", choices=["ragtruth", "wikibio"])
    args = parser.parse_args(argv)

    raw = verify_raw(args.dataset)
    if args.dataset == "ragtruth":
        built = build_ragtruth(raw["response.jsonl"], raw["source_info.jsonl"])
    else:
        import pyarrow.parquet as pq
        built = build_wikibio(pq.read_table(raw["evaluation.parquet"]).to_pylist())
    write(HARNESS / "data" / "derived" / args.dataset, built)
    print(json.dumps(built.report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
