# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LogNovations Holdings LLC
"""Score every item of the given splits with the groundedness baseline.

    python scripts/score_groundedness.py ragtruth --splits TRAIN DEV_SELECT DEV_CAL DEV_TEST

Reads the label-free item and source tables and appends one line per item to
data/derived/<dataset>/groundedness.jsonl: the item identifier and the support
probability of each of its sentences. It is resumable: items already in the
output are skipped. It never reads a label file.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

HARNESS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HARNESS / "src"))
os.environ.setdefault("NLTK_DATA", str(HARNESS / ".venv" / "nltk_data"))
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

from selfref.groundedness import MiniCheckScorer, claims_of  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("dataset", choices=["ragtruth"])
    parser.add_argument("--splits", nargs="+", required=True)
    parser.add_argument("--limit", type=int, default=None, help="stop after this many new items")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default=None)
    args = parser.parse_args(argv)

    derived = HARNESS / "data" / "derived" / args.dataset
    items = [json.loads(line) for line in open(derived / "items.jsonl", encoding="utf-8")]
    items = [item for item in items if item["split"] in set(args.splits)]
    sources = {}
    for line in open(derived / "sources.jsonl", encoding="utf-8"):
        record = json.loads(line)
        sources[record["source_id"]] = record["reference"]

    out_path = derived / "groundedness.jsonl"
    done = set()
    if out_path.exists():
        done = {json.loads(line)["id"] for line in open(out_path, encoding="utf-8")}
    todo = [item for item in items if item["id"] not in done]
    todo.sort(key=lambda item: item["source_id"])            # one source's responses together
    if args.limit is not None:
        todo = todo[:args.limit]
    print(f"{len(items)} items in the requested splits, {len(done)} already scored, "
          f"{len(todo)} to score", flush=True)
    if not todo:
        return 0

    scorer = MiniCheckScorer(device=args.device, batch_size=args.batch_size)
    print(f"model on {scorer.device}", flush=True)
    chunk_cache: dict[str, list[str]] = {}
    start, pairs = time.perf_counter(), 0
    with open(out_path, "a", encoding="utf-8") as out:
        for n, item in enumerate(todo, 1):
            source_id = item["source_id"]
            if source_id not in chunk_cache:
                chunk_cache = {source_id: scorer.chunks_of(sources[source_id])}
            chunks = chunk_cache[source_id]
            claims = claims_of(item["response"])
            support = scorer.support(chunks, claims)
            pairs += len(chunks) * len(claims)
            out.write(json.dumps({"id": item["id"], "support": [round(p, 6) for p in support]}) + "\n")
            out.flush()
            if n % 50 == 0 or n == len(todo):
                elapsed = time.perf_counter() - start
                print(f"{n}/{len(todo)} items  {pairs} pairs  {elapsed:.0f}s  "
                      f"{n / elapsed:.2f} items/s  {pairs / elapsed:.1f} pairs/s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
