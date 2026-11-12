# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LogNovations Holdings LLC
"""Score WikiBio sentences with the SelfCheck-NLI consistency baseline.

    python scripts/score_selfcheck.py --splits TRAIN DEV_SELECT DEV_CAL DEV_TEST

Appends one line per sentence to data/derived/wikibio/selfcheck_nli.jsonl: the
item identifier and the contradiction probability against each of its
passage's samples. Resumable. It never reads a label file.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HARNESS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HARNESS / "src"))

from selfref.selfcheck import SelfCheckNLI  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--splits", nargs="+", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args(argv)

    derived = HARNESS / "data" / "derived" / "wikibio"
    items = [json.loads(line) for line in open(derived / "items.jsonl", encoding="utf-8")]
    items = [item for item in items if item["split"] in set(args.splits)]
    samples = {}
    for line in open(derived / "sources.jsonl", encoding="utf-8"):
        record = json.loads(line)
        samples[record["source_id"]] = record["resamples"]
    out_path = derived / "selfcheck_nli.jsonl"
    done = {json.loads(line)["id"] for line in open(out_path)} if out_path.exists() else set()
    todo = [item for item in items if item["id"] not in done][:args.limit]
    print(f"{len(items)} sentences requested, {len(done)} done, {len(todo)} to score", flush=True)
    if not todo:
        return 0
    if args.threads:
        import torch
        torch.set_num_threads(args.threads)
    scorer = SelfCheckNLI(device=args.device)
    print(f"model on {scorer.device}", flush=True)
    start = time.perf_counter()
    with open(out_path, "a", encoding="utf-8") as out:
        for n, item in enumerate(todo, 1):
            probabilities = scorer.contradiction(item["sentence"], samples[item["group"]])
            out.write(json.dumps({"id": item["id"],
                                  "contradiction": [round(p, 6) for p in probabilities]}) + "\n")
            out.flush()
            if n % 50 == 0 or n == len(todo):
                elapsed = time.perf_counter() - start
                print(f"{n}/{len(todo)} sentences  {elapsed:.0f}s  "
                      f"{20 * n / elapsed:.1f} pairs/s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
