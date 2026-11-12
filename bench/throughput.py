# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LogNovations Holdings LLC
"""Throughput of the reference backend and the open compressors.

    python bench/throughput.py --out results/throughput.json

The corpus is synthetic and seeded: no dataset is read. Each item has a source
of about 4 KB of pseudo-words and an output of about 1 KB that mixes spans
copied from the source with fresh words, so that matches of many lengths occur.
The figures say how fast the code is on this machine. They say nothing about
whether the features are useful.

Reported per operation: wall-clock seconds, megabytes of input processed, and
seconds per megabyte. The JSON also records the machine, the Python and
package versions, the git commit with a dirty flag, and the seed.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import random
import subprocess
import sys
import time
from importlib import metadata
from pathlib import Path

HARNESS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HARNESS / "src"))

from selfref.extract import FeatureInput, extract_features  # noqa: E402
from selfref.matchlen import match_length_profile, self_match_length_profile  # noqa: E402
from selfref.ncd import GzipCompressor, ZstdCompressor, ncd  # noqa: E402

MB = 1_000_000


def make_corpus(n_items: int, seed: int, source_words: int = 700, output_words: int = 170):
    """Seeded synthetic (output, source) pairs of pseudo-words."""
    rng = random.Random(seed)
    letters = "abcdefghijklmnopqrstuvwxyz"
    vocabulary = ["".join(rng.choices(letters, k=rng.randint(2, 9))) for _ in range(2000)]
    items = []
    for _ in range(n_items):
        source = rng.choices(vocabulary, k=source_words)
        output: list[str] = []
        while len(output) < output_words:
            if rng.random() < 0.5:                       # copy a span of the source
                start = rng.randrange(len(source))
                output += source[start:start + rng.randint(1, 12)]
            else:                                        # fresh words
                output += rng.choices(vocabulary, k=rng.randint(1, 6))
        items.append((" ".join(output[:output_words]).encode(), " ".join(source).encode()))
    return items


def environment(seed: int, n_items: int) -> dict:
    def git(*args):
        done = subprocess.run(["git", "-C", str(HARNESS), *args], capture_output=True, text=True)
        return done.stdout.strip() if done.returncode == 0 else None

    cpu = platform.processor()
    if platform.system() == "Darwin":
        done = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                              capture_output=True, text=True)
        cpu = done.stdout.strip() or cpu
    status = git("status", "--porcelain")
    return {
        "cpu": cpu,
        "logical_cores": os.cpu_count(),
        "machine": platform.machine(),
        "system": f"{platform.system()} {platform.release()}",
        "python": platform.python_version(),
        "packages": {name: metadata.version(name) for name in ("numpy", "zstandard")},
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty": None if status is None else bool(status),
        "seed": seed,
        "n_items": n_items,
        "threads": 1,
    }


def timed(operation, items, megabytes: float) -> dict:
    start = time.perf_counter()
    for output, source in items:
        operation(output, source)
    seconds = time.perf_counter() - start
    return {"seconds": round(seconds, 4), "megabytes": round(megabytes, 4),
            "seconds_per_megabyte": round(seconds / megabytes, 3)}


def run_reference(items) -> dict:
    """Timings for the open code paths. Keys are operation names."""
    both = sum(len(o) + len(s) for o, s in items) / MB
    outputs = sum(len(o) for o, _ in items) / MB
    gzip_c, zstd_c = GzipCompressor(), ZstdCompressor()
    return {
        "profile_source[reference]": timed(lambda o, s: match_length_profile(o, [s]), items, both),
        "profile_self[reference]": timed(lambda o, s: self_match_length_profile(o), items, outputs),
        "ncd[gzip]": timed(lambda o, s: ncd(o, s, gzip_c), items, both),
        "ncd[zstd]": timed(lambda o, s: ncd(o, s, zstd_c), items, both),
        "extract_features[self+source]": timed(
            lambda o, s: extract_features(FeatureInput(output=o, source=s)), items, both),
    }


def describe(items) -> dict:
    return {"mean_output_bytes": round(sum(len(o) for o, _ in items) / len(items), 1),
            "mean_source_bytes": round(sum(len(s) for _, s in items) / len(items), 1)}


def print_table(results: dict) -> None:
    width = max(len(name) for name in results)
    print(f"{'operation':<{width}}  {'seconds':>9}  {'MB':>8}  {'s/MB':>9}")
    for name, row in results.items():
        print(f"{name:<{width}}  {row['seconds']:>9.3f}  {row['megabytes']:>8.3f}  "
              f"{row['seconds_per_megabyte']:>9.3f}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=HARNESS / "results" / "throughput.json")
    parser.add_argument("--items", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20261111)
    args = parser.parse_args(argv)

    items = make_corpus(args.items, args.seed)
    results = run_reference(items)
    report = {"environment": environment(args.seed, args.items), "corpus": describe(items),
              "results": results}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print_table(results)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
