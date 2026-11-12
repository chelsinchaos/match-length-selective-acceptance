# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LogNovations Holdings LLC
"""Runtime per megabyte of every score, on real items (pre-registration, Section 15).

    python scripts/runtime.py [--items 200] [--repeats 3]

The CPU scores are timed here, one thread, on the first ``--items`` TRAIN
responses of RAGTruth in a fixed hash order. Each score is timed ``--repeats``
times over the same items and the fastest pass is reported: another process
on the machine can only slow a pass down. A megabyte is a megabyte of input:
the output plus its source. The two neural baselines are not re-run: their
time is read from the logs of the scoring runs, which record the items scored
and the seconds taken. A missing log is reported as missing and a run that
stopped early as stopped; neither is estimated.
Writes controls/runtime.json.
"""
from __future__ import annotations

import argparse
import json
import platform
import re
import subprocess
import sys
import time
from pathlib import Path

HARNESS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HARNESS / "src"))

from selfref.baselines import lexical_baselines, rouge_l  # noqa: E402
from selfref.features import profile_features  # noqa: E402
from selfref.matchlen import match_length_profile, self_match_length_profile  # noqa: E402
from selfref.ncd import GzipCompressor, ZstdCompressor, ncd  # noqa: E402
from selfref.splits import unit_interval  # noqa: E402
from selfref.tokens import token_profile_features  # noqa: E402

DERIVED = HARNESS / "data" / "derived"
LOGS = HARNESS / "results" / "logs"
SALT = "selfref-prereg-v1/runtime-sample"
SCORES = {"ragtruth": "groundedness.jsonl", "wikibio": "selfcheck_nli.jsonl"}


def git(*args) -> str:
    return subprocess.run(["git", "-C", str(HARNESS), *args], capture_output=True,
                          text=True).stdout.strip()


def input_bytes(dataset: str, derived: Path) -> tuple[dict, dict]:
    """Split and bytes of input (output plus references) of every item, by id."""
    refs = {}
    for line in open(derived / dataset / "sources.jsonl", encoding="utf-8"):
        record = json.loads(line)
        refs[record["source_id"]] = (len(record["reference"].encode()) if dataset == "ragtruth"
                                     else sum(len(s.encode()) for s in record["resamples"]))
    split, size = {}, {}
    for line in open(derived / dataset / "items.jsonl", encoding="utf-8"):
        item = json.loads(line)
        split[item["id"]] = item["split"]
        size[item["id"]] = (len(item["response"].encode()) + refs[item["source_id"]] if dataset == "ragtruth"
                            else len(item["sentence"].encode()) + refs[item["group"]])
    return split, size


def from_log(name: str, pattern: str, dataset: str, splits, logs: Path = LOGS,
             derived: Path = DERIVED) -> dict:
    """Seconds per megabyte of a scoring run, from the last progress line of its log.

    The scorers append to their score file and skip what is already there, so a
    run that reached its end scored the last ``total`` records of that file. The
    megabytes are those of exactly these items. A run that resumed an earlier one
    is therefore timed on the items it scored itself; ``items_in_the_splits``
    says how many the splits hold. A log that stops early is reported, not timed.
    """
    path = logs / name
    if not path.exists():
        return {"status": "log missing", "log": name}
    matches = re.findall(pattern, path.read_text())
    if not matches:
        return {"status": "no progress line in the log", "log": name}
    done, total, seconds = (int(x) for x in matches[-1])
    if done != total:
        return {"status": f"the log stops at {done} of {total} items", "log": name}
    split, size = input_bytes(dataset, derived)
    scored = [json.loads(line)["id"] for line in open(derived / dataset / SCORES[dataset], encoding="utf-8")]
    timed = scored[-total:]
    if len(timed) != total or any(split.get(i) not in splits for i in timed):
        return {"status": f"the last {total} scores of {SCORES[dataset]} are not all from {', '.join(splits)}",
                "log": name}
    megabytes = sum(size[i] for i in timed) / 1e6
    return {"items": total, "items_in_the_splits": sum(s in splits for s in split.values()),
            "megabytes": megabytes, "seconds": seconds, "seconds_per_megabyte": seconds / megabytes,
            "log": name, "device": "mps"}


def time_score(fn, pairs, repeats: int, clock=time.perf_counter) -> dict:
    """Seconds for one pass of ``fn`` over ``pairs``: the fastest of ``repeats`` passes."""
    passes = []
    for _ in range(repeats):
        start = clock()
        for output, source in pairs:
            fn(output, source)
        passes.append(clock() - start)
    return {"seconds": min(passes), "passes": passes}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--items", type=int, default=200)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args(argv)

    sources = {}
    for line in open(DERIVED / "ragtruth" / "sources.jsonl", encoding="utf-8"):
        record = json.loads(line)
        sources[record["source_id"]] = record["reference"].encode("utf-8")
    items = [json.loads(line) for line in open(DERIVED / "ragtruth" / "items.jsonl", encoding="utf-8")]
    items = sorted((i for i in items if i["split"] == "TRAIN"),
                   key=lambda i: unit_interval(i["id"], SALT))[:args.items]
    pairs = [(i["response"].encode("utf-8"), sources[i["source_id"]]) for i in items]
    megabytes = sum(len(o) + len(s) for o, s in pairs) / 1e6
    gzip, zstd = GzipCompressor(), ZstdCompressor()
    text = lambda b: b.decode("utf-8", errors="replace")
    scores = {
        "SELF profile features (16 columns)":
            lambda o, s: profile_features(self_match_length_profile(o), prefix="self"),
        "SOURCE profile features (16 columns)":
            lambda o, s: profile_features(match_length_profile(o, [s]), prefix="source",
                                          reference_length=len(s)),
        "SOURCE and SELF profile features over word tokens": token_profile_features,
        "NCD, gzip": lambda o, s: ncd(o, s, gzip),
        "NCD, zstd": lambda o, s: ncd(o, s, zstd),
        "lexical baselines (length, n-gram novelty, ROUGE-L)": lexical_baselines,
        "ROUGE-L alone": lambda o, s: rouge_l(text(o), text(s)),
    }
    cpu = {}
    for name, fn in scores.items():
        timing = time_score(fn, pairs, args.repeats)
        cpu[name] = {**timing, "seconds_per_megabyte": timing["seconds"] / megabytes}
        print(f"{name:55s} {timing['seconds'] / megabytes:8.3f} s/MB", flush=True)

    progress = r"(\d+)/(\d+) (?:items|sentences)\s+(?:\d+ pairs\s+)?(\d+)s"
    record = {
        "provenance": {"git_commit": git("rev-parse", "HEAD"),
                       "git_dirty": bool(git("status", "--porcelain", "--", ".")),
                       "machine": platform.machine(), "processor": platform.processor(),
                       "python": platform.python_version(), "threads": 1},
        "megabyte": "one megabyte of input: the output plus its source (or its samples)",
        "cpu_scores": {"sample": f"first {len(pairs)} RAGTruth TRAIN responses in the hash order "
                                 f"of salt {SALT!r}", "items": len(pairs), "megabytes": megabytes,
                       "passes_per_score": args.repeats, "reported": "the fastest pass",
                       "scores": cpu},
        "neural_baselines": {
            "MiniCheck groundedness, RAGTruth CAL and TEST, first run": from_log(
                "groundedness_ragtruth_sealed.log", progress, "ragtruth", ("CAL", "TEST")),
            "MiniCheck groundedness, RAGTruth CAL and TEST, resumed run": from_log(
                "groundedness_ragtruth_sealed_resumed.log", progress, "ragtruth", ("CAL", "TEST")),
            "SelfCheck-NLI, WikiBio TEST": from_log(
                "selfcheck_wikibio_sealed.log", progress, "wikibio", ("TEST",))},
    }
    (HARNESS / "controls").mkdir(exist_ok=True)
    (HARNESS / "controls" / "runtime.json").write_text(json.dumps(record, indent=1) + "\n")
    print(json.dumps(record["neural_baselines"], indent=1))
    print("wrote controls/runtime.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
