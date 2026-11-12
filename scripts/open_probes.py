# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LogNovations Holdings LLC
"""Three probes that the independent audit asked for (exploratory).

    python scripts/open_probes.py [--reps 300] [--boots 400]

None of them reads a sealed label, and the guard is not asked. They were
written after the confirmatory results had been seen, and nothing in them can
change those results.

near_duplicates   label-free. Pairs of sources in different splits whose
                  prompts are similar. The study's rule dropped sources whose
                  content had a Jaccard similarity of at least 0.9 on word
                  5-grams; this lists what remains from 0.5 upward.
temperature       open splits only. How well the temperature covariate alone
                  ranks errors within a generator, and what BASELINE and
                  BASELINE+SOURCE do when they are refitted without it.
no_gain           open splits only. The decision rule applied to pairs of
                  refits of the frozen BASELINE configuration, at the CAL and
                  TEST sizes: what the rule returns when the true gain is zero.

Writes controls/open_probes.json.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HARNESS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HARNESS / "src"))

from selfref import controls  # noqa: E402
from selfref.combiner import SEARCH_SPACE, Config, fit_and_select, risk_scores  # noqa: E402
from selfref.dedup import shingles  # noqa: E402
from selfref.experiment import Sample, column_sets, rehearse  # noqa: E402
from selfref.selcert_pin import PINNED_SHA, load_selcert  # noqa: E402
from selfref.tables import categories_of, load_ragtruth_split  # noqa: E402

DERIVED = HARNESS / "data" / "derived" / "ragtruth"
PREREG = HARNESS / "prereg"
OPEN = ("TRAIN", "DEV_SELECT", "DEV_CAL", "DEV_TEST")
FIT_SEED = 20261111                 # the seed of the frozen arms
SEEDS = {"temperature_bootstrap": 20261151, "no_gain": 20261152}
TEMPERATURE = "cov.temperature"
REPORT_FROM = 0.5


def git(*args) -> str:
    return subprocess.run(["git", "-C", str(HARNESS), *args], capture_output=True,
                          text=True).stdout.strip()


# ------------------------------------------------------------- near-duplicates

def near_duplicates() -> dict:
    """Pairs of sources in different splits with similar prompts. Label-free."""
    split = {}
    for line in open(DERIVED / "items.jsonl", encoding="utf-8"):
        item = json.loads(line)
        split[item["source_id"]] = item["split"]
    sets = {}
    for line in open(DERIVED / "sources.jsonl", encoding="utf-8"):
        source = json.loads(line)
        if source["source_id"] in split:
            sets[source["source_id"]] = shingles(source["reference"])
    keys = sorted(sets)
    postings: dict[str, list[int]] = defaultdict(list)
    pairs = []
    for i, key in enumerate(keys):
        shared: Counter = Counter()
        for shingle in sets[key]:
            for j in postings[shingle]:
                shared[j] += 1
            postings[shingle].append(i)
        for j, intersection in shared.items():
            similarity = intersection / (len(sets[key]) + len(sets[keys[j]]) - intersection)
            if similarity >= REPORT_FROM and split[key] != split[keys[j]]:
                pairs.append({"first": keys[j], "split_first": split[keys[j]], "second": key,
                              "split_second": split[key], "jaccard": similarity})
    pairs.sort(key=lambda p: -p["jaccard"])
    sealed = {"CAL", "TEST"}
    cal_test = [p for p in pairs if {p["split_first"], p["split_second"]} == sealed]
    return {"text_compared": "the prompt of each source, as lower-cased word 5-grams; the study's rule "
                             "compared the source content without the prompt's instruction",
            "threshold_of_the_study": 0.9, "listed_from": REPORT_FROM, "sources": len(keys),
            "pairs_in_different_splits": pairs,
            "pairs_between_cal_and_test": len(cal_test),
            "largest_between_cal_and_test": cal_test[0]["jaccard"] if cal_test else None,
            "pairs_with_one_source_in_cal_or_test":
                sum(1 for p in pairs if {p["split_first"], p["split_second"]} & sealed)}


# ----------------------------------------------------------------- temperature

def temperature(tables: dict, selcert, alpha: float, delta: float, boots: int) -> dict:
    """The temperature covariate on the open splits."""
    names = tables["TRAIN"].names
    sets = column_sets(names)
    column = names.index(TEMPERATURE)
    train = tables["TRAIN"].sample(sets["BASELINE"])
    values = tables["TRAIN"].X[:, column]
    by_generator = {}
    for generator in np.unique(train.stratum):
        chosen = train.stratum == generator
        distinct = np.unique(np.round(values[chosen], 6))
        by_generator[str(generator)] = {
            "responses_in_train": int(chosen.sum()),
            "distinct_temperatures": [float(v) for v in distinct],
            "auroc_of_temperature_alone": (controls.auroc(train.loss[chosen], values[chosen])
                                           if distinct.size > 1 else None)}

    parts = {"DEV_SELECT": "DEV_SELECT", "DEV_CAL": "DEV_CAL", "DEV_TEST": "DEV_TEST"}
    arms, risk_on_test = {}, {}
    for arm in ("BASELINE", "BASELINE+SOURCE"):
        for label, columns in ((arm, sets[arm]),
                               (f"{arm} without temperature", [c for c in sets[arm] if c != TEMPERATURE])):
            data = {split: tables[split].sample(columns) for split in OPEN}
            chosen = fit_and_select(data["TRAIN"].X, data["TRAIN"].loss, data["DEV_SELECT"].X,
                                    data["DEV_SELECT"].loss, FIT_SEED, space=SEARCH_SPACE)
            risk = {split: risk_scores(chosen.model, data[split].X) for split in parts}
            rehearsal = rehearse(selcert, risk, data, (alpha,), delta)[alpha]
            risk_on_test[label] = risk["DEV_TEST"]
            arms[label] = {"columns": len(columns), "selected": chosen.config.as_dict(),
                           "log_loss_dev_select": chosen.table[0]["log_loss"],
                           "auroc_dev_test": controls.auroc(data["DEV_TEST"].loss, risk["DEV_TEST"]),
                           "dev_rehearsal_coverage_points": rehearsal["coverage_points"],
                           "dev_rehearsal_thresholds": rehearsal["thresholds"]}
    test = tables["DEV_TEST"].sample(sets["BASELINE"])
    difference = lambda first, second: controls.paired_metric_difference(
        test, risk_on_test[first], risk_on_test[second], n_boot=boots,
        seed=SEEDS["temperature_bootstrap"])
    return {"split_evaluated": "DEV_TEST", "alpha": alpha,
            "by_generator_in_train": by_generator, "arms": arms,
            "paired_difference_on_dev_test": {
                "BASELINE: without temperature minus with":
                    difference("BASELINE", "BASELINE without temperature"),
                "without temperature: BASELINE+SOURCE minus BASELINE":
                    difference("BASELINE without temperature", "BASELINE+SOURCE without temperature")}}


# --------------------------------------------------------------------- no gain

def block_counts(split: str) -> dict:
    counts: dict[str, set] = {}
    for line in open(DERIVED / "items.jsonl", encoding="utf-8"):
        item = json.loads(line)
        if item["split"] == split:
            counts.setdefault(item["task_type"], set()).add(item["group"])
    return {block: len(groups) for block, groups in counts.items()}


def no_gain(tables: dict, selcert, alpha: float, delta: float, bar: float, floor: float,
            reps: int, boots: int) -> dict:
    """The decision rule on pairs of refits of the frozen BASELINE configuration."""
    baseline = json.loads((PREREG / "baseline.json").read_text())
    selected = baseline["selected"]
    config = Config(selected["kind"], tuple(sorted((k, v) for k, v in selected.items() if k != "kind")))
    data = {split: tables[split].sample(baseline["columns"]) for split in OPEN}
    parts = [data["DEV_CAL"], data["DEV_TEST"]]
    pool = Sample(*(np.concatenate([getattr(p, field) for p in parts])
                    for field in ("X", "loss", "stratum", "group", "block")))
    counts = (block_counts("CAL"), block_counts("TEST"))
    result = controls.aa_decisions(selcert, config, data["TRAIN"], data["DEV_SELECT"], pool, *counts,
                                   alpha=alpha, delta=delta, bar=bar, floor=floor, n_rep=reps,
                                   n_boot=boots, seed=SEEDS["no_gain"])
    return {"configuration": selected, "calibration_groups": counts[0], "test_groups": counts[1],
            "alpha": alpha, "BAR_points": bar, "FLOOR_points": floor, **result}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--reps", type=int, default=300)
    parser.add_argument("--boots", type=int, default=400)
    parser.add_argument("--auroc-boots", type=int, default=10_000)
    args = parser.parse_args(argv)
    started = time.perf_counter()
    limits = json.loads((PREREG / "bar_floor.json").read_text())
    alpha, delta = limits["primary_alpha"], limits["delta"]
    selcert = load_selcert()
    categories = categories_of(DERIVED)
    tables = {split: load_ragtruth_split(DERIVED, split, categories) for split in OPEN}

    record = {
        "status": "EXPLORATORY. Added after the confirmatory results were seen. No sealed label is read.",
        "provenance": {"git_commit": git("rev-parse", "HEAD"),
                       "git_dirty": bool(git("status", "--porcelain", "--", ".")),
                       "selcert_commit": PINNED_SHA, "seeds": {"fit": FIT_SEED, **SEEDS}},
        "near_duplicates": near_duplicates()}
    print(f"near-duplicates done ({time.perf_counter() - started:.0f}s)", flush=True)
    record["temperature"] = temperature(tables, selcert, alpha, delta, args.auroc_boots)
    print(f"temperature done ({time.perf_counter() - started:.0f}s)", flush=True)
    record["no_gain"] = no_gain(tables, selcert, alpha, delta, limits["BAR_points"],
                                limits["FLOOR_points"], args.reps, args.boots)
    (HARNESS / "controls" / "open_probes.json").write_text(json.dumps(record, indent=1) + "\n")
    n = record["no_gain"]
    print(f"no gain: FAIL {n['p_fail']:.3f}  INCONCLUSIVE {n['p_inconclusive']:.3f}  PASS {n['p_pass']:.3f}")
    print(f"wrote controls/open_probes.json ({time.perf_counter() - started:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
