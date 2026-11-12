# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LogNovations Holdings LLC
"""The confirmatory analysis on RAGTruth, and its rehearsal inside DEV.

    python scripts/confirm.py rehearsal
    python scripts/confirm.py preflight
    python scripts/confirm.py confirmatory --i-have-prereg-approval --purpose "confirmatory run"

The rehearsal and the confirmatory run call the same function,
``selfref.confirm.analyse``, on the frozen models. They differ only in the two
samples they hand to it:

    rehearsal      calibrates on DEV_CAL and tests on DEV_TEST. Open labels.
                   Writes prereg/rehearsal.json.
    preflight      reads no label and asks nothing of the guard. It checks that
                   the confirmatory run has what it needs: every CAL and TEST
                   item has its label-free scores, every frozen model matches
                   its recorded hash, and every model scores every item.
    confirmatory   calibrates on CAL and tests on TEST. It repeats the
                   preflight, then asks the guard, which requires the flag, the
                   prereg-v1 tag, an unchanged pre-registration and a clean
                   tree, and records the access in RUNLEDGER.md. Writes
                   confirmatory/result.json.

The frozen models are loaded from disk and checked against the SHA-256 values
recorded when they were frozen.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np

HARNESS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HARNESS / "src"))

from selfref.combiner import risk_scores  # noqa: E402
from selfref.confirm import analyse  # noqa: E402
from selfref.datasets.build import sealed_ragtruth_labels  # noqa: E402
from selfref.guard import access_count, authorise_all  # noqa: E402
from selfref.selcert_pin import PINNED_SHA, load_selcert  # noqa: E402
from selfref.tables import categories_of, load_ragtruth_split  # noqa: E402

DERIVED = HARNESS / "data" / "derived" / "ragtruth"
RAW = HARNESS / "data" / "raw" / "ragtruth"
PREREG = HARNESS / "prereg"
LEDGER = HARNESS / "RUNLEDGER.md"
ALPHAS = (0.01, 0.05, 0.10)
N_BOOT = 10_000
SEED = 20261111 + 10
MODEL_FILES = {"BASELINE": "BASELINE", "BASELINE+SOURCE": "BASELINE_plus_SOURCE",
               "BASELINE+SELF": "BASELINE_plus_SELF"}


def git(*args) -> str:
    return subprocess.run(["git", "-C", str(HARNESS), *args], capture_output=True,
                          text=True).stdout.strip()


def frozen(arm: str, expected_sha: str):
    blob = (DERIVED / "models" / f"{MODEL_FILES[arm]}.pkl").read_bytes()
    if hashlib.sha256(blob).hexdigest() != expected_sha:
        raise SystemExit(f"the frozen model for {arm} does not match its recorded SHA-256")
    return pickle.loads(blob)


def require_label_free_inputs(split: str) -> None:
    """Stop before the guard is asked if a sealed split's scores are not all present."""
    path = DERIVED / f"features_{split}.npz"
    if not path.exists():
        raise SystemExit(f"{path.name} is missing: run scripts/compute_features.py for {split}")
    ids = {str(i) for i in np.load(path)["ids"]}
    scored = {json.loads(line)["id"] for line in open(DERIVED / "groundedness.jsonl")}
    if ids - scored:
        raise SystemExit(f"{len(ids - scored)} {split} items have no groundedness score: "
                         f"run scripts/score_groundedness.py for {split}")


def preflight(records: dict, categories: dict) -> None:
    """Fail now, before the guard records an access, if the run could not finish.

    No label is read. The table of a sealed split is assembled with a placeholder
    in place of its loss column, and that column is not used: only the score
    matrix goes to the frozen models.
    """
    for split in ("CAL", "TEST"):
        require_label_free_inputs(split)
        ids = [str(i) for i in np.load(DERIVED / f"features_{split}.npz")["ids"]]
        table = load_ragtruth_split(DERIVED, split, categories,
                                    sealed_labels={i: {"loss": 0} for i in ids})
        for arm, record in records.items():
            scores = risk_scores(frozen(arm, record["model_sha256"]),
                                 table.sample(record["columns"]).X)
            if scores.shape != (len(ids),) or not np.all(np.isfinite(scores)):
                raise SystemExit(f"{arm} does not give a finite score to every {split} item")
        print(f"  preflight: {split} has {len(ids)} items, scored by every frozen model")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("mode", choices=["rehearsal", "preflight", "confirmatory"])
    parser.add_argument("--i-have-prereg-approval", action="store_true", dest="flag")
    parser.add_argument("--purpose", default="")
    parser.add_argument("--boots", type=int, default=N_BOOT)
    args = parser.parse_args(argv)

    limits = json.loads((PREREG / "bar_floor.json").read_text())
    records = {"BASELINE": json.loads((PREREG / "baseline.json").read_text())}
    records.update(json.loads((PREREG / "profile.json").read_text())["arms"])
    categories = categories_of(DERIVED)
    select = load_ragtruth_split(DERIVED, "DEV_SELECT", categories)

    if args.mode == "rehearsal":
        cal = load_ragtruth_split(DERIVED, "DEV_CAL", categories)
        test = load_ragtruth_split(DERIVED, "DEV_TEST", categories)
        entries, out = [], PREREG / "rehearsal.json"
    else:
        preflight(records, categories)
        if args.mode == "preflight":
            print("preflight passed: no label was read and the guard was not asked")
            return 0
        entries = authorise_all(("CAL", "TEST"), args.purpose, flag=args.flag,
                                repo=HARNESS, ledger=LEDGER)
        by_split = {entry["split"]: entry for entry in entries}
        raw = (RAW / "response.jsonl", RAW / "source_info.jsonl")
        cal = load_ragtruth_split(DERIVED, "CAL", categories,
                                  sealed_labels=sealed_ragtruth_labels(*raw, "CAL", by_split["CAL"]))
        test = load_ragtruth_split(DERIVED, "TEST", categories,
                                   sealed_labels=sealed_ragtruth_labels(*raw, "TEST", by_split["TEST"]))
        out = HARNESS / "confirmatory" / "result.json"

    tables = {"select": select, "cal": cal, "test": test}
    risk, data = {}, None
    for arm, record in records.items():
        model = frozen(arm, record["model_sha256"])
        samples = {part: table.sample(record["columns"]) for part, table in tables.items()}
        risk[arm] = {part: risk_scores(model, sample.X) for part, sample in samples.items()}
        data = samples                       # loss, stratum, group and block are the same for every arm

    result = analyse(load_selcert(), risk, data, primary_alpha=limits["primary_alpha"],
                     alphas=ALPHAS, delta=limits["delta"], bar=limits["BAR_points"],
                     floor=limits["FLOOR_points"], n_boot=args.boots, seed=SEED)
    result = {"mode": args.mode,
              "provenance": {"git_commit": git("rev-parse", "HEAD"),
                             "git_dirty_before_run": bool(entries == [] and git("status", "--porcelain", "--", ".")),
                             "selcert_commit": PINNED_SHA, "seed": SEED,
                             "ledger_entries": entries,
                             "test_accesses_so_far": access_count(LEDGER, "TEST"),
                             "cal_accesses_so_far": access_count(LEDGER, "CAL")},
              **result}
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")

    h1 = result["hypotheses"]["H1"]
    print(f"mode: {args.mode}   alpha = {result['primary_alpha']}   "
          f"BAR = {result['BAR_points']}   FLOOR = {result['FLOOR_points']}")
    for arm in records:
        row = result["coverage"][arm][str(result["primary_alpha"])]
        print(f"  {arm:16s} coverage {row['coverage_points']:6.2f} points   "
              f"realised risk {row['realised_risk']}")
    for name in ("H1", "H3"):
        h = result["hypotheses"][name]
        print(f"  {name}: gain {h['gain_points']:+.2f} points, 95% interval "
              f"[{h['interval_low']:+.2f}, {h['interval_high']:+.2f}] -> {h['outcome']}")
    print(f"  TEST accesses recorded in the ledger: {access_count(LEDGER, 'TEST')}")
    print(f"wrote {out.relative_to(HARNESS)}")
    if args.mode == "confirmatory":
        print(f"\nPRIMARY RESULT (H1): {h1['outcome']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
