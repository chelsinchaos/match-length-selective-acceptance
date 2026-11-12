# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LogNovations Holdings LLC
"""Ranking analysis of the RESAMPLE features on WikiBio GPT-3 (pre-registration, Section 15).

    python scripts/wikibio_ranking.py rehearsal
    python scripts/wikibio_ranking.py confirmatory --i-have-prereg-approval --purpose "WikiBio ranking analysis"

H2 cannot be certified on this dataset (Section 4.3) and is recorded as not
testable. This is the secondary metric the pre-registration reports in its
place: AUROC and average precision, on the test sentences, of a baseline
combiner and of that combiner with the 16 RESAMPLE profile columns, with the
difference in AUROC and an interval from a bootstrap of whole passages. It
does not change the status of H2.

    baseline            the lexical baselines computed against the samples,
                        sentence position, gzip and zstd NCD, and SelfCheck-NLI
    + RESAMPLE          the same with the 16 RESAMPLE profile columns

Both are fitted on TRAIN and selected on DEV_SELECT with the search and seed of
Section 6. The rehearsal tests on DEV_TEST with open labels; the confirmatory
mode tests on TEST, whose labels it reads through the guard.

One exploratory variant is reported beside it: the same pair with the number of
sentences in the passage added to the baseline. The pre-registration's list
does not name that column; the variant shows the conclusion does not rest on
leaving it out.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

HARNESS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HARNESS / "src"))

from selfref import controls  # noqa: E402
from selfref.combiner import fit_and_select, risk_scores  # noqa: E402
from selfref.datasets.build import sealed_wikibio_labels  # noqa: E402
from selfref.experiment import Sample  # noqa: E402
from selfref.guard import access_count, authorise  # noqa: E402
from selfref.selfcheck import mean_contradiction  # noqa: E402
from selfref.tables import LOG_COLUMNS  # noqa: E402

DERIVED = HARNESS / "data" / "derived" / "wikibio"
RAW = HARNESS / "data" / "raw" / "wikibio" / "evaluation.parquet"
LEDGER = HARNESS / "RUNLEDGER.md"
FIT_SEED = 20261111
BOOTSTRAP_SEED = 20261132
SELFCHECK = "sc.nli"


def git(*args) -> str:
    return subprocess.run(["git", "-C", str(HARNESS), *args], capture_output=True,
                          text=True).stdout.strip()


def selfcheck_scores() -> dict:
    return {record["id"]: mean_contradiction(record["contradiction"])
            for record in map(json.loads, open(DERIVED / "selfcheck_nli.jsonl", encoding="utf-8"))}


def score_table(split: str, selfcheck: dict):
    """Identifiers, column names and the label-free score matrix of one split."""
    stored = np.load(DERIVED / f"features_{split}.npz")
    ids, names = [str(i) for i in stored["ids"]], [str(n) for n in stored["names"]]
    X = np.array(stored["X"], dtype=float)
    for column in LOG_COLUMNS:
        if column in names:
            X[:, names.index(column)] = np.log1p(X[:, names.index(column)])
    missing = [i for i in ids if i not in selfcheck]
    if missing:
        raise SystemExit(f"{len(missing)} {split} sentences have no SelfCheck-NLI score: "
                         f"run scripts/score_selfcheck.py --splits {split}")
    return ids, names + [SELFCHECK], np.column_stack([X, [selfcheck[i] for i in ids]])


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("mode", choices=["rehearsal", "confirmatory"])
    parser.add_argument("--i-have-prereg-approval", action="store_true", dest="flag")
    parser.add_argument("--purpose", default="")
    parser.add_argument("--boots", type=int, default=10_000)
    args = parser.parse_args(argv)
    test_split = "TEST" if args.mode == "confirmatory" else "DEV_TEST"

    # ---- label-free: score tables, and the models fitted on the open splits ---------------
    selfcheck = selfcheck_scores()
    open_labels = json.loads((DERIVED / "labels_open.json").read_text())
    scores = {part: score_table(split, selfcheck)
              for part, split in (("TRAIN", "TRAIN"), ("select", "DEV_SELECT"), ("test", test_split))}
    names = scores["TRAIN"][1]
    lexical = [n for n in names if n.startswith(("base.len_", "base.novel_", "base.rougeL_"))]
    baseline = lexical + ["base.position", "resample.ncd_gzip", "resample.ncd_zstd", SELFCHECK]
    resample = [n for n in names if n.startswith("resample.") and ".ncd_" not in n]
    assert len(resample) == 16, resample
    arms = {"baseline": baseline, "baseline + RESAMPLE": baseline + resample,
            "baseline with passage length": baseline + ["base.n_sentences"],
            "baseline with passage length + RESAMPLE": baseline + ["base.n_sentences"] + resample}
    take = lambda part, columns: scores[part][2][:, [names.index(c) for c in columns]]
    loss_of = lambda part: np.array([open_labels[i]["loss"] for i in scores[part][0]])
    fitted = {}
    for arm, columns in arms.items():
        selection = fit_and_select(take("TRAIN", columns), loss_of("TRAIN"),
                                   take("select", columns), loss_of("select"), FIT_SEED)
        fitted[arm] = {"model": selection.model, "selected": selection.config.as_dict(),
                       "log_loss_dev_select": selection.table[0]["log_loss"]}

    # ---- the test labels --------------------------------------------------------------------
    test_ids = scores["test"][0]
    if args.mode == "confirmatory":
        entry = authorise("TEST", args.purpose, flag=args.flag, repo=HARNESS, ledger=LEDGER)
        labels = sealed_wikibio_labels(pq.read_table(RAW).to_pylist(), "TEST", entry)
        entries, out_path = [entry], HARNESS / "controls" / "wikibio_result.json"
    else:
        labels, entries = open_labels, []
        out_path = HARNESS / "controls" / "wikibio_rehearsal.json"
    loss = np.array([labels[i]["loss"] for i in test_ids])
    passages = np.array([i.split(":")[0] for i in test_ids])
    test = Sample(X=np.zeros((loss.size, 1)), loss=loss, stratum=np.full(loss.size, "all"),
                  group=passages, block=np.full(loss.size, "all"))

    risk = {arm: risk_scores(fitted[arm]["model"], take("test", arms[arm])) for arm in arms}
    pairs = {"pre-registered": ("baseline", "baseline + RESAMPLE"),
             "exploratory, with passage length": ("baseline with passage length",
                                                  "baseline with passage length + RESAMPLE")}
    record = {
        "mode": args.mode, "test_split": test_split,
        "provenance": {"git_commit": git("rev-parse", "HEAD"),
                       "git_dirty_before_run": bool(entries == [] and git("status", "--porcelain", "--", ".")),
                       "seeds": {"fit": FIT_SEED, "bootstrap": BOOTSTRAP_SEED},
                       "bootstrap_draws": args.boots, "ledger_entries": entries,
                       "test_accesses_so_far": access_count(LEDGER, "TEST")},
        "h2_status": "NOT TESTABLE under the decision rule; this ranking analysis does not change that",
        "sizes": {"train_sentences": int(loss_of("TRAIN").size),
                  "select_sentences": int(loss_of("select").size),
                  "test_sentences": int(loss.size), "test_passages": int(np.unique(passages).size),
                  "test_error_rate": float(loss.mean())},
        "arms": {arm: {"columns": arms[arm], "selected": fitted[arm]["selected"],
                       "log_loss_dev_select": fitted[arm]["log_loss_dev_select"],
                       "auroc": controls.auroc(loss, risk[arm]),
                       "auprc": controls.average_precision(loss, risk[arm])} for arm in arms},
        "difference": {label: {"first": a, "second": b,
                               **controls.paired_metric_difference(test, risk[a], risk[b],
                                                                   n_boot=args.boots, seed=BOOTSTRAP_SEED)}
                       for label, (a, b) in pairs.items()},
        "single_scores": {name: {"auroc": controls.auroc(loss, scores["test"][2][:, k]),
                                 "auprc": controls.average_precision(loss, scores["test"][2][:, k])}
                          for k, name in enumerate(names)},
    }
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(record, indent=1) + "\n")
    for arm in arms:
        print(f"  {arm:42s} AUROC {record['arms'][arm]['auroc']:.4f}  AUPRC {record['arms'][arm]['auprc']:.4f}")
    for label, d in record["difference"].items():
        print(f"  {label}: AUROC difference {d['auroc']['difference']:+.4f} "
              f"[{d['auroc']['low']:+.4f}, {d['auroc']['high']:+.4f}]")
    print(f"wrote {out_path.relative_to(HARNESS)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
