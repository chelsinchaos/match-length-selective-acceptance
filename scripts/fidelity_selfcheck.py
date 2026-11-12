# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LogNovations Holdings LLC
"""Fidelity check: reproduce a published baseline number.

    python scripts/fidelity_selfcheck.py

Published (Manakul, Liusie and Gales, arXiv:2303.08896, Table 2): SelfCheckGPT
with NLI detects non-factual sentences of the WikiBio GPT-3 dataset with an
AUC-PR of 92.50, against a random baseline of 72.96, over all 1,908 sentences.

This study computes the same score with the same checkpoint and the authors'
reference procedure, on the sentences of its open splits only, because the
labels of its CAL and TEST splits are sealed until the pre-registration is
tagged. The comparison is therefore between the published figure on all
sentences and ours on about two thirds of them.

Criterion, fixed before the number was computed: the AUC-PR for non-factual
sentences on the open splits is within 2.0 points of 92.50. The result is
written to prereg/fidelity.json whether or not it meets the criterion.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import auc, average_precision_score, precision_recall_curve

HARNESS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HARNESS / "src"))

from selfref.selfcheck import CHECKPOINT, REVISION, mean_contradiction  # noqa: E402

PUBLISHED = {"nonfact_auc_pr": 92.50, "nonfact_random": 72.96,
             "factual_auc_pr": 66.08, "factual_random": 27.04, "sentences": 1908}
TOLERANCE_POINTS = 2.0
OPEN_SPLITS = ("TRAIN", "DEV_SELECT", "DEV_CAL", "DEV_TEST")


def pr_auc(positive, score) -> dict:
    precision, recall, _ = precision_recall_curve(positive, score)
    return {"trapezoid": 100 * float(auc(recall, precision)),
            "average_precision": 100 * float(average_precision_score(positive, score))}


def main() -> int:
    derived = HARNESS / "data" / "derived" / "wikibio"
    items = [json.loads(line) for line in open(derived / "items.jsonl", encoding="utf-8")]
    wanted = [item["id"] for item in items if item["split"] in OPEN_SPLITS]
    labels = json.loads((derived / "labels_open.json").read_text())
    scores = {}
    for line in open(derived / "selfcheck_nli.jsonl", encoding="utf-8"):
        record = json.loads(line)
        scores[record["id"]] = mean_contradiction(record["contradiction"])
    missing = [i for i in wanted if i not in scores]
    if missing:
        raise SystemExit(f"{len(missing)} open sentences have no SelfCheck-NLI score yet")

    loss = np.array([labels[i]["loss"] for i in wanted])
    score = np.array([scores[i] for i in wanted])
    nonfact, factual = pr_auc(loss, score), pr_auc(1 - loss, -score)
    difference = nonfact["trapezoid"] - PUBLISHED["nonfact_auc_pr"]
    git = lambda *a: subprocess.run(["git", "-C", str(HARNESS), *a], capture_output=True,
                                    text=True).stdout.strip()
    record = {
        "provenance": {"git_commit": git("rev-parse", "HEAD"),
                       "git_dirty": bool(git("status", "--porcelain", "--", ".")),
                       "checkpoint": CHECKPOINT, "revision": REVISION},
        "published": PUBLISHED,
        "ours": {"sentences": int(loss.size), "splits": list(OPEN_SPLITS),
                 "nonfact_share": 100 * float(loss.mean()),
                 "nonfact_auc_pr": nonfact, "factual_auc_pr": factual},
        "criterion": f"|ours - published| <= {TOLERANCE_POINTS} points on non-factual AUC-PR (trapezoid)",
        "difference_points": difference,
        "within_tolerance": bool(abs(difference) <= TOLERANCE_POINTS),
    }
    (HARNESS / "prereg").mkdir(exist_ok=True)
    (HARNESS / "prereg" / "fidelity.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({k: record[k] for k in ("ours", "difference_points", "within_tolerance")},
                     indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
