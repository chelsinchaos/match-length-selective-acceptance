# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LogNovations Holdings LLC
"""Assemble one split's design table from the derived files.

A table joins, by item identifier,

* the label-free score table written by ``scripts/compute_features.py``;
* the groundedness model's per-sentence support, summarised;
* the deployment covariates: generator, task type and temperature;
* and the 0/1 loss, from the open-split label file.

This is the only place where scores and labels meet. It refuses to load a
sealed split unless the caller supplies that split's labels, which only the
guarded confirmatory run can obtain. A missing score is an error and is never
filled in.

Counts (lengths, phrase counts, longest matches) are heavy-tailed, so they are
passed through log(1 + x). The same transform applies to every arm.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np

from selfref.experiment import Sample
from selfref.groundedness import summarise_support
from selfref.splits import SEALED

__all__ = ["LOG_COLUMNS", "SealedSplitError", "Table", "load_ragtruth_split", "categories_of"]

#: Columns passed through log(1 + x).
LOG_COLUMNS = ("base.len_bytes", "base.len_words", "base.len_sentences",
               "self.rlz_phrases", "self.lcs", "source.rlz_phrases", "source.lcs",
               "resample.rlz_phrases", "resample.lcs")


class SealedSplitError(RuntimeError):
    """A sealed split (CAL or TEST) was requested outside the confirmatory run."""


@dataclass
class Table:
    ids: list[str]
    names: list[str]
    X: np.ndarray
    loss: np.ndarray
    stratum: np.ndarray
    group: np.ndarray
    block: np.ndarray

    def sample(self, columns: Sequence[str]) -> Sample:
        """The rows with only the named columns, as the study procedures expect them."""
        position = {name: i for i, name in enumerate(self.names)}
        index = [position[name] for name in columns]         # KeyError for an unknown column
        return Sample(self.X[:, index], self.loss, self.stratum, self.group, self.block)


def categories_of(derived: Path, split: str = "TRAIN") -> dict[str, list[str]]:
    """The generators and task types seen in a split, sorted. Taken from TRAIN and then fixed."""
    items = [json.loads(line) for line in open(Path(derived) / "items.jsonl", encoding="utf-8")]
    items = [item for item in items if item["split"] == split]
    return {"model": sorted({item["model"] for item in items}),
            "task": sorted({item["task_type"] for item in items})}


def load_ragtruth_split(derived: Path, split: str, categories: Mapping[str, Sequence[str]],
                        sealed_labels: Optional[Mapping[str, Mapping]] = None) -> Table:
    """The design table of one split of RAGTruth.

    An open split takes its labels from the open-split label file. A sealed
    split (CAL or TEST) has no labels on disk: the confirmatory run obtains
    them through the guard and passes them as ``sealed_labels``. Without them a
    sealed split cannot be loaded.
    """
    if split in SEALED and sealed_labels is None:
        raise SealedSplitError(f"{split} is sealed until the pre-registration is tagged")
    if split not in SEALED and sealed_labels is not None:
        raise ValueError(f"{split} is an open split; its labels come from the label file")
    derived = Path(derived)
    items = {}
    for line in open(derived / "items.jsonl", encoding="utf-8"):
        item = json.loads(line)
        if item["split"] == split:
            items[item["id"]] = item
    stored = np.load(derived / f"features_{split}.npz")
    ids = [str(i) for i in stored["ids"]]
    names = [str(n) for n in stored["names"]]
    X = np.array(stored["X"], dtype=float)
    if set(ids) != set(items):
        raise ValueError(f"{split}: the score table and the item table hold different items")
    for column in LOG_COLUMNS:
        if column in names:
            X[:, names.index(column)] = np.log1p(X[:, names.index(column)])

    support = {}
    for line in open(derived / "groundedness.jsonl", encoding="utf-8"):
        record = json.loads(line)
        support[record["id"]] = record["support"]
    missing = [i for i in ids if i not in support]
    if missing:
        raise KeyError(f"{split}: {len(missing)} items have no groundedness score, e.g. {missing[:3]}")
    grounded = [summarise_support(support[i]) for i in ids]
    grounded_names = ["gnd.min", "gnd.mean", "gnd.frac_unsupported"]

    covariate_names = (["cov.temperature"] + [f"cov.model={m}" for m in categories["model"]] +
                       [f"cov.task={t}" for t in categories["task"]])
    covariates = []
    for i in ids:
        item = items[i]
        if item["model"] not in categories["model"]:
            raise ValueError(f"unknown model {item['model']!r}")
        if item["task_type"] not in categories["task"]:
            raise ValueError(f"unknown task {item['task_type']!r}")
        covariates.append([float(item["temperature"])] +
                          [float(item["model"] == m) for m in categories["model"]] +
                          [float(item["task_type"] == t) for t in categories["task"]])

    labels = sealed_labels if sealed_labels is not None else \
        json.loads((derived / "labels_open.json").read_text())
    return Table(
        ids=ids,
        names=names + grounded_names + covariate_names,
        X=np.hstack([X, np.array([[g[n] for n in grounded_names] for g in grounded]),
                     np.array(covariates)]),
        loss=np.array([labels[i]["loss"] for i in ids]),
        stratum=np.array([items[i]["model"] for i in ids]),
        group=np.array([items[i]["group"] for i in ids]),
        block=np.array([items[i]["task_type"] for i in ids]))
