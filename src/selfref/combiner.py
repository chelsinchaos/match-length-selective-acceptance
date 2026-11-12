# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LogNovations Holdings LLC
"""The combiner: one model class search, the same for every arm.

A combiner turns a set of scores into one risk score, the estimated
probability that an output is an error. Every arm of the study (BASELINE, and
BASELINE plus each set of tested features) is fitted with exactly this search
space and this selection rule, so that no arm gets more tuning than another.

Search space, 24 configurations:

* L2-regularised logistic regression on standardised columns, with
  C in {0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30};
* gradient-boosted trees (scikit-learn's histogram implementation), with
  max_depth in {2, 3}, learning_rate in {0.05, 0.1}, max_iter in {100, 300},
  min_samples_leaf in {20, 50}.

Fit on TRAIN. Select the configuration with the smallest log loss on
DEV_SELECT. Freeze.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Sequence

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

__all__ = ["Config", "SEARCH_SPACE", "LOGISTIC_SPACE", "Selection", "build_model",
           "fit_and_select", "risk_scores"]


@dataclass(frozen=True)
class Config:
    kind: str                       # "logistic" or "boosted_trees"
    params: tuple                   # sorted (name, value) pairs

    def as_dict(self) -> dict:
        return {"kind": self.kind, **dict(self.params)}


def _config(kind: str, **params) -> Config:
    return Config(kind, tuple(sorted(params.items())))


SEARCH_SPACE: tuple[Config, ...] = tuple(
    [_config("logistic", C=c) for c in (0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0)] +
    [_config("boosted_trees", max_depth=d, learning_rate=lr, max_iter=it, min_samples_leaf=leaf)
     for d, lr, it, leaf in product((2, 3), (0.05, 0.1), (100, 300), (20, 50))])

#: The logistic part of the search space, for the controls that the
#: pre-registration specifies as "a logistic model".
LOGISTIC_SPACE: tuple[Config, ...] = tuple(c for c in SEARCH_SPACE if c.kind == "logistic")


@dataclass
class Selection:
    model: object
    config: Config
    table: list                     # every configuration with its log loss, best first


def build_model(config: Config, seed: int):
    """An unfitted model for ``config``. Fitting it is deterministic for a seed."""
    params = dict(config.params)
    if config.kind == "logistic":
        return make_pipeline(StandardScaler(),
                             LogisticRegression(C=params["C"], max_iter=5000, random_state=seed))
    if config.kind == "boosted_trees":
        return HistGradientBoostingClassifier(early_stopping=False, random_state=seed, **params)
    raise ValueError(f"unknown model kind {config.kind!r}")


def risk_scores(model, X) -> np.ndarray:
    """Estimated probability of an error for each row. Lower means safer to accept."""
    return model.predict_proba(np.asarray(X, dtype=float))[:, 1]


def fit_and_select(X_train, y_train, X_select, y_select, seed: int,
                   space: Sequence[Config] = SEARCH_SPACE) -> Selection:
    """Fit every configuration on the training rows and keep the one with the
    smallest log loss on the selection rows. Ties go to the earlier configuration.
    Every arm of the study uses the whole search space, which is the default."""
    fitted, table = [], []
    for index, config in enumerate(space):
        model = build_model(config, seed).fit(np.asarray(X_train, dtype=float), np.asarray(y_train))
        loss = float(log_loss(y_select, risk_scores(model, X_select), labels=[0, 1]))
        fitted.append(model)
        table.append({"index": index, "config": config, "log_loss": loss})
    table.sort(key=lambda row: (row["log_loss"], row["index"]))
    best = table[0]
    return Selection(fitted[best["index"]], best["config"], table)
