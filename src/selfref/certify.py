# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LogNovations Holdings LLC
"""From a frozen score to certified coverage.

The calibrator, ``calibrate_bonferroni`` from the pinned selcert package,
accepts an output when its score is at or below a threshold, and requires the
score, the model and the grid of candidate thresholds to be fixed before the
calibration sample is drawn, and the calibration sample to be i.i.d. from the
deployment distribution. This module supplies the pieces around it.

Rank transform. A frozen risk score is mapped to [0, 1] by the empirical
distribution of that score on DEV_SELECT: the rank of a score is the share of
DEV_SELECT scores at or below it. It is fitted separately for each stratum.

Grid. Twenty uniform thresholds, 0.05, 0.10, ..., 1.00. It depends on no data.

Stratified calibration. The data hold several responses per source, one from
each generator, and their errors share that source, so the pooled sample is
not i.i.d. Within one generator, each source contributes one response, and
the sources are exchangeable, so that sub-sample is. The calibrator is
therefore run once per stratum (generator), each at confidence delta divided by
the number of strata. With probability at least 1 - delta every stratum's
certificate holds at once, and then the risk among all accepted outputs, being
a weighted average of the strata's risks, is at most alpha too.

Certified coverage on a sample is the share of all its items that the
calibrated thresholds accept.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional

import numpy as np

__all__ = ["GRID_POINTS", "uniform_grid", "RankTransform", "StratumCertificates",
           "certify_by_stratum", "accepted_mask", "certified_coverage", "selective_risk"]

GRID_POINTS = 20


def uniform_grid(points: int = GRID_POINTS) -> np.ndarray:
    """The fixed thresholds 1/points, 2/points, ..., 1."""
    return np.arange(1, points + 1) / points


class RankTransform:
    """Per-stratum empirical distribution of a score, fitted once and frozen."""

    def __init__(self, fitted: Mapping[str, np.ndarray]) -> None:
        self._fitted = {name: np.sort(np.asarray(values, dtype=float))
                        for name, values in fitted.items()}

    @classmethod
    def fit(cls, scores, strata) -> "RankTransform":
        scores, strata = np.asarray(scores, dtype=float), np.asarray(strata).astype(str)
        return cls({name: scores[strata == name] for name in np.unique(strata)})

    def transform(self, scores, strata) -> np.ndarray:
        """Share of the stratum's fitted scores at or below each score."""
        scores, strata = np.asarray(scores, dtype=float), np.asarray(strata).astype(str)
        ranks = np.empty(scores.shape, dtype=float)
        for name in np.unique(strata):
            fitted = self._fitted[str(name)]             # KeyError for a stratum never fitted
            chosen = strata == name
            ranks[chosen] = np.searchsorted(fitted, scores[chosen], side="right") / fitted.size
        return ranks


@dataclass
class StratumCertificates:
    thresholds: dict[str, Optional[float]]      # None: defer everything in that stratum
    details: dict                               # stratum -> selcert CalibrationResult
    alpha: float
    delta: float


def certify_by_stratum(selcert, ranks, losses, strata, grid, alpha: float,
                         delta: float) -> StratumCertificates:
    """Run selcert's calibrator once per stratum, at confidence delta / (number of strata)."""
    ranks, losses = np.asarray(ranks, dtype=float), np.asarray(losses, dtype=float)
    strata = np.asarray(strata).astype(str)
    names = [str(name) for name in np.unique(strata)]
    thresholds, details = {}, {}
    for name in names:
        chosen = strata == name
        result = selcert.calibrate_bonferroni(ranks[chosen], losses[chosen], grid,
                                              alpha, delta / len(names))
        thresholds[name], details[name] = result.tau_hat, result
    return StratumCertificates(thresholds, details, alpha, delta)


def accepted_mask(ranks, strata, thresholds: Mapping[str, Optional[float]]) -> np.ndarray:
    """True where an item's rank is at or below its stratum's threshold."""
    ranks, strata = np.asarray(ranks, dtype=float), np.asarray(strata).astype(str)
    mask = np.zeros(ranks.shape, dtype=bool)
    for name in np.unique(strata):
        threshold = thresholds[str(name)]                # KeyError for an uncalibrated stratum
        if threshold is not None:
            chosen = strata == name
            mask[chosen] = ranks[chosen] <= threshold
    return mask


def certified_coverage(mask) -> float:
    """Share of all items that are accepted."""
    mask = np.asarray(mask, dtype=bool)
    return float(mask.mean()) if mask.size else 0.0


def selective_risk(mask, losses) -> Optional[float]:
    """Share of errors among accepted items, or None when nothing is accepted."""
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return None
    return float(np.asarray(losses, dtype=float)[mask].mean())
