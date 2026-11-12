# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LogNovations Holdings LLC
"""Rank transform, fixed grid, stratified calibration and certified coverage."""
from __future__ import annotations

import numpy as np
import pytest

from selfref.certify import (GRID_POINTS, RankTransform, accepted_mask, certified_coverage,
                             selective_risk, certify_by_stratum, uniform_grid)
from selfref.selcert_pin import load_selcert


# ------------------------------------------------------------------------ grid

def test_grid_is_twenty_uniform_points_ending_at_one():
    grid = uniform_grid()
    assert GRID_POINTS == 20
    assert grid.tolist() == pytest.approx([k / 20 for k in range(1, 21)])
    assert grid[-1] == 1.0


def test_grid_does_not_depend_on_any_data():
    assert uniform_grid(4).tolist() == [0.25, 0.5, 0.75, 1.0]


# -------------------------------------------------------------- rank transform

def test_rank_is_the_share_of_fitted_scores_at_or_below_within_the_stratum():
    transform = RankTransform.fit(np.array([1.0, 2.0, 3.0, 4.0, 10.0, 20.0]),
                                  np.array(["a", "a", "a", "a", "b", "b"]))
    ranks = transform.transform(np.array([0.5, 1.0, 2.5, 4.0, 99.0, 10.0, 15.0, 5.0]),
                                np.array(["a", "a", "a", "a", "a", "b", "b", "b"]))
    assert ranks.tolist() == [0.0, 0.25, 0.5, 1.0, 1.0, 0.5, 0.5, 0.0]


def test_rank_transform_is_monotone_within_a_stratum():
    rng = np.random.default_rng(0)
    transform = RankTransform.fit(rng.normal(size=200), np.array(["s"] * 200))
    scores = np.sort(rng.normal(size=50))
    ranks = transform.transform(scores, np.array(["s"] * 50))
    assert np.all(np.diff(ranks) >= 0)
    assert ranks.min() >= 0.0 and ranks.max() <= 1.0


def test_an_unseen_stratum_is_an_error():
    transform = RankTransform.fit(np.array([1.0]), np.array(["a"]))
    with pytest.raises(KeyError):
        transform.transform(np.array([1.0]), np.array(["z"]))


# ------------------------------------------------------- stratified calibration

def two_strata():
    # stratum "good": 200 items at rank 0.1 with no error, 200 at rank 0.9 that are all errors
    # stratum "bad":  every item is an error
    ranks = np.array([0.1] * 200 + [0.9] * 200 + [0.1] * 50 + [0.9] * 50)
    losses = np.array([0] * 200 + [1] * 200 + [1] * 100)
    strata = np.array(["good"] * 400 + ["bad"] * 100)
    return ranks, losses, strata


def test_each_stratum_gets_its_own_threshold_from_selcert():
    ranks, losses, strata = two_strata()
    result = certify_by_stratum(load_selcert(), ranks, losses, strata,
                                  grid=np.array([0.5, 1.0]), alpha=0.05, delta=0.05)
    assert result.thresholds == {"bad": None, "good": 0.5}


def test_delta_is_split_evenly_across_strata():
    ranks, losses, strata = two_strata()
    result = certify_by_stratum(load_selcert(), ranks, losses, strata,
                                  grid=np.array([0.5, 1.0]), alpha=0.05, delta=0.05)
    # zero errors in 200 trials, at level 1 - (delta / strata) / grid points
    expected = 1 - (0.05 / 2 / 2) ** (1 / 200)
    assert result.details["good"].upper_bounds[0] == pytest.approx(expected)
    assert result.details["good"].delta == pytest.approx(0.025)


def test_accepted_items_are_those_at_or_below_their_stratum_threshold():
    ranks = np.array([0.1, 0.5, 0.6, 0.1, 0.0])
    strata = np.array(["good", "good", "good", "bad", "bad"])
    mask = accepted_mask(ranks, strata, {"good": 0.5, "bad": None})
    assert mask.tolist() == [True, True, False, False, False]


def test_coverage_is_the_accepted_share_of_all_items_and_risk_is_the_error_share_among_accepted():
    mask = np.array([True, True, False, False, False])
    losses = np.array([0, 1, 1, 0, 1])
    assert certified_coverage(mask) == pytest.approx(0.4)
    assert selective_risk(mask, losses) == pytest.approx(0.5)
    assert certified_coverage(np.array([False, False])) == 0.0
    assert selective_risk(np.array([False, False]), np.array([1, 0])) is None


def test_a_stratum_missing_from_the_thresholds_is_an_error():
    with pytest.raises(KeyError):
        accepted_mask(np.array([0.1]), np.array(["new"]), {"good": 0.5})


# ------------------------------------------------------------- the guarantee

def test_certified_risk_is_controlled_on_synthetic_data_with_known_risk():
    """Two strata with known risk curves. Over repeated calibration draws, the
    true risk among accepted items exceeds alpha in at most a delta share of
    draws. The true risk is computed in closed form, not estimated."""
    selcert = load_selcert()
    rng = np.random.default_rng(20261111)
    grid, alpha, delta = uniform_grid(), 0.10, 0.05

    def risk_at(rank, stratum):                       # P(error | rank), increasing in rank
        return 0.6 * rank ** 2 if stratum == "x" else 0.2 + 0.7 * rank

    def true_selective_risk(tau, stratum):            # mean of risk_at over rank in [0, tau]
        return 0.2 * tau ** 2 if stratum == "x" else 0.2 + 0.35 * tau

    violations, replications = 0, 300
    for _ in range(replications):
        strata = np.array(["x"] * 600 + ["y"] * 600)
        ranks = rng.uniform(size=1200)
        probabilities = np.array([risk_at(r, s) for r, s in zip(ranks, strata)])
        losses = (rng.uniform(size=1200) < probabilities).astype(int)
        result = certify_by_stratum(selcert, ranks, losses, strata, grid, alpha, delta)
        # the pooled risk among accepted items is a coverage-weighted mean of the strata's risks
        weights = {s: (t or 0.0) for s, t in result.thresholds.items()}
        if sum(weights.values()) == 0:
            continue                                  # defer everything: counts as success
        pooled = sum(weights[s] * true_selective_risk(weights[s], s) for s in weights
                     if weights[s] > 0) / sum(weights.values())
        violations += pooled > alpha
    assert violations / replications <= delta
