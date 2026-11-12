# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LogNovations Holdings LLC
"""Properties of the paired bootstrap and of the checks around it that the
pre-registration states and that a regression could break without notice.

Each test names the change it guards against. They were added after the
independent audit showed, by mutation, that the suite did not notice these
five changes (REPORT.md, Section 8).
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from selfref import controls
from selfref.confirm import analyse
from selfref.experiment import Sample, _resample_groups, bootstrap_gain
from selfref.selcert_pin import load_selcert

SELCERT = load_selcert()


def sample(n_groups: int, seed: int, own_block: bool = False) -> Sample:
    """Two strata, one response of each per group. With ``own_block`` every
    group is a block by itself, so a resample of whole groups within block
    returns the sample unchanged."""
    rng = np.random.default_rng(seed)
    loss = (rng.uniform(size=2 * n_groups) < 0.4).astype(int)
    groups = np.arange(n_groups)
    block = groups.astype(str) if own_block else np.where(groups % 2 == 0, "qa", "summary")
    return Sample(X=np.zeros((2 * n_groups, 1)), loss=loss,
                  stratum=np.array(["g1", "g2"] * n_groups),
                  group=np.repeat(groups.astype(str), 2), block=np.repeat(block, 2))


def arms(part: Sample, seed: int):
    """A noise score and a score that separates errors from non-errors."""
    rng = np.random.default_rng(seed)
    return rng.uniform(size=part.loss.size), 0.5 * part.loss + 0.5 * rng.uniform(size=part.loss.size)


def gain(cal: Sample, test: Sample, n_boot: int = 60) -> dict:
    return bootstrap_gain(SELCERT, arms(cal, 1), arms(test, 2), cal, test, alpha=0.10, delta=0.05,
                          n_boot=n_boot, seed=7)


def test_a_sample_whose_groups_are_their_own_blocks_is_returned_unchanged():
    fixed = sample(50, 3, own_block=True)
    index = _resample_groups(fixed, np.random.default_rng(0))
    assert sorted(index) == list(range(fixed.loss.size))


def test_the_calibration_sample_is_resampled_in_every_draw():
    """Guards against a bootstrap that resamples the test sample only."""
    draws = gain(sample(300, 4), sample(150, 5, own_block=True))["draws"]
    assert len(set(draws)) > 1


def test_the_test_sample_is_resampled_in_every_draw():
    """Guards against a bootstrap that resamples the calibration sample only."""
    draws = gain(sample(300, 4, own_block=True), sample(150, 5))["draws"]
    assert len(set(draws)) > 1


def test_groups_are_resampled_within_their_block():
    """Guards against a bootstrap that is not stratified: every block keeps its number of rows."""
    rng = np.random.default_rng(6)
    groups = np.arange(100)
    part = Sample(X=np.zeros((200, 1)), loss=np.zeros(200, dtype=int), stratum=np.array(["g1", "g2"] * 100),
                  group=np.repeat(groups.astype(str), 2),
                  block=np.repeat(np.where(groups < 30, "qa", "summary"), 2))
    for _ in range(20):
        drawn = part.block[_resample_groups(part, rng)]
        assert (int((drawn == "qa").sum()), int((drawn == "summary").sum())) == (60, 140)


def test_the_interval_is_the_2_5th_to_the_97_5th_percentile_of_the_draws():
    """Guards against any other pair of percentiles."""
    result = gain(sample(300, 4), sample(150, 5), n_boot=200)
    low, high = np.percentile(result["draws"], [2.5, 97.5])
    assert (result["low"], result["high"]) == (pytest.approx(low), pytest.approx(high))
    assert np.percentile(result["draws"], 5.0) != pytest.approx(low)        # the draws are spread out


def test_a_draw_exactly_at_the_floor_counts_as_at_or_below_it():
    """Guards against a strict inequality in the p-value of a gain."""
    data = {"select": sample(300, 1), "cal": sample(400, 2), "test": sample(200, 3)}
    noise = {part: arms(data[part], 10 + k)[0] for k, part in enumerate(data)}
    risk = {arm: noise for arm in ("BASELINE", "BASELINE+SOURCE", "BASELINE+SELF")}   # identical arms
    result = analyse(SELCERT, risk, data, primary_alpha=0.10, alphas=(0.10,), delta=0.05, bar=5.0,
                     floor=0.0, n_boot=50, seed=1)
    assert result["hypotheses"]["H1"]["gain_points"] == 0.0            # every draw is exactly 0.0
    assert result["hypotheses"]["H1"]["p_gain_at_most_floor"] == 1.0


def test_the_synthetic_check_reports_a_calibrator_that_accepts_everything():
    """Guards against a check that cannot fail. A calibrator with no bound accepts
    every output; the true risk among them is 0.175, above alpha, in every draw."""
    reckless = SimpleNamespace(calibrate_bonferroni=lambda ranks, losses, grid, alpha, delta:
                               SimpleNamespace(tau_hat=float(np.max(grid))))
    result = controls.synthetic_guarantee(reckless, n_draws=40, n_per_stratum=200, alpha=0.10,
                                          delta=0.05, seed=3)
    assert result["violations"] == 40 and result["violation_share"] == 1.0
    assert result["holds"] is False
    assert result["max_true_risk_when_accepting"] == pytest.approx(0.175)
