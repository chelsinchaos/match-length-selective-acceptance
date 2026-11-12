# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LogNovations Holdings LLC
"""Combiner search, the DEV rehearsal, the A/A procedure and the paired bootstrap.

All on synthetic data with a known structure. No dataset is read here.
"""
from __future__ import annotations

import numpy as np
import pytest

from selfref.combiner import SEARCH_SPACE, build_model, fit_and_select, risk_scores
from selfref.experiment import (Sample, aa_coverage_gap, bootstrap_gain, column_sets,
                                rehearse)
from selfref.selcert_pin import load_selcert

GENERATORS = ["g1", "g2"]


def synthetic(n_groups: int, seed: int, signal: float = 2.5):
    """Each group has one item per generator. Column 0 predicts the loss, column 1
    adds a little, column 2 is noise. Generator g2 errs more often than g1."""
    rng = np.random.default_rng(seed)
    rows = []
    for group in range(n_groups):
        for generator in GENERATORS:
            x = rng.normal(size=3)
            logit = signal * x[0] + 0.8 * x[1] + (1.0 if generator == "g2" else -1.0)
            loss = int(rng.uniform() < 1 / (1 + np.exp(-logit)))
            rows.append((f"{seed}-{group}", generator, x, loss))
    return Sample(
        X=np.array([r[2] for r in rows]),
        loss=np.array([r[3] for r in rows]),
        stratum=np.array([r[1] for r in rows]),
        group=np.array([r[0] for r in rows]),
        block=np.array(["only"] * len(rows)))


@pytest.fixture(scope="module")
def data():
    return {"TRAIN": synthetic(400, 1), "DEV_SELECT": synthetic(250, 2),
            "DEV_CAL": synthetic(500, 3), "DEV_TEST": synthetic(250, 4)}


# -------------------------------------------------------------------- combiner

def test_the_search_space_is_fixed_and_has_both_model_classes():
    kinds = [config.kind for config in SEARCH_SPACE]
    assert len(SEARCH_SPACE) == 24
    assert kinds.count("logistic") == 8 and kinds.count("boosted_trees") == 16
    assert len(set(SEARCH_SPACE)) == 24


def test_selection_minimises_log_loss_on_the_selection_split(data):
    chosen = fit_and_select(data["TRAIN"].X, data["TRAIN"].loss,
                            data["DEV_SELECT"].X, data["DEV_SELECT"].loss, seed=0)
    assert len(chosen.table) == 24
    assert chosen.table[0]["log_loss"] == min(row["log_loss"] for row in chosen.table)
    assert chosen.config == chosen.table[0]["config"]
    scores = risk_scores(chosen.model, data["DEV_SELECT"].X)
    assert scores.shape == (500,) and 0 <= scores.min() and scores.max() <= 1


def test_fitting_is_deterministic_for_a_seed(data):
    for config in (SEARCH_SPACE[0], SEARCH_SPACE[-1]):
        first = build_model(config, seed=7).fit(data["TRAIN"].X, data["TRAIN"].loss)
        second = build_model(config, seed=7).fit(data["TRAIN"].X, data["TRAIN"].loss)
        assert np.array_equal(risk_scores(first, data["DEV_CAL"].X),
                              risk_scores(second, data["DEV_CAL"].X))


def test_risk_score_is_higher_for_riskier_items(data):
    chosen = fit_and_select(data["TRAIN"].X, data["TRAIN"].loss,
                            data["DEV_SELECT"].X, data["DEV_SELECT"].loss, seed=0)
    scores = risk_scores(chosen.model, data["DEV_CAL"].X)
    losses = data["DEV_CAL"].loss
    assert scores[losses == 1].mean() > scores[losses == 0].mean() + 0.2


# ------------------------------------------------------------------ column sets

def test_column_sets_keep_the_tested_features_out_of_the_baseline():
    names = ["base.len_bytes", "base.random", "base.rougeL_f", "gnd.min", "cov.model=a",
             "source.ncd_gzip", "source.ncd_zstd", "source.cov_k1", "source.lcs",
             "self.cov_k1", "self.mlen_entropy"]
    sets = column_sets(names)
    assert sets["BASELINE"] == ["base.len_bytes", "base.rougeL_f", "gnd.min", "cov.model=a",
                                "source.ncd_gzip", "source.ncd_zstd"]
    assert sets["BASELINE+SOURCE"] == sets["BASELINE"] + ["source.cov_k1", "source.lcs"]
    assert sets["BASELINE+SELF"] == sets["BASELINE"] + ["self.cov_k1", "self.mlen_entropy"]
    assert "base.random" not in sets["BASELINE"]            # a control, reported on its own


# ------------------------------------------------------------------- rehearsal

def test_rehearsal_reports_coverage_and_realised_risk_per_alpha(data):
    selcert = load_selcert()
    chosen = fit_and_select(data["TRAIN"].X, data["TRAIN"].loss,
                            data["DEV_SELECT"].X, data["DEV_SELECT"].loss, seed=0)
    risk = {name: risk_scores(chosen.model, sample.X) for name, sample in data.items()}
    out = rehearse(selcert, risk, data, alphas=(0.01, 0.10, 0.30), delta=0.05)
    assert set(out) == {0.01, 0.10, 0.30}
    coverages = [out[a]["coverage_points"] for a in (0.01, 0.10, 0.30)]
    assert coverages == sorted(coverages)                    # a looser alpha never covers less
    assert out[0.30]["coverage_points"] > 10
    assert out[0.30]["realised_risk"] is None or out[0.30]["realised_risk"] <= 0.30 + 0.05
    assert set(out[0.10]["thresholds"]) == set(GENERATORS)


def test_a_random_score_certifies_nothing_at_a_strict_alpha(data):
    selcert = load_selcert()
    rng = np.random.default_rng(5)
    risk = {name: rng.uniform(size=len(sample.loss)) for name, sample in data.items()}
    out = rehearse(selcert, risk, data, alphas=(0.05,), delta=0.05)
    assert out[0.05]["coverage_points"] == 0.0


# ------------------------------------------------------------------- bootstrap

def test_identical_arms_have_zero_gain_and_a_degenerate_interval(data):
    selcert = load_selcert()
    rng = np.random.default_rng(6)
    cal_ranks, test_ranks = rng.uniform(size=1000), rng.uniform(size=500)
    result = bootstrap_gain(selcert, (cal_ranks, cal_ranks), (test_ranks, test_ranks),
                            data["DEV_CAL"], data["DEV_TEST"], alpha=0.3, delta=0.05,
                            n_boot=200, seed=1)
    assert (result["point"], result["low"], result["high"]) == (0.0, 0.0, 0.0)


def test_a_better_arm_shows_a_positive_gain(data):
    selcert = load_selcert()
    rng = np.random.default_rng(8)
    cal, test = data["DEV_CAL"], data["DEV_TEST"]
    noise_cal, noise_test = rng.uniform(size=1000), rng.uniform(size=500)
    good_cal = 0.5 * cal.loss + 0.5 * noise_cal            # separates errors from non-errors
    good_test = 0.5 * test.loss + 0.5 * noise_test
    result = bootstrap_gain(selcert, (noise_cal, good_cal), (noise_test, good_test), cal, test,
                            alpha=0.10, delta=0.05, n_boot=200, seed=1)
    assert result["point"] > 20 and result["low"] > 0
    assert result["low"] <= result["point"] <= result["high"]
    assert len(result["draws"]) == 200


def test_bootstrap_is_reproducible_and_resamples_whole_groups(data):
    selcert = load_selcert()
    rng = np.random.default_rng(9)
    cal, test = data["DEV_CAL"], data["DEV_TEST"]
    arms_cal = (rng.uniform(size=1000), 0.5 * cal.loss + 0.5 * rng.uniform(size=1000))
    arms_test = (rng.uniform(size=500), 0.5 * test.loss + 0.5 * rng.uniform(size=500))
    kwargs = dict(alpha=0.10, delta=0.05, n_boot=50)
    first = bootstrap_gain(selcert, arms_cal, arms_test, cal, test, seed=3, **kwargs)
    again = bootstrap_gain(selcert, arms_cal, arms_test, cal, test, seed=3, **kwargs)
    other = bootstrap_gain(selcert, arms_cal, arms_test, cal, test, seed=4, **kwargs)
    assert first["draws"] == again["draws"]
    assert first["draws"] != other["draws"]
    # every group has exactly two items, so a resample of whole groups has an even size
    assert all(size % 2 == 0 for size in first["resample_sizes"])


# ------------------------------------------------------------------------- A/A

def test_aa_gap_is_reproducible_and_usually_small(data):
    selcert = load_selcert()
    gaps = [aa_coverage_gap(selcert, SEARCH_SPACE[3], data, alpha=0.30, delta=0.05, run=r, seed=11)
            for r in range(6)]
    again = aa_coverage_gap(selcert, SEARCH_SPACE[3], data, alpha=0.30, delta=0.05, run=0, seed=11)
    assert gaps[0] == again
    assert len(set(gaps)) > 1                     # different runs do differ
    assert max(abs(g) for g in gaps) < 25


def test_the_rank_transform_comes_from_dev_select_not_from_the_calibration_split():
    """DEV_SELECT scores are uniform on [0, 1]. The calibration split has 300
    clean items at score 0.2 and 300 errors at score 0.9. Ranked against
    DEV_SELECT those sit near 0.2 and 0.9, so the largest certifiable threshold
    is 0.85. Ranked against the calibration split itself they would sit at 0.5
    and 1.0, and the threshold would be 0.95."""
    selcert = load_selcert()

    def sample(n, loss):
        return Sample(X=np.zeros((n, 1)), loss=np.asarray(loss), stratum=np.array(["g"] * n),
                      group=np.arange(n).astype(str), block=np.array(["b"] * n))

    data = {"DEV_SELECT": sample(1000, [0] * 1000),
            "DEV_CAL": sample(600, [0] * 300 + [1] * 300),
            "DEV_TEST": sample(4, [0, 0, 1, 1])}
    risk = {"DEV_SELECT": np.linspace(0.0005, 0.9995, 1000),
            "DEV_CAL": np.array([0.2] * 300 + [0.9] * 300),
            "DEV_TEST": np.array([0.1, 0.5, 0.88, 0.95])}
    out = rehearse(selcert, risk, data, alphas=(0.05,), delta=0.05)[0.05]
    assert out["thresholds"] == {"g": pytest.approx(0.85)}
    assert out["accepted"] == 2 and out["coverage_points"] == pytest.approx(50.0)
    assert out["realised_risk"] == 0.0
