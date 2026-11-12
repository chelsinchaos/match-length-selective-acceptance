# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LogNovations Holdings LLC
"""The power simulation, on a synthetic population."""
from __future__ import annotations

import numpy as np
import pytest

from selfref.experiment import Sample
from selfref.power import draw_groups, mean_gain, simulate_power, strength_for_gain, tilt
from selfref.selcert_pin import load_selcert


@pytest.fixture(scope="module")
def population():
    """600 groups, two generators each, in two task blocks. Ranks carry a weak signal."""
    rng = np.random.default_rng(42)
    n = 600
    loss = (rng.uniform(size=2 * n) < 0.35).astype(int)
    ranks = np.clip(0.5 + 0.12 * (loss - 0.35) + rng.normal(scale=0.28, size=2 * n), 0, 1)
    pool = Sample(X=np.zeros((2 * n, 1)), loss=loss,
                  stratum=np.array(["g1", "g2"] * n),
                  group=np.repeat(np.arange(n).astype(str), 2),
                  block=np.repeat(np.where(np.arange(n) % 2 == 0, "qa", "summary"), 2))
    return pool, ranks


COUNTS_CAL = {"qa": 150, "summary": 150}
COUNTS_TEST = {"qa": 75, "summary": 75}


def test_tilt_moves_only_correct_outputs_and_only_toward_zero():
    ranks, loss = np.array([0.8, 0.8, 0.2, 0.0]), np.array([0, 1, 0, 1])
    assert tilt(ranks, loss, 0.5).tolist() == [0.4, 0.8, 0.1, 0.0]
    assert tilt(ranks, loss, 0.0).tolist() == ranks.tolist()
    assert tilt(ranks, loss, 1.0).tolist() == [0.0, 0.8, 0.0, 0.0]


def test_drawn_groups_are_whole_come_from_the_right_block_and_get_fresh_labels(population):
    pool, _ = population
    rows, labels = draw_groups(pool, {"qa": 40, "summary": 10}, np.random.default_rng(0))
    assert rows.size == 100 and len(set(labels)) == 50          # two rows per drawn group
    assert (pool.block[rows] == "qa").sum() == 80
    for label in set(labels):
        assert len(set(pool.group[rows][labels == label])) == 1  # a drawn group is one source


def test_zero_strength_gains_nothing_and_gain_grows_with_strength(population):
    pool, ranks = population
    selcert = load_selcert()
    args = (selcert, pool, ranks, COUNTS_CAL, COUNTS_TEST)
    kwargs = dict(alpha=0.10, delta=0.05, n_rep=12, seed=1)
    weak, strong = mean_gain(*args, 0.5, **kwargs), mean_gain(*args, 1.0, **kwargs)
    assert mean_gain(*args, 0.0, **kwargs) == 0.0
    assert 0.0 <= weak <= strong and strong > 20


def test_strength_search_hits_a_reachable_target_and_reports_an_unreachable_one(population):
    pool, ranks = population
    selcert = load_selcert()
    args = (selcert, pool, ranks, COUNTS_CAL, COUNTS_TEST)
    found = strength_for_gain(*args, target=10.0, alpha=0.10, delta=0.05, n_rep=12, seed=1,
                              tolerance=1.0)
    assert found["reachable"] and 0.0 < found["strength"] < 1.0
    # the gain is a step function of the strength, so the search returns a bracket
    (low_strength, low_gain), (high_strength, high_gain) = found["below"], found["above"]
    assert low_gain <= 10.0 <= high_gain and low_strength <= high_strength
    assert found["gain"] in (low_gain, high_gain)
    assert abs(found["gain"] - 10.0) == min(abs(low_gain - 10.0), abs(high_gain - 10.0))
    assert high_strength - low_strength < 1e-3 or abs(found["gain"] - 10.0) <= 1.0
    too_much = strength_for_gain(*args, target=500.0, alpha=0.10, delta=0.05, n_rep=4, seed=1)
    assert not too_much["reachable"] and too_much["strength"] == 1.0


def test_power_is_near_zero_without_an_effect_and_high_with_a_large_one(population):
    pool, ranks = population
    selcert = load_selcert()
    args = (selcert, pool, ranks, COUNTS_CAL, COUNTS_TEST)
    kwargs = dict(alpha=0.10, delta=0.05, bar=5.0, floor=2.0, n_rep=10, n_boot=60, seed=2)
    none = simulate_power(*args, strength=0.0, **kwargs)
    large = simulate_power(*args, strength=1.0, **kwargs)
    assert none["p_pass"] == 0.0 and none["p_fail"] == 1.0 and none["mean_gain"] == 0.0
    assert large["p_pass"] >= 0.9 and large["p_interval_above_floor"] >= 0.9
    for result in (none, large):
        assert result["p_pass"] + result["p_fail"] + result["p_inconclusive"] == pytest.approx(1.0)
