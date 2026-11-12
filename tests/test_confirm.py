# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LogNovations Holdings LLC
"""The confirmatory analysis as a function of arrays, on synthetic data."""
from __future__ import annotations

import numpy as np
import pytest

from selfref.confirm import analyse
from selfref.experiment import Sample
from selfref.selcert_pin import load_selcert

ARMS = ("BASELINE", "BASELINE+SOURCE", "BASELINE+SELF")


def sample(n_groups, seed):
    rng = np.random.default_rng(seed)
    loss = (rng.uniform(size=2 * n_groups) < 0.4).astype(int)
    return Sample(X=np.zeros((2 * n_groups, 1)), loss=loss,
                  stratum=np.array(["g1", "g2"] * n_groups),
                  group=np.repeat(np.arange(n_groups).astype(str), 2),
                  block=np.repeat(np.where(np.arange(n_groups) % 2 == 0, "qa", "summary"), 2))


def scores(part: Sample, quality: float, seed: int):
    """A risk score: ``quality`` of it is the truth and the rest is noise."""
    noise = np.random.default_rng(seed).uniform(size=part.loss.size)
    return quality * part.loss + (1 - quality) * noise


@pytest.fixture(scope="module")
def setting():
    data = {"select": sample(300, 1), "cal": sample(400, 2), "test": sample(200, 3)}
    quality = {"BASELINE": 0.15, "BASELINE+SOURCE": 0.6, "BASELINE+SELF": 0.15}
    risk = {arm: {part: scores(data[part], quality[arm], seed=10 * i + j)
                  for j, part in enumerate(("select", "cal", "test"))}
            for i, arm in enumerate(ARMS)}
    # the SELF arm is the baseline itself: no gain at all
    risk["BASELINE+SELF"] = {part: risk["BASELINE"][part].copy() for part in risk["BASELINE"]}
    return data, risk


def run(setting, **overrides):
    data, risk = setting
    kwargs = dict(primary_alpha=0.10, alphas=(0.01, 0.05, 0.10), delta=0.05, bar=5.0, floor=2.0,
                  n_boot=200, seed=1)
    kwargs.update(overrides)
    return analyse(load_selcert(), risk, data, **kwargs)


def test_a_clearly_better_arm_passes_and_an_identical_arm_fails(setting):
    result = run(setting)
    h1, h3 = result["hypotheses"]["H1"], result["hypotheses"]["H3"]
    assert h1["tested_arm"] == "BASELINE+SOURCE" and h1["outcome"] == "PASS"
    assert h1["gain_points"] > 5 and h1["interval_low"] > 2
    assert (h3["gain_points"], h3["interval_low"], h3["interval_high"]) == (0.0, 0.0, 0.0)
    assert h3["outcome"] == "FAIL"


def test_the_table_has_every_arm_at_every_alpha(setting):
    result = run(setting)
    assert set(result["coverage"]) == set(ARMS)
    for arm in ARMS:
        assert set(result["coverage"][arm]) == {"0.01", "0.05", "0.1"}
        row = result["coverage"][arm]["0.1"]
        assert {"coverage_points", "realised_risk", "accepted", "thresholds"} <= set(row)
    assert result["coverage"]["BASELINE+SOURCE"]["0.1"]["coverage_points"] > \
        result["coverage"]["BASELINE"]["0.1"]["coverage_points"]


def test_the_primary_gain_is_the_difference_of_the_two_coverages_at_the_primary_alpha(setting):
    result = run(setting)
    table = result["coverage"]
    difference = table["BASELINE+SOURCE"]["0.1"]["coverage_points"] - \
        table["BASELINE"]["0.1"]["coverage_points"]
    assert result["hypotheses"]["H1"]["gain_points"] == pytest.approx(difference)


def test_secondary_p_values_are_holm_adjusted_over_two_hypotheses(setting):
    result = run(setting)
    h3 = result["hypotheses"]["H3"]
    assert h3["p_gain_at_most_floor"] == 1.0                 # every draw is exactly zero
    assert h3["p_holm"] == 1.0 and h3["supported"] is False
    assert result["secondary_family"] == ["H2", "H3"]
    assert result["hypotheses"]["H2"]["outcome"] == "NOT TESTABLE"
    # a small raw p-value is doubled, because H2 counts in the family with p = 1
    data, risk = setting
    swapped = dict(risk)
    swapped["BASELINE+SELF"] = risk["BASELINE+SOURCE"]
    better = analyse(load_selcert(), swapped, data, primary_alpha=0.10, alphas=(0.10,),
                     delta=0.05, bar=5.0, floor=2.0, n_boot=200, seed=1)["hypotheses"]["H3"]
    assert better["p_holm"] == pytest.approx(min(1.0, 2 * better["p_gain_at_most_floor"]))
    assert better["supported"] is (better["outcome"] == "PASS" and better["p_holm"] < 0.05)


def test_the_analysis_is_reproducible_for_a_seed(setting):
    assert run(setting, seed=5)["hypotheses"] == run(setting, seed=5)["hypotheses"]


def test_holm_adjustment_textbook_example():
    from selfref.confirm import holm
    # sorted p: 0.01, 0.03, 0.04 -> 3*0.01, 2*0.03, 1*0.04, made monotone: 0.03, 0.06, 0.06
    adjusted = holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert adjusted == pytest.approx({"a": 0.03, "c": 0.06, "b": 0.06})


def test_holm_with_an_untestable_hypothesis_doubles_the_other_p_value():
    from selfref.confirm import holm
    assert holm({"H2": 1.0, "H3": 0.02}) == pytest.approx({"H2": 1.0, "H3": 0.04})
    assert holm({"H2": 1.0, "H3": 0.03}) == pytest.approx({"H2": 1.0, "H3": 0.06})
    assert holm({"H2": 1.0, "H3": 0.8}) == pytest.approx({"H2": 1.0, "H3": 1.0})
