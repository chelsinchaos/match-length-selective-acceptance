# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LogNovations Holdings LLC
"""The controls and robustness analyses, as functions of arrays, on synthetic data."""
from __future__ import annotations

import numpy as np
import pytest

from selfref import controls
from selfref.certify import RankTransform, certify_by_stratum, uniform_grid
from selfref.confirm import analyse
from selfref.experiment import Sample
from selfref.selcert_pin import load_selcert

SELCERT = load_selcert()


def sample(n_groups: int, seed: int, error_rate: float = 0.4) -> Sample:
    """Two strata, one response of each per group; two blocks of groups."""
    rng = np.random.default_rng(seed)
    loss = (rng.uniform(size=2 * n_groups) < error_rate).astype(int)
    return Sample(X=np.zeros((2 * n_groups, 1)), loss=loss,
                  stratum=np.array(["g1", "g2"] * n_groups),
                  group=np.repeat(np.arange(n_groups).astype(str), 2),
                  block=np.repeat(np.where(np.arange(n_groups) % 2 == 0, "qa", "summary"), 2))


def score(part: Sample, quality: float, seed: int) -> np.ndarray:
    """A risk score in [0, 1]: ``quality`` of it is the truth and the rest is noise."""
    noise = np.random.default_rng(seed).uniform(size=part.loss.size)
    return quality * part.loss + (1 - quality) * noise


@pytest.fixture(scope="module")
def setting():
    data = {"select": sample(300, 1), "cal": sample(400, 2), "test": sample(200, 3)}
    risk = {arm: {part: score(data[part], quality, seed=10 * i + j)
                  for j, part in enumerate(("select", "cal", "test"))}
            for i, (arm, quality) in enumerate({"GOOD": 0.6, "WEAK": 0.15}.items())}
    ranks = {}
    for arm, scores in risk.items():
        transform = RankTransform.fit(scores["select"], data["select"].stratum)
        ranks[arm] = (transform.transform(scores["cal"], data["cal"].stratum),
                      transform.transform(scores["test"], data["test"].stratum))
    return data, risk, ranks


# ------------------------------------------------------------------ coverage

def test_coverage_row_is_what_the_confirmatory_analysis_reports(setting):
    data, risk, ranks = setting
    arms = {"BASELINE": risk["WEAK"], "BASELINE+SOURCE": risk["GOOD"], "BASELINE+SELF": risk["WEAK"]}
    reported = analyse(SELCERT, arms, data, primary_alpha=0.10, alphas=(0.10,), delta=0.05,
                       bar=5.0, floor=2.0, n_boot=5, seed=1)["coverage"]["BASELINE+SOURCE"]["0.1"]
    row = controls.coverage_row(SELCERT, ranks["GOOD"][0], data["cal"], ranks["GOOD"][1],
                                data["test"], 0.10, 0.05)
    assert row == reported
    assert row["coverage_points"] > 0


# ---------------------------------------------------------- negative controls

def test_permuting_within_strata_keeps_each_stratums_values_and_moves_them():
    rng = np.random.default_rng(0)
    values, strata = np.arange(40), np.array(["a", "b"] * 20)
    permuted = controls.permute_within(values, strata, rng)
    for name in ("a", "b"):
        assert sorted(permuted[strata == name]) == sorted(values[strata == name])
    assert not np.array_equal(permuted, values)
    assert np.array_equal(values, np.arange(40))                    # the input is not changed


def test_a_good_score_certifies_nothing_once_the_labels_are_permuted(setting):
    data, _, ranks = setting
    result = controls.permuted_label_control(SELCERT, {"GOOD": ranks["GOOD"]}, data["cal"],
                                             data["test"], 0.10, 0.05, n_perm=40, seed=7)
    assert result["true_labels"]["GOOD"]["coverage_points"] > 0
    assert result["permutations"] == 40
    assert result["arms"]["GOOD"]["share_with_any_coverage"] <= 0.05
    assert result["arms"]["GOOD"]["first_permutation"]["coverage_points"] == 0.0
    again = controls.permuted_label_control(SELCERT, {"GOOD": ranks["GOOD"]}, data["cal"],
                                            data["test"], 0.10, 0.05, n_perm=40, seed=7)
    assert again == result


def test_a_derangement_sends_every_key_to_another_key_of_its_block():
    keys = [f"s{i}" for i in range(30)]
    blocks = ["qa"] * 12 + ["summary"] * 11 + ["data"] * 7
    mapping = controls.derangement(keys, blocks, seed=3)
    block_of = dict(zip(keys, blocks))
    assert set(mapping) == set(keys)
    assert all(mapping[k] != k and block_of[mapping[k]] == block_of[k] for k in keys)
    assert sorted(mapping.values()) == sorted(keys)                 # a bijection
    assert mapping == controls.derangement(keys[::-1], blocks[::-1], seed=3)   # order-independent
    assert mapping != controls.derangement(keys, blocks, seed=4)


def test_a_block_with_one_key_cannot_be_deranged():
    with pytest.raises(ValueError):
        controls.derangement(["a", "b", "c"], ["x", "x", "y"], seed=0)


# -------------------------------------------------------------- ranking metrics

def test_auroc_and_average_precision_take_an_error_as_the_positive_class():
    loss, risk = np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9])
    assert controls.auroc(loss, risk) == 1.0 and controls.average_precision(loss, risk) == 1.0
    assert controls.auroc(loss, -risk) == 0.0
    assert controls.auroc(np.zeros(4), risk) is None                # one class only
    assert controls.average_precision(np.zeros(4), risk) is None


def test_auroc_by_bin_uses_the_given_edges():
    values = np.array([1, 2, 3, 4, 11, 12, 13, 14, 21, 22])
    loss = np.array([0, 1, 0, 1, 0, 0, 1, 1, 0, 0])
    risk = np.array([.1, .9, .2, .8, .1, .2, .8, .9, .5, .5])
    rows = controls.auroc_by_bin(loss, risk, values, edges=[10, 20])
    assert [row["items"] for row in rows] == [4, 4, 2]
    assert [row["errors"] for row in rows] == [2, 2, 0]
    assert rows[0]["auroc"] == 1.0 and rows[1]["auroc"] == 1.0 and rows[2]["auroc"] is None


def test_the_risk_coverage_curve_of_a_perfect_score_is_zero_until_the_errors_start():
    loss = np.array([0] * 60 + [1] * 40)
    curve = controls.risk_coverage_curve(loss, loss + np.linspace(0, 0.1, 100), points=10)
    assert [row["coverage"] for row in curve] == pytest.approx([0.1 * k for k in range(1, 11)])
    assert [row["risk"] for row in curve[:6]] == [0.0] * 6
    assert curve[-1]["risk"] == pytest.approx(0.4)
    assert curve[7]["risk"] == pytest.approx(20 / 80)


def test_the_paired_difference_is_zero_for_identical_scores_and_positive_for_a_better_one(setting):
    data, risk, _ = setting
    test = data["test"]
    same = controls.paired_metric_difference(test, risk["WEAK"]["test"], risk["WEAK"]["test"],
                                             n_boot=50, seed=5)
    assert same["auroc"]["difference"] == 0.0
    assert (same["auroc"]["low"], same["auroc"]["high"]) == (0.0, 0.0)
    better = controls.paired_metric_difference(test, risk["WEAK"]["test"], risk["GOOD"]["test"],
                                               n_boot=50, seed=5)
    assert better["auroc"]["difference"] == pytest.approx(
        controls.auroc(test.loss, risk["GOOD"]["test"]) - controls.auroc(test.loss, risk["WEAK"]["test"]))
    assert better["auroc"]["low"] > 0 and better["auprc"]["low"] > 0
    assert better == controls.paired_metric_difference(test, risk["WEAK"]["test"],
                                                       risk["GOOD"]["test"], n_boot=50, seed=5)


# ------------------------------------------------------------ guarantee checks

def test_the_synthetic_check_counts_draws_whose_true_risk_exceeds_alpha():
    result = controls.synthetic_guarantee(SELCERT, n_draws=300, n_per_stratum=504, alpha=0.10,
                                          delta=0.05, seed=11)
    assert result["draws"] == 300
    assert result["violation_share"] == result["violations"] / 300
    assert result["violation_share"] <= 0.05 and result["holds"] is True
    assert result["draws_accepting_anything"] > 0                   # the check is not vacuous
    assert result == controls.synthetic_guarantee(SELCERT, n_draws=300, n_per_stratum=504,
                                                  alpha=0.10, delta=0.05, seed=11)


def test_the_lower_bound_on_a_risk_has_its_closed_form_at_the_extremes():
    assert controls.lower_risk_bound(SELCERT, errors=0, trials=50, level=0.99) == 0.0
    # every trial an error: the bound p solves p ** n = 1 - level
    assert controls.lower_risk_bound(SELCERT, 50, 50, 0.99) == pytest.approx(0.01 ** (1 / 50))
    middle = controls.lower_risk_bound(SELCERT, 10, 50, 0.99)
    assert 0.0 < middle < 10 / 50


def test_a_held_out_part_is_flagged_only_when_a_stratum_is_clearly_above_alpha():
    strata = np.array(["a"] * 100 + ["b"] * 100)
    accepted = np.array([True] * 100 + [False] * 100)
    clean, bad = np.zeros(200, dtype=int), np.array([1] * 60 + [0] * 140)
    assert controls.flagged(SELCERT, accepted, clean, strata, alpha=0.10, level=0.05) is False
    assert controls.flagged(SELCERT, accepted, bad, strata, alpha=0.10, level=0.05) is True
    assert controls.flagged(SELCERT, np.zeros(200, dtype=bool), bad, strata, 0.10, 0.05) is False
    # 13 errors in 100 is above alpha but not clearly: the lower bound stays below 0.10
    mild = np.array([1] * 13 + [0] * 187)
    assert controls.flagged(SELCERT, accepted, mild, strata, alpha=0.10, level=0.05) is False


def test_resplitting_exchangeable_data_rarely_flags(setting):
    data, _, ranks = setting
    pool = Sample(*(np.concatenate([getattr(data[p], f) for p in ("cal", "test")])
                    for f in ("X", "loss", "stratum", "group", "block")))
    pool.group = np.concatenate([np.char.add("c", data["cal"].group), np.char.add("t", data["test"].group)])
    pooled_ranks = np.concatenate(ranks["GOOD"])
    result = controls.resplit_guarantee(SELCERT, pooled_ranks, pool, {"qa": 200, "summary": 200},
                                        alpha=0.10, delta=0.05, n_resplits=60, seed=2)
    assert result["resplits"] == 60 and result["calibration_groups"] == 400
    assert result["held_out_groups"] == 200
    assert result["flagged_share"] <= result["allowed_share"] == pytest.approx(0.10)
    assert result["passes"] is True
    assert result["resplits_accepting_anything"] > 0


# ----------------------------------------------------- cheap alternative, stress

def test_more_calibration_data_does_not_lower_average_coverage(setting):
    data, _, ranks = setting
    result = controls.extra_calibration(SELCERT, ranks["GOOD"][0], data["cal"], ranks["GOOD"][1],
                                        data["test"], 0.10, 0.05, multiples=(1.0, 4.0),
                                        n_draws=40, seed=9)
    assert result["1.0"]["calibration_groups"] == 400 and result["4.0"]["calibration_groups"] == 1600
    assert result["4.0"]["mean_coverage_points"] >= result["1.0"]["mean_coverage_points"]
    assert result["1.0"]["draws"] == 40


def test_thresholds_for_a_held_out_block_ignore_that_blocks_calibration_rows(setting):
    data, _, ranks = setting
    cal, test = data["cal"], data["test"]
    rows = controls.cross_block(SELCERT, ranks["GOOD"][0], cal, ranks["GOOD"][1], test, 0.10, 0.05)
    assert sorted(rows) == ["qa", "summary"]
    others = cal.block != "qa"
    expected = certify_by_stratum(SELCERT, ranks["GOOD"][0][others], cal.loss[others],
                                  cal.stratum[others], uniform_grid(), 0.10, 0.05).thresholds
    assert rows["qa"]["thresholds"] == expected
    assert rows["qa"]["items"] == int((test.block == "qa").sum())
    # corrupting the held-out block's own calibration labels changes nothing for it
    spoiled = Sample(cal.X, np.where(cal.block == "qa", 1, cal.loss), cal.stratum, cal.group, cal.block)
    again = controls.cross_block(SELCERT, ranks["GOOD"][0], spoiled, ranks["GOOD"][1], test, 0.10, 0.05)
    assert again["qa"] == rows["qa"]


def test_breakdown_gives_coverage_and_risk_for_each_label():
    mask = np.array([True, True, False, False, True, False])
    loss = np.array([0, 1, 1, 0, 0, 0])
    rows = controls.breakdown(mask, loss, np.array(["a", "a", "a", "b", "b", "b"]))
    assert rows["a"] == {"items": 3, "accepted": 2, "coverage_points": pytest.approx(200 / 3),
                         "realised_risk": 0.5, "error_rate_of_all_items": pytest.approx(2 / 3)}
    assert rows["b"]["realised_risk"] == 0.0 and rows["b"]["accepted"] == 1
    none = controls.breakdown(np.zeros(2, dtype=bool), np.array([0, 1]), np.array(["a", "a"]))
    assert none["a"]["realised_risk"] is None and none["a"]["coverage_points"] == 0.0


def test_acceptance_overlap_counts_what_each_rule_accepts_and_what_both_accept():
    masks = {"a": np.array([True, True, False, False]), "b": np.array([True, False, True, False]),
             "c": np.zeros(4, dtype=bool)}
    assert controls.acceptance_overlap(masks) == [
        {"first": "a", "second": "b", "accepted_first": 2, "accepted_second": 2, "accepted_both": 1},
        {"first": "a", "second": "c", "accepted_first": 2, "accepted_second": 0, "accepted_both": 0},
        {"first": "b", "second": "c", "accepted_first": 2, "accepted_second": 0, "accepted_both": 0}]


def test_the_decision_rule_on_pairs_of_refits_reports_shares_of_each_outcome():
    """Two refits of one configuration have no true gain: the rule's outcomes on such
    pairs say what it returns when there is nothing to find."""
    from selfref.combiner import LOGISTIC_SPACE
    rng = np.random.default_rng(5)

    def part(n_groups, seed):
        s = sample(n_groups, seed)
        signal = s.loss + rng.normal(scale=1.0, size=s.loss.size)       # one informative column
        return Sample(X=signal[:, None], loss=s.loss, stratum=s.stratum, group=s.group, block=s.block)

    train, select, pool = part(150, 1), part(80, 2), part(300, 3)
    kwargs = dict(cal_counts={"qa": 60, "summary": 60}, test_counts={"qa": 30, "summary": 30},
                  alpha=0.30, delta=0.05, bar=50.0, floor=50.0, n_rep=4, n_boot=20, seed=9)
    result = controls.aa_decisions(SELCERT, LOGISTIC_SPACE[0], train, select, pool, **kwargs)
    assert result["replications"] == 4 and result["bootstrap_draws"] == 20
    assert result["p_pass"] + result["p_fail"] + result["p_inconclusive"] == pytest.approx(1.0)
    assert result["p_pass"] == 0.0                         # no pair of refits gains 50 points
    assert len(result["gains"]) == 4 and len(result["interval_high"]) == 4
    assert all(low <= point <= high for low, point, high
               in zip(result["interval_low"], result["gains"], result["interval_high"]))
    assert result == controls.aa_decisions(SELCERT, LOGISTIC_SPACE[0], train, select, pool, **kwargs)
