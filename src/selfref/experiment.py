# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LogNovations Holdings LLC
"""The procedures of the study, as functions of arrays.

* ``column_sets``       which columns each arm may use
* ``rehearse``          the confirmatory procedure run inside DEV
* ``aa_coverage_gap``   one baseline-versus-baseline run
* ``bootstrap_gain``    the coverage gain of one arm over another, with a
                        paired, clustered, stratified bootstrap interval

Everything here works on a ``Sample``: the design matrix of a split together
with each row's 0/1 loss, its calibration stratum (the generator), its group
(the source, never split across resamples) and its bootstrap block (the task
type, within which groups are resampled).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from selfref.certify import (RankTransform, accepted_mask, certified_coverage,
                             certify_by_stratum, selective_risk, uniform_grid)
from selfref.combiner import Config, build_model, risk_scores

__all__ = ["Sample", "column_sets", "rehearse", "aa_coverage_gap", "bootstrap_gain"]


@dataclass
class Sample:
    X: np.ndarray           # design matrix
    loss: np.ndarray        # 0/1 loss per row
    stratum: np.ndarray     # calibration stratum per row (generator)
    group: np.ndarray       # group key per row (source)
    block: np.ndarray       # bootstrap block per row (task type)

    def rows(self, index) -> "Sample":
        return Sample(self.X[index], self.loss[index], self.stratum[index],
                      self.group[index], self.block[index])


def column_sets(names: Sequence[str]) -> dict[str, list[str]]:
    """The columns of each arm.

    BASELINE: the lexical baselines, the groundedness model, the deployment
    covariates, and gzip and zstd NCD. ``base.random`` is a control reported on
    its own and enters no combiner. The tested features are the match-length
    profile columns of a frame; NCD is not among them.
    """
    def is_profile(name: str, frame: str) -> bool:
        return name.startswith(frame + ".") and ".ncd_" not in name

    baseline = [n for n in names
                if (n.startswith(("base.", "gnd.", "cov.")) and n != "base.random")
                or n in ("source.ncd_gzip", "source.ncd_zstd")]
    return {"BASELINE": baseline,
            "BASELINE+SOURCE": baseline + [n for n in names if is_profile(n, "source")],
            "BASELINE+SELF": baseline + [n for n in names if is_profile(n, "self")]}


def rehearse(selcert, risk: Mapping[str, np.ndarray], data: Mapping[str, Sample],
             alphas: Sequence[float], delta: float) -> dict:
    """The confirmatory procedure, run inside DEV.

    The rank transform is fitted on DEV_SELECT, thresholds are certified on
    DEV_CAL, and coverage is measured on DEV_TEST, each per stratum.
    """
    grid = uniform_grid()
    transform = RankTransform.fit(risk["DEV_SELECT"], data["DEV_SELECT"].stratum)
    cal_ranks = transform.transform(risk["DEV_CAL"], data["DEV_CAL"].stratum)
    test_ranks = transform.transform(risk["DEV_TEST"], data["DEV_TEST"].stratum)
    out = {}
    for alpha in alphas:
        certificates = certify_by_stratum(selcert, cal_ranks, data["DEV_CAL"].loss,
                                          data["DEV_CAL"].stratum, grid, alpha, delta)
        mask = accepted_mask(test_ranks, data["DEV_TEST"].stratum, certificates.thresholds)
        out[alpha] = {
            "coverage_points": 100.0 * certified_coverage(mask),
            "realised_risk": selective_risk(mask, data["DEV_TEST"].loss),
            "accepted": int(mask.sum()),
            "thresholds": dict(certificates.thresholds),
            "coverage_points_by_stratum": {
                str(s): 100.0 * float(mask[data["DEV_TEST"].stratum == s].mean())
                for s in np.unique(data["DEV_TEST"].stratum)},
        }
    return out


def _resample_groups(sample: Sample, rng: np.random.Generator) -> np.ndarray:
    """Row indices of a resample of whole groups, drawn with replacement within each block."""
    chosen = []
    for block in np.unique(sample.block):
        in_block = np.flatnonzero(sample.block == block)
        groups, inverse = np.unique(sample.group[in_block], return_inverse=True)
        members = [in_block[inverse == g] for g in range(groups.size)]
        for g in rng.integers(0, groups.size, size=groups.size):
            chosen.append(members[g])
    return np.concatenate(chosen)


def aa_coverage_gap(selcert, config: Config, data: Mapping[str, Sample], alpha: float,
                    delta: float, run: int, seed: int) -> float:
    """One A/A run: the DEV-rehearsal coverage of one baseline minus another.

    The two differ only in their random seed and in the bootstrap resample of
    TRAIN groups they are fitted on. The configuration is the frozen one.
    """
    coverages = []
    for arm in (0, 1):
        rng = np.random.default_rng([seed, run, arm])
        train = data["TRAIN"].rows(_resample_groups(data["TRAIN"], rng))
        model = build_model(config, seed=int(rng.integers(2 ** 31))).fit(train.X, train.loss)
        risk = {name: risk_scores(model, data[name].X)
                for name in ("DEV_SELECT", "DEV_CAL", "DEV_TEST")}
        coverages.append(rehearse(selcert, risk, data, (alpha,), delta)[alpha]["coverage_points"])
    return coverages[0] - coverages[1]


def _gain(selcert, cal_ranks, test_ranks, cal: Sample, test: Sample, alpha, delta, grid) -> float:
    coverage = []
    for arm in (0, 1):
        certificates = certify_by_stratum(selcert, cal_ranks[arm], cal.loss, cal.stratum,
                                          grid, alpha, delta)
        mask = accepted_mask(test_ranks[arm], test.stratum, certificates.thresholds)
        coverage.append(100.0 * certified_coverage(mask))
    return coverage[1] - coverage[0]


def bootstrap_gain(selcert, cal_ranks, test_ranks, cal: Sample, test: Sample, alpha: float,
                   delta: float, n_boot: int, seed: int) -> dict:
    """Coverage gain of the second arm over the first, in points, with a 95% interval.

    ``cal_ranks`` and ``test_ranks`` are pairs (first arm, second arm) of rank
    arrays on the calibration and test samples. The point estimate certifies
    thresholds on the calibration sample and measures coverage on the test
    sample. Each bootstrap draw resamples whole groups with replacement within
    each block, in both samples, recertifies both arms on the resampled
    calibration sample and measures both on the resampled test sample, so the
    two arms always see the same resample. The interval is the 2.5th to the
    97.5th percentile of the draws.
    """
    grid = uniform_grid()
    cal_ranks = tuple(np.asarray(r, dtype=float) for r in cal_ranks)
    test_ranks = tuple(np.asarray(r, dtype=float) for r in test_ranks)
    point = _gain(selcert, cal_ranks, test_ranks, cal, test, alpha, delta, grid)
    rng = np.random.default_rng(seed)
    draws, sizes = [], []
    for _ in range(n_boot):
        ci, ti = _resample_groups(cal, rng), _resample_groups(test, rng)
        draws.append(_gain(selcert, tuple(r[ci] for r in cal_ranks),
                           tuple(r[ti] for r in test_ranks),
                           cal.rows(ci), test.rows(ti), alpha, delta, grid))
        sizes.append(int(ci.size))
    low, high = np.percentile(draws, [2.5, 97.5])
    return {"point": float(point), "low": float(low), "high": float(high),
            "draws": [float(d) for d in draws], "resample_sizes": sizes}
