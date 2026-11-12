# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LogNovations Holdings LLC
"""The confirmatory analysis, as a function of arrays.

The same function runs the rehearsal inside DEV and the confirmatory run on
the sealed splits; only the data handed to it differ. Given, for each arm, its
frozen risk scores on three samples,

    select   the sample the rank transform is fitted on (DEV_SELECT)
    cal      the calibration sample
    test     the test sample

it produces

* a table of certified coverage and realised risk on the test sample, for
  every arm at every alpha;
* H1, the primary endpoint: the coverage gain of BASELINE+SOURCE over BASELINE
  at the primary alpha, its paired bootstrap interval, and PASS, FAIL or
  INCONCLUSIVE under the decision rule;
* H3, a secondary hypothesis: the same for BASELINE+SELF, with a Holm-adjusted
  p-value. The secondary family is {H2, H3}. H2 cannot be certified on its
  dataset and enters the family with p = 1, so H3's p-value is doubled.

The p-value of a gain is the share of bootstrap draws at or below FLOOR. A
secondary hypothesis is supported when its outcome is PASS and its
Holm-adjusted p-value is below 0.05.
"""
from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from selfref.certify import (RankTransform, accepted_mask, certified_coverage,
                             certify_by_stratum, selective_risk, uniform_grid)
from selfref.decision import decide
from selfref.experiment import Sample, bootstrap_gain

__all__ = ["BASELINE", "TESTED", "holm", "analyse"]

BASELINE = "BASELINE"
#: hypothesis -> the arm whose gain over BASELINE it concerns
TESTED = {"H1": "BASELINE+SOURCE", "H3": "BASELINE+SELF"}
SECONDARY_FAMILY = ["H2", "H3"]


def holm(p_values: Mapping[str, float]) -> dict[str, float]:
    """Holm's step-down adjustment (Holm, 1979). The k-th smallest of m p-values
    is multiplied by m - k + 1, the results are made non-decreasing in that
    order, and each is capped at 1."""
    ordered = sorted(p_values, key=lambda name: p_values[name])
    adjusted, running = {}, 0.0
    for rank, name in enumerate(ordered):
        running = max(running, (len(ordered) - rank) * p_values[name])
        adjusted[name] = min(1.0, running)
    return adjusted


def analyse(selcert, risk: Mapping[str, Mapping[str, np.ndarray]], data: Mapping[str, Sample], *,
            primary_alpha: float, alphas: Sequence[float], delta: float, bar: float,
            floor: float, n_boot: int, seed: int) -> dict:
    """Run the confirmatory analysis. See the module docstring."""
    grid = uniform_grid()
    cal, test = data["cal"], data["test"]
    ranks = {}
    for arm, scores in risk.items():
        transform = RankTransform.fit(scores["select"], data["select"].stratum)
        ranks[arm] = (transform.transform(scores["cal"], cal.stratum),
                      transform.transform(scores["test"], test.stratum))

    coverage = {}
    for arm, (cal_ranks, test_ranks) in ranks.items():
        coverage[arm] = {}
        for alpha in alphas:
            certificates = certify_by_stratum(selcert, cal_ranks, cal.loss, cal.stratum,
                                              grid, alpha, delta)
            mask = accepted_mask(test_ranks, test.stratum, certificates.thresholds)
            coverage[arm][str(alpha)] = {
                "coverage_points": 100.0 * certified_coverage(mask),
                "realised_risk": selective_risk(mask, test.loss),
                "accepted": int(mask.sum()),
                "thresholds": dict(certificates.thresholds)}

    hypotheses = {"H2": {"outcome": "NOT TESTABLE",
                         "reason": "no certificate is attainable on its dataset at any "
                                   "pre-specified alpha", "p_gain_at_most_floor": 1.0}}
    for offset, (name, arm) in enumerate(TESTED.items()):
        result = bootstrap_gain(selcert, (ranks[BASELINE][0], ranks[arm][0]),
                                (ranks[BASELINE][1], ranks[arm][1]), cal, test,
                                primary_alpha, delta, n_boot=n_boot, seed=seed + offset)
        hypotheses[name] = {
            "tested_arm": arm, "alpha": primary_alpha,
            "gain_points": result["point"],
            "interval_low": result["low"], "interval_high": result["high"],
            "bootstrap_draws": n_boot,
            "p_gain_at_most_floor": float(np.mean(np.asarray(result["draws"]) <= floor)),
            "outcome": decide(result["point"], result["low"], result["high"], bar, floor)}

    adjusted = holm({name: hypotheses[name]["p_gain_at_most_floor"] for name in SECONDARY_FAMILY})
    h3 = hypotheses["H3"]
    h3["p_holm"] = adjusted["H3"]
    h3["supported"] = bool(h3["outcome"] == "PASS" and h3["p_holm"] < 0.05)
    return {"primary_alpha": primary_alpha, "delta": delta, "BAR_points": bar,
            "FLOOR_points": floor, "coverage": coverage, "hypotheses": hypotheses,
            "secondary_family": list(SECONDARY_FAMILY),
            "sizes": {part: {"items": int(sample.loss.size),
                             "groups": int(np.unique(sample.group).size)}
                      for part, sample in data.items()}}
