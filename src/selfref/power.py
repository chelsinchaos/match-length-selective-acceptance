# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LogNovations Holdings LLC
"""Power analysis by simulation, at the sizes of the confirmatory run.

The simulation needs data on which the truth is known. It uses the open DEV
splits as a population and builds a second, better arm whose advantage over
the baseline is under control:

* Population. The baseline's ranks and losses on DEV_CAL and DEV_TEST.
* A replication. Draw, with replacement and within task type, as many source
  groups as the real CAL split has, and independently as many as the real
  TEST split has. A group drawn twice counts as two groups.
* The better arm. Every correct output has its rank multiplied by
  (1 - strength); errors keep theirs. strength = 0 is the baseline itself and
  strength = 1 puts every correct output ahead of every error.
* Calibration of the effect. ``strength_for_gain`` finds the strength at which
  the mean coverage gain over replications equals a target, such as BAR.
* The decision. Each replication is put through the paired bootstrap and the
  decision rule, exactly as the confirmatory run will be, with fewer bootstrap
  draws to keep the simulation affordable.

Reported per target gain: the share of replications that PASS, FAIL and are
INCONCLUSIVE, and the share whose whole interval lies above FLOOR, which is the
conventional meaning of detecting the effect.
"""
from __future__ import annotations

from typing import Mapping

import numpy as np

from selfref.certify import uniform_grid
from selfref.decision import decide
from selfref.experiment import Sample, _gain, bootstrap_gain

__all__ = ["tilt", "draw_groups", "mean_gain", "strength_for_gain", "simulate_power"]


def tilt(ranks, loss, strength: float) -> np.ndarray:
    """Ranks of a better arm: correct outputs move toward 0 by the factor (1 - strength)."""
    ranks = np.asarray(ranks, dtype=float)
    return np.where(np.asarray(loss) == 0, ranks * (1.0 - strength), ranks)


def draw_groups(pool: Sample, counts: Mapping[str, int], rng: np.random.Generator):
    """Row indices and fresh group labels of ``counts[block]`` groups drawn with
    replacement from each block of ``pool``."""
    rows, labels, serial = [], [], 0
    for block in sorted(counts):
        in_block = np.flatnonzero(pool.block == block)
        groups, inverse = np.unique(pool.group[in_block], return_inverse=True)
        for g in rng.integers(0, groups.size, size=counts[block]):
            members = in_block[inverse == g]
            rows.append(members)
            labels.append(np.full(members.size, serial))
            serial += 1
    return np.concatenate(rows), np.concatenate(labels).astype(str)


def _replication(pool: Sample, ranks, cal_counts, test_counts, strength, rng):
    samples, arms = [], []
    for counts in (cal_counts, test_counts):
        rows, labels = draw_groups(pool, counts, rng)
        sample = pool.rows(rows)
        sample.group = labels
        samples.append(sample)
        arms.append((ranks[rows], tilt(ranks[rows], sample.loss, strength)))
    return samples[0], samples[1], arms[0], arms[1]


def mean_gain(selcert, pool: Sample, ranks, cal_counts, test_counts, strength: float,
              alpha: float, delta: float, n_rep: int, seed: int) -> float:
    """Mean coverage gain, in points, of the tilted arm over the baseline."""
    grid, ranks = uniform_grid(), np.asarray(ranks, dtype=float)
    gains = []
    for rep in range(n_rep):
        rng = np.random.default_rng([seed, rep])            # the same draws for every strength
        cal, test, cal_arms, test_arms = _replication(pool, ranks, cal_counts, test_counts,
                                                      strength, rng)
        gains.append(_gain(selcert, cal_arms, test_arms, cal, test, alpha, delta, grid))
    return float(np.mean(gains))


def strength_for_gain(selcert, pool: Sample, ranks, cal_counts, test_counts, target: float,
                      alpha: float, delta: float, n_rep: int, seed: int,
                      tolerance: float = 0.1) -> dict:
    """A tilt strength whose mean gain is close to ``target`` points, by bisection.

    The mean gain is a step function of the strength, because thresholds sit on
    a grid, so the target may not be hit exactly. The search keeps a bracket:
    the largest strength found whose gain is at most the target and the smallest
    whose gain is at least the target. It returns the end of the bracket whose
    gain is closer to the target, and both ends, so that a caller can run the
    power simulation at each. If even strength = 1 gains less than the target,
    ``reachable`` is False and the strongest tilt is returned.
    """
    def gain(strength):
        return mean_gain(selcert, pool, ranks, cal_counts, test_counts, strength,
                         alpha, delta, n_rep, seed)

    below, above = (0.0, 0.0), (1.0, gain(1.0))
    if above[1] < target:
        return {"strength": 1.0, "gain": above[1], "reachable": False,
                "below": below, "above": above}
    for _ in range(24):
        if min(abs(below[1] - target), abs(above[1] - target)) <= tolerance:
            break
        middle = (below[0] + above[0]) / 2
        achieved = gain(middle)
        if achieved < target:
            below = (middle, achieved)
        else:
            above = (middle, achieved)
    closest = min((below, above), key=lambda end: abs(end[1] - target))
    return {"strength": closest[0], "gain": closest[1], "reachable": True,
            "below": below, "above": above}


def simulate_power(selcert, pool: Sample, ranks, cal_counts, test_counts, strength: float,
                   alpha: float, delta: float, bar: float, floor: float, n_rep: int,
                   n_boot: int, seed: int) -> dict:
    """Run the decision procedure on ``n_rep`` replications at one tilt strength."""
    ranks = np.asarray(ranks, dtype=float)
    outcomes, points, lows = [], [], []
    for rep in range(n_rep):
        rng = np.random.default_rng([seed, rep])
        cal, test, cal_arms, test_arms = _replication(pool, ranks, cal_counts, test_counts,
                                                      strength, rng)
        result = bootstrap_gain(selcert, cal_arms, test_arms, cal, test, alpha, delta,
                                n_boot=n_boot, seed=int(rng.integers(2 ** 31)))
        outcomes.append(decide(result["point"], result["low"], result["high"], bar, floor))
        points.append(result["point"])
        lows.append(result["low"])
    share = lambda name: float(np.mean([o == name for o in outcomes]))
    return {"strength": strength, "replications": n_rep, "bootstrap_draws": n_boot,
            "mean_gain": float(np.mean(points)), "sd_gain": float(np.std(points, ddof=1)),
            "p_pass": share("PASS"), "p_fail": share("FAIL"),
            "p_inconclusive": share("INCONCLUSIVE"),
            "p_interval_above_floor": float(np.mean(np.array(lows) > floor))}
