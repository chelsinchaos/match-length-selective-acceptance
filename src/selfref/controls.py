# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LogNovations Holdings LLC
"""Controls and robustness analyses, as functions of arrays.

These are the analyses of Sections 14 and 15 of the pre-registration. Like
``selfref.confirm.analyse`` they take ranks, scores and ``Sample`` objects and
know nothing of files, so the same code runs in the DEV rehearsal and on the
sealed splits.

Throughout, a *risk score* is higher for outputs more likely to be errors, a
*rank* is a risk score mapped to [0, 1] by the frozen per-generator transform,
and an output is accepted when its rank is at or below its generator's
certified threshold. Every certificate comes from the pinned calibrator through
``selfref.certify.certify_by_stratum``; the one confidence bound used here that
the calibrator does not compute itself, a lower bound on a risk, is obtained
from the calibrator package's own upper bound by symmetry.
"""
from __future__ import annotations

from typing import Mapping, Optional, Sequence

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from selfref.certify import (RankTransform, accepted_mask, certified_coverage, certify_by_stratum,
                             selective_risk, uniform_grid)
from selfref.combiner import Config, build_model, risk_scores
from selfref.decision import decide
from selfref.experiment import Sample, _resample_groups, bootstrap_gain
from selfref.power import draw_groups

__all__ = ["accept", "coverage_row", "permute_within", "permuted_label_control", "derangement",
           "auroc", "average_precision", "auroc_by_bin", "risk_coverage_curve",
           "paired_metric_difference", "synthetic_guarantee", "lower_risk_bound", "flagged",
           "resplit_guarantee", "extra_calibration", "cross_block", "breakdown",
           "acceptance_overlap", "aa_decisions"]


# ------------------------------------------------------------------ coverage

def accept(selcert, cal_ranks, cal: Sample, test_ranks, test: Sample, alpha: float, delta: float):
    """Certify thresholds on the calibration sample; return the mask of accepted
    test items and the thresholds."""
    certificates = certify_by_stratum(selcert, cal_ranks, cal.loss, cal.stratum,
                                      uniform_grid(), alpha, delta)
    return accepted_mask(test_ranks, test.stratum, certificates.thresholds), \
        dict(certificates.thresholds)


def coverage_row(selcert, cal_ranks, cal: Sample, test_ranks, test: Sample, alpha: float,
                 delta: float) -> dict:
    """Certified coverage and realised risk on the test sample, as the confirmatory table reports them."""
    mask, thresholds = accept(selcert, cal_ranks, cal, test_ranks, test, alpha, delta)
    return {"coverage_points": 100.0 * certified_coverage(mask),
            "realised_risk": selective_risk(mask, test.loss),
            "accepted": int(mask.sum()),
            "thresholds": thresholds}


def breakdown(mask, loss, labels) -> dict:
    """Coverage and realised risk within each value of ``labels`` (a task type or a generator)."""
    mask, loss, labels = np.asarray(mask, dtype=bool), np.asarray(loss), np.asarray(labels).astype(str)
    rows = {}
    for name in np.unique(labels):
        chosen = labels == name
        rows[str(name)] = {"items": int(chosen.sum()), "accepted": int(mask[chosen].sum()),
                           "coverage_points": 100.0 * certified_coverage(mask[chosen]),
                           "realised_risk": selective_risk(mask[chosen], loss[chosen]),
                           "error_rate_of_all_items": float(loss[chosen].mean())}
    return rows


def acceptance_overlap(masks: Mapping[str, np.ndarray]) -> list:
    """For every pair of acceptance rules on the same items: how many each accepts, and how many both do."""
    names = list(masks)
    return [{"first": a, "second": b,
             "accepted_first": int(np.sum(masks[a])), "accepted_second": int(np.sum(masks[b])),
             "accepted_both": int(np.sum(np.asarray(masks[a], dtype=bool) & np.asarray(masks[b], dtype=bool)))}
            for k, a in enumerate(names) for b in names[k + 1:]]


# ---------------------------------------------------------- negative controls

def permute_within(values, strata, rng: np.random.Generator) -> np.ndarray:
    """A copy of ``values`` permuted at random within each stratum."""
    values, strata = np.asarray(values), np.asarray(strata).astype(str)
    out = values.copy()
    for name in np.unique(strata):
        chosen = np.flatnonzero(strata == name)
        out[chosen] = values[rng.permutation(chosen)]
    return out


def _with_loss(sample: Sample, loss) -> Sample:
    return Sample(sample.X, np.asarray(loss), sample.stratum, sample.group, sample.block)


def permuted_label_control(selcert, ranks: Mapping[str, tuple], cal: Sample, test: Sample,
                           alpha: float, delta: float, n_perm: int, seed: int) -> dict:
    """Certified coverage of each arm after the labels are permuted within generator.

    ``ranks`` maps an arm to its (calibration ranks, test ranks). The labels of
    both samples are permuted, so a score keeps nothing but each generator's
    error rate. The first permutation is the control; the others show how often
    chance alone certifies anything.
    """
    out = {"permutations": n_perm,
           "true_labels": {arm: coverage_row(selcert, c, cal, t, test, alpha, delta)
                           for arm, (c, t) in ranks.items()},
           "arms": {}}
    coverage = {arm: [] for arm in ranks}
    first = {}
    for p in range(n_perm):
        rng = np.random.default_rng([seed, p])
        cal_p = _with_loss(cal, permute_within(cal.loss, cal.stratum, rng))
        test_p = _with_loss(test, permute_within(test.loss, test.stratum, rng))
        for arm, (c, t) in ranks.items():
            row = coverage_row(selcert, c, cal_p, t, test_p, alpha, delta)
            coverage[arm].append(row["coverage_points"])
            if p == 0:
                first[arm] = row
    for arm, values in coverage.items():
        values = np.array(values)
        out["arms"][arm] = {"first_permutation": first[arm],
                            "share_with_any_coverage": float((values > 0).mean()),
                            "mean_coverage_points": float(values.mean()),
                            "max_coverage_points": float(values.max())}
    return out


def derangement(keys: Sequence[str], blocks: Sequence[str], seed: int) -> dict[str, str]:
    """A fixed map sending every key to a different key of the same block.

    Within each block the keys are sorted, shuffled with a seeded generator and
    arranged in a cycle; each key maps to the next one. The result depends on
    the set of (key, block) pairs and the seed, not on the order they are given in.
    """
    by_block: dict[str, set] = {}
    for key, block in zip(keys, blocks):
        by_block.setdefault(str(block), set()).add(str(key))
    mapping = {}
    for index, block in enumerate(sorted(by_block)):
        members = sorted(by_block[block])
        if len(members) < 2:
            raise ValueError(f"block {block!r} has one key; it cannot be paired with another")
        order = np.random.default_rng([seed, index]).permutation(len(members))
        cycle = [members[i] for i in order]
        for position, key in enumerate(cycle):
            mapping[key] = cycle[(position + 1) % len(cycle)]
    return mapping


# -------------------------------------------------------------- ranking metrics

def auroc(loss, score) -> Optional[float]:
    """Area under the ROC curve with an error as the positive class; None if one class is absent."""
    loss = np.asarray(loss)
    if loss.size == 0 or loss.min() == loss.max():
        return None
    return float(roc_auc_score(loss, np.asarray(score, dtype=float)))


def average_precision(loss, score) -> Optional[float]:
    """Average precision with an error as the positive class; None if one class is absent."""
    loss = np.asarray(loss)
    if loss.size == 0 or loss.min() == loss.max():
        return None
    return float(average_precision_score(loss, np.asarray(score, dtype=float)))


def auroc_by_bin(loss, score, values, edges: Sequence[float]) -> list[dict]:
    """AUROC within each bin of ``values``. Bin k holds edges[k-1] <= value < edges[k]."""
    loss, score = np.asarray(loss), np.asarray(score, dtype=float)
    bins = np.searchsorted(np.asarray(edges, dtype=float), np.asarray(values, dtype=float),
                           side="right")
    rows = []
    for k in range(len(edges) + 1):
        chosen = bins == k
        rows.append({"bin": k, "items": int(chosen.sum()), "errors": int(loss[chosen].sum()),
                     "auroc": auroc(loss[chosen], score[chosen])})
    return rows


def risk_coverage_curve(loss, score, points: int = 100) -> list[dict]:
    """Error rate among the lowest-risk share of items, at ``points`` evenly spaced shares."""
    loss = np.asarray(loss, dtype=float)
    order = np.argsort(np.asarray(score, dtype=float), kind="stable")
    errors = np.concatenate([[0.0], np.cumsum(loss[order])])
    curve = []
    for k in range(1, points + 1):
        kept = max(1, int(round(k / points * loss.size)))
        curve.append({"coverage": k / points, "risk": float(errors[kept] / kept)})
    return curve


def paired_metric_difference(test: Sample, first, second, n_boot: int, seed: int) -> dict:
    """AUROC and average precision of two risk scores on one sample, and second
    minus first, with a 95% percentile interval from a paired bootstrap of whole
    groups within block. Both scores see the same resample in every draw."""
    first, second = np.asarray(first, dtype=float), np.asarray(second, dtype=float)
    metrics = {"auroc": auroc, "auprc": average_precision}
    rng = np.random.default_rng(seed)
    draws = {name: [] for name in metrics}
    skipped = 0
    for _ in range(n_boot):
        rows = _resample_groups(test, rng)
        loss = test.loss[rows]
        if loss.min() == loss.max():
            skipped += 1
            continue
        for name, metric in metrics.items():
            draws[name].append(metric(loss, second[rows]) - metric(loss, first[rows]))
    out = {"bootstrap_draws": n_boot, "draws_skipped_for_one_class": skipped}
    for name, metric in metrics.items():
        a, b = metric(test.loss, first), metric(test.loss, second)
        low, high = np.percentile(draws[name], [2.5, 97.5])
        out[name] = {"first": a, "second": b, "difference": b - a,
                     "low": float(low), "high": float(high)}
    return out


# ------------------------------------------------------------ guarantee checks

def synthetic_guarantee(selcert, n_draws: int, n_per_stratum: int, alpha: float, delta: float,
                        seed: int) -> dict:
    """The stated bound on synthetic data whose risk is known in closed form.

    Two strata of equal size. Ranks are uniform on [0, 1]. The probability of
    an error at rank r is 0.6 r^2 in the first stratum and 0.3 r in the second,
    so the risk among outputs accepted at threshold t is 0.2 t^2 and 0.15 t.
    Each draw is a fresh calibration sample put through the per-stratum
    calibrator. A draw is a violation when the true pooled risk among accepted
    outputs, the acceptance-weighted mean of the two strata's risks, exceeds
    alpha. A draw that accepts nothing is not a violation.
    """
    rng = np.random.default_rng(seed)
    grid = uniform_grid()
    strata = np.array(["x"] * n_per_stratum + ["y"] * n_per_stratum)
    first = strata == "x"
    true_risk = {"x": lambda t: 0.2 * t ** 2, "y": lambda t: 0.15 * t}
    violations, accepting, pooled_risks = 0, 0, []
    for _ in range(n_draws):
        ranks = rng.uniform(size=strata.size)
        probability = np.where(first, 0.6 * ranks ** 2, 0.3 * ranks)
        losses = (rng.uniform(size=strata.size) < probability).astype(int)
        thresholds = certify_by_stratum(selcert, ranks, losses, strata, grid, alpha,
                                        delta).thresholds
        mass = {name: (t or 0.0) for name, t in thresholds.items()}
        if sum(mass.values()) == 0:
            continue
        accepting += 1
        pooled = sum(mass[name] * true_risk[name](mass[name]) for name in mass) / sum(mass.values())
        pooled_risks.append(pooled)
        violations += pooled > alpha
    return {"draws": n_draws, "n_per_stratum": n_per_stratum, "alpha": alpha, "delta": delta,
            "violations": int(violations), "violation_share": violations / n_draws,
            "holds": bool(violations / n_draws <= delta),
            "draws_accepting_anything": accepting,
            "mean_true_risk_when_accepting": float(np.mean(pooled_risks)) if pooled_risks else None,
            "max_true_risk_when_accepting": float(np.max(pooled_risks)) if pooled_risks else None}


def lower_risk_bound(selcert, errors: int, trials: int, level: float) -> float:
    """One-sided lower confidence bound, at ``level``, on a risk estimated by
    ``errors`` in ``trials``. By symmetry it is one minus the calibrator
    package's upper bound for the ``trials - errors`` non-errors."""
    return 1.0 - float(selcert.clopper_pearson_upper(trials - errors, trials, level))


def flagged(selcert, accepted, loss, strata, alpha: float, level: float) -> bool:
    """True when, in some stratum with accepted outputs, the lower bound on the
    risk among them exceeds alpha. Each stratum is tested at 1 - level / (number
    of strata), so a sample whose risks are all at most alpha is flagged with
    probability at most ``level``."""
    accepted, loss = np.asarray(accepted, dtype=bool), np.asarray(loss)
    strata = np.asarray(strata).astype(str)
    names = np.unique(strata)
    for name in names:
        chosen = accepted & (strata == name)
        if chosen.any():
            bound = lower_risk_bound(selcert, int(loss[chosen].sum()), int(chosen.sum()),
                                     1.0 - level / names.size)
            if bound > alpha:
                return True
    return False


def resplit_guarantee(selcert, ranks, pool: Sample, cal_counts: Mapping[str, int], alpha: float,
                      delta: float, n_resplits: int, seed: int, flag_level: float = 0.05) -> dict:
    """Re-split pooled calibration and test groups and count flagged held-out parts.

    Each re-split sends ``cal_counts[block]`` groups of each block, chosen at
    random, to a calibration part and the rest to a held-out part; thresholds
    are certified on the calibration part and applied to the held-out part,
    which is flagged as ``flagged`` describes. If the guarantee holds, a
    re-split is flagged with probability at most delta + flag_level.
    """
    ranks = np.asarray(ranks, dtype=float)
    members = {}
    for block in sorted(cal_counts):
        in_block = np.flatnonzero(pool.block == block)
        groups, inverse = np.unique(pool.group[in_block], return_inverse=True)
        members[block] = [in_block[inverse == g] for g in range(groups.size)]
    flags, accepting, coverage, risks = 0, 0, [], []
    for r in range(n_resplits):
        rng = np.random.default_rng([seed, r])
        cal_rows, held_rows = [], []
        for block, groups in members.items():
            order = rng.permutation(len(groups))
            cal_rows += [groups[g] for g in order[:cal_counts[block]]]
            held_rows += [groups[g] for g in order[cal_counts[block]:]]
        cal_rows, held_rows = np.concatenate(cal_rows), np.concatenate(held_rows)
        cal, held = pool.rows(cal_rows), pool.rows(held_rows)
        mask, _ = accept(selcert, ranks[cal_rows], cal, ranks[held_rows], held, alpha, delta)
        coverage.append(100.0 * certified_coverage(mask))
        if mask.any():
            accepting += 1
            risks.append(selective_risk(mask, held.loss))
        flags += flagged(selcert, mask, held.loss, held.stratum, alpha, flag_level)
    total_groups = sum(len(groups) for groups in members.values())
    allowed = delta + flag_level
    return {"resplits": n_resplits, "calibration_groups": int(sum(cal_counts.values())),
            "held_out_groups": int(total_groups - sum(cal_counts.values())),
            "alpha": alpha, "delta": delta, "flag_level": flag_level,
            "flagged": int(flags), "flagged_share": flags / n_resplits,
            "allowed_share": allowed, "passes": bool(flags / n_resplits <= allowed),
            "resplits_accepting_anything": accepting,
            "mean_held_out_coverage_points": float(np.mean(coverage)),
            "mean_held_out_risk_when_accepting": float(np.mean(risks)) if risks else None,
            "share_of_resplits_with_held_out_risk_above_alpha":
                float(np.sum(np.array(risks) > alpha) / n_resplits) if risks else 0.0}


# ----------------------------------------------------- cheap alternative, stress

def extra_calibration(selcert, cal_ranks, cal: Sample, test_ranks, test: Sample, alpha: float,
                      delta: float, multiples: Sequence[float], n_draws: int, seed: int) -> dict:
    """Coverage on the test sample when the calibration sample is resampled, with
    replacement and within block, to ``multiple`` times its number of groups."""
    cal_ranks = np.asarray(cal_ranks, dtype=float)
    members = {}
    for block in np.unique(cal.block):
        in_block = np.flatnonzero(cal.block == block)
        groups, inverse = np.unique(cal.group[in_block], return_inverse=True)
        members[str(block)] = [in_block[inverse == g] for g in range(groups.size)]
    out = {}
    for index, multiple in enumerate(multiples):
        sizes = {block: int(round(multiple * len(groups))) for block, groups in members.items()}
        coverage = []
        for draw in range(n_draws):
            rng = np.random.default_rng([seed, index, draw])
            rows = np.concatenate([groups[g] for block, groups in members.items()
                                   for g in rng.integers(0, len(groups), size=sizes[block])])
            mask, _ = accept(selcert, cal_ranks[rows], cal.rows(rows), test_ranks, test, alpha, delta)
            coverage.append(100.0 * certified_coverage(mask))
        coverage = np.array(coverage)
        out[str(float(multiple))] = {
            "calibration_groups": int(sum(sizes.values())), "draws": n_draws,
            "mean_coverage_points": float(coverage.mean()),
            "median_coverage_points": float(np.median(coverage)),
            "share_with_any_coverage": float((coverage > 0).mean())}
    return out


def cross_block(selcert, cal_ranks, cal: Sample, test_ranks, test: Sample, alpha: float,
                delta: float) -> dict:
    """Thresholds certified on every block but one, applied to the test items of
    the block left out. Exchangeability between the two is not expected to hold."""
    cal_ranks, test_ranks = np.asarray(cal_ranks, dtype=float), np.asarray(test_ranks, dtype=float)
    rows = {}
    for block in np.unique(test.block):
        others, held = cal.block != block, test.block == block
        mask, thresholds = accept(selcert, cal_ranks[others], cal.rows(others),
                                  test_ranks[held], test.rows(held), alpha, delta)
        rows[str(block)] = {"calibrated_on": sorted(str(b) for b in np.unique(cal.block[others])),
                            "thresholds": thresholds, "items": int(held.sum()),
                            "accepted": int(mask.sum()),
                            "coverage_points": 100.0 * certified_coverage(mask),
                            "realised_risk": selective_risk(mask, test.loss[held])}
    return rows


# ------------------------------------------------- the rule when there is no gain

def aa_decisions(selcert, config: Config, train: Sample, select: Sample, pool: Sample,
                 cal_counts: Mapping[str, int], test_counts: Mapping[str, int], alpha: float,
                 delta: float, bar: float, floor: float, n_rep: int, n_boot: int, seed: int) -> dict:
    """The decision rule applied to pairs of arms that differ by refitting only.

    Each replication fits two models of one configuration, as the A/A test
    does: each on its own bootstrap resample of the TRAIN groups and with its
    own seed. Neither arm has anything the other lacks, so the true gain is
    zero. A calibration sample and a test sample of the given sizes are then
    drawn from ``pool`` (whole groups, with replacement, within block), as the
    power simulation draws them, and the pair goes through the paired bootstrap
    and the decision rule. The shares of the three outcomes say what the rule
    returns when there is nothing to find.
    """
    outcomes, gains, lows, highs = [], [], [], []
    for rep in range(n_rep):
        rng = np.random.default_rng([seed, rep])
        ranks = []
        for _arm in (0, 1):
            fitted_on = train.rows(_resample_groups(train, rng))
            model = build_model(config, seed=int(rng.integers(2 ** 31))).fit(fitted_on.X, fitted_on.loss)
            transform = RankTransform.fit(risk_scores(model, select.X), select.stratum)
            ranks.append(transform.transform(risk_scores(model, pool.X), pool.stratum))
        samples, arms = [], []
        for counts in (cal_counts, test_counts):
            rows, labels = draw_groups(pool, counts, rng)
            part = pool.rows(rows)
            part.group = labels
            samples.append(part)
            arms.append((ranks[0][rows], ranks[1][rows]))
        result = bootstrap_gain(selcert, arms[0], arms[1], samples[0], samples[1], alpha, delta,
                                n_boot=n_boot, seed=int(rng.integers(2 ** 31)))
        outcomes.append(decide(result["point"], result["low"], result["high"], bar, floor))
        gains.append(result["point"])
        lows.append(result["low"])
        highs.append(result["high"])
    share = lambda name: float(np.mean([o == name for o in outcomes]))
    return {"replications": n_rep, "bootstrap_draws": n_boot,
            "p_pass": share("PASS"), "p_fail": share("FAIL"), "p_inconclusive": share("INCONCLUSIVE"),
            "mean_gain": float(np.mean(gains)), "mean_absolute_gain": float(np.mean(np.abs(gains))),
            "interval_high_median": float(np.median(highs)),
            "interval_high_p90": float(np.percentile(highs, 90)),
            "gains": [float(g) for g in gains], "interval_low": [float(x) for x in lows],
            "interval_high": [float(x) for x in highs]}

