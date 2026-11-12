# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LogNovations Holdings LLC
"""EXPLORATORY. BASELINE only. Open splits only (TRAIN and the three DEV splits).

    python scripts/explore_design.py [--jobs 8]

Run after the power gate of Phase 2 triggered, to see where the A/A noise
comes from. It is not part of the pre-registered procedure and it changes
nothing in it. It asks what BASELINE's DEV-rehearsal coverage and the A/A gap
would be under other grid sizes and under the other way of meeting the
calibrator's i.i.d. requirement:

    per_generator   calibrate each generator separately at delta / 6
                    (the design in force, at K = 20)
    pooled_one      keep one response per source, its generator chosen by a
                    hash of the source id; calibrate once at delta; one
                    threshold on a rank pooled over generators; coverage is
                    still measured on every DEV_TEST response

No tested column (SOURCE, SELF or RESAMPLE) and no CAL or TEST row is touched.
The A/A models are fitted exactly as ``phase2.py aa`` fits them, so the
per_generator K = 20 gaps must equal prereg/aa.json; the record says whether
they do. Writes prereg/exploratory_design.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parent))
import phase2  # noqa: E402  (also puts src/ on the path)

from selfref.certify import (RankTransform, accepted_mask, certified_coverage,  # noqa: E402
                             certify_by_stratum, selective_risk, uniform_grid)
from selfref.combiner import build_model, risk_scores  # noqa: E402
from selfref.decision import compute_bar, compute_floor  # noqa: E402
from selfref.experiment import _resample_groups  # noqa: E402
from selfref.selcert_pin import load_selcert  # noqa: E402

DESIGNS = [(kind, points) for kind in ("per_generator", "pooled_one") for points in (20, 50, 100)]
ALPHA = 0.10
PARTS = ("DEV_SELECT", "DEV_CAL", "DEV_TEST")


def name_of(design) -> str:
    return f"{design[0]}_K{design[1]}"


def one_per_source(sample) -> np.ndarray:
    """True for one response of each source: the generator picked by a hash of the source id."""
    generators = sorted(set(sample.stratum.astype(str)))
    keep = np.zeros(sample.loss.size, dtype=bool)
    for row, (group, stratum) in enumerate(zip(sample.group.astype(str), sample.stratum.astype(str))):
        digest = hashlib.sha256(b"selfref-explore/one-per-source\0" + group.encode()).digest()
        keep[row] = generators.index(stratum) == int.from_bytes(digest[:8], "big") % len(generators)
    return keep


def coverage(selcert, design, risk, data, alpha: float, keep_cal) -> dict:
    kind, points = design
    grid = uniform_grid(points)
    strata = {part: (data[part].stratum.astype(str) if kind == "per_generator"
                     else np.full(data[part].loss.size, "pooled")) for part in PARTS}
    transform = RankTransform.fit(risk["DEV_SELECT"], strata["DEV_SELECT"])
    cal_ranks = transform.transform(risk["DEV_CAL"], strata["DEV_CAL"])
    test_ranks = transform.transform(risk["DEV_TEST"], strata["DEV_TEST"])
    rows = np.ones(cal_ranks.size, dtype=bool) if kind == "per_generator" else keep_cal
    certificates = certify_by_stratum(selcert, cal_ranks[rows], data["DEV_CAL"].loss[rows],
                                      strata["DEV_CAL"][rows], grid, alpha, phase2.DELTA)
    mask = accepted_mask(test_ranks, strata["DEV_TEST"], certificates.thresholds)
    return {"coverage_points": 100.0 * certified_coverage(mask),
            "realised_risk": selective_risk(mask, data["DEV_TEST"].loss),
            "thresholds": dict(certificates.thresholds), "calibration_items": int(rows.sum())}


def _aa_run(run: int, config_record: dict) -> dict:
    if "data" not in _aa_run.__dict__:                          # once per worker process
        _aa_run.data = phase2.arm(phase2.load_tables(), "BASELINE")[0]
        _aa_run.selcert = load_selcert()
        _aa_run.keep = one_per_source(_aa_run.data["DEV_CAL"])
    data, config = _aa_run.data, phase2.config_from(config_record)
    covered = {name_of(d): [] for d in DESIGNS}
    for arm in (0, 1):                                          # as in selfref.experiment.aa_coverage_gap
        rng = np.random.default_rng([phase2.SEED, run, arm])
        train = data["TRAIN"].rows(_resample_groups(data["TRAIN"], rng))
        model = build_model(config, seed=int(rng.integers(2 ** 31))).fit(train.X, train.loss)
        risk = {part: risk_scores(model, data[part].X) for part in PARTS}
        for design in DESIGNS:
            covered[name_of(design)].append(
                coverage(_aa_run.selcert, design, risk, data, ALPHA, _aa_run.keep)["coverage_points"])
    return covered


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--jobs", type=int, default=8)
    args = parser.parse_args(argv)

    baseline = phase2.read_record("baseline.json")
    data, selcert = phase2.arm(phase2.load_tables(), "BASELINE")[0], load_selcert()
    keep = one_per_source(data["DEV_CAL"])
    model = phase2.load_model("BASELINE")
    risk = {part: risk_scores(model, data[part].X) for part in PARTS}
    runs = Parallel(n_jobs=args.jobs)(delayed(_aa_run)(run, baseline["selected"])
                                      for run in range(phase2.AA_RUNS))

    designs = {}
    for design in DESIGNS:
        name = name_of(design)
        frozen = {str(alpha): coverage(selcert, design, risk, data, alpha, keep)
                  for alpha in phase2.ALPHAS}
        first = np.array([run[name][0] for run in runs])
        gaps = first - np.array([run[name][1] for run in runs])
        absolute = np.abs(gaps)
        c_b, floor = frozen[str(ALPHA)]["coverage_points"], compute_floor(gaps)
        designs[name] = {
            "kind": design[0], "grid_points": design[1],
            "frozen_baseline_dev_rehearsal": frozen,
            "aa": {"alpha": ALPHA, "runs": int(gaps.size),
                   "abs_p50": float(np.percentile(absolute, 50)),
                   "abs_p90": float(np.percentile(absolute, 90)),
                   "abs_p95": float(np.percentile(absolute, 95)),
                   "abs_p99": float(np.percentile(absolute, 99)),
                   "abs_max": float(absolute.max()), "sd": float(gaps.std(ddof=1)),
                   "refit_coverage_mean": float(first.mean()),
                   "refit_coverage_sd": float(first.std(ddof=1)),
                   "refit_coverage_share_zero": float((first == 0).mean()),
                   "gap_points": [float(g) for g in gaps]},
            "implied_by_the_rule": {"c_B_points": c_b, "FLOOR_points": floor,
                                    "BAR_points": compute_bar(c_b, floor)}}
    committed = np.array(phase2.read_record("aa.json")["gap_points"])
    reproduces = bool(np.array_equal(np.array(designs["per_generator_K20"]["aa"]["gap_points"]),
                                     committed))
    phase2.write_record("exploratory_design.json", {
        "status": "EXPLORATORY. BASELINE only, open splits only. Not part of the pre-registered "
                  "procedure; the design in force is per_generator_K20.",
        "per_generator_K20_reproduces_prereg_aa_json": reproduces,
        "designs": designs})
    print("per_generator_K20 reproduces prereg/aa.json:", reproduces)
    for name, record in designs.items():
        frozen, aa, implied = (record["frozen_baseline_dev_rehearsal"], record["aa"],
                               record["implied_by_the_rule"])
        print(f"{name:20s} coverage at alpha 0.05 {frozen['0.05']['coverage_points']:5.2f}, "
              f"at 0.10 {frozen['0.1']['coverage_points']:5.2f} | A/A |gap| p50 {aa['abs_p50']:5.2f} "
              f"p95 {aa['abs_p95']:5.2f} p99 {aa['abs_p99']:5.2f} | FLOOR {implied['FLOOR_points']:5.2f} "
              f"BAR {implied['BAR_points']:5.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
