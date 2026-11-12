# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LogNovations Holdings LLC
"""A small smoke experiment on synthetic items, for CI.

    python scripts/smoke.py

It needs no dataset and no neural model. It makes synthetic sources and
outputs, computes the profile features and the lexical baselines with the
reference backend, fits the three arms with the study's combiner search,
and runs the confirmatory analysis with the pinned calibrator, exactly as the
real pipeline does. A synthetic output copies a span of its source with some
words replaced: none or one in a correct output, one to six in an erroneous
one. The two overlap, so no arm is perfect, the arms differ and the bootstrap
has something to resample. The numbers mean nothing; the run shows that the
pieces fit together.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HARNESS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HARNESS / "src"))

from selfref.baselines import lexical_baselines  # noqa: E402
from selfref.combiner import fit_and_select, risk_scores  # noqa: E402
from selfref.confirm import analyse  # noqa: E402
from selfref.experiment import Sample, column_sets  # noqa: E402
from selfref.extract import FeatureInput, extract_features  # noqa: E402
from selfref.selcert_pin import load_selcert  # noqa: E402

ERROR_RATE = {"g1": 0.15, "g2": 0.5}
SHARES = (("TRAIN", 2 / 6), ("select", 1 / 6), ("cal", 2 / 6), ("test", 1 / 6))


def run(groups: int = 600, n_boot: int = 200, seed: int = 20261111) -> dict:
    rng = np.random.default_rng(seed)
    vocabulary = [f"w{k}" for k in range(400)]
    rows, names = [], None
    for g in range(groups):
        source_words = list(rng.choice(vocabulary[:300], size=120))
        source = " ".join(source_words).encode()
        for stratum, rate in ERROR_RATE.items():
            wrong = int(rng.uniform() < rate)
            start = int(rng.integers(0, 90))
            words = source_words[start:start + 30]
            replaced = int(rng.integers(1, 7)) if wrong else int(rng.integers(0, 2))
            for position in rng.choice(30, size=replaced, replace=False):
                words[int(position)] = str(rng.choice(vocabulary[:300]))
            output = " ".join(words).encode()
            scores = extract_features(FeatureInput(output=output, source=source))
            scores.update(lexical_baselines(output, source))
            scores["cov.noise"] = float(rng.uniform())
            names = names or sorted(scores)
            rows.append(([scores[n] for n in names], wrong, stratum, str(g), "qa" if g % 2 else "summary"))
    X = np.array([r[0] for r in rows])
    whole = Sample(X, np.array([r[1] for r in rows]), np.array([r[2] for r in rows]),
                   np.array([r[3] for r in rows]), np.array([r[4] for r in rows]))
    group_index = whole.group.astype(int)
    data, edge = {}, 0
    for part, share in SHARES:
        upper = edge + int(round(share * groups))
        data[part] = whole.rows((group_index >= edge) & (group_index < upper))
        edge = upper
    risk = {}
    for arm, columns in column_sets(names).items():
        index = [names.index(c) for c in columns]
        selection = fit_and_select(data["TRAIN"].X[:, index], data["TRAIN"].loss,
                                   data["select"].X[:, index], data["select"].loss, seed)
        risk[arm] = {part: risk_scores(selection.model, data[part].X[:, index])
                     for part in ("select", "cal", "test")}
    return analyse(load_selcert(), risk, {p: data[p] for p in ("select", "cal", "test")},
                   primary_alpha=0.10, alphas=(0.05, 0.10), delta=0.05, bar=5.0, floor=2.0,
                   n_boot=n_boot, seed=seed)


def main() -> int:
    result = run()
    for arm, table in result["coverage"].items():
        print(f"{arm:16s} coverage at alpha 0.10: {table['0.1']['coverage_points']:.2f} points")
    for name in ("H1", "H3"):
        h = result["hypotheses"][name]
        print(f"{name}: gain {h['gain_points']:+.2f} [{h['interval_low']:+.2f}, "
              f"{h['interval_high']:+.2f}] -> {h['outcome']}")
    print("smoke experiment completed; these numbers are synthetic and mean nothing")
    json.dumps(result)                      # the record is serialisable
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
