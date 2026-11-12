# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LogNovations Holdings LLC
"""The smoke experiment that CI runs: the whole pipeline on synthetic items."""
from __future__ import annotations

import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "smoke.py"


def load():
    spec = importlib.util.spec_from_file_location("smoke", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_smoke_experiment_runs_end_to_end_and_reports_the_three_arms():
    result = load().run(groups=240, n_boot=40, seed=1)
    assert set(result["coverage"]) == {"BASELINE", "BASELINE+SOURCE", "BASELINE+SELF"}
    for arm in result["coverage"].values():
        assert 0.0 <= arm["0.1"]["coverage_points"] <= 100.0
    assert result["hypotheses"]["H1"]["outcome"] in ("PASS", "FAIL", "INCONCLUSIVE")
    assert result["hypotheses"]["H2"]["outcome"] == "NOT TESTABLE"
    assert result["sizes"]["test"]["groups"] == 40


def test_the_smoke_experiment_is_reproducible():
    smoke = load()
    assert smoke.run(groups=240, n_boot=20, seed=3) == smoke.run(groups=240, n_boot=20, seed=3)


def test_the_smoke_experiment_is_not_degenerate():
    """The synthetic errors must be hard enough that no arm is perfect: otherwise the
    three arms tie exactly, every bootstrap draw is zero, and the run exercises little."""
    result = load().run()
    coverage = [arm["0.1"]["coverage_points"] for arm in result["coverage"].values()]
    assert 0.0 < min(coverage) and max(coverage) < 60.0
    h1 = result["hypotheses"]["H1"]
    assert h1["interval_low"] < h1["interval_high"]
