# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LogNovations Holdings LLC
"""The certified-deployment calibrator is a pinned external dependency.

``calibrate_bonferroni`` comes from the ``selcert`` package in the companion
repository, at one fixed commit. These tests check that the checkout really is
that commit and is unmodified, that the function the harness imports is the one
in that checkout, that its behaviour matches what the pre-registration assumes
(low scores accepted; the largest passing grid point chosen; "defer everything"
when nothing passes), and that the harness contains no calibrator of its own.

If the checkout is missing, run ``make fetch-selcert`` first.
"""
from __future__ import annotations

import ast
import inspect
import subprocess
from pathlib import Path

import pytest

from selfref.selcert_pin import (PINNED_SHA, VENDOR_DIR, SelcertPinError,
                                 load_selcert, verify_checkout)

SRC = Path(__file__).resolve().parents[1] / "src" / "selfref"


def git(cwd, *args):
    return subprocess.run(["git", "-C", str(cwd), *args], check=True,
                          capture_output=True, text=True).stdout.strip()


# -------------------------------------------------------------------- the pin

def test_pinned_commit_is_the_one_recorded_in_recon():
    assert PINNED_SHA == "a27fe4fd3fd8726f98036cfb3d7d7fe24edd3fdf"


def test_vendored_checkout_is_at_the_pinned_commit_and_unmodified():
    verify_checkout()          # raises SelcertPinError with the reason otherwise
    assert git(VENDOR_DIR, "rev-parse", "HEAD") == PINNED_SHA


def test_a_missing_checkout_is_rejected(tmp_path):
    with pytest.raises(SelcertPinError, match="fetch"):
        verify_checkout(tmp_path / "nothing-here")


def test_a_checkout_at_another_commit_is_rejected(tmp_path):
    other = tmp_path / "other"
    other.mkdir()
    git(other, "init", "-q")
    (other / "f.txt").write_text("x")
    git(other, "add", "f.txt")
    git(other, "-c", "user.name=t", "-c", "user.email=t@example.invalid",
        "commit", "-q", "-m", "unrelated")
    with pytest.raises(SelcertPinError, match="commit"):
        verify_checkout(other)


def test_a_modified_checkout_is_rejected(tmp_path):
    copy = tmp_path / "copy"
    git(tmp_path, "clone", "-q", str(VENDOR_DIR), str(copy))
    git(copy, "checkout", "-q", "--detach", PINNED_SHA)
    verify_checkout(copy)                                   # pristine copy passes
    target = copy / "src" / "selcert" / "calibrate.py"
    target.write_text(target.read_text() + "\n# tampered\n")
    with pytest.raises(SelcertPinError, match="modified"):
        verify_checkout(copy)


def test_an_added_source_file_is_rejected(tmp_path):
    copy = tmp_path / "copy"
    git(tmp_path, "clone", "-q", str(VENDOR_DIR), str(copy))
    git(copy, "checkout", "-q", "--detach", PINNED_SHA)
    (copy / "src" / "selcert" / "extra.py").write_text("x = 1\n")
    with pytest.raises(SelcertPinError, match="modified"):
        verify_checkout(copy)


# ------------------------------------------------------ what the harness imports

def test_loaded_module_comes_from_the_pinned_checkout():
    selcert = load_selcert()
    assert Path(selcert.__file__).resolve().is_relative_to(VENDOR_DIR.resolve())
    assert selcert.__version__ == "0.1.0"


def test_calibrate_bonferroni_has_the_signature_the_design_assumes():
    selcert = load_selcert()
    parameters = list(inspect.signature(selcert.calibrate_bonferroni).parameters)
    assert parameters == ["scores", "losses", "grid", "alpha", "delta"]


def test_low_scores_are_accepted():
    # 200 items at score 0.1 with no errors, 200 at 0.9 that are all errors.
    selcert = load_selcert()
    scores = [0.1] * 200 + [0.9] * 200
    losses = [0] * 200 + [1] * 200
    result = selcert.calibrate_bonferroni(scores, losses, [0.5, 1.0], alpha=0.05, delta=0.05)
    assert result.tau_hat == 0.5
    assert list(result.accepted) == [200, 400]
    assert list(result.errors) == [0, 200]
    # Clopper-Pearson with zero errors in m trials at level 1 - delta/K
    assert result.upper_bounds[0] == pytest.approx(1 - (0.05 / 2) ** (1 / 200))
    assert result.empirical_coverage == pytest.approx(0.5)


def test_tau_hat_is_the_largest_passing_grid_point_not_the_first_before_a_failure():
    # tau=0.2: 0 errors in 300 (passes). tau=0.4: 20 in 320 (fails).
    # tau=0.6: 20 in 3320 (passes again), so 0.6 is chosen.
    selcert = load_selcert()
    scores = [0.1] * 300 + [0.3] * 20 + [0.5] * 3000
    losses = [0] * 300 + [1] * 20 + [0] * 3000
    result = selcert.calibrate_bonferroni(scores, losses, [0.2, 0.4, 0.6], alpha=0.05, delta=0.05)
    assert [bool(u <= 0.05) for u in result.upper_bounds] == [True, False, True]
    assert result.tau_hat == 0.6


def test_nothing_passing_means_defer_everything():
    selcert = load_selcert()
    result = selcert.calibrate_bonferroni([0.2, 0.4, 0.6], [1, 1, 1], [0.5, 1.0],
                                          alpha=0.05, delta=0.05)
    assert result.tau_hat is None
    assert result.empirical_coverage == 0.0


# ------------------------------------------------------------ no reimplementation

def test_the_harness_defines_no_calibrator_of_its_own():
    offenders = []
    for path in sorted(SRC.rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.FunctionDef) and \
                    any(word in node.name.lower() for word in ("calibrate", "clopper", "bonferroni")):
                offenders.append(f"{path.name}:{node.name}")
    assert not offenders, f"calibration must come from selcert, found {offenders}"


def test_the_harness_computes_no_binomial_confidence_bound_of_its_own():
    # A Clopper-Pearson bound is a beta quantile. Nothing under src/ may compute one.
    forbidden = {"beta", "betaincinv", "betainc", "ppf", "binom", "binomtest"}
    offenders = []
    for path in sorted(SRC.rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            name = getattr(node, "id", None) or getattr(node, "attr", None)
            if isinstance(node, ast.ImportFrom):
                offenders += [f"{path.name}: from {node.module} import {a.name}"
                              for a in node.names if a.name in forbidden or "scipy.stats" in (node.module or "")]
            elif name in forbidden:
                offenders.append(f"{path.name}: {name}")
    assert not offenders, offenders
