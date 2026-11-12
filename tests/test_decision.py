# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LogNovations Holdings LLC
"""The decision rule, exactly as fixed in the brief.

    BAR   = the smaller of 5 points and 10% of (100 - c_B)
    FLOOR = the larger of 2 points and the 99th percentile of the A/A gap
    if BAR < FLOOR: BAR = FLOOR

    PASS          point estimate >= BAR and the whole 95% interval above FLOOR
    FAIL          upper end of the interval below BAR
    INCONCLUSIVE  anything else
"""
from __future__ import annotations

import numpy as np
import pytest

from selfref.decision import compute_bar, compute_floor, decide


# ----------------------------------------------------------------------- FLOOR

def test_floor_is_two_points_when_the_aa_gap_is_small():
    assert compute_floor(np.array([0.1, -0.3, 0.0, 0.5])) == 2.0


def test_floor_is_the_99th_percentile_of_the_absolute_gap_when_that_is_larger():
    gaps = np.concatenate([np.zeros(980), np.full(20, -6.0)])      # 2% of runs differ by 6 points
    assert np.percentile(np.abs(gaps), 99) == pytest.approx(6.0)
    assert compute_floor(gaps) == pytest.approx(6.0)
    # with only 1% of runs that far apart, the 99th percentile sits below them and 2 points governs
    assert compute_floor(np.concatenate([np.zeros(990), np.full(10, -6.0)])) == 2.0


def test_floor_uses_the_size_of_the_gap_not_its_sign():
    assert compute_floor(np.full(100, -7.0)) == pytest.approx(7.0)


# ------------------------------------------------------------------------- BAR

@pytest.mark.parametrize("c_b, floor, expected", [
    (0.0, 2.0, 5.0),        # 10% of 100 = 10, capped at 5
    (40.0, 2.0, 5.0),       # 10% of 60 = 6, capped at 5
    (50.0, 2.0, 5.0),       # 10% of 50 = 5
    (70.0, 2.0, 3.0),       # 10% of 30 = 3
    (90.0, 2.0, 2.0),       # 10% of 10 = 1, raised to the floor
    (70.0, 4.0, 4.0),       # 3 is below the floor of 4
    (0.0, 6.5, 6.5),        # the floor can exceed 5
])
def test_bar(c_b, floor, expected):
    assert compute_bar(c_b, floor) == pytest.approx(expected)


def test_bar_rejects_a_coverage_outside_zero_to_one_hundred():
    with pytest.raises(ValueError):
        compute_bar(120.0, 2.0)


# -------------------------------------------------------------------- outcomes

BAR, FLOOR = 5.0, 2.0


@pytest.mark.parametrize("point, low, high, expected", [
    (6.0, 2.5, 9.0, "PASS"),            # at or above BAR, whole interval above FLOOR
    (5.0, 2.1, 8.0, "PASS"),            # the point estimate may equal BAR
    (6.0, 2.0, 9.0, "INCONCLUSIVE"),    # the interval must be strictly above FLOOR
    (6.0, 1.0, 9.0, "INCONCLUSIVE"),    # large estimate, interval reaches below FLOOR
    (4.0, 2.5, 6.0, "INCONCLUSIVE"),    # below BAR but the interval reaches BAR
    (3.0, 2.2, 4.9, "FAIL"),            # a real effect that is too small
    (0.0, 0.0, 0.0, "FAIL"),            # neither arm certifies anything
    (-1.0, -3.0, 1.0, "FAIL"),
    (4.0, 1.0, 5.0, "INCONCLUSIVE"),    # the upper end equals BAR: not below it
])
def test_outcomes(point, low, high, expected):
    assert decide(point, low, high, BAR, FLOOR) == expected


def test_an_interval_with_its_ends_swapped_is_rejected():
    with pytest.raises(ValueError):
        decide(6.0, 9.0, 7.0, BAR, FLOOR)


def test_the_rule_reads_the_three_numbers_separately():
    # A percentile interval need not contain the point estimate. The rule is applied as written.
    assert decide(6.0, 6.5, 9.0, BAR, FLOOR) == "PASS"
    assert decide(4.9, 5.5, 9.0, BAR, FLOOR) == "INCONCLUSIVE"
