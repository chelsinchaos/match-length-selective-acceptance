# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LogNovations Holdings LLC
"""The decision rule of the study, as fixed before any data was read.

All quantities are in percentage points of certified coverage.

    c_B    BASELINE certified coverage in the DEV rehearsal
    BAR    the smaller of 5 points and 10% of (100 - c_B)
    FLOOR  the larger of 2 points and the 99th percentile of the
           baseline-versus-baseline (A/A) gap
    if BAR is below FLOOR, BAR is set equal to FLOOR

    PASS          the point estimate is at or above BAR and the entire 95%
                  interval is above FLOOR
    FAIL          the upper end of the 95% interval is below BAR
    INCONCLUSIVE  anything else

Two readings are fixed here because the rule's text leaves them open. The A/A
gap is taken in absolute value, which can only raise FLOOR. "Above" and
"below" are strict.
"""
from __future__ import annotations

import numpy as np

__all__ = ["compute_floor", "compute_bar", "decide"]


def compute_floor(aa_gaps) -> float:
    """max(2, 99th percentile of the absolute A/A coverage gap), in points."""
    return max(2.0, float(np.percentile(np.abs(np.asarray(aa_gaps, dtype=float)), 99)))


def compute_bar(c_b: float, floor: float) -> float:
    """min(5, 10% of (100 - c_B)), raised to FLOOR if it falls below it."""
    if not 0.0 <= c_b <= 100.0:
        raise ValueError("c_B is a coverage in percentage points, between 0 and 100")
    return max(min(5.0, 0.10 * (100.0 - c_b)), floor)


def decide(point: float, low: float, high: float, bar: float, floor: float) -> str:
    """PASS, FAIL or INCONCLUSIVE for a coverage gain and its 95% interval."""
    if low > high:
        raise ValueError("the interval's lower end is above its upper end")
    if point >= bar and low > floor:
        return "PASS"
    if high < bar:
        return "FAIL"
    return "INCONCLUSIVE"
