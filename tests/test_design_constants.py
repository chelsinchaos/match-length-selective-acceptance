# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LogNovations Holdings LLC
"""Constants of the design that the pre-registration and the report state as numbers.

They follow from the sample sizes, the grid and delta alone. They are recomputed
here from the calibrator package's own bound, so the typed numbers are checked.
"""
from __future__ import annotations

import pytest

from selfref.certify import GRID_POINTS
from selfref.selcert_pin import load_selcert

SELCERT = load_selcert()
DELTA, GENERATORS = 0.05, 6


def bound_with_no_error(accepted: int, delta: float) -> float:
    """The calibrator's upper bound on the risk when ``accepted`` items are accepted and none is an error."""
    return float(SELCERT.clopper_pearson_upper(0, accepted, 1.0 - delta / GRID_POINTS))


@pytest.mark.parametrize("alpha, smallest", [(0.10, 74), (0.05, 152), (0.01, 775)])
def test_the_smallest_error_free_set_that_one_generator_can_certify(alpha, smallest):
    per_generator = DELTA / GENERATORS
    assert bound_with_no_error(smallest, per_generator) <= alpha
    assert bound_with_no_error(smallest - 1, per_generator) > alpha


def test_fifty_two_error_free_sentences_cannot_be_certified_at_any_alpha_of_the_study():
    bound = bound_with_no_error(52, DELTA)
    assert bound == pytest.approx(0.1088, abs=5e-5)
    assert bound > 0.10
