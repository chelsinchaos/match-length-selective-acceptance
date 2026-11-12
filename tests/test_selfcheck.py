# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LogNovations Holdings LLC
"""The consistency baseline's parts that need no model weights."""
from __future__ import annotations

import pytest

from selfref.selfcheck import CHECKPOINT, REVISION, mean_contradiction, pairs_for


def test_the_pinned_checkpoint():
    assert CHECKPOINT == "potsawee/deberta-v3-large-mnli"
    assert REVISION == "83f4359862aba52c07506cd23d4454db77945b36"


def test_pairs_put_the_sentence_first_as_the_reference_code_does():
    assert pairs_for("He was born in 1900.", ["Sample one.", "Sample two."]) == [
        ("He was born in 1900.", "Sample one."), ("He was born in 1900.", "Sample two.")]
    assert pairs_for("x", []) == []


def test_sentence_score_is_the_mean_over_samples():
    assert mean_contradiction([0.2, 0.4, 0.9]) == pytest.approx(0.5)
    with pytest.raises(ValueError):
        mean_contradiction([])
