# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LogNovations Holdings LLC
"""The match-length profile over word tokens instead of bytes."""
from __future__ import annotations

import pytest

from selfref.tokens import token_ids, token_profile_features

PROFILE_COLUMNS = 16


def test_tokens_share_one_code_table_across_the_texts():
    out, src = token_ids("The cat sat.", "A cat; the mat")
    assert out == [0, 1, 2] and src == [3, 1, 0, 4]


def test_there_are_sixteen_columns_for_each_of_the_two_frames():
    feats = token_profile_features(b"one two three", b"one two four")
    assert sum(name.startswith("tok.source.") for name in feats) == PROFILE_COLUMNS
    assert sum(name.startswith("tok.self.") for name in feats) == PROFILE_COLUMNS
    assert len(feats) == 2 * PROFILE_COLUMNS


def test_a_worked_example_against_the_source():
    # output tokens a b c a b, source tokens a b x: L = [2, 1, 0, 2, 1]
    feats = token_profile_features(b"a b c a b", b"a b x")
    assert feats["tok.source.lcs"] == 2.0
    assert feats["tok.source.cov_k1"] == pytest.approx(4 / 5)
    assert feats["tok.source.cov_k2"] == pytest.approx(4 / 5)
    assert feats["tok.source.cov_k4"] == 0.0
    assert feats["tok.source.rlz_phrases"] == 3.0          # (a b) (c) (a b)


def test_a_worked_example_against_the_outputs_own_prefix():
    feats = token_profile_features(b"a b a b", b"")        # L = [0, 0, 2, 1]
    assert feats["tok.self.lcs"] == 2.0
    assert feats["tok.self.cov_k2"] == pytest.approx(2 / 4)
    assert feats["tok.source.cov_k1"] == 0.0               # an empty source matches nothing


def test_case_and_punctuation_do_not_matter():
    feats = token_profile_features(b"The CAT, sat!", b"the cat sat")
    assert feats["tok.source.cov_k1"] == 1.0 and feats["tok.source.lcs"] == 3.0


def test_disjoint_vocabularies_match_nothing_and_an_empty_output_is_all_zeros():
    feats = token_profile_features(b"alpha beta gamma", b"one two three")
    assert feats["tok.source.cov_k1"] == 0.0 and feats["tok.source.rlz_phrases"] == 3.0
    assert set(token_profile_features(b"", b"one two").values()) == {0.0}
