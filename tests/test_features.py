# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LogNovations Holdings LLC
"""Features derived from a match-length profile."""
from __future__ import annotations

import math

import pytest
from hypothesis import given, settings, strategies as st

from selfref.features import (DYADIC_SCALES, dyadic_coverage, longest_match,
                              match_length_entropy, profile_features,
                              rlz_phrase_count, rlz_phrases,
                              self_match_length_entropy)
from selfref.matchlen import match_length_profile

profiles = st.lists(st.integers(min_value=0, max_value=40), max_size=60)


def brute_coverage(profile, k):
    n = len(profile)
    if n == 0:
        return 0.0
    covered = sum(
        1 for j in range(n)
        if any(profile[i] >= k and i <= j < i + profile[i] for i in range(j + 1)))
    return covered / n


# ---------------------------------------------------------------------- scales

def test_scales_are_the_eleven_powers_of_two_up_to_1024():
    assert DYADIC_SCALES == (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024)


# -------------------------------------------------------------------- coverage

def test_coverage_hand_worked_example():
    # L = [3, 2, 1, 0, 1], n = 5
    # k=1: starts 0,1,2,4 qualify -> positions {0,1,2,4} covered -> 4/5
    # k=2: starts 0,1 qualify     -> positions {0,1,2} covered   -> 3/5
    # k=4: nothing qualifies      -> 0
    assert dyadic_coverage([3, 2, 1, 0, 1], scales=(1, 2, 4)) == [0.8, 0.6, 0.0]


def test_coverage_when_target_equals_reference():
    n = 20
    profile = match_length_profile(b"x" * 7 + b"abcdefghijklm", [b"x" * 7 + b"abcdefghijklm"])
    assert profile == [n - i for i in range(n)]
    cov = dyadic_coverage(profile)
    assert cov == [1.0 if k <= n else 0.0 for k in DYADIC_SCALES]


def test_coverage_with_disjoint_alphabets_is_zero_at_every_scale():
    assert dyadic_coverage([0, 0, 0, 0]) == [0.0] * len(DYADIC_SCALES)


def test_coverage_of_empty_profile_is_zero_at_every_scale():
    assert dyadic_coverage([]) == [0.0] * len(DYADIC_SCALES)


def test_scale_larger_than_the_output_gives_zero_not_an_error():
    assert dyadic_coverage([2, 1], scales=(1024,)) == [0.0]


@settings(max_examples=300, deadline=None)
@given(profiles, st.sampled_from([1, 2, 4, 8, 16]))
def test_coverage_equals_brute_force_definition(profile, k):
    assert dyadic_coverage(profile, scales=(k,)) == [brute_coverage(profile, k)]


@settings(max_examples=200, deadline=None)
@given(profiles)
def test_coverage_is_non_increasing_in_scale_and_within_unit_interval(profile):
    cov = dyadic_coverage(profile)
    assert all(0.0 <= c <= 1.0 for c in cov)
    assert all(a >= b for a, b in zip(cov, cov[1:]))


# ------------------------------------------------- relative Lempel-Ziv parsing

def test_rlz_hand_worked_example():
    # L = [3, 2, 1, 0, 1]: phrase of 3, then a literal (no match), then a phrase of 1
    assert rlz_phrases([3, 2, 1, 0, 1]) == [3, 1, 1]
    assert rlz_phrase_count([3, 2, 1, 0, 1]) == 3


def test_rlz_reproduces_the_worked_example_of_the_1993_article():
    """Ziv and Merhav (1993), Section II, p. 1271: with z = 01111000110 and
    x = 10010100110, "parsing z with respect to x yields: (011, 110, 00110)".
    The article prints the count beside it as 2; the phrases it lists are three."""
    z, x = b"01111000110", b"10010100110"
    lengths = rlz_phrases(match_length_profile(z, [x]))
    cuts = [sum(lengths[:k]) for k in range(len(lengths) + 1)]
    assert [z[a:b] for a, b in zip(cuts, cuts[1:])] == [b"011", b"110", b"00110"]
    assert rlz_phrase_count(match_length_profile(z, [x])) == 3


def test_rlz_is_one_phrase_when_target_equals_reference():
    assert rlz_phrase_count([5, 4, 3, 2, 1]) == 1


def test_rlz_is_one_literal_per_symbol_with_disjoint_alphabets():
    assert rlz_phrase_count([0, 0, 0, 0]) == 4


def test_rlz_of_empty_profile_is_zero_phrases():
    assert rlz_phrases([]) == []
    assert rlz_phrase_count([]) == 0


@settings(max_examples=200, deadline=None)
@given(st.binary(max_size=40), st.binary(max_size=40))
def test_rlz_phrases_tile_the_target_exactly(target, reference):
    profile = match_length_profile(target, [reference])
    phrases = rlz_phrases(profile)
    assert sum(phrases) == len(target)
    assert all(p >= 1 for p in phrases)


# ------------------------------------------------------ longest common substring

def test_longest_match_closed_forms():
    assert longest_match([5, 4, 3, 2, 1]) == 5
    assert longest_match([0, 0, 0]) == 0
    assert longest_match([]) == 0


# ------------------------------------------------------- match-length entropy

def test_entropy_with_no_matches_is_log2_of_reference_length():
    # n * log2(N) / sum(L + 1) = 4 * log2(16) / 4
    assert match_length_entropy([0, 0, 0, 0], reference_length=16) == pytest.approx(4.0)


def test_entropy_hand_worked_example():
    # n = 4, N = 4, sum(L + 1) = 4 + 3 + 2 + 1 = 10  ->  4 * 2 / 10
    assert match_length_entropy([3, 2, 1, 0], reference_length=4) == pytest.approx(0.8)


@pytest.mark.parametrize("profile, n_ref", [([], 100), ([1, 0], 0), ([1, 0], 1)])
def test_entropy_degenerate_cases_are_zero(profile, n_ref):
    assert match_length_entropy(profile, reference_length=n_ref) == 0.0


def test_self_entropy_hand_worked_example():
    # Kontoyiannis et al. (1998), Theorem 1(b): (1/n) * sum_{i=2}^{n} Lambda_i / log2(i),
    # with Lambda_i = L[i] + 1 and n the last position index. Four symbols -> n = 3.
    # Terms: 1/log2(2) at i=2 and 1/log2(3) at i=3; normalised by n = 3, not by the 2 terms.
    expected = 1.0 / ((1.0 / math.log2(2) + 1.0 / math.log2(3)) / 3.0)
    assert self_match_length_entropy([0, 0, 0, 0]) == pytest.approx(expected)


def test_self_entropy_second_hand_worked_example():
    # L = [0, 0, 0, 3, 2, 1] (the profile of b"abcabc"), n = 5
    # sum = 1/log2(2) + 4/log2(3) + 3/log2(4) + 2/log2(5)
    total = 1 / math.log2(2) + 4 / math.log2(3) + 3 / math.log2(4) + 2 / math.log2(5)
    assert self_match_length_entropy([0, 0, 0, 3, 2, 1]) == pytest.approx(5 / total)


@pytest.mark.parametrize("profile", [[], [0], [0, 1]])
def test_self_entropy_needs_three_symbols(profile):
    assert self_match_length_entropy(profile) == 0.0


# ------------------------------------------------------------ feature assembly

def test_profile_features_keys_for_a_fixed_reference():
    feats = profile_features([3, 2, 1, 0, 1], prefix="source", reference_length=50)
    expected = {f"source.cov_k{k}" for k in DYADIC_SCALES} | {
        "source.rlz_phrases", "source.rlz_phrases_per_symbol",
        "source.lcs", "source.lcs_frac", "source.mlen_entropy"}
    assert set(feats) == expected
    assert feats["source.rlz_phrases"] == 3.0
    assert feats["source.rlz_phrases_per_symbol"] == pytest.approx(3 / 5)
    assert feats["source.lcs"] == 3.0
    assert feats["source.lcs_frac"] == pytest.approx(3 / 5)
    assert feats["source.mlen_entropy"] == pytest.approx(
        match_length_entropy([3, 2, 1, 0, 1], reference_length=50))


def test_profile_features_without_reference_length_uses_the_self_estimator():
    profile = [0, 0, 0, 3, 2, 1]
    feats = profile_features(profile, prefix="self")
    assert feats["self.mlen_entropy"] == pytest.approx(self_match_length_entropy(profile))


def test_profile_features_of_empty_profile_are_all_zero():
    feats = profile_features([], prefix="source", reference_length=10)
    assert feats and all(v == 0.0 for v in feats.values())


@settings(max_examples=200, deadline=None)
@given(profiles, st.one_of(st.none(), st.integers(min_value=0, max_value=10_000)))
def test_every_feature_is_a_finite_float(profile, reference_length):
    feats = profile_features(profile, prefix="p", reference_length=reference_length)
    assert all(isinstance(v, float) and math.isfinite(v) for v in feats.values())
