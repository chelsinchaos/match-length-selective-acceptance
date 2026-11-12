# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LogNovations Holdings LLC
"""The match-length profile, reference backend.

L[i] is the length of the longest substring of the target starting at i that
occurs in a reference. Closed-form cases first, then agreement with the
brute-force definition on generated inputs.
"""
from __future__ import annotations

import pytest
from hypothesis import given, settings, strategies as st

from oracles import brute_profile, brute_self_profile
from selfref.matchlen import match_length_profile, self_match_length_profile

ALPHABETS = [b"ab", b"abcd", bytes(range(256))]


def strings(max_size: int):
    """Byte strings over a small, medium or full alphabet."""
    return st.sampled_from(ALPHABETS).flatmap(
        lambda a: st.lists(st.sampled_from(list(a)), max_size=max_size).map(bytes))


def same_alphabet_case(max_target: int, max_ref: int, max_refs: int = 3):
    """(target, references) drawn over one shared alphabet, so matches are common."""
    def build(a):
        s = lambda m: st.lists(st.sampled_from(list(a)), max_size=m).map(bytes)
        return st.tuples(s(max_target), st.lists(s(max_ref), min_size=1, max_size=max_refs))
    return st.sampled_from(ALPHABETS).flatmap(build)


# ---------------------------------------------------------------- closed forms

def test_target_equal_to_reference_matches_to_the_end():
    t = b"the cat sat on the mat"
    assert match_length_profile(t, [t]) == [len(t) - i for i in range(len(t))]


def test_disjoint_alphabets_give_all_zeros():
    assert match_length_profile(b"aaaa", [b"bbbb"]) == [0, 0, 0, 0]


def test_empty_target_gives_empty_profile():
    assert match_length_profile(b"", [b"abc"]) == []


def test_empty_reference_gives_zeros():
    assert match_length_profile(b"abc", [b""]) == [0, 0, 0]


def test_no_references_gives_zeros():
    assert match_length_profile(b"abc", []) == [0, 0, 0]


def test_hand_worked_example():
    # target  a b c x b c d      reference  z b c d a b
    # i=0 "ab" (ref[4:6]); i=1 "b"... "bc" then "bcx" fails -> 2; i=2 "c" -> 1;
    # i=3 "x" absent -> 0; i=4 "bcd" -> 3; i=5 "cd" -> 2; i=6 "d" -> 1
    assert match_length_profile(b"abcxbcd", [b"zbcdab"]) == [2, 2, 1, 0, 3, 2, 1]


def test_match_never_spans_two_references():
    # "ab" would exist only by gluing the end of one reference to the start of the next.
    assert match_length_profile(b"ab", [b"xa", b"by"]) == [1, 1]


def test_longest_match_is_taken_across_references():
    assert match_length_profile(b"abcd", [b"ab", b"abc", b"d"]) == [3, 2, 1, 1]


def test_arbitrary_byte_values_are_ordinary_symbols():
    t = bytes([0x00, 0xFF, 0x24, 0x80, 0x00])            # NUL, 0xFF, '$', high bit
    r = bytes([0x24, 0x80, 0x00, 0xFF, 0x24])
    assert match_length_profile(t, [r]) == brute_profile(t, [r])


def test_symbols_above_255_are_supported():
    target = [1000, 2000, 3000, 7]
    reference = [5, 1000, 2000, 9, 3000]
    assert match_length_profile(target, [reference]) == [2, 1, 1, 0]


def test_negative_symbol_in_target_is_rejected():
    with pytest.raises(ValueError):
        match_length_profile([1, -1, 2], [[1, 2]])


def test_negative_symbol_in_reference_is_rejected():
    with pytest.raises(ValueError):
        match_length_profile([1, 2], [[1, -5]])


# ------------------------------------------------------------------ SELF frame

def test_self_profile_hand_worked_example():
    # a b c a b c : nothing before position 3 repeats; at 3 "abc" lies in the prefix "abc"
    assert self_match_length_profile(b"abcabc") == [0, 0, 0, 3, 2, 1]


def test_self_profile_match_must_lie_wholly_in_the_prefix():
    # At i=1 the prefix is "a". "aa" starts at 0 but would overlap position 1,
    # so only "a" counts.
    assert self_match_length_profile(b"aaaa") == [0, 1, 2, 1]


def test_self_profile_of_a_long_single_letter_run():
    n = 2000
    assert self_match_length_profile(b"a" * n) == [min(i, n - i) for i in range(n)]


def test_self_profile_of_empty_and_single_symbol():
    assert self_match_length_profile(b"") == []
    assert self_match_length_profile(b"x") == [0]


def test_self_profile_rejects_negative_symbols():
    with pytest.raises(ValueError):
        self_match_length_profile([3, -2])


# ------------------------------------------------------------------ properties

@settings(max_examples=300, deadline=None)
@given(same_alphabet_case(max_target=24, max_ref=24))
def test_profile_equals_brute_force_definition(case):
    target, references = case
    assert match_length_profile(target, references) == brute_profile(target, references)


@settings(max_examples=150, deadline=None)
@given(strings(24), st.lists(strings(24), max_size=3))
def test_profile_equals_brute_force_on_unrelated_alphabets(target, references):
    assert match_length_profile(target, references) == brute_profile(target, references)


@settings(max_examples=300, deadline=None)
@given(strings(40))
def test_self_profile_equals_brute_force_definition(target):
    assert self_match_length_profile(target) == brute_self_profile(target)


@settings(max_examples=200, deadline=None)
@given(same_alphabet_case(max_target=60, max_ref=60))
def test_profile_drops_by_at_most_one_per_step(case):
    # If target[i:i+l] occurs in a reference, so does target[i+1:i+l].
    target, references = case
    profile = match_length_profile(target, references)
    assert all(b >= a - 1 for a, b in zip(profile, profile[1:]))


@settings(max_examples=200, deadline=None)
@given(same_alphabet_case(max_target=60, max_ref=60))
def test_profile_never_runs_past_the_end_of_the_target(case):
    target, references = case
    profile = match_length_profile(target, references)
    assert len(profile) == len(target)
    assert all(0 <= l <= len(target) - i for i, l in enumerate(profile))
