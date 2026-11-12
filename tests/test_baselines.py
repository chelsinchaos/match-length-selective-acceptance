# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LogNovations Holdings LLC
"""Lexical baselines: length, n-gram novelty and ROUGE-L against the source."""
from __future__ import annotations

import math

import pytest
from hypothesis import given, settings, strategies as st

from selfref.baselines import (lcs_length, lexical_baselines, novel_ngram_rate, rouge_l,
                               word_tokens)

token_lists = st.lists(st.sampled_from(["a", "b", "c", "d"]), max_size=25)


def plain_lcs(a, b):
    table = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i, x in enumerate(a, 1):
        for j, y in enumerate(b, 1):
            table[i][j] = table[i - 1][j - 1] + 1 if x == y else max(table[i - 1][j], table[i][j - 1])
    return table[len(a)][len(b)]


# ---------------------------------------------------------------------- tokens

def test_tokens_are_lowercased_alphanumeric_runs():
    assert word_tokens("The store's hours: 9:0-22:30, Mon–Fri!") == \
        ["the", "store", "s", "hours", "9", "0", "22", "30", "mon", "fri"]
    assert word_tokens("") == []


# ------------------------------------------------------------------------- LCS

def test_lcs_hand_worked_examples():
    assert lcs_length(list("abcbdab"), list("bdcaba")) == 4          # the textbook example
    assert lcs_length(list("abc"), list("abc")) == 3
    assert lcs_length(list("abc"), list("xyz")) == 0
    assert lcs_length([], list("abc")) == 0
    assert lcs_length(list("abc"), []) == 0


@settings(max_examples=400, deadline=None)
@given(token_lists, token_lists)
def test_lcs_equals_plain_dynamic_programming(a, b):
    assert lcs_length(a, b) == plain_lcs(a, b)


# --------------------------------------------------------------------- ROUGE-L

def test_rouge_l_hand_worked_example():
    # candidate: the cat sat        reference: the cat was on the mat   LCS = "the cat" = 2
    scores = rouge_l("the cat sat", "the cat was on the mat")
    assert scores["precision"] == pytest.approx(2 / 3)        # LCS / candidate length
    assert scores["recall"] == pytest.approx(2 / 6)           # LCS / reference length
    assert scores["f"] == pytest.approx(2 * (2 / 3) * (2 / 6) / (2 / 3 + 2 / 6))


def test_rouge_l_closed_forms():
    assert rouge_l("a b c", "a b c") == {"precision": 1.0, "recall": 1.0, "f": 1.0}
    assert rouge_l("a b c", "x y z") == {"precision": 0.0, "recall": 0.0, "f": 0.0}
    assert rouge_l("", "a b") == {"precision": 0.0, "recall": 0.0, "f": 0.0}
    assert rouge_l("a b", "") == {"precision": 0.0, "recall": 0.0, "f": 0.0}


# ------------------------------------------------------------- n-gram novelty

def test_novel_ngram_rate_hand_worked_example():
    # output bigrams: (the cat) (cat sat) (sat down); the source has only (the cat)
    assert novel_ngram_rate("the cat sat down", "see the cat run", 2) == pytest.approx(2 / 3)
    assert novel_ngram_rate("the cat sat down", "see the cat run", 1) == pytest.approx(2 / 4)


def test_novel_ngram_rate_closed_forms():
    assert novel_ngram_rate("a b c d", "a b c d", 3) == 0.0
    assert novel_ngram_rate("a b c d", "w x y z", 1) == 1.0
    assert novel_ngram_rate("a b", "a b", 4) == 0.0            # no 4-grams in the output: defined as 0
    assert novel_ngram_rate("", "a b", 1) == 0.0


# ------------------------------------------------------------------- assembly

def test_lexical_baselines_keys_and_values():
    out = lexical_baselines(b"The store opens at nine. It closes at ten!",
                            b"hours: the store opens at nine and closes late")
    assert set(out) == {"base.len_bytes", "base.len_words", "base.len_sentences",
                        "base.novel_1gram", "base.novel_2gram", "base.novel_3gram",
                        "base.novel_4gram", "base.rougeL_precision", "base.rougeL_recall",
                        "base.rougeL_f"}
    assert out["base.len_bytes"] == 42.0
    assert out["base.len_words"] == 9.0
    assert out["base.len_sentences"] == 2.0
    assert out["base.novel_1gram"] == pytest.approx(2 / 9)        # "it" and "ten" are not in the source


def test_lexical_baselines_accept_bytes_that_are_not_valid_utf8():
    out = lexical_baselines(b"caf\xff\xfe latte", b"caf latte")
    assert all(isinstance(v, float) and math.isfinite(v) for v in out.values())


@settings(max_examples=100, deadline=None)
@given(st.binary(max_size=120), st.binary(max_size=200))
def test_every_baseline_is_a_finite_float_in_range(output, source):
    out = lexical_baselines(output, source)
    assert all(isinstance(v, float) and math.isfinite(v) for v in out.values())
    for name, value in out.items():
        if "novel" in name or "rouge" in name:
            assert 0.0 <= value <= 1.0
