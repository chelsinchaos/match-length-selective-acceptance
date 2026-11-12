# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LogNovations Holdings LLC
"""The groundedness baseline's text handling.

These tests cover the parts that do not need model weights: sentence splitting,
document chunking and the summary statistics. The chunking follows MiniCheck's
reference inference code, so its quirks are pinned here.
"""
from __future__ import annotations

import pytest

from selfref.groundedness import (CHECKPOINT, CHUNK_TOKENS, MAX_MODEL_LEN, REVISION,
                                  chunk_document, claims_of, sentences_with_newlines,
                                  summarise_support)


def words(text: str) -> int:
    """A stand-in tokenizer for the tests: one token per whitespace word."""
    return len(text.split())


def test_the_pinned_model_and_its_reference_settings():
    assert CHECKPOINT == "lytang/MiniCheck-DeBERTa-v3-Large"
    assert REVISION == "2f2d01a54fa022a7ffadb76260e1ea8bc88c82bb"
    assert (CHUNK_TOKENS, MAX_MODEL_LEN) == (400, 2048)


def test_sentences_keep_a_marker_between_lines():
    assert sentences_with_newlines("One. Two.\nThree.") == ["One.", "Two.", "\n", "Three."]
    assert sentences_with_newlines("") == []


def test_claims_are_the_sentences_without_line_markers_or_blanks():
    assert claims_of("One. Two!\n\n  \nThree?") == ["One.", "Two!", "Three?"]
    assert claims_of("") == []
    assert claims_of("no terminal punctuation") == ["no terminal punctuation"]


def test_chunks_fill_up_to_the_budget_and_never_split_a_sentence():
    sentences = ["a a a", "b b", "c c c c", "d"]           # 3, 2, 4, 1 tokens
    assert chunk_document(sentences, words, chunk_tokens=5) == ["a a a b b", "c c c c d"]


def test_a_sentence_that_would_overflow_starts_a_new_chunk():
    sentences = ["a a a", "b b b", "c"]                      # 3 + 3 > 5
    assert chunk_document(sentences, words, chunk_tokens=5) == ["a a a", "b b b c"]


def test_an_oversized_first_sentence_does_not_leave_an_empty_chunk():
    # The reference code yields an empty first chunk here and then filters it out.
    assert chunk_document(["a a a a a a a", "b"], words, chunk_tokens=5) == ["a a a a a a a", "b"]


def test_line_markers_become_real_newlines_inside_a_chunk():
    sentences = sentences_with_newlines("One two.\nThree four.")
    assert chunk_document(sentences, words, chunk_tokens=50) == ["One two.\nThree four."]


def test_an_empty_document_has_no_chunks():
    assert chunk_document([], words, chunk_tokens=5) == []


def test_summary_of_support_probabilities():
    out = summarise_support([0.9, 0.2, 0.6, 0.4])
    assert out == {"gnd.min": pytest.approx(0.2), "gnd.mean": pytest.approx(0.525),
                   "gnd.frac_unsupported": pytest.approx(0.5)}


def test_an_output_with_no_claims_is_vacuously_supported():
    assert summarise_support([]) == {"gnd.min": 1.0, "gnd.mean": 1.0, "gnd.frac_unsupported": 0.0}
