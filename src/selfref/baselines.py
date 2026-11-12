# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LogNovations Holdings LLC
"""Lexical baselines: scores that existing practice would try first.

These are baselines, not the features under test. They take the same inputs as
the features (the output and its source, as bytes) and see nothing else.

* Length: bytes, words and sentences of the output.
* N-gram novelty, n = 1..4: the fraction of the output's word n-gram
  occurrences that do not occur anywhere in the source.
* ROUGE-L against the source (Lin, 2004): with LCS the length of the longest
  common subsequence of the two word sequences, precision = LCS / |output|,
  recall = LCS / |source|, and F their harmonic mean.

Words are maximal runs of ``[a-z0-9]`` after lower-casing, with no stemming.
Bytes that are not valid UTF-8 are replaced before tokenising.
"""
from __future__ import annotations

import re
from typing import Sequence

import numpy as np

__all__ = ["word_tokens", "lcs_length", "rouge_l", "novel_ngram_rate", "lexical_baselines"]

_WORD = re.compile(r"[a-z0-9]+")
_SENTENCE_END = re.compile(r"[.!?]+(?:\s|$)")


def word_tokens(text: str) -> list[str]:
    """Lower-cased alphanumeric runs."""
    return _WORD.findall(text.lower())


def lcs_length(a: Sequence, b: Sequence) -> int:
    """Length of the longest common subsequence of two sequences.

    Row-by-row dynamic programming with each row computed by array operations:
    row[j] = max(previous[j], previous[j-1] + 1 where the items match), followed
    by a running maximum, which supplies the row[j-1] term.
    """
    if not a or not b:
        return 0
    codes: dict = {}
    short, long = (a, b) if len(a) <= len(b) else (b, a)
    long_ids = np.fromiter((codes.setdefault(x, len(codes)) for x in long),
                           dtype=np.int64, count=len(long))
    previous = np.zeros(len(long) + 1, dtype=np.int64)
    for item in short:
        code = codes.get(item)
        if code is None:
            continue                                   # matches nothing: the row is unchanged
        candidate = np.where(long_ids == code, previous[:-1] + 1, 0)
        previous[1:] = np.maximum.accumulate(np.maximum(previous[1:], candidate))
    return int(previous[-1])


def _rouge_l_tokens(candidate: Sequence[str], reference: Sequence[str]) -> dict[str, float]:
    if not candidate or not reference:
        return {"precision": 0.0, "recall": 0.0, "f": 0.0}
    lcs = lcs_length(candidate, reference)
    precision, recall = lcs / len(candidate), lcs / len(reference)
    f = 2 * precision * recall / (precision + recall) if lcs else 0.0
    return {"precision": precision, "recall": recall, "f": f}


def rouge_l(candidate: str, reference: str) -> dict[str, float]:
    """ROUGE-L precision, recall and F of ``candidate`` against ``reference``."""
    return _rouge_l_tokens(word_tokens(candidate), word_tokens(reference))


def _novel_rate(output: Sequence[str], source: Sequence[str], n: int) -> float:
    grams = [tuple(output[i:i + n]) for i in range(len(output) - n + 1)]
    if not grams:
        return 0.0
    known = {tuple(source[i:i + n]) for i in range(len(source) - n + 1)}
    return sum(1 for gram in grams if gram not in known) / len(grams)


def novel_ngram_rate(output: str, source: str, n: int) -> float:
    """Fraction of the output's word n-gram occurrences absent from the source;
    0.0 when the output has no n-gram of that order."""
    return _novel_rate(word_tokens(output), word_tokens(source), n)


def lexical_baselines(output: bytes, source: bytes) -> dict[str, float]:
    """All lexical baselines of one item, keyed ``"base.<name>"``."""
    out_text = output.decode("utf-8", errors="replace")
    out_tokens = word_tokens(out_text)
    src_tokens = word_tokens(source.decode("utf-8", errors="replace"))
    scores = {
        "base.len_bytes": float(len(output)),
        "base.len_words": float(len(out_tokens)),
        "base.len_sentences": float(len(_SENTENCE_END.findall(out_text))),
    }
    for n in (1, 2, 3, 4):
        scores[f"base.novel_{n}gram"] = _novel_rate(out_tokens, src_tokens, n)
    for name, value in _rouge_l_tokens(out_tokens, src_tokens).items():
        scores[f"base.rougeL_{name}"] = float(value)
    return scores
