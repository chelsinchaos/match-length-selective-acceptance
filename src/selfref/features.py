# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LogNovations Holdings LLC
"""Features derived from a match-length profile.

Every function here takes a profile L (see ``selfref.matchlen``) and nothing
else about the item, apart from the total reference length where a formula
needs it. All of them are defined for an empty profile and return finite
values, so a feature table never contains NaN.

Definitions

* Dyadic coverage at scale k: the fraction of positions j for which some
  start i <= j has L[i] >= k and i + L[i] > j. It is the share of the target
  that lies inside a match at least k symbols long.
* Relative Lempel-Ziv parse (the Ziv-Merhav cross-parse): starting at i = 0,
  take a phrase of length max(L[i], 1) and advance by it. A symbol that occurs
  nowhere in the reference is a phrase of length one.
* Longest common substring: max(L).
* Match-length entropy estimate in the SELF frame, where the reference at
  position i is the prefix of length i. This is estimator (b) of Theorem 1 in
  Kontoyiannis, Algoet, Suhov and Wyner (1998):

      1 / H  ~  (1/n) * sum_{i=2}^{n} Lambda_i / log2(i),   Lambda_i = L[i] + 1,

  where Lambda_i is the length of the shortest substring starting at i that
  does not appear within the previous i symbols, and n is the index of the
  last position. Logarithms are to base 2, as in the paper.
* Match-length entropy estimate against a fixed reference of total length N:

      1 / H  ~  (1/m) * sum_i (L[i] + 1) / log2(N),   m = len(L).

  This has the form of estimator (a) of the same theorem with the window of
  previous symbols replaced by the reference. The paper proves consistency for
  matches within one sequence; nothing is claimed here about what this
  quantity estimates for two different sequences. It is used as a feature.

Sources, and how each formula was checked against them, are listed in
REFERENCES.md at the top of the harness.
"""
from __future__ import annotations

import math
from typing import Sequence

__all__ = ["DYADIC_SCALES", "dyadic_coverage", "rlz_phrases", "rlz_phrase_count",
           "longest_match", "match_length_entropy", "self_match_length_entropy",
           "profile_features"]

#: The fixed scales k = 1, 2, 4, ..., 1024.
DYADIC_SCALES: tuple[int, ...] = tuple(2 ** m for m in range(11))


def dyadic_coverage(profile: Sequence[int],
                    scales: Sequence[int] = DYADIC_SCALES) -> list[float]:
    """Fraction of positions covered by a match of length >= k, for each k in ``scales``."""
    n = len(profile)
    if n == 0:
        return [0.0] * len(scales)
    out = []
    for k in scales:
        reach = 0            # one past the last position covered by a qualifying match so far
        covered = 0
        for i, length in enumerate(profile):
            if length >= k and i + length > reach:
                reach = i + length
            if i < reach:
                covered += 1
        out.append(covered / n)
    return out


def rlz_phrases(profile: Sequence[int]) -> list[int]:
    """Phrase lengths of the greedy relative Lempel-Ziv parse. They sum to ``len(profile)``."""
    n = len(profile)
    out: list[int] = []
    i = 0
    while i < n:
        step = min(max(profile[i], 1), n - i)
        out.append(step)
        i += step
    return out


def rlz_phrase_count(profile: Sequence[int]) -> int:
    """Number of phrases in the greedy relative Lempel-Ziv parse."""
    return len(rlz_phrases(profile))


def longest_match(profile: Sequence[int]) -> int:
    """Length of the longest common substring of target and reference."""
    return max(profile, default=0)


def match_length_entropy(profile: Sequence[int], reference_length: int) -> float:
    """n * log2(N) / sum(L[i] + 1), in bits per symbol; 0.0 when n = 0 or N < 2."""
    n = len(profile)
    if n == 0 or reference_length < 2:
        return 0.0
    return n * math.log2(reference_length) / sum(length + 1 for length in profile)


def self_match_length_entropy(profile: Sequence[int]) -> float:
    """n / sum_{i=2}^{n} (L[i] + 1) / log2(i), with n the last position index;
    0.0 when the profile has fewer than three positions."""
    last = len(profile) - 1
    if last < 2:
        return 0.0
    return last / sum((profile[i] + 1) / math.log2(i) for i in range(2, last + 1))


def profile_features(profile: Sequence[int], *, prefix: str,
                     reference_length: int | None = None) -> dict[str, float]:
    """All profile-derived features, keyed ``"<prefix>.<name>"``.

    ``reference_length`` is the total length of the fixed reference. Leave it
    as ``None`` for the SELF frame, where the reference grows with the position.
    """
    n = len(profile)
    phrases = rlz_phrase_count(profile)
    lcs = longest_match(profile)
    if reference_length is None:
        entropy = self_match_length_entropy(profile)
    else:
        entropy = match_length_entropy(profile, reference_length)
    feats = {f"{prefix}.cov_k{k}": c
             for k, c in zip(DYADIC_SCALES, dyadic_coverage(profile))}
    feats[f"{prefix}.rlz_phrases"] = float(phrases)
    feats[f"{prefix}.rlz_phrases_per_symbol"] = phrases / n if n else 0.0
    feats[f"{prefix}.lcs"] = float(lcs)
    feats[f"{prefix}.lcs_frac"] = lcs / n if n else 0.0
    feats[f"{prefix}.mlen_entropy"] = float(entropy)
    return feats
