# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LogNovations Holdings LLC
"""Brute-force definitions used only by tests.

These are the definitions in the plainest form that can be read against the
text of the pre-registration. They are quadratic or worse and exist so that the
suffix-automaton implementations in ``selfref.matchlen`` have something
independent to be checked against.
"""
from __future__ import annotations

from typing import Sequence


def contains(haystack: Sequence[int], needle: Sequence[int]) -> bool:
    """True when ``needle`` is a contiguous subsequence of ``haystack``."""
    h, n = tuple(haystack), tuple(needle)
    if not n:
        return True
    return any(h[p:p + len(n)] == n for p in range(len(h) - len(n) + 1))


def brute_profile(target: Sequence[int],
                  references: Sequence[Sequence[int]]) -> list[int]:
    """L[i] = longest l such that target[i:i+l] lies inside ONE reference."""
    t = tuple(target)
    out = []
    for i in range(len(t)):
        best = 0
        for length in range(1, len(t) - i + 1):
            if any(contains(r, t[i:i + length]) for r in references):
                best = length
            else:
                break          # occurrence is monotone in the length
        out.append(best)
    return out


def brute_self_profile(target: Sequence[int]) -> list[int]:
    """L[i] = longest l such that target[i:i+l] lies inside target[:i]."""
    t = tuple(target)
    return [brute_profile(t[i:], [t[:i]])[0] if i < len(t) else 0
            for i in range(len(t))]
