# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LogNovations Holdings LLC
"""Match-length profiles: the reference backend.

The primitive of this harness. For a target sequence T and a set of reference
sequences, the *match-length profile* is

    L[i] = the largest l such that T[i:i+l] occurs inside a single reference,

with L[i] = 0 when the symbol T[i] occurs in no reference. A match never spans
the boundary between two references.

The *SELF profile* uses the target's own prefix as the reference:

    L[i] = the largest l such that T[i:i+l] occurs inside T[:i].

The earlier occurrence must lie wholly inside the prefix, so it cannot overlap
position i.

Sequences are sequences of non-negative integers; ``bytes`` qualifies, and so
does a list of token ids. Everything here is pure Python with no dependency
outside the standard library, so that the definitions can be audited and the
results reproduced anywhere.
"""
from __future__ import annotations

import operator
from typing import Sequence

__all__ = ["SuffixAutomaton", "match_length_profile", "self_match_length_profile"]


class SuffixAutomaton:
    """Suffix automaton of a sequence that grows one symbol at a time.

    After ``extend`` has been called with s[0], ..., s[m-1], a walk from the
    root that follows a sequence of symbols succeeds exactly when that sequence
    is a contiguous subsequence of s. (Blumer et al., 1985.)
    """

    __slots__ = ("_next", "_link", "_len", "_last")

    def __init__(self) -> None:
        self._next: list[dict[int, int]] = [{}]   # transitions per state
        self._link: list[int] = [-1]              # suffix link per state
        self._len: list[int] = [0]                # longest string reaching the state
        self._last = 0

    def extend(self, symbol: int) -> None:
        """Append one symbol to the indexed sequence."""
        nxt, link, length = self._next, self._link, self._len
        cur = len(nxt)
        nxt.append({})
        length.append(length[self._last] + 1)
        link.append(0)
        p = self._last
        while p != -1 and symbol not in nxt[p]:
            nxt[p][symbol] = cur
            p = link[p]
        if p != -1:
            q = nxt[p][symbol]
            if length[p] + 1 == length[q]:
                link[cur] = q
            else:
                clone = len(nxt)
                nxt.append(dict(nxt[q]))
                length.append(length[p] + 1)
                link.append(link[q])
                while p != -1 and nxt[p].get(symbol) == q:
                    nxt[p][symbol] = clone
                    p = link[p]
                link[q] = clone
                link[cur] = clone
        self._last = cur

    def longest_prefix_match(self, seq: Sequence[int], start: int) -> int:
        """Length of the longest prefix of ``seq[start:]`` that occurs in the indexed sequence."""
        nxt = self._next
        state, i, n = 0, start, len(seq)
        while i < n:
            following = nxt[state].get(seq[i])
            if following is None:
                break
            state = following
            i += 1
        return i - start

    def longest_suffix_matches(self, seq: Sequence[int]) -> list[int]:
        """For each j, the length of the longest suffix of ``seq[:j+1]`` that occurs
        in the indexed sequence. Linear in ``len(seq)``."""
        nxt, link, length = self._next, self._link, self._len
        out: list[int] = []
        state, matched = 0, 0
        for symbol in seq:
            while state != 0 and symbol not in nxt[state]:
                state = link[state]
                matched = length[state]
            following = nxt[state].get(symbol)
            if following is None:
                state, matched = 0, 0
            else:
                state, matched = following, matched + 1
            out.append(matched)
        return out


def _symbols(seq: Sequence[int]) -> Sequence[int]:
    """Validate a symbol sequence: integers, none negative."""
    if isinstance(seq, (bytes, bytearray, memoryview)):
        return bytes(seq)
    out = [operator.index(s) for s in seq]
    if any(s < 0 for s in out):
        raise ValueError("symbols must be non-negative integers")
    return out


def match_length_profile(target: Sequence[int],
                         references: Sequence[Sequence[int]]) -> list[int]:
    """L[i] = length of the longest substring of ``target`` starting at i that
    occurs inside one of ``references``.

    Method. A substring starting at i in the target is a substring ending at
    n-1-i in the reversed target. So the reversed target is scanned through a
    suffix automaton of the reversed references, which yields the longest match
    *ending* at each position in one linear pass. References are joined by
    distinct negative separators; the target holds no negative symbol, so no
    match can cross from one reference into the next.
    """
    t = _symbols(target)
    refs = [_symbols(r) for r in references]
    if not t:
        return []
    automaton = SuffixAutomaton()
    for k, ref in enumerate(refs):
        if k:
            automaton.extend(-k)                 # unique separator, never in a target
        for symbol in reversed(ref):
            automaton.extend(symbol)
    ending_at = automaton.longest_suffix_matches(t[::-1])
    return ending_at[::-1]


def self_match_length_profile(target: Sequence[int]) -> list[int]:
    """L[i] = length of the longest substring of ``target`` starting at i that
    occurs inside ``target[:i]``.

    Method. The automaton holds exactly ``target[:i]`` when position i is
    queried, and is extended with ``target[i]`` afterwards. The cost is the sum
    of the match lengths, which is quadratic for a single repeated symbol and
    near-linear for ordinary text.
    """
    t = _symbols(target)
    automaton = SuffixAutomaton()
    out: list[int] = []
    for i, symbol in enumerate(t):
        out.append(automaton.longest_prefix_match(t, i))
        automaton.extend(symbol)
    return out
