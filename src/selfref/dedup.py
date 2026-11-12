# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LogNovations Holdings LLC
"""Near-duplicate clustering of source texts.

Two texts are near-duplicates when the Jaccard similarity of their sets of
lower-cased word 5-grams is at least a threshold. Near-duplicate texts are
merged into clusters by transitive closure, and one member of each cluster is
kept, so that no near-identical source can sit on both sides of a split.

The comparison is exact: every pair of texts that shares at least one 5-gram is
compared. Candidate pairs come from an inverted index, which is what keeps the
cost manageable for a few thousand texts.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Iterable, Mapping

__all__ = ["shingles", "jaccard", "near_duplicate_clusters", "representatives"]


def shingles(text: str, n: int = 5) -> set[str]:
    """The set of lower-cased word n-grams of ``text``. A text with fewer than
    n words is a single shingle; an empty text has none."""
    words = text.lower().split()
    if not words:
        return set()
    if len(words) < n:
        return {" ".join(words)}
    return {" ".join(words[i:i + n]) for i in range(len(words) - n + 1)}


def jaccard(a: set, b: set) -> float:
    """|a & b| / |a | b|, and 0.0 when both are empty."""
    union = len(a | b)
    return len(a & b) / union if union else 0.0


def near_duplicate_clusters(texts: Mapping[str, str], threshold: float = 0.9,
                            n: int = 5) -> list[list[str]]:
    """Clusters of two or more identifiers whose texts are near-duplicates.

    Each cluster is sorted, and the clusters are sorted. Identifiers that have
    no near-duplicate do not appear.
    """
    keys = sorted(texts)
    sets = [shingles(texts[key], n) for key in keys]
    parent = list(range(len(keys)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    postings: dict[str, list[int]] = defaultdict(list)
    for i, current in enumerate(sets):
        shared: Counter[int] = Counter()
        for shingle in current:
            for j in postings[shingle]:
                shared[j] += 1
            postings[shingle].append(i)
        for j, intersection in shared.items():
            union = len(current) + len(sets[j]) - intersection
            if intersection / union >= threshold:
                parent[find(i)] = find(j)

    members: dict[int, list[str]] = defaultdict(list)
    for i, key in enumerate(keys):
        members[find(i)].append(key)
    return sorted(sorted(group) for group in members.values() if len(group) > 1)


def _order(identifier: str):
    return (0, int(identifier), "") if identifier.isdigit() else (1, 0, identifier)


def representatives(identifiers: Iterable[str], clusters: Iterable[Iterable[str]]) -> dict[str, str]:
    """Map every identifier to the one kept for its cluster: the smallest, with
    all-digit identifiers compared as numbers. An identifier in no cluster maps
    to itself."""
    kept = {identifier: identifier for identifier in identifiers}
    for cluster in clusters:
        cluster = list(cluster)
        smallest = min(cluster, key=_order)
        for identifier in cluster:
            kept[identifier] = smallest
    return kept
