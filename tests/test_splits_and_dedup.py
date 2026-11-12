# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LogNovations Holdings LLC
"""Group-hash splits and near-duplicate clustering."""
from __future__ import annotations

import hashlib
from collections import Counter

import pytest
from hypothesis import given, settings, strategies as st

from selfref.dedup import jaccard, near_duplicate_clusters, representatives, shingles
from selfref.splits import FRACTIONS, SEALED, SPLITS, assign_split, unit_interval


# ---------------------------------------------------------------------- splits

def test_the_six_splits_and_their_fractions():
    assert SPLITS == ("TRAIN", "DEV_SELECT", "DEV_CAL", "DEV_TEST", "CAL", "TEST")
    assert FRACTIONS == (0.30, 0.10, 0.20, 0.10, 0.20, 0.10)
    assert sum(FRACTIONS) == pytest.approx(1.0)
    assert SEALED == ("CAL", "TEST")


def test_unit_interval_is_the_first_eight_bytes_of_sha256():
    digest = hashlib.sha256(b"salt\x00key-1").digest()
    assert unit_interval("key-1", "salt") == int.from_bytes(digest[:8], "big") / 2 ** 64


def test_assignment_is_deterministic_and_depends_on_the_salt():
    assert assign_split("group-42", "salt-a") == assign_split("group-42", "salt-a")
    keys = [f"g{i}" for i in range(400)]
    assert [assign_split(k, "salt-a") for k in keys] != [assign_split(k, "salt-b") for k in keys]


def test_assignment_follows_the_cumulative_fractions():
    for key in (f"g{i}" for i in range(300)):
        u = unit_interval(key, "s")
        edges = [0.30, 0.40, 0.60, 0.70, 0.90, 1.0]
        expected = SPLITS[next(i for i, edge in enumerate(edges) if u < edge)]
        assert assign_split(key, "s") == expected


def test_proportions_are_close_to_the_fractions_on_many_groups():
    counts = Counter(assign_split(f"group-{i}", "prop") for i in range(20_000))
    for name, fraction in zip(SPLITS, FRACTIONS):
        assert counts[name] / 20_000 == pytest.approx(fraction, abs=0.012)


def test_every_member_of_a_group_lands_in_the_same_split():
    rows = [("g1", "r1"), ("g1", "r2"), ("g2", "r3"), ("g1", "r4"), ("g2", "r5")]
    by_group = {}
    for group, _ in rows:
        by_group.setdefault(group, set()).add(assign_split(group, "x"))
    assert all(len(v) == 1 for v in by_group.values())


# ------------------------------------------------------------------- shingles

def test_shingles_are_lowercased_word_five_grams():
    assert shingles("The Cat sat on the mat today") == {
        "the cat sat on the", "cat sat on the mat", "sat on the mat today"}


def test_a_text_shorter_than_five_words_is_one_shingle():
    assert shingles("Too   short") == {"too short"}
    assert shingles("") == set()


def test_jaccard_closed_forms():
    assert jaccard({"a", "b"}, {"a", "b"}) == 1.0
    assert jaccard({"a"}, {"b"}) == 0.0
    assert jaccard({"a", "b", "c"}, {"b", "c", "d"}) == pytest.approx(2 / 4)
    assert jaccard(set(), set()) == 0.0


# ------------------------------------------------------------------- clusters

BASE = " ".join(f"w{i}" for i in range(60))


def test_identical_texts_form_one_cluster():
    clusters = near_duplicate_clusters({"a": BASE, "b": BASE, "c": "something else entirely here now"})
    assert clusters == [["a", "b"]]


def test_whitespace_and_case_differences_do_not_matter():
    clusters = near_duplicate_clusters({"a": BASE, "b": BASE.upper().replace(" ", "  \n")})
    assert clusters == [["a", "b"]]


def test_one_changed_word_in_sixty_is_a_near_duplicate_but_a_rewrite_is_not():
    one_word = BASE.replace("w30", "CHANGED")          # 5 of 56 shingles change -> J = 51/61
    assert jaccard(shingles(BASE), shingles(one_word)) == pytest.approx(51 / 61)
    assert near_duplicate_clusters({"a": BASE, "b": one_word}, threshold=0.8) == [["a", "b"]]
    assert near_duplicate_clusters({"a": BASE, "b": one_word}, threshold=0.9) == []
    half = " ".join(BASE.split()[:30] + [f"x{i}" for i in range(30)])
    assert near_duplicate_clusters({"a": BASE, "b": half}, threshold=0.8) == []


def test_clusters_are_transitive():
    # a~b and b~c (each 54/58 = 0.931), but a and c are below the threshold (52/60 = 0.867).
    words = BASE.split()
    b = " ".join(words[:58] + ["B1", "B2"])
    c = " ".join(["C1", "C2"] + words[2:58] + ["B1", "B2"])
    assert jaccard(shingles(BASE), shingles(b)) == pytest.approx(54 / 58)
    assert jaccard(shingles(b), shingles(c)) == pytest.approx(54 / 58)
    assert jaccard(shingles(BASE), shingles(c)) == pytest.approx(52 / 60)
    clusters = near_duplicate_clusters({"a": BASE, "b": b, "c": c}, threshold=0.9)
    assert clusters == [["a", "b", "c"]]


def test_representative_is_the_smallest_identifier_in_each_cluster():
    assert representatives(["10", "9", "x"], [["10", "9"]]) == {"10": "9", "9": "9", "x": "x"}
    # numeric identifiers compare as numbers, not as strings
    assert representatives(["100", "20"], [["100", "20"]]) == {"100": "20", "20": "20"}


@settings(max_examples=60, deadline=None)
@given(st.lists(st.lists(st.sampled_from("abcdef"), min_size=0, max_size=14).map(" ".join),
                min_size=0, max_size=7))
def test_clusters_match_a_brute_force_pairwise_definition(texts):
    named = {f"t{i}": t for i, t in enumerate(texts)}
    sets = {k: shingles(v) for k, v in named.items()}
    parent = {k: k for k in named}

    def find(x):
        while parent[x] != x:
            x = parent[x]
        return x

    keys = sorted(named)
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            if jaccard(sets[a], sets[b]) >= 0.5:
                parent[find(a)] = find(b)
    groups = {}
    for k in keys:
        groups.setdefault(find(k), []).append(k)
    expected = sorted(sorted(g) for g in groups.values() if len(g) > 1)
    assert near_duplicate_clusters(named, threshold=0.5) == expected
