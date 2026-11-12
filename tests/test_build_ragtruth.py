# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LogNovations Holdings LLC
"""Building the study tables from raw RAGTruth files.

What must hold: reserved rows are never parsed; near-duplicate sources collapse
to one; no group crosses splits; label-free item records carry nothing derived
from the annotations; and the labels of the sealed splits (CAL, TEST) are not
kept anywhere.
"""
from __future__ import annotations

import json

import pytest

from selfref.datasets.build import SALT_RAGTRUTH, build_ragtruth
from selfref.splits import SEALED, SPLITS, assign_split

MODELS = ["gpt-4-0613", "mistral-7B-instruct"]
RESERVED_UNPARSEABLE = b'{"id": "r", "source_id": "reserved-src", "split": "test", NOT JSON'


def text_for(i: int) -> str:
    return " ".join(f"doc{i}word{j}" for j in range(40))


@pytest.fixture
def raw(tmp_path):
    n_sources = 120
    responses, sources = [], []
    rid = 0
    for i in range(n_sources):
        content = text_for(i)
        sources.append(json.dumps({"source_id": str(i), "task_type": "Summary", "source": "CNN/DM",
                                   "source_info": content, "prompt": f"Summarize: {content}"}))
        for model in MODELS:
            rid += 1
            spans = [{"start": 0, "end": 3, "text": "abc", "label_type": "Evident Conflict",
                      "implicit_true": False, "due_to_null": False, "meta": "note"}] if rid % 3 == 0 else []
            responses.append(json.dumps({
                "id": str(rid), "source_id": str(i), "model": model, "temperature": 0.7,
                "labels": spans, "split": "train", "quality": "good",
                "response": f"Response {rid} about doc{i}word1."}))
    # source 500 duplicates source 7 exactly; source 501 is unrelated to everything
    sources.append(json.dumps({"source_id": "500", "task_type": "Summary", "source": "CNN/DM",
                               "source_info": text_for(7), "prompt": f"Summarize: {text_for(7)}"}))
    responses.append(json.dumps({"id": "9001", "source_id": "500", "model": MODELS[0],
                                 "temperature": 0.7, "labels": [], "split": "train",
                                 "quality": "good", "response": "Duplicate-source response."}))
    # a source used only by reserved rows; it is unparseable too
    source_lines = [s.encode() for s in sources] + [b'{"source_id": "reserved-src", NOT JSON']
    response_lines = [r.encode() for r in responses]
    response_lines[5:5] = [RESERVED_UNPARSEABLE, RESERVED_UNPARSEABLE]
    response_path, source_path = tmp_path / "response.jsonl", tmp_path / "source_info.jsonl"
    response_path.write_bytes(b"\n".join(response_lines) + b"\n")
    source_path.write_bytes(b"\n".join(source_lines) + b"\n")
    return response_path, source_path


def test_reserved_rows_are_counted_and_never_parsed(raw):
    built = build_ragtruth(*raw)                 # would raise on the unparseable reserved lines
    assert built.report["reserved_rows"] == 2
    assert "reserved-src" not in built.sources
    assert all(item["source_id"] != "reserved-src" for item in built.items)


def test_near_duplicate_sources_collapse_to_the_smallest_identifier(raw):
    built = build_ragtruth(*raw)
    assert built.report["near_duplicate_clusters"] == 1
    assert built.report["sources_dropped_as_duplicates"] == 1
    assert built.report["responses_dropped_with_duplicate_sources"] == 1
    assert "500" not in {item["source_id"] for item in built.items}
    assert "7" in {item["source_id"] for item in built.items}
    assert "9001" not in {item["id"] for item in built.items}


def test_split_is_the_hash_of_the_group_and_no_group_crosses_splits(raw):
    built = build_ragtruth(*raw)
    splits_of_group = {}
    for item in built.items:
        assert item["split"] == assign_split(item["group"], SALT_RAGTRUTH)
        splits_of_group.setdefault(item["group"], set()).add(item["split"])
    assert all(len(s) == 1 for s in splits_of_group.values())
    assert {item["split"] for item in built.items} <= set(SPLITS)


def test_item_records_carry_nothing_derived_from_the_annotations(raw):
    built = build_ragtruth(*raw)
    allowed = {"id", "source_id", "group", "split", "model", "temperature",
               "task_type", "corpus", "response"}
    assert all(set(item) == allowed for item in built.items)


def test_labels_exist_for_open_splits_only(raw):
    built = build_ragtruth(*raw)
    split_of = {item["id"]: item["split"] for item in built.items}
    assert set(built.labels) == {i for i, s in split_of.items() if s not in SEALED}
    assert any(s in SEALED for s in split_of.values())           # the fixture does reach them
    assert all(set(v) == {"loss", "n_spans", "n_spans_not_implicit_true", "quality"}
               for v in built.labels.values())
    assert {v["loss"] for v in built.labels.values()} == {0, 1}


def test_loss_matches_the_span_rule_on_open_rows(raw):
    built = build_ragtruth(*raw)
    for item_id, label in built.labels.items():
        assert label["loss"] == int(int(item_id) % 3 == 0)


def test_report_counts_add_up(raw):
    built = build_ragtruth(*raw)
    report = built.report
    assert report["open_rows_in_file"] == 120 * 2 + 1
    assert report["items"] == 120 * 2
    assert sum(report["items_per_split"].values()) == report["items"]
    assert sum(report["groups_per_split"].values()) == report["groups"] == 120
    assert set(report["loss_rate_per_open_split"]) == set(SPLITS) - set(SEALED)
    assert not set(report["loss_rate_per_open_split"]) & set(SEALED)


# ------------------------------------------------- reading sealed labels, guarded

def authorisation(split):
    return {"split": split, "commit": "0" * 40, "number": 1, "access_number": 1,
            "time": "2026-01-01T00:00:00Z", "purpose": "test"}


def test_sealed_labels_need_an_authorisation_for_that_very_split(raw):
    from selfref.datasets.build import sealed_ragtruth_labels
    from selfref.guard import SealedAccessRefused
    for bad in (None, {}, authorisation("TEST"), {"split": "CAL"}):
        with pytest.raises(SealedAccessRefused):
            sealed_ragtruth_labels(*raw, "CAL", bad)


def test_only_sealed_splits_can_be_read_this_way(raw):
    from selfref.datasets.build import sealed_ragtruth_labels
    with pytest.raises(ValueError):
        sealed_ragtruth_labels(*raw, "TRAIN", authorisation("TRAIN"))


def test_sealed_labels_cover_exactly_the_items_of_that_split_and_follow_the_same_rule(raw):
    from selfref.datasets.build import sealed_ragtruth_labels
    built = build_ragtruth(*raw)
    for split in SEALED:
        expected = {item["id"] for item in built.items if item["split"] == split}
        labels = sealed_ragtruth_labels(*raw, split, authorisation(split))
        assert set(labels) == expected and expected
        assert all(labels[i]["loss"] == int(int(i) % 3 == 0) for i in labels)
        assert not set(labels) & set(built.labels)            # disjoint from the open labels


def test_open_and_sealed_labels_together_cover_every_item(raw):
    from selfref.datasets.build import sealed_ragtruth_labels
    built = build_ragtruth(*raw)
    everything = dict(built.labels)
    for split in SEALED:
        everything.update(sealed_ragtruth_labels(*raw, split, authorisation(split)))
    assert set(everything) == {item["id"] for item in built.items}
