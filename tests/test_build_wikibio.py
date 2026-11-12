# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LogNovations Holdings LLC
"""Building the sentence-level WikiBio GPT-3 tables used for H2."""
from __future__ import annotations

import pytest

from selfref.datasets.build import SALT_WIKIBIO, build_wikibio
from selfref.splits import SEALED, assign_split

LABELS = ["accurate", "minor_inaccurate", "major_inaccurate"]


def passages(n: int):
    rows = []
    for i in range(n):
        sentences = [f"Person{i} was born in town{i}.", f"Person{i} wrote book{i}.",
                     f"Person{i} died in year{i}."]
        rows.append({
            "gpt3_text": " ".join(sentences),
            "wiki_bio_text": f"Person{i} is a real person from town{i}.",
            "gpt3_sentences": sentences,
            "annotation": [LABELS[(i + k) % 3] for k in range(3)],
            "wiki_bio_test_idx": 1000 + i,
            "gpt3_text_samples": [f"Sample {j} about person{i} from town{i}." for j in range(20)],
        })
    return rows


def test_one_item_per_sentence_grouped_by_passage():
    built = build_wikibio(passages(40))
    assert len(built.items) == 120
    assert built.report["groups"] == 40
    first = built.items[0]
    assert set(first) == {"id", "group", "split", "position", "n_sentences", "sentence"}
    assert first["id"] == "1000:0" and first["group"] == "1000"
    assert (first["position"], first["n_sentences"]) == (0, 3)


def test_split_is_the_hash_of_the_passage():
    built = build_wikibio(passages(40))
    for item in built.items:
        assert item["split"] == assign_split(item["group"], SALT_WIKIBIO)


def test_resamples_are_stored_per_passage_and_there_are_twenty():
    built = build_wikibio(passages(5))
    assert set(built.sources) == {"1000", "1001", "1002", "1003", "1004"}
    assert all(len(samples) == 20 for samples in built.sources.values())
    assert built.sources["1002"][3] == b"Sample 3 about person2 from town2."


def test_loss_is_one_unless_the_sentence_is_labelled_accurate():
    built = build_wikibio(passages(40))
    for item_id, label in built.labels.items():
        passage, position = (int(part) for part in item_id.split(":"))
        annotation = LABELS[(passage - 1000 + position) % 3]
        assert label == {"loss": int(annotation != "accurate"),
                         "major": int(annotation == "major_inaccurate")}


def test_labels_of_sealed_splits_are_not_kept():
    built = build_wikibio(passages(120))
    split_of = {item["id"]: item["split"] for item in built.items}
    assert any(split in SEALED for split in split_of.values())
    assert set(built.labels) == {i for i, s in split_of.items() if s not in SEALED}


def test_a_passage_whose_sentences_and_annotations_disagree_in_number_is_an_error():
    rows = passages(1)
    rows[0]["annotation"] = rows[0]["annotation"][:2]
    with pytest.raises(ValueError):
        build_wikibio(rows)


def test_an_unknown_annotation_value_is_an_error():
    rows = passages(1)
    rows[0]["annotation"][0] = "mostly_fine"
    with pytest.raises(ValueError):
        build_wikibio(rows)


def test_sealed_wikibio_labels_are_guarded_and_follow_the_same_rule():
    from selfref.datasets.build import sealed_wikibio_labels
    from selfref.guard import SealedAccessRefused
    rows = passages(120)
    built = build_wikibio(rows)
    with pytest.raises(SealedAccessRefused):
        sealed_wikibio_labels(rows, "CAL", None)
    with pytest.raises(ValueError):
        sealed_wikibio_labels(rows, "DEV_CAL", {"split": "DEV_CAL", "commit": "0" * 40})
    for split in SEALED:
        labels = sealed_wikibio_labels(rows, split, {"split": split, "commit": "0" * 40})
        assert set(labels) == {i["id"] for i in built.items if i["split"] == split}
        for item_id, label in labels.items():
            passage, position = (int(part) for part in item_id.split(":"))
            assert label["loss"] == int(LABELS[(passage - 1000 + position) % 3] != "accurate")
