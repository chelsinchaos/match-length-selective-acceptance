# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LogNovations Holdings LLC
"""Reading RAGTruth without ever parsing a reserved row.

The authors' test split is the reserved replication set. The reader decides a
row's split from the raw bytes and parses a row as JSON only if it is not
reserved. The fixtures make every reserved row invalid JSON apart from its
split field, so any attempt to parse one fails loudly.
"""
from __future__ import annotations

import json

import pytest

from selfref.datasets.ragtruth import (ReservedRowError, dedup_text, open_response_lines,
                                       parse_response, read_sources, response_loss,
                                       source_id_of_line, source_reference, split_of_line)


def response_line(rid, source_id, split, labels=(), response="An answer.", model="gpt-4-0613"):
    return json.dumps({"id": rid, "source_id": source_id, "model": model, "temperature": 0.7,
                       "labels": list(labels), "split": split, "quality": "good",
                       "response": response}).encode()


RESERVED_UNPARSEABLE = b'{"id": "9", "source_id": "s9", "split": "test", THIS IS NOT JSON'


@pytest.fixture
def response_file(tmp_path):
    lines = [
        response_line("1", "s1", "train"),
        RESERVED_UNPARSEABLE,
        response_line("2", "s1", "train", labels=[{"start": 0, "end": 2, "text": "An",
                                                   "label_type": "Evident Conflict",
                                                   "implicit_true": False, "due_to_null": False,
                                                   "meta": ""}]),
        RESERVED_UNPARSEABLE,
        response_line("3", "s2", "train"),
    ]
    path = tmp_path / "response.jsonl"
    path.write_bytes(b"\n".join(lines) + b"\n")
    return path


# ------------------------------------------------------------ split screening

def test_split_is_read_from_the_raw_line():
    assert split_of_line(response_line("1", "s1", "train")) == "train"
    assert split_of_line(RESERVED_UNPARSEABLE) == "test"


def test_split_field_without_a_space_after_the_colon_is_recognised():
    assert split_of_line(b'{"id":"1","split":"test","response":"x"}') == "test"


def test_text_that_merely_mentions_a_split_does_not_change_the_split():
    # Inside a JSON string the quotes are escaped, so this is not a second split field.
    line = response_line("1", "s1", "train", response='he wrote "split": "test" in the file')
    assert split_of_line(line) == "train"


@pytest.mark.parametrize("line", [
    b'{"id": "1", "response": "no split field"}',
    b'{"split": "train", "x": {"split": "test"}}',
    b'{"split": "validation"}',
])
def test_missing_ambiguous_or_unknown_split_is_an_error(line):
    with pytest.raises(ValueError):
        split_of_line(line)


def test_only_unreserved_lines_are_returned_and_reserved_ones_are_counted(response_file):
    lines, n_reserved = open_response_lines(response_file)
    assert n_reserved == 2
    assert [json.loads(line)["id"] for line in lines] == ["1", "2", "3"]


def test_parsing_a_reserved_row_is_refused():
    with pytest.raises(ReservedRowError):
        parse_response(response_line("7", "s7", "test"))


def test_parsing_an_open_row_returns_its_fields():
    row = parse_response(response_line("1", "s1", "train"))
    assert (row["id"], row["source_id"], row["split"]) == ("1", "s1", "train")


# --------------------------------------------------------------------- sources

def test_source_id_is_read_from_the_raw_line():
    assert source_id_of_line(b'{"source_id": "14312", "task_type": "QA"}') == "14312"


def test_only_wanted_sources_are_parsed(tmp_path):
    wanted = json.dumps({"source_id": "s1", "task_type": "Summary", "source": "CNN/DM",
                         "source_info": "Some news.", "prompt": "Summarize: Some news."}).encode()
    unwanted_unparseable = b'{"source_id": "s9", BROKEN'
    path = tmp_path / "source_info.jsonl"
    path.write_bytes(wanted + b"\n" + unwanted_unparseable + b"\n")
    sources = read_sources(path, {"s1"})
    assert set(sources) == {"s1"}
    assert sources["s1"]["task_type"] == "Summary"


def test_a_wanted_source_that_is_absent_is_an_error(tmp_path):
    path = tmp_path / "source_info.jsonl"
    path.write_bytes(b'{"source_id": "s1", "prompt": "p", "source_info": "x", "task_type": "QA"}\n')
    with pytest.raises(KeyError):
        read_sources(path, {"s1", "s2"})


def test_source_reference_is_the_prompt_as_utf8_bytes():
    assert source_reference({"prompt": "Résumé: café", "source_info": "ignored"}) == \
        "Résumé: café".encode("utf-8")


def test_dedup_text_is_the_source_content_not_the_prompt_template():
    assert dedup_text({"prompt": "Summarize: A b c.", "source_info": "A b c."}) == "A b c."
    as_dict = dedup_text({"prompt": "p", "source_info": {"b": 2, "a": "x"}})
    assert as_dict == dedup_text({"prompt": "q", "source_info": {"a": "x", "b": 2}})   # key order ignored


# ------------------------------------------------------------------------ loss

def test_loss_is_one_when_any_span_is_annotated():
    assert response_loss({"labels": []}) == 0
    assert response_loss({"labels": [{"label_type": "Subtle Conflict", "implicit_true": False}]}) == 1
    # an implicit_true span still counts: the dataset's own published count includes it
    assert response_loss({"labels": [{"label_type": "Evident Baseless Info", "implicit_true": True}]}) == 1


def test_loss_requires_the_labels_field():
    with pytest.raises(KeyError):
        response_loss({"response": "sealed row with labels stripped"})
