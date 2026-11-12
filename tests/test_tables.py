# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LogNovations Holdings LLC
"""Assembling a split's design table from the derived files."""
from __future__ import annotations

import json
import math

import numpy as np
import pytest

from selfref.tables import LOG_COLUMNS, SealedSplitError, load_ragtruth_split


@pytest.fixture
def derived(tmp_path):
    items = [
        {"id": "1", "source_id": "s1", "group": "s1", "split": "TRAIN", "model": "gpt-4-0613",
         "temperature": 0.7, "task_type": "QA", "corpus": "MARCO", "response": "a"},
        {"id": "2", "source_id": "s1", "group": "s1", "split": "TRAIN", "model": "mistral-7B-instruct",
         "temperature": 1.0, "task_type": "QA", "corpus": "MARCO", "response": "b"},
        {"id": "3", "source_id": "s2", "group": "s2", "split": "TRAIN", "model": "gpt-4-0613",
         "temperature": 0.7, "task_type": "Summary", "corpus": "CNN/DM", "response": "c"},
        {"id": "9", "source_id": "s9", "group": "s9", "split": "CAL", "model": "gpt-4-0613",
         "temperature": 0.7, "task_type": "QA", "corpus": "MARCO", "response": "z"},
    ]
    (tmp_path / "items.jsonl").write_text("\n".join(json.dumps(i) for i in items) + "\n")
    names = np.array(["base.len_bytes", "base.random", "self.cov_k1", "source.lcs", "source.ncd_zstd"])
    X = np.array([[10.0, 0.3, 0.5, 4.0, 0.9],
                  [20.0, 0.6, 0.1, 0.0, 0.8],
                  [30.0, 0.9, 0.2, 7.0, 0.7]])
    np.savez_compressed(tmp_path / "features_TRAIN.npz", ids=np.array(["1", "2", "3"]), names=names, X=X)
    np.savez_compressed(tmp_path / "features_CAL.npz", ids=np.array(["9"]), names=names, X=X[:1])
    (tmp_path / "groundedness.jsonl").write_text(
        json.dumps({"id": "1", "support": [0.9, 0.3]}) + "\n" +
        json.dumps({"id": "2", "support": [0.2]}) + "\n" +
        json.dumps({"id": "3", "support": []}) + "\n")
    (tmp_path / "labels_open.json").write_text(json.dumps({
        "1": {"loss": 0, "n_spans": 0, "n_spans_not_implicit_true": 0, "quality": "good"},
        "2": {"loss": 1, "n_spans": 2, "n_spans_not_implicit_true": 1, "quality": "good"},
        "3": {"loss": 0, "n_spans": 0, "n_spans_not_implicit_true": 0, "quality": "good"}}))
    return tmp_path


CATEGORIES = {"model": ["gpt-4-0613", "mistral-7B-instruct"], "task": ["QA", "Summary"]}


def test_table_has_scores_groundedness_and_covariates(derived):
    table = load_ragtruth_split(derived, "TRAIN", CATEGORIES)
    assert table.names == ["base.len_bytes", "base.random", "self.cov_k1", "source.lcs",
                           "source.ncd_zstd", "gnd.min", "gnd.mean", "gnd.frac_unsupported",
                           "cov.temperature", "cov.model=gpt-4-0613",
                           "cov.model=mistral-7B-instruct", "cov.task=QA", "cov.task=Summary"]
    assert table.ids == ["1", "2", "3"]
    row = dict(zip(table.names, table.X[0]))
    assert row["gnd.min"] == pytest.approx(0.3) and row["gnd.mean"] == pytest.approx(0.6)
    assert row["gnd.frac_unsupported"] == pytest.approx(0.5)
    assert (row["cov.model=gpt-4-0613"], row["cov.model=mistral-7B-instruct"]) == (1.0, 0.0)
    assert (row["cov.task=QA"], row["cov.task=Summary"]) == (1.0, 0.0)
    assert row["cov.temperature"] == 0.7


def test_counts_are_log_transformed_and_nothing_else_is(derived):
    table = load_ragtruth_split(derived, "TRAIN", CATEGORIES)
    assert {"base.len_bytes", "source.lcs"} <= set(LOG_COLUMNS)
    row = dict(zip(table.names, table.X[0]))
    assert row["base.len_bytes"] == pytest.approx(math.log1p(10.0))
    assert row["source.lcs"] == pytest.approx(math.log1p(4.0))
    assert row["self.cov_k1"] == 0.5 and row["source.ncd_zstd"] == 0.9


def test_loss_stratum_group_and_block_come_with_the_rows(derived):
    table = load_ragtruth_split(derived, "TRAIN", CATEGORIES)
    assert table.loss.tolist() == [0, 1, 0]
    assert table.stratum.tolist() == ["gpt-4-0613", "mistral-7B-instruct", "gpt-4-0613"]
    assert table.group.tolist() == ["s1", "s1", "s2"]
    assert table.block.tolist() == ["QA", "QA", "Summary"]


def test_a_sealed_split_cannot_be_loaded(derived):
    with pytest.raises(SealedSplitError):
        load_ragtruth_split(derived, "CAL", CATEGORIES)


def test_a_missing_groundedness_score_is_an_error_not_a_default(derived):
    lines = (derived / "groundedness.jsonl").read_text().splitlines()[:2]
    (derived / "groundedness.jsonl").write_text("\n".join(lines) + "\n")
    with pytest.raises(KeyError, match="groundedness"):
        load_ragtruth_split(derived, "TRAIN", CATEGORIES)


def test_an_unknown_category_is_an_error(derived):
    with pytest.raises(ValueError, match="model"):
        load_ragtruth_split(derived, "TRAIN", {"model": ["gpt-4-0613"], "task": ["QA", "Summary"]})


def test_sample_selects_columns_by_name(derived):
    table = load_ragtruth_split(derived, "TRAIN", CATEGORIES)
    sample = table.sample(["gnd.min", "self.cov_k1"])
    assert sample.X.shape == (3, 2)
    assert sample.X[:, 0].tolist() == pytest.approx([0.3, 0.2, 1.0])       # no claims: fully supported
    with pytest.raises(KeyError):
        table.sample(["no.such.column"])


def test_a_sealed_split_loads_only_with_labels_supplied_by_the_caller(derived):
    """The confirmatory run obtains sealed labels through the guard and hands them in.
    Without them a sealed split cannot be loaded; with them it loads like any other."""
    lines = (derived / "groundedness.jsonl").read_text()
    (derived / "groundedness.jsonl").write_text(lines + json.dumps({"id": "9", "support": [0.8]}) + "\n")
    with pytest.raises(SealedSplitError):
        load_ragtruth_split(derived, "CAL", CATEGORIES)
    table = load_ragtruth_split(derived, "CAL", CATEGORIES, sealed_labels={"9": {"loss": 1}})
    assert table.ids == ["9"] and table.loss.tolist() == [1]


def test_supplying_labels_for_an_open_split_is_refused(derived):
    with pytest.raises(ValueError, match="open"):
        load_ragtruth_split(derived, "TRAIN", CATEGORIES, sealed_labels={"1": {"loss": 0}})


def test_sealed_labels_must_cover_every_item_of_the_split(derived):
    lines = (derived / "groundedness.jsonl").read_text()
    (derived / "groundedness.jsonl").write_text(lines + json.dumps({"id": "9", "support": [0.8]}) + "\n")
    with pytest.raises(KeyError):
        load_ragtruth_split(derived, "CAL", CATEGORIES, sealed_labels={"other": {"loss": 1}})
