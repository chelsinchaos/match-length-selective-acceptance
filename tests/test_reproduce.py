# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LogNovations Holdings LLC
"""How a reproduction's records are compared with the committed ones."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "reproduce.py"


@pytest.fixture(scope="module")
def reproduce():
    spec = importlib.util.spec_from_file_location("reproduce", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_records_that_differ_only_in_stamps_and_timings_are_identical(reproduce):
    a = {"provenance": {"git_commit": "aaa"}, "coverage": 8.84, "seconds": 3.0}
    b = {"provenance": {"git_commit": "bbb"}, "coverage": 8.84, "seconds": 9.0}
    assert reproduce.first_difference(reproduce.strip(a), reproduce.strip(b)) is None


def test_a_different_number_is_found_and_located(reproduce):
    a, b = {"arms": {"A": {"auroc": 0.90}}}, {"arms": {"A": {"auroc": 0.91}}}
    assert "/arms/A/auroc" in reproduce.first_difference(reproduce.strip(a), reproduce.strip(b))


def test_the_hash_of_a_model_file_is_compared_apart_from_the_numbers(reproduce):
    """The file of a fitted model holds the number of threads it was fitted with, so its
    hash changes from machine to machine while every number stays the same."""
    a = {"arms": {"A": {"model_sha256": "aa", "auroc": 0.9}}, "model_sha256": "cc"}
    b = {"arms": {"A": {"model_sha256": "bb", "auroc": 0.9}}, "model_sha256": "cc"}
    assert reproduce.first_difference(reproduce.strip(a), reproduce.strip(b)) is None
    assert reproduce.model_hashes(a) == ["aa", "cc"]
    assert reproduce.model_hashes(a) != reproduce.model_hashes(b)


def test_the_comparison_record_counts_the_records_and_the_reads_the_reproduction_added(reproduce):
    rows = [{"record": "x.json", "first_difference": None, "model_hash_differs": False},
            {"record": "y.json", "first_difference": "/a: 1 != 2", "model_hash_differs": False},
            {"record": "z.json", "first_difference": None, "model_hash_differs": True}]
    before = [["1", "2026-11-12T14:33:03Z", "CAL", "1", "c1", "confirmatory run"],
              ["2", "2026-11-12T14:33:03Z", "TEST", "1", "c1", "confirmatory run"]]
    now = before + [["3", "2026-11-12T15:21:25Z", "TEST", "2", "c2", "reproduction of the confirmatory run"]]
    record = reproduce.comparison_record("abc", rows, before, now)
    assert (record["records_compared"], record["records_identical"]) == (3, 2)
    assert record["records_identical_with_a_different_model_hash"] == 1
    assert record["sealed_reads_by_the_reproduction"]["TEST"] == 1
    assert record["sealed_reads_by_the_reproduction"]["CAL"] == 0
    assert record["sealed_reads_by_the_reproduction"]["rows"][0]["purpose"] == "reproduction of the confirmatory run"
    assert record["ledger_rows_at_the_reference"] == 2


def test_ledger_rows_are_read_from_the_text_of_the_ledger(reproduce):
    text = ("# Run ledger\n\nprose\n\n| # | time (UTC) | split | access to this split | commit | purpose |\n"
            "|---|---|---|---|---|---|\n| 1 | 2026-11-12T14:33:03Z | CAL | 1 | c1 | confirmatory run |\n")
    assert reproduce.ledger_rows(text) == [["1", "2026-11-12T14:33:03Z", "CAL", "1", "c1", "confirmatory run"]]
    assert reproduce.ledger_rows("") == []
