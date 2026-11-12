# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LogNovations Holdings LLC
"""The report takes every number from results.json, through placeholders."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from selfref.render import lookup, md_table, render

HARNESS = Path(__file__).resolve().parents[1]
RESULTS = {"ledger": {"test_accesses": 2},
           "records": {"a.json": {"gain": -0.4255, "arms": {"BASELINE+SOURCE": {"0.1": [9.148, None]}},
                                  "n": 1584, "name": "x"}}}


def test_a_placeholder_is_a_path_into_the_results_with_an_optional_format():
    assert render("gain {{a.json > gain | +.2f}} points", RESULTS) == "gain -0.43 points"
    assert render("{{a.json > arms > BASELINE+SOURCE > 0.1 > 0 | .1f}}", RESULTS) == "9.1"
    assert render("{{a.json > n | ,d}} items, {{a.json > name}}", RESULTS) == "1,584 items, x"
    assert render("{{ledger > test_accesses}}", RESULTS) == "2"


def test_a_missing_value_is_shown_as_a_dash_and_a_wrong_path_is_an_error():
    assert render("{{a.json > arms > BASELINE+SOURCE > 0.1 > 1 | .2f}}", RESULTS) == "—"
    with pytest.raises(KeyError):
        render("{{a.json > no_such_key}}", RESULTS)
    with pytest.raises(KeyError):
        render("{{b.json > gain}}", RESULTS)


def test_a_table_placeholder_calls_the_named_table_with_the_results():
    tables = {"demo": lambda results: md_table(["k", "v"], [["gain", results["records"]["a.json"]["gain"]]])}
    assert render("{{table: demo}}", RESULTS, tables) == "| k | v |\n|---|---|\n| gain | -0.4255 |"
    with pytest.raises(KeyError):
        render("{{table: other}}", RESULTS, tables)


def test_lookup_walks_lists_by_index():
    assert lookup(RESULTS, "a.json > arms > BASELINE+SOURCE > 0.1 > 0") == 9.148


def test_the_committed_report_is_exactly_the_template_rendered_from_results_json():
    template, results, report = (HARNESS / "report" / "REPORT.template.md", HARNESS / "results.json",
                                 HARNESS / "REPORT.md")
    if not (template.exists() and results.exists() and report.exists()):
        pytest.skip("the report has not been written yet")
    import importlib.util
    spec = importlib.util.spec_from_file_location("make_report", HARNESS / "scripts" / "make_report.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    loaded = json.loads(results.read_text())
    assert report.read_text() == render(template.read_text(), loaded, module.TABLES)


def test_the_reproduction_tables_say_so_when_their_record_is_absent():
    """A reproduction in a clone has no record of an earlier reproduction check."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("make_report", HARNESS / "scripts" / "make_report.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for name in ("reproduction", "reproduction_ledger"):
        text = module.TABLES[name]({"records": {}})
        assert "not in this tree" in text and "|" not in text


def test_the_report_renders_without_the_records_that_a_public_reproduction_lacks():
    """A reproduction from the public tree has no record of an earlier reproduction
    check. The template must not quote it outside the tables that stand in for it."""
    template, results = HARNESS / "report" / "REPORT.template.md", HARNESS / "results.json"
    if not (template.exists() and results.exists()):
        pytest.skip("the report has not been written yet")
    import importlib.util
    spec = importlib.util.spec_from_file_location("make_report", HARNESS / "scripts" / "make_report.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    loaded = json.loads(results.read_text())
    loaded["records"].pop("reproduction/check.json", None)
    text = render(template.read_text(), loaded, module.TABLES)
    assert "not in this tree" in text
