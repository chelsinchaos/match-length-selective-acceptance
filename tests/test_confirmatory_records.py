# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LogNovations Holdings LLC
"""The confirmatory record, the controls record and the run ledger are consistent
with each other, with the records frozen before the tag, and with the rule.

Written before the confirmatory run. Each test skips until its record exists.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from selfref.decision import decide
from selfref.guard import PREREG_TAG, _rows

HARNESS = Path(__file__).resolve().parents[1]
ARMS = ("BASELINE", "BASELINE+SOURCE", "BASELINE+SELF")
FROZEN_BEFORE_THE_TAG = ("prereg/baseline.json", "prereg/profile.json", "prereg/bar_floor.json",
                         "prereg/aa.json", "prereg/power.json", "PREREGISTRATION.md")
# The code of the confirmatory analysis: the decision rule, the calibration per
# stratum, the bootstrap, the analysis itself and its script, the guard, the
# scores, the data builders and the splits. combiner.py and tables.py are not
# listed: each gained one addition after the tag (an optional search space; a
# reader for the supplementary tables) and is otherwise unchanged.
ANALYSIS_CODE_FROZEN_AT_THE_TAG = (
    "src/selfref/decision.py", "src/selfref/certify.py", "src/selfref/experiment.py",
    "src/selfref/confirm.py", "src/selfref/guard.py", "src/selfref/selcert_pin.py",
    "src/selfref/features.py", "src/selfref/matchlen.py", "src/selfref/extract.py",
    "src/selfref/baselines.py", "src/selfref/ncd.py", "src/selfref/groundedness.py",
    "src/selfref/selfcheck.py", "src/selfref/splits.py", "src/selfref/dedup.py",
    "src/selfref/power.py", "src/selfref/datasets/build.py", "src/selfref/datasets/ragtruth.py",
    "scripts/confirm.py", "scripts/phase2.py", "scripts/build_tables.py",
    "scripts/score_selfcheck.py")


def record(name: str) -> dict:
    path = HARNESS / name
    if not path.exists():
        pytest.skip(f"{name} has not been produced yet")
    return json.loads(path.read_text())


def git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(HARNESS), *args], capture_output=True, text=True)


def require_tag() -> None:
    if git("rev-parse", "--verify", "--quiet", f"refs/tags/{PREREG_TAG}^{{commit}}").returncode != 0:
        pytest.skip("no git history with the pre-registration tag here")


def require_commits(*commits: str) -> None:
    """The records name commits of the repository the study ran in. A public
    export of the tree has its own, shorter history, and those commits are not
    in it; a check of their order can only run where they exist."""
    for commit in commits:
        if git("cat-file", "-e", f"{commit}^{{commit}}").returncode != 0:
            pytest.skip("this tree is an export: the commits its records name are in the study's repository")


def test_the_rule_was_applied_with_the_bar_and_floor_committed_before_the_tag():
    result, limits = record("confirmatory/result.json"), record("prereg/bar_floor.json")
    assert result["mode"] == "confirmatory"
    assert result["BAR_points"] == limits["BAR_points"]
    assert result["FLOOR_points"] == limits["FLOOR_points"]
    assert result["primary_alpha"] == limits["primary_alpha"] and result["delta"] == limits["delta"]


def test_each_outcome_follows_from_its_gain_and_interval_by_the_decision_rule():
    result = record("confirmatory/result.json")
    for name in ("H1", "H3"):
        h = result["hypotheses"][name]
        assert h["outcome"] == decide(h["gain_points"], h["interval_low"], h["interval_high"],
                                      result["BAR_points"], result["FLOOR_points"])
        assert h["bootstrap_draws"] == 10_000
        assert h["interval_low"] <= h["interval_high"]
    assert result["hypotheses"]["H2"]["outcome"] == "NOT TESTABLE"


def test_each_gain_is_the_difference_of_two_rows_of_the_coverage_table():
    result = record("confirmatory/result.json")
    table, alpha = result["coverage"], str(result["primary_alpha"])
    for name, arm in (("H1", "BASELINE+SOURCE"), ("H3", "BASELINE+SELF")):
        difference = table[arm][alpha]["coverage_points"] - table["BASELINE"][alpha]["coverage_points"]
        assert result["hypotheses"][name]["tested_arm"] == arm
        assert result["hypotheses"][name]["gain_points"] == pytest.approx(difference, abs=1e-9)


def test_the_secondary_p_value_is_holm_adjusted_over_two_hypotheses():
    h3 = record("confirmatory/result.json")["hypotheses"]["H3"]
    assert h3["p_holm"] == pytest.approx(min(1.0, 2 * h3["p_gain_at_most_floor"]))
    assert h3["supported"] is bool(h3["outcome"] == "PASS" and h3["p_holm"] < 0.05)


def test_the_confirmatory_run_read_the_sealed_splits_at_a_commit_that_descends_from_the_tag():
    result = record("confirmatory/result.json")
    require_tag()
    stamp = result["provenance"]
    assert sorted(entry["split"] for entry in stamp["ledger_entries"]) == ["CAL", "TEST"]
    assert {entry["commit"] for entry in stamp["ledger_entries"]} == {stamp["git_commit"]}
    require_commits(stamp["git_commit"])
    assert git("merge-base", "--is-ancestor", PREREG_TAG, stamp["git_commit"]).returncode == 0


def test_nothing_frozen_before_the_tag_has_changed_since():
    record("confirmatory/result.json")
    require_tag()
    changed = git("diff", "--name-only", PREREG_TAG, "HEAD", "--", *FROZEN_BEFORE_THE_TAG).stdout.split()
    assert changed == []


def test_the_code_of_the_confirmatory_analysis_has_not_changed_since_the_tag():
    record("confirmatory/result.json")
    require_tag()
    changed = git("diff", "--name-only", PREREG_TAG, "HEAD", "--",
                  *ANALYSIS_CODE_FROZEN_AT_THE_TAG).stdout.split()
    assert changed == []
    for name in ANALYSIS_CODE_FROZEN_AT_THE_TAG:
        assert (HARNESS / name).exists(), name


def test_the_first_reads_in_the_ledger_are_the_confirmatory_run_and_every_read_follows_the_tag():
    result = record("confirmatory/result.json")
    require_tag()
    rows = _rows(HARNESS / "RUNLEDGER.md")
    confirmatory = {(e["split"], e["commit"]) for e in result["provenance"]["ledger_entries"]}
    assert {(row[2], row[4]) for row in rows[:2]} == confirmatory
    assert [row[0] for row in rows] == [str(n) for n in range(1, len(rows) + 1)]
    require_commits(*(row[4] for row in rows))
    for row in rows:
        assert git("merge-base", "--is-ancestor", PREREG_TAG, row[4]).returncode == 0


def test_the_controls_start_from_the_same_coverage_table_as_the_confirmatory_run():
    result, controls = record("confirmatory/result.json"), record("controls/result.json")
    assert controls["mode"] == "confirmatory"
    for arm in ARMS:
        assert controls["coverage"][arm] == result["coverage"][arm]
    assert controls["sizes"]["test"]["items"] == result["sizes"]["test"]["items"]


def test_the_controls_ran_after_the_confirmatory_result_was_committed():
    result, controls = record("confirmatory/result.json"), record("controls/result.json")
    require_tag()
    assert result["provenance"]["test_accesses_so_far"] == 1
    assert controls["provenance"]["test_accesses_so_far"] == 2
    require_commits(controls["provenance"]["git_commit"])
    committed = git("log", "-1", "--format=%H", "--", "confirmatory/result.json").stdout.strip()
    assert committed, "confirmatory/result.json is not committed"
    assert git("merge-base", "--is-ancestor", committed, controls["provenance"]["git_commit"]).returncode == 0
    assert result["provenance"]["test_accesses_so_far"] == 1
    assert controls["provenance"]["test_accesses_so_far"] == 2
