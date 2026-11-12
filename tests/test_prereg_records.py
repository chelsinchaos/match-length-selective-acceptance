# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LogNovations Holdings LLC
"""The committed pre-registration records follow mechanically from one another.

BAR and FLOOR must be derivable, exactly, from the records committed before
them: c_B from BASELINE's DEV rehearsal, and the 99th percentile from the
1,000 recorded A/A gaps. A reader can redo the arithmetic from the JSON alone.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

from selfref.decision import compute_bar, compute_floor

PREREG = Path(__file__).resolve().parents[1] / "prereg"


def record(name: str) -> dict:
    path = PREREG / name
    if not path.exists():
        pytest.skip(f"prereg/{name} has not been produced yet")
    return json.loads(path.read_text())


def test_the_primary_alpha_is_the_same_in_every_record():
    baseline, aa, limits = record("baseline.json"), record("aa.json"), record("bar_floor.json")
    assert baseline["primary_alpha"] == aa["alpha"] == limits["primary_alpha"]
    assert limits["primary_alpha"] in (0.01, 0.05, 0.10)


def test_c_b_is_baseline_coverage_in_the_dev_rehearsal_at_the_primary_alpha():
    baseline, limits = record("baseline.json"), record("bar_floor.json")
    rehearsal = baseline["dev_rehearsal"][str(limits["primary_alpha"])]
    assert limits["c_B_points"] == rehearsal["coverage_points"]


def test_the_aa_record_holds_one_gap_per_run_and_its_summary_comes_from_them():
    aa = record("aa.json")
    gaps = np.abs(np.array(aa["gap_points"]))
    assert aa["runs"] == gaps.size == 1000
    assert aa["summary"]["abs_p99"] == float(np.percentile(gaps, 99))
    assert aa["summary"]["abs_max"] == float(gaps.max())


def test_floor_is_the_larger_of_two_points_and_the_recorded_99th_percentile():
    aa, limits = record("aa.json"), record("bar_floor.json")
    assert limits["aa_abs_gap_p99_points"] == aa["summary"]["abs_p99"]
    assert limits["FLOOR_points"] == max(2.0, aa["summary"]["abs_p99"])
    assert limits["FLOOR_points"] == compute_floor(aa["gap_points"])


def test_bar_follows_from_c_b_and_floor():
    limits = record("bar_floor.json")
    by_hand = max(min(5.0, 0.10 * (100.0 - limits["c_B_points"])), limits["FLOOR_points"])
    assert limits["BAR_points"] == by_hand
    assert limits["BAR_points"] == compute_bar(limits["c_B_points"], limits["FLOOR_points"])
    assert limits["BAR_points"] >= limits["FLOOR_points"] >= 2.0


# --- the tested arms: same search as BASELINE, fitted only after BAR and FLOOR were committed

TESTED_ARMS = {"BASELINE+SOURCE": "source.", "BASELINE+SELF": "self."}


def searched(arm_record: dict) -> list[str]:
    return sorted(json.dumps(row["config"], sort_keys=True) for row in arm_record["search"])


def test_each_tested_arm_was_searched_over_the_same_configurations_as_baseline():
    baseline, arms = record("baseline.json"), record("profile.json")["arms"]
    assert set(arms) == set(TESTED_ARMS)
    assert len(set(searched(baseline))) == 24
    for arm in arms.values():
        assert searched(arm) == searched(baseline)


def test_each_tested_arm_adds_exactly_the_sixteen_profile_columns_of_its_frame():
    baseline, arms = record("baseline.json"), record("profile.json")["arms"]
    shared = len(baseline["columns"])
    for name, frame in TESTED_ARMS.items():
        columns = arms[name]["columns"]
        assert columns[:shared] == baseline["columns"]
        added = columns[shared:]
        assert len(added) == len(set(added)) == 16
        assert all(column.startswith(frame) and ".ncd_" not in column for column in added)


def test_every_frozen_configuration_has_the_smallest_log_loss_of_its_search():
    records = [record("baseline.json"), *record("profile.json")["arms"].values()]
    for arm in records:
        best = min(arm["search"], key=lambda row: row["log_loss_dev_select"])
        assert arm["selected"] == best["config"]


def test_the_tested_arms_were_fitted_after_bar_and_floor_were_committed():
    stamp = record("profile.json")["provenance"]
    harness = str(PREREG.parent)
    last_change = subprocess.run(["git", "-C", harness, "log", "-1", "--format=%H", "--",
                                  "prereg/bar_floor.json"], capture_output=True, text=True)
    if last_change.returncode != 0 or not last_change.stdout.strip():
        pytest.skip("no git history here to check the order of the commits against")
    assert stamp["git_dirty"] is False
    is_ancestor = subprocess.run(["git", "-C", harness, "merge-base", "--is-ancestor",
                                  last_change.stdout.strip(), stamp["git_commit"]])
    assert is_ancestor.returncode == 0, "BAR and FLOOR changed after the tested arms were fitted"
