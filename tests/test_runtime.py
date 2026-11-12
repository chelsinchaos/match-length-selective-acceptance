# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LogNovations Holdings LLC
"""The time of a neural baseline, read from the log of its scoring run."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "runtime.py"
PROGRESS = r"(\d+)/(\d+) (?:items|sentences)\s+(?:\d+ pairs\s+)?(\d+)s"


@pytest.fixture(scope="module")
def runtime():
    spec = importlib.util.spec_from_file_location("runtime", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_lines(path: Path, records) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in records))


@pytest.fixture
def tree(tmp_path):
    """Four sealed responses of 1,000 bytes each on one source of 9,000 bytes; one open response."""
    derived, logs = tmp_path / "derived", tmp_path / "logs"
    logs.mkdir()
    write_lines(derived / "ragtruth" / "sources.jsonl", [{"source_id": "s", "reference": "r" * 9000}])
    write_lines(derived / "ragtruth" / "items.jsonl",
                [{"id": "open", "split": "TRAIN", "source_id": "s", "response": "x" * 1000}]
                + [{"id": f"t{i}", "split": "TEST", "source_id": "s", "response": "x" * 1000} for i in range(4)])
    write_lines(derived / "ragtruth" / "groundedness.jsonl",
                [{"id": i, "support": [0.5]} for i in ("open", "t0", "t1", "t2", "t3")])
    return derived, logs


def test_a_run_that_scored_the_whole_split_is_timed_on_all_of_it(runtime, tree):
    derived, logs = tree
    (logs / "run.log").write_text("4 items in the requested splits, 1 already scored, 4 to score\n"
                                  "2/4 items  10 pairs  20s  0.10 items/s\n4/4 items  20 pairs  40s  0.10 items/s\n")
    timing = runtime.from_log("run.log", PROGRESS, "ragtruth", ("TEST",), logs=logs, derived=derived)
    assert timing["items"] == 4 and timing["items_in_the_splits"] == 4
    assert timing["megabytes"] == pytest.approx(0.04)
    assert timing["seconds"] == 40
    assert timing["seconds_per_megabyte"] == pytest.approx(1000.0)


def test_a_resumed_run_is_timed_on_the_items_it_scored_and_says_how_many(runtime, tree):
    derived, logs = tree
    (logs / "run.log").write_text("4 items in the requested splits, 3 already scored, 2 to score\n"
                                  "2/2 items  10 pairs  30s  0.07 items/s\n")
    timing = runtime.from_log("run.log", PROGRESS, "ragtruth", ("TEST",), logs=logs, derived=derived)
    assert timing["items"] == 2 and timing["items_in_the_splits"] == 4
    assert timing["megabytes"] == pytest.approx(0.02)
    assert timing["seconds_per_megabyte"] == pytest.approx(1500.0)


def test_an_interrupted_run_is_reported_and_not_timed(runtime, tree):
    derived, logs = tree
    (logs / "run.log").write_text("4 items in the requested splits, 1 already scored, 4 to score\n"
                                  "2/4 items  10 pairs  20s  0.10 items/s\nexit=137\n")
    timing = runtime.from_log("run.log", PROGRESS, "ragtruth", ("TEST",), logs=logs, derived=derived)
    assert "seconds_per_megabyte" not in timing
    assert "2 of 4" in timing["status"]


def test_a_missing_log_is_reported_as_missing(runtime, tree):
    derived, logs = tree
    timing = runtime.from_log("absent.log", PROGRESS, "ragtruth", ("TEST",), logs=logs, derived=derived)
    assert timing["status"] == "log missing"


def test_scores_from_another_split_at_the_end_of_the_file_are_refused(runtime, tree):
    derived, logs = tree
    (logs / "run.log").write_text("5 items in the requested splits, 0 already scored, 5 to score\n"
                                  "5/5 items  10 pairs  50s  0.10 items/s\n")
    timing = runtime.from_log("run.log", PROGRESS, "ragtruth", ("TEST",), logs=logs, derived=derived)
    assert "seconds_per_megabyte" not in timing
    assert "not all" in timing["status"]


def test_a_score_is_timed_several_times_and_the_fastest_pass_is_kept(runtime):
    """Another process on the machine can only slow a pass down, so the fastest of
    several passes is the one least disturbed."""
    ticks = iter([0.0, 5.0, 10.0, 12.0, 20.0, 24.0])          # three passes: 5 s, 2 s, 4 s
    calls = []
    timing = runtime.time_score(lambda o, s: calls.append((o, s)), [(b"a", b"b"), (b"c", b"d")],
                                repeats=3, clock=lambda: next(ticks))
    assert timing == {"seconds": 2.0, "passes": [5.0, 2.0, 4.0]}
    assert len(calls) == 6
