# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LogNovations Holdings LLC
"""Gather every record of the study into results.json.

    python scripts/collect_results.py

The report takes every number from results.json and from nowhere else. Each
record inside it keeps its own provenance stamp (the commit it was computed
at, and whether the tree was dirty); the file as a whole is stamped again.
A record that does not exist is listed as absent, with the reason when the
reason is known. The run ledger is included, so that the number of reads of
the sealed splits is a number like any other.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

HARNESS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HARNESS / "src"))

from selfref.guard import _rows  # noqa: E402

RECORDS = [
    "prereg/baseline.json", "prereg/aa.json", "prereg/bar_floor.json", "prereg/power.json",
    "prereg/power_small_gains.json", "prereg/exploratory_design.json", "prereg/fidelity.json",
    "prereg/profile.json", "prereg/rehearsal.json",
    "confirmatory/result.json", "confirmatory/replication.json",
    "controls/rehearsal.json", "controls/result.json",
    "controls/wikibio_rehearsal.json", "controls/wikibio_result.json",
    "controls/acceptance_overlap.json", "controls/open_probes.json",
    "controls/paraphrase.json", "controls/runtime.json", "reproduction/check.json",
    "results/throughput.json",
    "data/derived/ragtruth/build_report.json", "data/derived/wikibio/build_report.json",
    "datasets.manifest.json",
]
ABSENT_BECAUSE = {
    "confirmatory/replication.json": "the replication runs only if H1 is INCONCLUSIVE",
    "reproduction/check.json": "written when a reproduction in a clone is compared with the study's records",
}


def git(*args) -> str:
    return subprocess.run(["git", "-C", str(HARNESS), *args], capture_output=True,
                          text=True).stdout.strip()


def main() -> int:
    records, absent = {}, {}
    for name in RECORDS:
        path = HARNESS / name
        if path.exists():
            records[name] = json.loads(path.read_text())
        else:
            absent[name] = ABSENT_BECAUSE.get(name, "not produced")
    rows = _rows(HARNESS / "RUNLEDGER.md")
    ledger = {"entries": [dict(zip(("number", "time_utc", "split", "access_to_this_split",
                                    "commit", "purpose"), row)) for row in rows],
              "accesses": len(rows),
              "test_accesses": sum(1 for row in rows if row[2] == "TEST"),
              "cal_accesses": sum(1 for row in rows if row[2] == "CAL")}
    results = {"provenance": {"git_commit": git("rev-parse", "HEAD"),
                              "git_dirty": bool(git("status", "--porcelain", "--", ".")),
                              "prereg_tag_commit": git("rev-parse", "--verify", "--quiet",
                                                       "refs/tags/prereg-v1^{commit}")},
               "ledger": ledger, "absent": absent, "records": records}
    (HARNESS / "results.json").write_text(json.dumps(results, indent=1) + "\n")
    print(f"wrote results.json: {len(records)} records, {len(absent)} absent "
          f"({', '.join(absent) or 'none'}); {ledger['accesses']} ledger rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
