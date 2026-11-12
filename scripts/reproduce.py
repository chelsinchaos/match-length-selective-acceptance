# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LogNovations Holdings LLC
"""Rebuild every number, table and figure of the report from the raw data.

    python scripts/reproduce.py --commit                      # everything, from raw data
    python scripts/reproduce.py --commit --scores-from DIR    # reuse the neural baselines' scores
    python scripts/reproduce.py --compare REF                 # compare the records with those at REF
    python scripts/reproduce.py --compare REF --record FILE   # and write the comparison as a record

Run it in a fresh clone, on its own branch. The steps are the ones the study
ran, in the same order and with the same seeds:

    data        fetch the raw files and check their SHA-256; build the item tables
    scores      the label-free score tables; the two neural baselines
    phase 2     BASELINE, the A/A test, BAR and FLOOR, power, the tested arms, the DEV rehearsals
    confirm     the confirmatory analysis on CAL and TEST, once
    controls    the controls and secondary metrics; the WikiBio ranking analysis; the paraphrase stress
    report      results.json, the tables and figures, REPORT.md

Three things to know before running it.

1. Reads of the sealed splits go through the guard, here as everywhere. The
   guard wants a clean tree, and each read adds a row to RUNLEDGER.md, so the
   regenerated records have to be committed between steps. ``--commit`` does
   that, with messages that start "reproduction:". Without it the script stops
   at the first step that needs a clean tree and says what to commit.
2. The neural baselines take hours (about nine on an Apple M1 Max) and the
   paraphrase model about two more. ``--scores-from DIR`` copies their score
   files from the derived-data directory of an earlier run instead; everything
   else is still rebuilt from the raw files.
3. ``--compare REF`` loads every record at the git revision REF and in the
   working tree and reports where they differ, ignoring provenance stamps,
   ledger rows and timings. On the machine the study ran on, nothing else
   should differ. On another backend the neural scores can differ in their
   last decimals, and numbers downstream of them can move.

   The SHA-256 of a fitted model's file is compared apart from the numbers.
   The file holds the number of threads the model was fitted with, so the hash
   differs between machines with different core counts while the model's
   scores are the same. A record whose numbers agree and whose model hash
   differs is reported as identical, with a note.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

HARNESS = Path(__file__).resolve().parents[1]
PY = sys.executable
OPEN = ["TRAIN", "DEV_SELECT", "DEV_CAL", "DEV_TEST"]
SEALED = ["CAL", "TEST"]
VOLATILE = {"provenance", "ledger_entries", "ledger", "seconds", "seconds_per_megabyte",
            "environment", "megabytes", "cpu_scores", "neural_baselines"}
MODEL_HASH = "model_sha256"         # compared apart from the numbers: see the docstring
RECORD_DIRS = ("prereg", "confirmatory", "controls")
PARAPHRASE = "data/derived/ragtruth_paraphrase"


def sh(*command: str) -> None:
    print("\n$ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=HARNESS, check=True)


def script(name: str, *args: str) -> None:
    sh(PY, f"scripts/{name}", *args)


def git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(HARNESS), *args], capture_output=True,
                          text=True).stdout.strip()


def checkpoint(message: str, commit: bool) -> None:
    """Commit the regenerated records, so that the next step starts from a clean tree."""
    if not git("status", "--porcelain", "--", "."):
        return
    if not commit:
        raise SystemExit(f"\nThe next step needs a clean tree. Commit the regenerated records "
                         f"({message}), or run with --commit, then start again.")
    subprocess.run(["git", "-C", str(HARNESS), "add", "--", "."], check=True)
    subprocess.run(["git", "-C", str(HARNESS), "commit", "-q", "-m", f"reproduction: {message}"],
                   check=True)
    print(f"committed: reproduction: {message}")


def copy_scores(source: Path) -> None:
    for relative in ("ragtruth/groundedness.jsonl", "wikibio/selfcheck_nli.jsonl",
                     "ragtruth_paraphrase/paraphrases.jsonl", "ragtruth_paraphrase/groundedness.jsonl"):
        target = HARNESS / "data" / "derived" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / relative, target)
        print(f"copied {relative} from {source}")


def run(args) -> int:
    reuse = args.scores_from is not None
    # data
    script("fetch_selcert.py")
    script("fetch_data.py")
    script("build_tables.py", "ragtruth")
    script("build_tables.py", "wikibio")
    # scores
    script("compute_features.py", "ragtruth", "--splits", *OPEN, *SEALED)
    script("compute_features.py", "wikibio", "--splits", *OPEN, "TEST")
    script("compute_control_features.py", "tokens", "--splits", *OPEN, *SEALED)
    script("compute_control_features.py", "shuffled", "--splits", "DEV_TEST", "TEST")
    if reuse:
        copy_scores(args.scores_from)
    else:
        script("score_groundedness.py", "ragtruth", "--splits", *OPEN, *SEALED)
        script("score_selfcheck.py", "--splits", *OPEN, "TEST")
    script("fidelity_selfcheck.py")
    # phase 2
    script("phase2.py", "baseline")
    script("phase2.py", "aa")
    script("phase2.py", "barfloor")
    checkpoint("BASELINE, the A/A test, BAR and FLOOR", args.commit)   # the tested arms need them committed
    script("phase2.py", "power")
    script("phase2.py", "power", "--multiples", "0.25", "0.5", "0.75", "--out", "power_small_gains.json")
    script("explore_design.py")
    script("phase2.py", "profile")
    script("confirm.py", "rehearsal")
    script("controls.py", "rehearsal")
    script("wikibio_ranking.py", "rehearsal")
    # confirm
    checkpoint("power, the tested arms and the DEV rehearsals", args.commit)
    script("confirm.py", "confirmatory", "--i-have-prereg-approval", "--purpose",
           "reproduction of the confirmatory run")
    # controls
    checkpoint("the confirmatory result", args.commit)
    script("controls.py", "confirmatory", "--i-have-prereg-approval", "--purpose",
           "reproduction of the controls and secondary metrics")
    script("acceptance_overlap.py")          # label-free; uses the thresholds the controls run recorded
    script("open_probes.py")                 # open splits and label-free tables only
    checkpoint("the controls", args.commit)
    script("wikibio_ranking.py", "confirmatory", "--i-have-prereg-approval", "--purpose",
           "reproduction of the WikiBio ranking analysis")
    script("paraphrase.py", "generate")      # with reused scores it only writes the item tables
    script("compute_features.py", "ragtruth", "--splits", "TEST", "--derived", PARAPHRASE)
    if not reuse:
        script("score_groundedness.py", "ragtruth", "--splits", "TEST", "--derived", PARAPHRASE)
    script("paraphrase.py", "analyse")
    script("runtime.py")
    sh(PY, "bench/throughput.py", "--out", "results/throughput.json")
    # report
    script("collect_results.py")
    script("make_report.py")
    checkpoint("the remaining records, results.json and the report", args.commit)
    print("\nDone. Compare with the published records: python scripts/reproduce.py --compare <REF>")
    return 0


def strip(value):
    """The record without its stamps, timings and model-file hashes."""
    if isinstance(value, dict):
        return {k: strip(v) for k, v in value.items() if k not in VOLATILE and k != MODEL_HASH}
    if isinstance(value, list):
        return [strip(v) for v in value]
    return value


def model_hashes(value) -> list:
    """Every model-file hash in a record, in the order of its keys."""
    if isinstance(value, dict):
        found = []
        for key, inner in value.items():
            found += [inner] if key == MODEL_HASH else model_hashes(inner)
        return found
    if isinstance(value, list):
        return [h for inner in value for h in model_hashes(inner)]
    return []


def ledger_rows(text: str) -> list:
    """The rows of a run ledger, read as the guard reads them: six cells, the first a number."""
    rows = []
    for line in text.splitlines():
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) == 6 and cells[0].isdigit():
            rows.append(cells)
    return rows


def comparison_record(reference: str, rows: list, ledger_before: list, ledger_now: list) -> dict:
    """The outcome of a comparison, as a record. ``rows`` has one entry per
    compared record; the two ledgers are the rows of RUNLEDGER.md at the
    reference and in the working tree."""
    added = ledger_now[len(ledger_before):]
    fields = ("number", "time_utc", "split", "access_to_this_split", "commit", "purpose")
    return {
        "reference": reference,
        "ignored": sorted(VOLATILE) + [MODEL_HASH + " (compared apart)"],
        "records": rows,
        "records_compared": len(rows),
        "records_identical": sum(r["first_difference"] is None for r in rows),
        "records_identical_with_a_different_model_hash":
            sum(r["first_difference"] is None and r["model_hash_differs"] for r in rows),
        "ledger_rows_at_the_reference": len(ledger_before),
        "sealed_reads_by_the_reproduction": {
            "CAL": sum(row[2] == "CAL" for row in added),
            "TEST": sum(row[2] == "TEST" for row in added),
            "rows": [dict(zip(fields, row)) for row in added]}}


def first_difference(a, b, path=""):
    if type(a) is not type(b):
        return f"{path}: {a!r} != {b!r}"
    if isinstance(a, dict):
        for key in sorted(set(a) | set(b)):
            if key not in a or key not in b:
                return f"{path}/{key}: present on one side only"
            found = first_difference(a[key], b[key], f"{path}/{key}")
            if found:
                return found
    elif isinstance(a, list):
        if len(a) != len(b):
            return f"{path}: lengths {len(a)} and {len(b)}"
        for index, (x, y) in enumerate(zip(a, b)):
            found = first_difference(x, y, f"{path}[{index}]")
            if found:
                return found
    elif a != b:
        return f"{path}: {a!r} != {b!r}"
    return None


def compare(ref: str, record_path=None) -> int:
    top = Path(git("rev-parse", "--show-toplevel")).resolve()
    prefix = str(HARNESS.resolve().relative_to(top))
    show = lambda name: subprocess.run(["git", "-C", str(top), "show", f"{ref}:{prefix}/{name}"],
                                       capture_output=True, text=True)
    names = sorted(str(p.relative_to(HARNESS)) for d in RECORD_DIRS
                   for p in (HARNESS / d).glob("*.json"))
    rows = []
    for name in names:
        shown = show(name)
        if shown.returncode != 0:
            print(f"ABSENT AT {ref}  {name}")
            rows.append({"record": name, "first_difference": f"absent at {ref}", "model_hash_differs": False})
            continue
        old, new = json.loads(shown.stdout), json.loads((HARNESS / name).read_text())
        found = first_difference(strip(old), strip(new))
        hash_differs = model_hashes(old) != model_hashes(new)
        note = "   first difference at " + found if found else (
            "   (model file hash differs; numbers agree)" if hash_differs else "")
        print(f"{'DIFFERENT' if found else 'identical'}  {name}{note}")
        rows.append({"record": name, "first_difference": found, "model_hash_differs": hash_differs})
    differing = sum(r["first_difference"] is not None for r in rows)
    print(f"\n{len(names) - differing} of {len(names)} records identical to {ref} "
          f"(provenance stamps, ledger rows, timings and model file hashes ignored)")
    if record_path is not None:
        record = comparison_record(git("rev-parse", ref), rows, ledger_rows(show("RUNLEDGER.md").stdout),
                                   ledger_rows((HARNESS / "RUNLEDGER.md").read_text()))
        record = {"provenance": {"git_commit": git("rev-parse", "HEAD"),
                                 "git_dirty": bool(git("status", "--porcelain", "--", "."))}, **record}
        Path(record_path).write_text(json.dumps(record, indent=1) + "\n")
        print(f"wrote {record_path}")
    return 1 if differing else 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--commit", action="store_true",
                        help="commit regenerated records where a step needs a clean tree")
    parser.add_argument("--scores-from", type=Path, default=None, metavar="DIR",
                        help="a data/derived directory holding the neural baselines' score files")
    parser.add_argument("--compare", default=None, metavar="REF")
    parser.add_argument("--record", type=Path, default=None, metavar="FILE",
                        help="with --compare: write the comparison to FILE as a record")
    args = parser.parse_args(argv)
    return compare(args.compare, args.record) if args.compare else run(args)


if __name__ == "__main__":
    raise SystemExit(main())
