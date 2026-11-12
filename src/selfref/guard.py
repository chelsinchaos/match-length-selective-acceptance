# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LogNovations Holdings LLC
"""Guard on the sealed splits.

The labels of CAL and TEST may be read only by the confirmatory run. Before
anything reads them, ``authorise`` checks that

* the caller passed the explicit flag;
* the git tag ``prereg-v1`` exists;
* HEAD descends from that tag;
* PREREGISTRATION.md at HEAD is byte-identical to the tagged version;
* there is no uncommitted change in the harness directory,

and then appends a row to RUNLEDGER.md with the time, the split, the commit and
the stated purpose. A refusal writes nothing. The ledger is how the number of
TEST accesses is reported.
"""
from __future__ import annotations

import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from selfref.splits import SEALED

__all__ = ["PREREG_TAG", "PREREG_FILE", "SealedAccessRefused", "authorise",
           "authorise_all", "access_count"]

PREREG_TAG = "prereg-v1"
PREREG_FILE = "PREREGISTRATION.md"
_HEADER = "| # | time (UTC) | split | access to this split | commit | purpose |"


class SealedAccessRefused(RuntimeError):
    """A sealed split was requested and a condition for reading it does not hold."""


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True)


def _rows(ledger: Path) -> list[list[str]]:
    if not Path(ledger).exists():
        return []
    rows = []
    for line in Path(ledger).read_text(encoding="utf-8").splitlines():
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) == 6 and cells[0].isdigit():
            rows.append(cells)
    return rows


def access_count(ledger: Path, split: Optional[str] = None) -> int:
    """Number of recorded accesses, in total or to one split."""
    return sum(1 for row in _rows(ledger) if split is None or row[2] == split)


def _check(splits, purpose: str, flag: bool, repo: Path) -> str:
    """Raise unless every condition for reading these sealed splits holds. Returns HEAD."""
    for split in splits:
        if split not in SEALED:
            raise ValueError(f"{split} is not a sealed split")
    if not purpose.strip():
        raise ValueError("state the purpose of the access")
    if not flag:
        raise SealedAccessRefused("reading a sealed split needs the explicit flag")
    tagged = _git(repo, "rev-parse", "--verify", "--quiet", f"refs/tags/{PREREG_TAG}^{{commit}}")
    if tagged.returncode != 0:
        raise SealedAccessRefused(f"there is no {PREREG_TAG} tag: the pre-registration "
                                  f"has not been approved and tagged")
    if _git(repo, "merge-base", "--is-ancestor", PREREG_TAG, "HEAD").returncode != 0:
        raise SealedAccessRefused(f"HEAD does not descend from {PREREG_TAG}")
    if _git(repo, "diff", "--quiet", PREREG_TAG, "HEAD", "--", PREREG_FILE).returncode != 0:
        raise SealedAccessRefused(f"{PREREG_FILE} has changed since it was tagged {PREREG_TAG}")
    if _git(repo, "status", "--porcelain", "--", ".").stdout.strip():
        raise SealedAccessRefused("there are uncommitted changes; commit them first so the "
                                  "run is tied to one commit")
    return _git(repo, "rev-parse", "HEAD").stdout.strip()


def _record(split: str, purpose: str, commit: str, ledger: Path) -> dict:
    entry = {"number": access_count(ledger) + 1,
             "time": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
             "split": split, "access_number": access_count(ledger, split) + 1,
             "commit": commit, "purpose": " ".join(purpose.split()).replace("|", "/")}
    ledger = Path(ledger)
    text = ledger.read_text(encoding="utf-8") if ledger.exists() else "# Run ledger\n"
    if _HEADER not in text:
        text = text.rstrip("\n") + "\n\n" + _HEADER + "\n|---|---|---|---|---|---|\n"
    text += (f"| {entry['number']} | {entry['time']} | {split} | {entry['access_number']} "
             f"| {commit} | {entry['purpose']} |\n")
    ledger.write_text(text, encoding="utf-8")
    return entry


def authorise(split: str, purpose: str, *, flag: bool, repo: Path, ledger: Path) -> dict:
    """Check every condition for reading one sealed split, then record the access."""
    commit = _check((split,), purpose, flag, repo)
    return _record(split, purpose, commit, ledger)


def authorise_all(splits, purpose: str, *, flag: bool, repo: Path, ledger: Path) -> list[dict]:
    """Check the conditions once, then record one access per split.

    The confirmatory run reads CAL and TEST together. They are authorised in
    one step because the first ledger row would otherwise make the tree dirty
    and block the second.
    """
    commit = _check(tuple(splits), purpose, flag, repo)
    return [_record(split, purpose, commit, ledger) for split in splits]
