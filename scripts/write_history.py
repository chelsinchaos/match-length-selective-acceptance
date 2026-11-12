# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LogNovations Holdings LLC
"""Write HISTORY.md: the commits of the study's repository that touched this tree.

    python scripts/write_history.py

Run it in the repository the study ran in. The public repository is an export
with a history of its own, so running it there would list that history instead
of the study's; the script refuses to do that.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

HARNESS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HARNESS / "src"))

from selfref.guard import PREREG_TAG, _rows  # noqa: E402

HEAD = """# History of the study's repository

The study was run in a private repository that also holds proprietary code. The
public repository is an export of the harness and has a history of its own: one
commit with the tree as it was when the pre-registration was tagged, and one
with the final tree.

The commit hashes in the records, in `RUNLEDGER.md` and in `REPORT.md` are those
of the private repository. This file lists every commit there that touched the
public part of the harness, oldest first, with its time and its subject, so
that a reader can follow the order of events. The list cannot be checked
against the public repository, where these commits do not exist. The commit
that holds this file is not in it: a file cannot name the commit that contains
it.

Times are UTC.

| Commit | Time (UTC) | Subject | Note |
|---|---|---|---|
"""


def git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(HARNESS), *args], capture_output=True, text=True,
                          env={"TZ": "UTC", "PATH": "/usr/bin:/bin:/usr/local/bin:/opt/homebrew/bin"}).stdout.strip()


def main() -> int:
    tagged = git("rev-parse", "--verify", "--quiet", f"refs/tags/{PREREG_TAG}^{{commit}}")
    rows = _rows(HARNESS / "RUNLEDGER.md")
    if not tagged or any(git("cat-file", "-t", row[4]) != "commit" for row in rows):
        raise SystemExit("this is not the repository the study ran in; HISTORY.md is left as it is")
    private = ":(exclude)" + "fastcodec" + "_backend"          # paths are relative to this directory
    log = git("log", "--reverse", "--date=format-local:%Y-%m-%d %H:%M:%S", "--format=%H|%ad|%s",
              "--", ".", private)
    reads = {}
    for row in rows:
        reads.setdefault(row[4], []).append(row)
    lines = []
    for entry in log.splitlines():
        commit, when, subject = entry.split("|", 2)
        notes = []
        if commit == tagged:
            tag_time = git("for-each-ref", f"refs/tags/{PREREG_TAG}",
                           "--format=%(taggerdate:format-local:%Y-%m-%d %H:%M:%S)")
            notes.append(f"tagged `{PREREG_TAG}` at {tag_time}")
        if commit in reads:
            times = sorted({row[1] for row in reads[commit]})
            purposes = sorted({row[5] for row in reads[commit]})
            notes.append(f"sealed labels read at this commit ({', '.join(times)}): {'; '.join(purposes)}")
        lines.append(f"| `{commit[:7]}` | {when} | {subject.replace('|', '/')} | {'; '.join(notes)} |")
    (HARNESS / "HISTORY.md").write_text(HEAD + "\n".join(lines) + "\n")
    print(f"wrote HISTORY.md: {len(lines)} commits")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
