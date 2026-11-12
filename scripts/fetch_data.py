# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LogNovations Holdings LLC
"""Fetch the raw dataset files listed in datasets.manifest.json and check them.

    python scripts/fetch_data.py            # download what is missing; verify SHA-256
    python scripts/fetch_data.py --record   # first fetch: write sizes and SHA-256 into the manifest

Each file is pinned to an upstream commit or revision by its URL. A file whose
SHA-256 differs from the manifest stops the script. Raw files live under
data/raw/, which is git-ignored; only their hashes are committed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

HARNESS = Path(__file__).resolve().parents[1]
MANIFEST = HARNESS / "datasets.manifest.json"


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def download(url: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "selfref-harness/0.1"})
    partial = target.with_suffix(target.suffix + ".part")
    with urllib.request.urlopen(request, timeout=120) as response, open(partial, "wb") as out:
        for block in iter(lambda: response.read(1 << 20), b""):
            out.write(block)
    partial.replace(target)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--record", action="store_true",
                        help="write sizes and SHA-256 values that the manifest does not have yet")
    args = parser.parse_args(argv)

    manifest = json.loads(MANIFEST.read_text())
    changed = False
    for entry in manifest["files"]:
        target = HARNESS / entry["local"]
        if not target.exists():
            print(f"downloading {entry['local']}")
            download(entry["url"], target)
        size, digest = target.stat().st_size, sha256_of(target)
        if entry["sha256"] is None:
            if not args.record:
                raise SystemExit(f"{entry['local']}: no SHA-256 in the manifest; "
                                 f"run with --record to write {digest}")
            entry["bytes"], entry["sha256"] = size, digest
            changed = True
            print(f"recorded  {entry['local']}  {size} bytes  sha256 {digest}")
        elif (entry["sha256"], entry["bytes"]) != (digest, size):
            raise SystemExit(f"{entry['local']}: SHA-256 or size differs from the manifest\n"
                             f"  manifest {entry['bytes']} bytes {entry['sha256']}\n"
                             f"  on disk  {size} bytes {digest}")
        else:
            print(f"verified  {entry['local']}  {size} bytes  sha256 {digest}")
    if changed:
        MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
