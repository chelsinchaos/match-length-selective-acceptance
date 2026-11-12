# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LogNovations Holdings LLC
"""Clone the selcert companion repository at the pinned commit into vendor/.

    python scripts/fetch_selcert.py

Idempotent: an existing checkout is moved to the pinned commit and verified.
The only network access is a clone of the public companion repository.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from selfref.selcert_pin import PINNED_SHA, REPO_URL, VENDOR_DIR, verify_checkout  # noqa: E402


def main() -> int:
    if not (VENDOR_DIR / ".git").exists():
        VENDOR_DIR.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--quiet", REPO_URL, str(VENDOR_DIR)], check=True)
    subprocess.run(["git", "-C", str(VENDOR_DIR), "checkout", "--quiet", "--detach", PINNED_SHA],
                   check=True)
    verify_checkout()
    print(f"selcert at {PINNED_SHA} in {VENDOR_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
