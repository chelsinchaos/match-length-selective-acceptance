# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LogNovations Holdings LLC
"""Pinned-commit loader for the ``selcert`` package.

Certified selective deployment is not implemented in this harness. It comes
from ``calibrate_bonferroni`` in the companion repository, at one fixed commit.
That repository carries no packaging metadata, so it cannot be installed with
pip; instead ``scripts/fetch_selcert.py`` clones it into ``vendor/`` and this
module refuses to import it unless the checkout is exactly the pinned commit
with no modified, added or deleted file.

Usage::

    from selfref.selcert_pin import load_selcert
    selcert = load_selcert()
    result = selcert.calibrate_bonferroni(scores, losses, grid, alpha, delta)
"""
from __future__ import annotations

import importlib
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType

__all__ = ["PINNED_SHA", "REPO_URL", "VENDOR_DIR", "SelcertPinError",
           "verify_checkout", "load_selcert"]

#: Commit of github.com/chelsinchaos/explanation-is-not-certification read in Phase 0.
PINNED_SHA = "a27fe4fd3fd8726f98036cfb3d7d7fe24edd3fdf"
REPO_URL = "https://github.com/chelsinchaos/explanation-is-not-certification.git"

_HARNESS_ROOT = Path(__file__).resolve().parents[2]
#: Where the checkout lives. ``SELFREF_SELCERT_DIR`` overrides it, for an installed package.
VENDOR_DIR = Path(os.environ.get(
    "SELFREF_SELCERT_DIR",
    _HARNESS_ROOT / "vendor" / "explanation-is-not-certification"))


class SelcertPinError(RuntimeError):
    """The selcert checkout is missing, at another commit, or modified."""


def _git(path: Path, *args: str) -> str:
    done = subprocess.run(["git", "-C", str(path), *args],
                          capture_output=True, text=True)
    if done.returncode != 0:
        raise SelcertPinError(f"git {' '.join(args)} failed in {path}: {done.stderr.strip()}")
    return done.stdout.strip()


def verify_checkout(path: Path = VENDOR_DIR) -> None:
    """Raise ``SelcertPinError`` unless ``path`` is the pinned commit, untouched."""
    path = Path(path)
    if not (path / ".git").exists():
        raise SelcertPinError(
            f"no selcert checkout at {path}; fetch it with `python scripts/fetch_selcert.py`")
    head = _git(path, "rev-parse", "HEAD")
    if head != PINNED_SHA:
        raise SelcertPinError(
            f"selcert checkout at {path} is at commit {head}, expected {PINNED_SHA}")
    changes = _git(path, "status", "--porcelain")
    if changes:
        raise SelcertPinError(
            f"selcert checkout at {path} has been modified:\n{changes}")


def load_selcert(path: Path = VENDOR_DIR) -> ModuleType:
    """Import ``selcert`` from the verified checkout and return the module."""
    path = Path(path)
    verify_checkout(path)
    src = str(path / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    module = importlib.import_module("selcert")
    origin = Path(module.__file__).resolve()
    if not origin.is_relative_to(path.resolve()):
        raise SelcertPinError(
            f"selcert was imported from {origin}, not from the pinned checkout at {path}")
    return module
