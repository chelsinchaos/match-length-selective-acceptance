# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LogNovations Holdings LLC
"""Splits by hash of a group key.

Every item belongs to a group (a source document, or a cluster of
near-duplicate sources). A group's split is a pure function of its key and a
fixed salt, so no group can cross splits and the assignment cannot depend on
any outcome.

    u = first 8 bytes of SHA-256(salt + NUL + key), as a fraction of 2**64

and the split is the interval of the cumulative fractions that u falls in.

The six splits

    TRAIN        fit the combiners
    DEV_SELECT   choose among fitted combiners; fit the rank transform
    DEV_CAL      rehearsal of calibration
    DEV_TEST     rehearsal of the test
    CAL          calibration, sealed until the pre-registration is tagged
    TEST         the confirmatory test, sealed until then

DEV_CAL and DEV_TEST have the same fractions as CAL and TEST, so that the
rehearsal has the sample sizes of the confirmatory run.
"""
from __future__ import annotations

import hashlib
from typing import Sequence

__all__ = ["SPLITS", "FRACTIONS", "SEALED", "unit_interval", "assign_split"]

SPLITS: tuple[str, ...] = ("TRAIN", "DEV_SELECT", "DEV_CAL", "DEV_TEST", "CAL", "TEST")
FRACTIONS: tuple[float, ...] = (0.30, 0.10, 0.20, 0.10, 0.20, 0.10)
#: Splits whose labels may not be read before the pre-registration is tagged.
SEALED: tuple[str, ...] = ("CAL", "TEST")


def unit_interval(key: str, salt: str) -> float:
    """A number in [0, 1) determined by ``salt`` and ``key`` alone."""
    digest = hashlib.sha256(salt.encode("utf-8") + b"\x00" + key.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / 2 ** 64


def assign_split(group_key: str, salt: str, fractions: Sequence[float] = FRACTIONS) -> str:
    """The split of a group."""
    u = unit_interval(group_key, salt)
    edge = 0.0
    for name, fraction in zip(SPLITS, fractions):
        edge += fraction
        if u < edge:
            return name
    return SPLITS[len(fractions) - 1]          # u fell in the rounding gap below 1.0
