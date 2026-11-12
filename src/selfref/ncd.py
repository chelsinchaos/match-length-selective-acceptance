# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LogNovations Holdings LLC
"""Normalized compression distance (Cilibrasi and Vitanyi, 2005).

    NCD(T, R) = ( C(R + T) - min(C(T), C(R)) ) / max(C(T), C(R))

where C is the length in bytes of a compressor's output. The reference comes
first in the concatenation, so the compressor meets the target with the
reference already in its window. Real compressors are not symmetric, so the
order is part of the definition.

A compressor is any object with a ``name`` and a ``compressed_len(data)``
method. Two open ones are provided, both deterministic:

* gzip at level 9, with the header timestamp fixed at zero;
* zstd at level 19, one frame per call.

gzip's window is 32 KiB, so it cannot see a match between a target and a part
of the reference more than 32 KiB behind it.
"""
from __future__ import annotations

import gzip
from typing import Protocol

import zstandard

__all__ = ["Compressor", "GzipCompressor", "ZstdCompressor", "ncd"]


class Compressor(Protocol):
    name: str

    def compressed_len(self, data: bytes) -> int: ...


class GzipCompressor:
    name = "gzip"
    level = 9

    def compressed_len(self, data: bytes) -> int:
        return len(gzip.compress(data, compresslevel=self.level, mtime=0))


class ZstdCompressor:
    name = "zstd"
    level = 19

    def __init__(self) -> None:
        self._compressor = zstandard.ZstdCompressor(level=self.level)

    def compressed_len(self, data: bytes) -> int:
        return len(self._compressor.compress(data))


def ncd(target: bytes, reference: bytes, compressor: Compressor) -> float:
    """Normalized compression distance between ``target`` and ``reference``."""
    c_target = compressor.compressed_len(target)
    c_reference = compressor.compressed_len(reference)
    largest = max(c_target, c_reference)
    if largest == 0:
        return 0.0
    c_joint = compressor.compressed_len(reference + target)
    return (c_joint - min(c_target, c_reference)) / largest
