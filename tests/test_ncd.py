# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LogNovations Holdings LLC
"""Normalized compression distance and the two open compressors."""
from __future__ import annotations

import math
import random

import pytest

from selfref.ncd import GzipCompressor, ZstdCompressor, ncd

TEXT = (b"The court gave the Palestinian Authority jurisdiction over alleged crimes "
        b"in the occupied territories, a step that both sides had expected. ") * 6
NOISE = random.Random(20261111).randbytes(len(TEXT))

COMPRESSORS = [GzipCompressor(), ZstdCompressor()]


class FixedLengths:
    """A compressor whose output lengths are given by a table."""
    name = "stub"

    def __init__(self, table):
        self.table = table

    def compressed_len(self, data: bytes) -> int:
        return self.table[bytes(data)]


def test_compressor_names_and_levels_are_fixed():
    assert (GzipCompressor().name, GzipCompressor().level) == ("gzip", 9)
    assert (ZstdCompressor().name, ZstdCompressor().level) == ("zstd", 19)


@pytest.mark.parametrize("compressor", COMPRESSORS, ids=lambda c: c.name)
@pytest.mark.parametrize("data", [b"", b"a", TEXT, NOISE], ids=["empty", "one", "text", "noise"])
def test_compressed_length_is_positive_and_repeatable(compressor, data):
    first = compressor.compressed_len(data)
    assert isinstance(first, int) and first > 0
    assert compressor.compressed_len(data) == first
    assert type(compressor)().compressed_len(data) == first      # a fresh instance agrees


@pytest.mark.parametrize("compressor", COMPRESSORS, ids=lambda c: c.name)
def test_text_compresses_and_noise_does_not(compressor):
    assert compressor.compressed_len(TEXT) < len(TEXT) / 2
    assert compressor.compressed_len(NOISE) >= len(NOISE)


def test_ncd_formula_with_the_reference_first_in_the_concatenation():
    # C(R + T) = 70, C(T) = 40, C(R) = 60  ->  (70 - 40) / 60 = 0.5
    # Only b"RT" is in the table, so concatenating in the other order would raise.
    stub = FixedLengths({b"T": 40, b"R": 60, b"RT": 70})
    assert ncd(b"T", b"R", stub) == pytest.approx(0.5)


def test_ncd_is_zero_when_the_compressor_reports_zero_lengths():
    stub = FixedLengths({b"": 0})
    assert ncd(b"", b"", stub) == 0.0


@pytest.mark.parametrize("compressor", COMPRESSORS, ids=lambda c: c.name)
def test_ncd_to_itself_is_smaller_than_ncd_to_noise(compressor):
    assert ncd(TEXT, TEXT, compressor) < ncd(TEXT, NOISE, compressor)


@pytest.mark.parametrize("compressor", COMPRESSORS, ids=lambda c: c.name)
@pytest.mark.parametrize("target, reference", [(b"", b""), (b"abc", b""), (b"", b"abc")])
def test_ncd_with_empty_inputs_is_finite(compressor, target, reference):
    assert math.isfinite(ncd(target, reference, compressor))
