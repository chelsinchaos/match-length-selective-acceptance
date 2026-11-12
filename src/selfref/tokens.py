# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LogNovations Holdings LLC
"""The match-length profile over word tokens instead of bytes.

A stress analysis of the study recomputes the profile features on sequences of
lower-cased word tokens. The tokens are the same ones the lexical baselines
use. Each distinct token of an item gets an integer code, shared by the output
and its source, and the profile and its features are then computed exactly as
for bytes: a symbol is a word where it was a byte.

Columns are named ``tok.source.<feature>`` and ``tok.self.<feature>``.
"""
from __future__ import annotations

from selfref.baselines import word_tokens
from selfref.features import profile_features
from selfref.matchlen import match_length_profile, self_match_length_profile

__all__ = ["token_ids", "token_profile_features"]


def token_ids(*texts: str) -> list[list[int]]:
    """The word tokens of each text as integer codes from one shared table."""
    codes: dict[str, int] = {}
    return [[codes.setdefault(token, len(codes)) for token in word_tokens(text)]
            for text in texts]


def token_profile_features(output: bytes, source: bytes) -> dict[str, float]:
    """The SOURCE and SELF profile features of one item over word tokens."""
    out_ids, src_ids = token_ids(output.decode("utf-8", errors="replace"),
                                 source.decode("utf-8", errors="replace"))
    feats = profile_features(self_match_length_profile(out_ids), prefix="tok.self")
    feats.update(profile_features(match_length_profile(out_ids, [src_ids]),
                                  prefix="tok.source", reference_length=len(src_ids)))
    return feats
