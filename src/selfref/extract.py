# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LogNovations Holdings LLC
"""Feature extraction over the three reference frames.

Frames

* SELF      the reference at position i is the output's own prefix.
* SOURCE    the reference is the context the output should be grounded in.
* RESAMPLE  the references are other samples drawn for the same prompt.

What a feature function is allowed to see. ``extract_features`` takes a
``FeatureInput`` and nothing else. A ``FeatureInput`` holds the output bytes,
optionally the source bytes, and optionally the resample bytes. It has no
identifier and no field for a judgment about the output, it is immutable, and
it has no instance dictionary, so nothing further can be attached to it. Code
that knows whether an output was right or wrong therefore has no way to hand
that knowledge to this module.
"""
from __future__ import annotations

from dataclasses import dataclass
from statistics import fmean
from typing import Protocol, Sequence

from selfref.features import profile_features
from selfref.matchlen import match_length_profile, self_match_length_profile
from selfref.ncd import Compressor, GzipCompressor, ZstdCompressor, ncd

__all__ = ["FeatureInput", "MatchBackend", "ReferenceMatchBackend",
           "DEFAULT_COMPRESSORS", "extract_features"]


@dataclass(frozen=True, slots=True)
class FeatureInput:
    """Everything a feature function may see about one item."""

    output: bytes
    source: bytes | None = None
    resamples: tuple[bytes, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.output, bytes):
            raise TypeError("output must be bytes")
        if self.source is not None and not isinstance(self.source, bytes):
            raise TypeError("source must be bytes or None")
        if not isinstance(self.resamples, tuple) or \
                not all(isinstance(sample, bytes) for sample in self.resamples):
            raise TypeError("resamples must be a tuple of bytes")


class MatchBackend(Protocol):
    """Computes the match-length profile of a target against fixed references."""

    name: str

    def profile(self, target: bytes, references: Sequence[bytes]) -> list[int]: ...


class ReferenceMatchBackend:
    """The open suffix-automaton backend. Every headline number uses this one."""

    name = "reference-suffix-automaton"

    def profile(self, target: bytes, references: Sequence[bytes]) -> list[int]:
        return match_length_profile(target, references)


DEFAULT_COMPRESSORS: tuple[Compressor, ...] = (GzipCompressor(), ZstdCompressor())


def extract_features(item: FeatureInput,
                     compressors: Sequence[Compressor] = DEFAULT_COMPRESSORS,
                     match_backend: MatchBackend | None = None) -> dict[str, float]:
    """Features of one item, keyed ``"<frame>.<name>"``.

    The SELF features are always present. The SOURCE features are present when
    ``item.source`` is not ``None`` (an empty source is a source with nothing
    in it, and yields zeros). The RESAMPLE features are present when at least
    one resample is given; their NCD is the mean of the pairwise distances.

    ``match_backend`` replaces the profile computation for the fixed-reference
    frames. The SELF frame always uses the reference implementation.
    """
    if type(item) is not FeatureInput:
        raise TypeError("extract_features accepts a FeatureInput and nothing else")
    backend = match_backend if match_backend is not None else ReferenceMatchBackend()

    feats = profile_features(self_match_length_profile(item.output), prefix="self")

    if item.source is not None:
        profile = backend.profile(item.output, [item.source])
        feats.update(profile_features(profile, prefix="source",
                                      reference_length=len(item.source)))
        for compressor in compressors:
            feats[f"source.ncd_{compressor.name}"] = float(
                ncd(item.output, item.source, compressor))

    if item.resamples:
        profile = backend.profile(item.output, list(item.resamples))
        feats.update(profile_features(
            profile, prefix="resample",
            reference_length=sum(len(sample) for sample in item.resamples)))
        for compressor in compressors:
            feats[f"resample.ncd_{compressor.name}"] = float(fmean(
                ncd(item.output, sample, compressor) for sample in item.resamples))

    return feats
