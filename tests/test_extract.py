# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LogNovations Holdings LLC
"""The three reference frames, assembled into one feature dictionary."""
from __future__ import annotations

import math
from statistics import fmean

import pytest
from hypothesis import given, settings, strategies as st

from selfref.extract import FeatureInput, extract_features
from selfref.features import DYADIC_SCALES, dyadic_coverage
from selfref.matchlen import match_length_profile, self_match_length_profile
from selfref.ncd import GzipCompressor, ZstdCompressor, ncd

PROFILE_NAMES = [f"cov_k{k}" for k in DYADIC_SCALES] + [
    "rlz_phrases", "rlz_phrases_per_symbol", "lcs", "lcs_frac", "mlen_entropy"]
NCD_NAMES = ["ncd_gzip", "ncd_zstd"]

OUTPUT = b"The store opens at nine and closes at ten thirty on weekdays."
SOURCE = b"hours: Monday 9:0-22:30. The store opens at nine. Parking: lot."
SAMPLES = (b"The store opens at nine and closes late.", b"It opens at 9 on weekdays.")


def keys(frame, names):
    return {f"{frame}.{n}" for n in names}


def test_self_frame_is_always_present_and_alone_without_other_inputs():
    feats = extract_features(FeatureInput(output=OUTPUT))
    assert set(feats) == keys("self", PROFILE_NAMES)


def test_source_frame_appears_only_when_a_source_is_given():
    feats = extract_features(FeatureInput(output=OUTPUT, source=SOURCE))
    assert set(feats) == keys("self", PROFILE_NAMES) | keys("source", PROFILE_NAMES + NCD_NAMES)


def test_resample_frame_appears_only_when_resamples_are_given():
    feats = extract_features(FeatureInput(output=OUTPUT, resamples=SAMPLES))
    assert set(feats) == keys("self", PROFILE_NAMES) | keys("resample", PROFILE_NAMES + NCD_NAMES)


def test_an_empty_source_is_a_source_with_nothing_in_it():
    feats = extract_features(FeatureInput(output=OUTPUT, source=b""))
    assert keys("source", PROFILE_NAMES + NCD_NAMES) <= set(feats)
    assert feats["source.cov_k1"] == 0.0
    assert feats["source.lcs"] == 0.0
    assert feats["source.rlz_phrases"] == float(len(OUTPUT))


def test_output_identical_to_source_is_fully_covered():
    feats = extract_features(FeatureInput(output=SOURCE, source=SOURCE))
    assert feats["source.cov_k1"] == 1.0
    assert feats["source.rlz_phrases"] == 1.0
    assert feats["source.lcs_frac"] == 1.0


def test_frames_use_the_documented_references():
    feats = extract_features(FeatureInput(output=OUTPUT, source=SOURCE, resamples=SAMPLES))
    assert feats["self.cov_k4"] == dyadic_coverage(self_match_length_profile(OUTPUT))[2]
    assert feats["source.lcs"] == float(max(match_length_profile(OUTPUT, [SOURCE])))
    assert feats["resample.lcs"] == float(max(match_length_profile(OUTPUT, list(SAMPLES))))
    assert feats["source.ncd_zstd"] == ncd(OUTPUT, SOURCE, ZstdCompressor())
    assert feats["resample.ncd_gzip"] == pytest.approx(
        fmean(ncd(OUTPUT, s, GzipCompressor()) for s in SAMPLES))


def test_a_different_match_backend_can_be_supplied_for_fixed_references():
    class AlwaysZero:
        name = "zero"

        def profile(self, target, references):
            return [0] * len(target)

    feats = extract_features(FeatureInput(output=OUTPUT, source=SOURCE), match_backend=AlwaysZero())
    assert feats["source.lcs"] == 0.0
    assert feats["self.lcs"] == float(max(self_match_length_profile(OUTPUT)))   # SELF is unaffected


@pytest.mark.parametrize("bad", [
    {"output": OUTPUT, "source": SOURCE},
    (OUTPUT, SOURCE),
    OUTPUT,
    None,
])
def test_only_a_feature_input_is_accepted(bad):
    with pytest.raises(TypeError):
        extract_features(bad)


@pytest.mark.parametrize("kwargs", [
    {"output": "a string, not bytes"},
    {"output": b"ok", "source": "a string"},
    {"output": b"ok", "resamples": [b"a list, not a tuple"]},
    {"output": b"ok", "resamples": ("a string",)},
])
def test_feature_input_validates_its_field_types(kwargs):
    with pytest.raises(TypeError):
        FeatureInput(**kwargs)


@settings(max_examples=60, deadline=None)
@given(st.binary(max_size=80), st.one_of(st.none(), st.binary(max_size=80)),
       st.lists(st.binary(max_size=40), max_size=3).map(tuple))
def test_every_value_is_a_finite_float(output, source, resamples):
    feats = extract_features(FeatureInput(output=output, source=source, resamples=resamples))
    assert all(isinstance(v, float) and math.isfinite(v) for v in feats.values())
