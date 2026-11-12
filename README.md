# selfref — match-length profile features for selective acceptance

A pre-registered research harness. It tests one question:

> Do multi-scale match-length statistics of an LLM output add certified
> coverage, over a strong combination of existing scores, when the output is
> accepted or deferred by a rule with a finite-sample error bound?

The outcome is not known in advance. A clean null result is as useful as a
positive one, and both will be reported in full.

**What this is and is not.** The quantities computed here are candidate
*scores for selective acceptance*. The acceptance rule built on them carries an
error bound that holds under exchangeability of the calibration and deployment
data. Nothing here verifies that an output is true.

## Status

Phase 1 of five: the feature library, its tests and a throughput benchmark. No
dataset has been read. Hypotheses, the decision rule and the analysis plan are
fixed in a pre-registration before any calibration or test data is opened.

## The primitive

For a target byte string `T` and a reference `R`, the match-length profile is

    L[i] = length of the longest substring of T starting at i that occurs in R.

From it: coverage of `T` by matches of length at least `k` at the dyadic scales
`k = 1, 2, 4, …, 1024`; the relative Lempel–Ziv phrase count; a match-length
entropy estimate; and the longest common substring. Normalized compression
distance with gzip and zstd is computed alongside.

Three reference frames:

| Frame | Reference |
|---|---|
| SELF | the output's own prefix |
| SOURCE | the context the output should be grounded in |
| RESAMPLE | other samples drawn for the same prompt |

Exact definitions are in the module docstrings of `src/selfref/`.

## Layout

```
src/selfref/matchlen.py     suffix automaton; the two profiles
src/selfref/features.py     profile -> derived features
src/selfref/ncd.py          gzip and zstd compressors; NCD
src/selfref/extract.py      FeatureInput; the three frames; extract_features
src/selfref/selcert_pin.py  loads the pinned calibrator
scripts/fetch_selcert.py    clones the calibrator at the pinned commit
bench/throughput.py         seconds per megabyte, with environment capture
tests/                      closed forms, brute-force oracles, property tests
```

## Run

```bash
make venv            # .venv with the pinned requirements
make fetch-selcert   # the calibrator, at the pinned commit
make test
make bench
```

## Design rules the tests enforce

- **Feature code cannot see labels.** `extract_features` accepts only a
  `FeatureInput`, which holds the output, the source and the resamples and has
  nowhere to put anything else. Static checks reject any parameter or identifier
  in the feature modules that could carry a judgment about an output, and any
  import outside a short allowlist. See `tests/test_label_blindness.py`.
- **Every definition is checked against a brute-force oracle** on generated
  inputs, as well as on closed-form cases (target equal to reference, disjoint
  alphabets, empty strings). See `tests/oracles.py`.
- **The calibrator is not reimplemented.** Certified selective deployment comes
  from `calibrate_bonferroni` in the companion repository
  `chelsinchaos/explanation-is-not-certification`, pinned at commit
  `a27fe4fd3fd8726f98036cfb3d7d7fe24edd3fdf`. The loader refuses a checkout at
  any other commit or with any modified file.
- **This tree stands alone.** Every number in the eventual report is
  reproducible with the open reference backend in this directory. A second
  backend built on a proprietary codec is reported separately for agreement and
  throughput, and is not part of this release.

## Licence

MIT; see `LICENSE`.
