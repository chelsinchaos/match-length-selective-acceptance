# selfref — match-length profile features for selective acceptance

A pre-registered research harness. It tests one question:

> Do multi-scale match-length statistics of an LLM output add certified
> coverage, over a strong combination of existing scores, when the output is
> accepted or deferred by a rule with a finite-sample error bound?

The outcome was not known in advance. A clean null result is as useful as a
positive one, and the result is reported in full, whichever it is.

**What this is and is not.** The quantities computed here are candidate
*scores for selective acceptance*. The acceptance rule built on them carries an
error bound that holds under exchangeability of the calibration and deployment
data. Nothing here verifies that an output is true.

## Status

The study is complete. `paper/paper.pdf` is the paper. `REPORT.md` is the full
record behind it: the result, every control, the limitations, the deviations
and the findings of an independent audit.

- **Primary hypothesis (H1): FAIL** under the pre-registered decision rule.
  The gain in certified coverage from the SOURCE features is below the bar the
  rule set. The pre-registration fixed in advance how that is read: as "the
  gain is below the bar", which includes a real but smaller gain, and not as
  evidence that the features carry no signal.
- H3 (SELF features): FAIL. H2 (RESAMPLE features): not testable on the
  available data.
- The design could not resolve small gains. That was established by a power
  analysis before any tested model was fitted, and the report says what the
  outcome does and does not show.

The pre-registration (`PREREGISTRATION.md`) was approved and tagged
`prereg-v1` before any calibration or test label was read. Every result in the
report is read from `results.json`. Departures from the pre-registration and
incidents are logged in `DEVIATIONS.md`, and every read of a sealed split in
`RUNLEDGER.md`.

## Where this repository comes from

The study was run in a private repository that also holds proprietary code,
none of which is here. The public repository,
<https://github.com/chelsinchaos/match-length-selective-acceptance>, is an export of the harness. It has two commits: the tree as it was
when the pre-registration was tagged, which carries the tag `prereg-v1`, and
the final tree.

The commit hashes in the records, in `RUNLEDGER.md` and in `REPORT.md` are
those of the private repository, and `HISTORY.md` lists them with their times
and subjects. They cannot be checked against the public history. The
pre-registration was not deposited with a third party before the confirmatory
run, so that it came first rests on the private repository's history. The
report says so among its limitations.

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
PREREGISTRATION.md          the plan, frozen at the tag prereg-v1
REPORT.md                   the results, rendered from report/REPORT.template.md
paper/paper.pdf             the paper; paper.tex is rendered from paper/paper.template.tex
results.json                every record of the study, and the run ledger
RUNLEDGER.md, DEVIATIONS.md reads of the sealed splits; departures from the plan
HISTORY.md                  the commits of the repository the study ran in
REFERENCES.md               each formula, and what it was checked against

src/selfref/matchlen.py     suffix automaton; the two profiles
src/selfref/features.py     profile -> derived features
src/selfref/extract.py      FeatureInput; the three frames; extract_features
src/selfref/ncd.py, baselines.py, groundedness.py, selfcheck.py, tokens.py
                            the baselines, and the profile over word tokens
src/selfref/datasets/       readers and builders; sealed labels are dropped at parse
src/selfref/splits.py, dedup.py, tables.py
src/selfref/combiner.py     the one model search every arm uses
src/selfref/certify.py      rank transform, grid, per-generator calibration
src/selfref/selcert_pin.py  loads the pinned calibrator
src/selfref/experiment.py, decision.py, power.py, confirm.py, controls.py
                            the procedures, as functions of arrays
src/selfref/guard.py        the guard on the sealed splits
src/selfref/render.py       report placeholders into results.json

scripts/                    one script per step; scripts/reproduce.py runs them all
prereg/, confirmatory/, controls/   the records each step wrote, with provenance
reproduction/               the comparison of a reproduction in a clone with those records
tests/                      closed forms, brute-force oracles, property tests, record checks
```

## Run

```bash
make venv            # .venv with the pinned requirements (requirements.lock is the full lock)
make fetch-selcert   # the calibrator, at the pinned commit
make test            # the test suite
make smoke           # a small synthetic end-to-end experiment
make docker          # the same two, inside the container
make report          # results.json -> REPORT.md and its figures
make paper           # results.json -> paper/paper.tex, its figures and paper/paper.pdf (needs pdflatex)
```

To rebuild every number from the raw data, clone the repository, and in the
clone run `make reproduce`, then `make compare REF=<revision of the published
records>`. `scripts/reproduce.py` explains what it does, how long it takes,
and why it commits as it goes.

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
- **This tree stands alone.** Every number in the report is
  reproducible with the open reference backend in this directory. A second
  backend built on a proprietary codec was compared with it privately. Its
  measurements are not published (`DEVIATIONS.md`), and it is not part of this
  release.

## Licence

MIT; see `LICENSE`.
