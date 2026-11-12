# Sources, and how each formula was checked

The rule for this harness is that every formula and citation is checked against
its primary source before it enters the report. This file records, for each
formula in the code, what it was checked against, where in that source, on what
date, and what remains unchecked.

Status words: **checked against the primary source**; **checked against a
restatement** (a later peer-reviewed or arXiv paper that states the definition,
when the original could not be obtained); **not yet checked**.

## 1. Match-length entropy estimators — `selfref.features`

**Source.** I. Kontoyiannis, P. H. Algoet, Yu. M. Suhov and A. J. Wyner,
"Nonparametric entropy estimation for stationary processes and random fields,
with applications to English text", IEEE Transactions on Information Theory,
vol. 44, no. 3, May 1998, starting at p. 1319. Read from the author-hosted copy
at `https://www.dpmms.cam.ac.uk/~ik355/PAPERS/suhov2.pdf` on 2026-11-11; the
running head of that copy gives the journal, volume, number and month.

**What the source says** (Section I, p. 1320).

- "For n ≥ 1 let L_n denote the minimum length k such that the string X_0^{k−1}
  that starts at time 0 does not appear as a continuous substring within the
  past X_{−n}^{−1}. Alternatively, L_n is obtained by adding 1 to the longest
  match-length."
- "Λ_i^n(X) = L_n(T^i X), the length of the shortest substring X_i^{i+k−1}
  starting at position i that does not appear as a contiguous substring of the
  previous n symbols X_{i−n}^{i−1}."
- "logarithms are to base 2 throughout this correspondence".
- Theorem 1, for a stationary ergodic process with entropy rate H > 0, under a
  Doeblin condition:
  - (a) lim (1/n) Σ_{i=1}^{n} Λ_i^n / log n = 1/H
  - (b) lim (1/n) Σ_{i=2}^{n} Λ_i^i / log i = 1/H
  - (c) lim (1/n) Σ_{i=1}^{n} Λ_i^i / log n = 1/H

**How the code corresponds.**

- The SELF profile requires the earlier occurrence to lie wholly inside the
  prefix, as "within the past" does, so `L[i] + 1` is Λ_i^i.
  **Checked against the primary source.**
- `self_match_length_entropy` is the reciprocal of the average in (b), with n
  the index of the last position. **Checked against the primary source.** The
  first implementation divided by the number of terms instead of by n; that was
  corrected on 2026-11-11 after this check.
- `match_length_entropy` (fixed reference of length N) has the form of (a) with
  the window of previous symbols replaced by the reference. That substitution
  is ours. The source proves nothing about two different sequences, and the
  harness claims nothing about what the quantity estimates. It is a feature.

**Caveat.** The theorem concerns an unbounded realization. On a finite output a
match is cut short by the end of the data, so values near the end are biased
downward. The features are used as scores, not as entropy estimates.

## 2. Relative Lempel–Ziv parse — `selfref.features.rlz_phrases`

**Original source.** J. Ziv and N. Merhav, "A measure of relative entropy
between individual sequences with application to universal classification",
IEEE Transactions on Information Theory, vol. 39, no. 4, pp. 1270–1279, 1993.
**Not obtained**: no open copy was found on 2026-11-11. The volume and page
numbers above come from a bibliographic index and from the restatement below,
not from the article itself.

**Restatement used.** N. Barnfield, R. Grondin, G. Pozzoli and R. Raquépas,
"On the Ziv–Merhav theorem beyond Markovianity", arXiv:2310.01367v1, Section 1,
read from the arXiv HTML on 2026-11-11 through a fetch agent that returned
verbatim quotations.

- c_N(y|x) is "the number of words in a sequential parsing of y_1^N using the
  longest possible substrings of x_1^N".
- "if there is no such substring of x_1^N, the parsed word is set to be one
  letter long."
- The estimator built on it is Q̂_N(y, x) := c_N(y|x) ln N / N, which the paper
  states converges to the specific cross entropy under its assumptions.

**How the code corresponds.** `rlz_phrases` takes, at each position, the
longest match in the reference, or one symbol when there is none, and advances
by it. That is the parse quoted above. **Checked against a restatement.** The
harness uses the phrase count and the count per symbol as features. It does not
use the estimator Q̂_N, and it does not require the two strings to have equal
length, which the restatement assumes.

**Still to do.** Check the parse against the 1993 article itself.

## 3. Normalized compression distance — `selfref.ncd.ncd`

**Source.** R. Cilibrasi and P. Vitányi, "Clustering by compression",
arXiv:cs/0312044v2 (revised 9 April 2004), Section 3, read from the arXiv HTML
on 2026-11-11 through a fetch agent that returned verbatim quotations.

- Equation (3.1): NCD(x, y) = (C(xy) − min{C(x), C(y)}) / max{C(x), C(y)}, where
  "C(xy) denotes the compressed size of the concatenation of x and y".
- Remark 3.1: a real compressor may give C(xy) different from C(yx); the authors
  "simply use C(xy)".
- The stated range is 0 ≤ r ≤ 1 + ε.
- Section 5.2 keeps inputs within gzip's 32-kilobyte window in one experiment.
  The paper gives no general statement about inputs longer than the window.

**How the code corresponds.** `ncd(target, reference, compressor)` computes
(3.1) with the reference first in the concatenation. **Checked against the
arXiv version.**

**Still to do.** The journal version (IEEE Transactions on Information Theory,
2005) was not read, and its volume and page numbers are not confirmed.

## 4. Suffix automaton — `selfref.matchlen.SuffixAutomaton`

The construction is the standard online algorithm usually credited to Blumer,
Blumer, Haussler, Ehrenfeucht, Chen and Seiferas (1985). **Not yet checked**
against that paper. Correctness here does not rest on the citation: both
profiles are tested against brute-force definitions on generated inputs
(`tests/oracles.py`, `tests/test_matchlen.py`).

## 5. Certified selective deployment

Not implemented here. `calibrate_bonferroni` is imported from the companion
repository at a pinned commit; see `src/selfref/selcert_pin.py`. The harness's
reading of its signature and hypotheses is recorded in the pre-registration.
