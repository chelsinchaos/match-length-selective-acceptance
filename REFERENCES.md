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
vol. 44, no. 3, May 1998, pp. 1319–1327 (the page range and DOI 10.1109/18.669425 are from the
Crossref record, fetched 2026-11-12). Read from the author-hosted copy
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

**Source.** J. Ziv and N. Merhav, "A measure of relative entropy between
individual sequences with application to universal classification", IEEE
Transactions on Information Theory, vol. 39, no. 4, July 1993, pp. 1270–1279,
DOI 10.1109/18.243444 (Crossref record). Read on 2026-11-12 from the scan
hosted by the second author, `https://webee.technion.ac.il/people/merhav/papers/zmm.pdf`
(SHA-256 `e9575d25f1df74451ecaf35a48037f742cc86ff64141ff6cae6af45b6a460855`),
whose running head gives the journal, volume, number and month. The scan's
text layer is unreliable, so the passage was read from the page image.

**What the source says** (Section II, p. 1271).

- "First, find the longest prefix of z that appears as a string in x, i.e., the
  largest integer m such that (z_1, z_2 ⋯ z_m) = (x_i, x_{i+1} ⋯ x_{i+m−1}) for
  some i. The string (z_1, z_2 ⋯ z_m) is defined as the first phrase of z with
  respect to x. If m=0 (i.e. z_1 does not appear in x), the first phrase of z
  with respect to x is z_1. Thus, the case m = 0 is treated as though m = 1.
  Next, start from z_{m+1} and find, in a similar manner, the longest prefix of
  z_{m+1}, z_{m+2} ⋯ z_n, which appears in x, and so on."
- "Let c(z|x) denote the number of phrases in z with respect to x."
- Worked example: with z = (01111000110) and x = (10010100110), "parsing z with
  respect to x yields: (011, 110, 00110), that is c(z|x) = 2."

**How the code corresponds.** `rlz_phrases` takes, at each position, the
longest match in the reference, or one symbol when there is none, and advances
by it. That is the parse quoted above. **Checked against the primary source.**
`tests/test_features.py` runs the article's worked example and obtains the
three phrases it lists, (011, 110, 00110).

**A discrepancy inside the source.** The example lists three phrases and prints
the count as 2. The definition gives 3 for those phrases, and so does the code.
The printed 2 is taken to be a misprint; it is not reproduced.

The harness uses the phrase count and the count per symbol as features. It
does not use the article's relative-entropy estimator, and it does not require
the two strings to have equal length, as the article does.

**Restatement also read.** N. Barnfield, R. Grondin, G. Pozzoli and R. Raquépas,
"On the Ziv-Merhav theorem beyond Markovianity", arXiv:2310.01367v1, Section 1
(arXiv HTML, 2026-11-11, through a fetch agent that returned verbatim
quotations). It states the same parse. The arXiv record gives the journal
version as Canadian Journal of Mathematics 77(3):891–915 (2025).

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

**Journal version.** IEEE Transactions on Information Theory, vol. 51, no. 4,
April 2005, pp. 1523–1545, DOI 10.1109/TIT.2005.844059 (Crossref record,
fetched 2026-11-12; the arXiv record carries no journal reference). The
formula was re-read on 2026-11-12 from the arXiv v2 PDF, p. 6, Eq. (3.1).
**Still not done:** the text of the journal version itself was not read.

## 4. Suffix automaton — `selfref.matchlen.SuffixAutomaton`

The construction is the standard online algorithm usually credited to A.
Blumer, J. Blumer, D. Haussler, A. Ehrenfeucht, M. T. Chen and J. Seiferas, "The
smallest automaton recognizing the subwords of a text", Theoretical Computer
Science 40 (1985), pp. 31–55, DOI 10.1016/0304-3975(85)90157-4 (zbMATH and
Crossref records, fetched 2026-11-12; the Crossref title has "automation").
The article itself was **not read**, so the code is not checked against it. Correctness here does not rest on the citation: both
profiles are tested against brute-force definitions on generated inputs
(`tests/oracles.py`, `tests/test_matchlen.py`).

## 5. Certified selective deployment

Not implemented here. `calibrate_bonferroni` is imported from the companion
repository at a pinned commit; see `src/selfref/selcert_pin.py`. The harness's
reading of its signature and hypotheses is recorded in the pre-registration.

## 6. Other works the report cites

Each bibliographic record below was fetched on 2026-11-12 by a fetch agent that
returned the field values verbatim, from the arXiv API, Crossref, the ACL
Anthology or zbMATH. Where the report quotes a fact from a paper, the sentence
or table cell was read from the paper's PDF.

- **RAGTruth.** C. Niu, Y. Wu, J. Zhu, S. Xu, K. Shum, R. Zhong, J. Song and T.
  Zhang, "RAGTruth: A Hallucination Corpus for Developing Trustworthy
  Retrieval-Augmented Language Models", arXiv:2401.00396v2. Read from the PDF:
  the six generators (GPT-3.5-turbo-0613, GPT-4-0613, Mistral-7b-Instruct,
  Llama-2-7B-chat, Llama-2-13B-chat, and Llama-2-70B-chat, 4-bit quantized);
  the three tasks (question answering, data-to-text writing, news
  summarization); "each response is independently labeled by two annotators
  ... In cases where there is a considerable difference between the two
  annotations, a third review is undertaken"; 17,790 responses.
- **SelfCheckGPT.** P. Manakul, A. Liusie and M. J. F. Gales, "SelfCheckGPT:
  Zero-Resource Black-Box Hallucination Detection for Generative Large Language
  Models", arXiv:2303.08896v3 (EMNLP 2023). Read from the PDF: Table 2, row
  "SelfCheckGPT w/ NLI", sentence-level AUC-PR 92.50 (NonFact) and 66.08
  (Factual), random 72.96 and 27.04; 238 passages and 1,908 sentences; 761
  (39.9%) major-inaccurate, 631 (33.1%) minor-inaccurate, 516 (27.0%) accurate.
- **MiniCheck.** L. Tang, P. Laban and G. Durrett, "MiniCheck: Efficient
  Fact-Checking of LLMs on Grounding Documents", Proceedings of EMNLP 2024,
  pp. 8818–8847, DOI 10.18653/v1/2024.emnlp-main.499; arXiv:2404.10774.
- **Clopper–Pearson.** C. J. Clopper and E. S. Pearson, "The use of confidence
  or fiducial limits illustrated in the case of the binomial", Biometrika
  26(4), 1934, pp. 404–413, DOI 10.1093/biomet/26.4.404 (Crossref). The article
  was not read. The bound used in this study is the calibrator package's
  function, whose source was read.
- **Holm.** S. Holm, "A simple sequentially rejective multiple test procedure",
  Scandinavian Journal of Statistics 6, 1979, pp. 65–70 (zbMATH record). The
  issue number was not confirmed from a primary record, and the article was
  not read. The adjustment as implemented is stated in full in
  `selfref.confirm.holm` and tested.
- **Learn then Test.** A. N. Angelopoulos, S. Bates, E. J. Candès, M. I. Jordan
  and L. Lei, "Learn then Test: Calibrating Predictive Algorithms to Achieve
  Risk Control", arXiv:2110.01052. Cited by the calibrator package; not read
  for this study.
- **Companion paper.** C. McElveen (2026), "Explanation is not certification:
  information-theoretic limits of interpretability-based assurance for machine
  learning systems", in the companion repository at the pinned commit. Theorem
  5 and the calibrator's source were read in Phase 0.

## 7. Works cited by the paper only

Two works are cited in the introduction of `paper/paper.pdf` for context. Their
bibliographic records were fetched on 2026-11-12. The articles were not read
for this study, and nothing in the code or the analysis rests on them.

- **Selective classification.** Y. Geifman and R. El-Yaniv, "Selective
  Classification for Deep Neural Networks", arXiv:1705.08500 (submitted
  2017-05-23; arXiv API record). The record carries no journal reference, so
  the paper cites the preprint.
- **Semantic entropy.** S. Farquhar, J. Kossen, L. Kuhn and Y. Gal, "Detecting
  hallucinations in large language models using semantic entropy", Nature 630,
  625–630 (2024), DOI 10.1038/s41586-024-07421-0 (Crossref record). Cited as
  the source of a baseline that the brief named and that could not be run.

