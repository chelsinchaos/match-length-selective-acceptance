# Pre-registration: do match-length profile features add certified coverage?

**Status: APPROVED by the owner on 2026-11-12 (recorded at 03:03 PST; 11:03 UTC) and tagged `prereg-v1`.** Sections 1 to 8 and 10 to 19 were written on 2026-11-11 before any combiner had been fitted. The values of Sections 8, 9 and 12 were then computed from BASELINE alone on the open splits and committed. Section 12 shows that the design does not detect a gain the size of BAR with 80% power. The study stopped there, as the brief requires, and on 2026-11-11 at 21:22 PST the owner decided to proceed with the design as written (Section 12). The additions made on that decision are listed in Section 17. They were committed before any combiner that includes a tested column was fitted. The two tested arms were then fitted, frozen and rehearsed in DEV; what that showed is recorded in Section 17. No calibration or test label has been read.

Everything here was fixed without knowledge of any outcome on the calibration (CAL) or test (TEST) splits. Dates are local (PST) unless marked UTC. After tagging, any departure is entered in the deviations log, with a timestamp and a reason (Section 19).

---

## 1. Question and wording

Do multi-scale match-length statistics of an LLM output add **certified coverage**, over a strong combination of existing scores, when the output is accepted or deferred by a rule with a finite-sample error bound?

The quantities studied are **scores for selective acceptance**. The acceptance rule built on them carries an error bound that holds under exchangeability of the calibration and deployment data. Nothing in this study verifies that an output is true, and no result will be described that way.

The outcome is not known. A null result will be reported in full.

## 2. Hypotheses, hierarchy and priors

The primitive is the match-length profile of a target byte string T against a reference R: L[i] is the length of the longest substring of T starting at i that occurs in R. Three reference frames:

| Frame | Reference |
|---|---|
| SOURCE | the context the output should be grounded in |
| RESAMPLE | other samples drawn for the same prompt |
| SELF | the output's own prefix |

Hypotheses, in a fixed hierarchy:

| | Statement | Role | Prior that it is supported | Set by |
|---|---|---|---|---|
| H1 | SOURCE features add certified coverage over the best baseline combination | primary | 35–55% | owner, in the brief |
| H2 | RESAMPLE features do | secondary | 2–8% | set at the owner's request on 2026-11-11, before the dataset was opened |
| H3 | SELF features do | secondary, expected to fail | 5–15% | owner, in the brief |

## 3. The calibrator

Certified selective deployment is not implemented in this harness. It is `calibrate_bonferroni` from the `selcert` package of `github.com/chelsinchaos/explanation-is-not-certification`, pinned at commit `a27fe4fd3fd8726f98036cfb3d7d7fe24edd3fdf`. The loader refuses any other commit or a modified checkout, and tests reject any calibrator or binomial bound defined in this harness.

What it does. `calibrate_bonferroni(scores, losses, grid, alpha, delta)` accepts an output when its score is at or below a threshold. For each of the K grid thresholds it computes a one-sided Clopper–Pearson upper bound, at level 1 − delta/K, on the error rate among accepted calibration items. It returns the largest threshold whose bound is at most alpha, or none, which means defer everything.

What it guarantees (Theorem 5 of the companion paper). If the score, the model and the grid are fixed independently of the calibration sample, and the calibration pairs are i.i.d. from the deployment distribution, then with probability at least 1 − delta over the calibration draw the error rate among accepted outputs is at most alpha, or everything is deferred. Deferring everything counts as success. The guarantee is marginal: a subgroup can carry a higher risk.

How its hypotheses are met here is the subject of Section 7.

## 4. Data

### 4.1 RAGTruth: the confirmatory dataset

Source: Niu et al., arXiv:2401.00396; repository `ParticleMedia/RAGTruth` at commit `c103204b9ce28d6bbad859304bf30de72b8ed8fe`, MIT licence. Raw files and their SHA-256 values:

| File | Bytes | SHA-256 |
|---|---|---|
| `dataset/response.jsonl` | 21,458,735 | `e4c2e4ac24fff676d8984cc61c35d791612fadc58015335d97dd632375e18073` |
| `dataset/source_info.jsonl` | 15,117,971 | `0dffc26ea9f3c1c3d7c7e8336b56ef1646e3cec876edffcca3c9c624d12d578b` |

- **Unit.** One LLM response to one prompt. Each source has six responses, one from each of six generators.
- **Ground truth.** The dataset's human span annotations: two annotators per response, with a third review on considerable disagreement.
- **0/1 loss.** 1 when the response has at least one annotated span of any type; 0 otherwise. No row is excluded. This is the dataset's own published count of responses with hallucinations. Its paper gives no response-level definition in words, so this one is stated here.
- **SOURCE reference.** The `prompt` field, the exact text the generator was given, as UTF-8 bytes. The same reference is used for every task type, for the features under test and for every baseline.
- **Group.** The source (`source_id`). All six responses of a source share a split.
- **Near-duplicates.** Two sources are near-duplicates when the Jaccard similarity of their sets of lower-cased word 5-grams, computed on the source content, is at least 0.9. Clusters are formed by transitive closure and the member with the smallest identifier is kept. One cluster was found; one source and its six responses were dropped.
- **Deployment covariates.** Generator, task type and temperature are known when an output is produced. They are baseline inputs (Section 5.2).

**Splits.** A group's split is determined by the first eight bytes of SHA-256 of the salt `selfref-prereg-v1/ragtruth`, a NUL byte and the group key, read as a fraction, against cumulative fractions 0.30, 0.10, 0.20, 0.10, 0.20, 0.10.

| Split | Role | Groups | Responses |
|---|---|---|---|
| TRAIN | fit combiners | 747 | 4,482 |
| DEV_SELECT | select among fitted combiners; fit the rank transform | 238 | 1,428 |
| DEV_CAL | rehearsal of calibration | 526 | 3,156 |
| DEV_TEST | rehearsal of the test | 235 | 1,410 |
| CAL | calibration; **sealed** | 504 | 3,024 |
| TEST | confirmatory test; **sealed** | 264 | 1,584 |

**Sealing.** The labels of CAL and TEST are discarded in the statement that parses each row and are written to no file. They can be read only through `selfref.guard`, which requires an explicit flag, the `prereg-v1` tag, HEAD descending from it, this file unchanged since the tag, and a clean tree, and which appends every access to `RUNLEDGER.md`.

### 4.2 The reserved replication set

The RAGTruth authors' own `test` split: 450 sources and 2,700 responses, identified by the dataset's `split` field. It was chosen by the owner on 2026-11-11. Its rows share a file with the rest, so "not loaded" means this: a row's split is read from its raw bytes by a regular expression, and a row whose split is `test` is never parsed. The reader's tests make such rows unparseable to prove it. Only their number, 2,700, is known.

### 4.3 WikiBio GPT-3 hallucination: the dataset for H2

Source: Manakul, Liusie and Gales, arXiv:2303.08896; Hugging Face `potsawee/wiki_bio_gpt3_hallucination` at revision `b3cfb73209a8c51582fa1d9b7fe7e45fec5529b2`, CC BY-SA 3.0. One file, 2,561,507 bytes, SHA-256 `dc95a59a899ee48deb40c711346b75f05706ae1ebaec4315688e31ae790625fa`. Chosen by the owner on 2026-11-11 in place of generating outputs for this study.

- **Unit.** One sentence of a GPT-3 passage: 1,908 sentences in 238 passages.
- **0/1 loss.** 1 unless the sentence is labelled `accurate`. This groups major and minor inaccuracies as non-factual, as the paper does.
- **References.** The passage's 20 stochastic samples.
- **Group.** The passage. Splits use the salt `selfref-prereg-v1/wikibio` and the same fractions: 78 / 17 / 41 / 25 / 52 / 25 passages.

**H2 cannot be certified on this dataset.** Sentences of one passage are not independent, so an i.i.d. calibration sample has one sentence per passage: 52 items in CAL. If all 52 were accepted and none were an error, the calibrator's bound at 20 grid points and delta 0.05 would be 0.1088, which is above every alpha in this study. This follows from the sample size alone and was established before any outcome was examined. H2 is therefore recorded as **not testable** under the decision rule. The ranking analysis in Section 15 is reported for the RESAMPLE features as a secondary metric and does not change that status.

### 4.4 Leads examined and not used

| Lead | Why it is not confirmatory ground truth |
|---|---|
| OpenHalDet (arXiv:2606.06959) | Its labels come from a GPT-4o-mini judge, and its labelled data is not distributed |
| HaluEval | Its source-grounded examples are hallucinations written by ChatGPT on instruction; its human-labelled part has no source |
| FaithBench | 660 summaries of 66 passages selected because detectors disagreed on them; non-commercial licence |

## 5. Scores

Every score is computed from the output and its references alone. The code that computes scores cannot receive a label: its input type has three fields (output, source, resamples), and static checks reject any parameter, identifier or import through which a label could arrive.

### 5.1 Features under test

From a match-length profile L of length n:

| Feature | Definition |
|---|---|
| Dyadic coverage, k = 1, 2, 4, …, 1024 (11 columns) | the share of positions j for which some start i ≤ j has L[i] ≥ k and i + L[i] > j |
| Relative Lempel–Ziv phrase count, and per symbol | greedy parse: from i = 0, take a phrase of length max(L[i], 1) and advance |
| Longest common substring, and as a share of n | max L |
| Match-length entropy estimate | fixed reference of length N: n · log2 N / Σ (L[i] + 1). SELF: n′ / Σ_{i=2}^{n′} (L[i] + 1) / log2 i with n′ the last index |

That is 16 columns per frame. The reference backend is a suffix automaton in pure Python; every number in the report comes from it. Sources for the formulas, and how each was checked, are in `REFERENCES.md`.

- **H1** tests the 16 SOURCE columns on RAGTruth.
- **H3** tests the 16 SELF columns on RAGTruth.
- **H2** would test the 16 RESAMPLE columns on WikiBio.

Normalized compression distance with gzip and zstd is **not** among the tested features. It is a baseline.

### 5.2 Baselines

On RAGTruth, BASELINE has these inputs:

| Baseline | Columns |
|---|---|
| Output length | bytes, words, sentences |
| N-gram overlap with the source | share of the output's word n-gram occurrences absent from the source, n = 1..4 |
| ROUGE-L against the source | precision, recall, F |
| Open groundedness model | MiniCheck-DeBERTa-v3-Large (`lytang/MiniCheck-DeBERTa-v3-Large` at `2f2d01a54fa022a7ffadb76260e1ea8bc88c82bb`), used as its reference inference code uses it: 400-token chunks that never split a sentence, maximum support over chunks. Each sentence of the response is a claim; minimum, mean and share below 0.5 |
| Compression distance | NCD with gzip and with zstd |
| Deployment covariates | generator, task type, temperature |

A fixed pseudo-random score is reported on its own as a control and enters no combiner.

### 5.3 Baselines in the brief that are not run, and why

| Baseline | Reason |
|---|---|
| Verbalized confidence | RAGTruth holds none, and its generators cannot be asked after the fact |
| Token log-probability, token entropy | need generations of our own; none are run, by the owner's choice of WikiBio for H2. The WikiBio release holds no token probabilities |
| Semantic entropy | needs token probabilities or clustered whole-prompt samples; not available for RAGTruth, and not defined per sentence on WikiBio |
| SelfCheck-style consistency | needs resamples; available and run on WikiBio only (Section 13 and 15) |

## 6. Combiner and arms

A combiner maps a set of columns to one risk score, the estimated probability of an error. Counts (lengths, phrase counts, longest matches) pass through log(1 + x) in every arm.

Arms on RAGTruth:

| Arm | Columns |
|---|---|
| BASELINE | Section 5.2 |
| BASELINE+SOURCE | BASELINE and the 16 SOURCE columns |
| BASELINE+SELF | BASELINE and the 16 SELF columns |

Every arm is fitted with the same search and the same seed (20261111): 8 L2 logistic regressions on standardised columns (C from 0.01 to 30) and 16 gradient-boosted tree models (depth 2 or 3; learning rate 0.05 or 0.1; 100 or 300 iterations; leaf size 20 or 50). Fit on TRAIN. Select the smallest log loss on DEV_SELECT. Freeze, and record the SHA-256 of the frozen model.

BAR and FLOOR are computed and committed before any combiner that includes a tested column is fitted. The script that fits those arms refuses to run otherwise.

## 7. From score to certified coverage

1. **Rank transform.** A frozen risk score is mapped to [0, 1] by the empirical distribution of that score on DEV_SELECT, separately for each generator: the rank of a score is the share of that generator's DEV_SELECT scores at or below it.
2. **Grid.** Twenty uniform thresholds, 0.05, 0.10, …, 1.00. The grid depends on no data.
3. **Calibration per generator.** The six responses of a source share that source, so the pooled calibration sample is not i.i.d. Within one generator each source contributes one response, and sources are exchangeable under the hash split, so that sub-sample is. The calibrator is run once per generator at confidence delta/6. With probability at least 1 − delta all six certificates hold together, and then the error rate among all accepted outputs, a weighted average of the six, is at most alpha.
4. **Certified coverage** on a sample is the share of all its items that the six thresholds accept, in percentage points.

Everything in steps 1 to 3 is fixed before CAL is read: the frozen models, the rank transforms, the grid and K, alpha and delta.

**An alternative that was examined and not adopted.** The defaults accepted on 2026-11-11 named a fallback in case the power analysis did not support calibration per generator: keep one response per source, its generator chosen by a hash of the source, and calibrate once at delta. After the power gate triggered (Section 12), the fallback was examined with BASELINE alone on the open splits (`prereg/exploratory_design.json`). BASELINE's coverage in the DEV rehearsal rises to 31.56 points, but the 99th percentile of the A/A gap rises to 16.46, so the rule would set BAR = FLOOR = 16.46 points. Its power was not simulated. The owner's decision to proceed with the design as written keeps calibration per generator.

## 8. Error levels and the primary alpha

alpha ∈ {0.01, 0.05, 0.10}; delta = 0.05.

From sample size alone: with calibration per generator, the smallest error-free accepted set that can be certified has 74 items at alpha = 0.10, 152 at alpha = 0.05 and 775 at alpha = 0.01. CAL has 504 sources per generator, so **alpha = 0.01 cannot be certified** and will show zero coverage in every arm. It is kept in the tables to document this.

**Rule for the primary alpha**, written into the analysis script before any rehearsal was run: the strictest alpha at which BASELINE certifies at least 5 points of coverage in the DEV rehearsal; failing that, the loosest alpha at which it certifies anything; failing that, none, and the study stops.

**BASELINE-only evidence** (`prereg/baseline.json`). The frozen BASELINE is the boosted-tree model with depth 3, learning rate 0.1, 100 iterations and leaf size 50, the smallest log loss on DEV_SELECT (0.4285) among the 24 configurations; SHA-256 of the frozen model `a5d0dcd2cc79eb07b43abc1eac62c25c1b9558e28ecd4e4eca0318c28951e41a`. Its AUROC is 0.879 on DEV_SELECT, 0.900 on DEV_CAL and 0.906 on DEV_TEST. In the DEV rehearsal (thresholds certified on DEV_CAL, coverage measured on DEV_TEST):

| alpha | Certified coverage, points | Accepted of 1,410 | Realised risk among accepted |
|---|---|---|---|
| 0.01 | 0.00 | 0 | none accepted |
| 0.05 | 0.00 | 0 | none accepted |
| 0.10 | 9.15 | 129 | 0.039 |

All of the coverage at alpha = 0.10 comes from one generator: GPT-3.5 is certified at threshold 0.55, which accepts 54.9% of its outputs, and the other five generators defer everything.

Primary alpha: **0.10**, the strictest alpha at which BASELINE certifies at least 5 points.

## 9. Primary endpoint and decision rule

**Endpoint.** The certified coverage of BASELINE+SOURCE minus that of BASELINE, in percentage points, at the primary alpha and delta = 0.05, with thresholds certified on CAL and coverage measured on TEST.

**Interval.** A paired bootstrap of 10,000 draws. Each draw resamples whole source groups with replacement within task type, in CAL and in TEST, recertifies both arms on the resampled CAL and measures both on the resampled TEST, so both arms always see the same resample. The interval is the 2.5th to the 97.5th percentile. Seed 20261121. The brief asks for stratification by dataset; with one confirmatory dataset, the strata are its three task types.

**Decision rule, fixed in the brief.**

- c_B = BASELINE certified coverage in the DEV rehearsal at the primary alpha.
- BAR = the smaller of 5 points and 10% of (100 − c_B).
- FLOOR = the larger of 2 points and the 99th percentile of the baseline-versus-baseline gap from the A/A test.
- If BAR is below FLOOR, BAR equals FLOOR.

| Outcome | Condition |
|---|---|
| PASS | the point estimate is at or above BAR and the entire 95% interval is above FLOOR |
| FAIL | the upper end of the 95% interval is below BAR; this includes a real but too-small effect |
| INCONCLUSIVE | anything else; it triggers exactly one replication (Section 11) |

Two readings the rule's text leaves open are fixed as follows. The A/A gap is taken in absolute value, which can only raise FLOOR. "Above" and "below" are strict.

**A/A test.** 1,000 runs. Each fits two BASELINE models with the frozen configuration, differing only in their seed and in the bootstrap resample of TRAIN groups they are fitted on, puts both through the DEV rehearsal at the primary alpha, and records the difference in coverage.

The A/A distribution (`prereg/aa.json`; all 1,000 gaps are recorded, and a seeded rerun reproduced them): absolute gap median 1.28 points, 90th percentile 6.88, 95th 9.08, 99th 10.78, maximum 15.74.

| Quantity | Value, points | Exact value in `prereg/bar_floor.json` |
|---|---|---|
| c_B | 9.15 | 9.148936170212766 |
| 99th percentile of the absolute A/A gap | 10.78 | 10.780851063829786 |
| FLOOR = max(2, 10.78) | **10.78** | 10.780851063829786 |
| BAR = min(5, 10% of (100 − 9.15)) = 5, which is below FLOOR, so BAR = FLOOR | **10.78** | 10.780851063829786 |

These were committed (commit `0273903`) before any combiner that includes a tested column was fitted. A test in the suite recomputes them from the committed records.

If neither arm certifies any coverage, the gain is exactly zero with a degenerate interval, and the rule returns FAIL.

## 10. Secondary hypotheses

- **H3.** The same endpoint, interval and rule with BASELINE+SELF in place of BASELINE+SOURCE.
- **H2.** Not testable (Section 4.3).

**Holm correction.** The secondary family is {H2, H3}. The p-value of a gain is the share of its bootstrap draws at or below FLOOR. H2 enters with p = 1, so H3's p-value is doubled. A secondary hypothesis is supported when its outcome is PASS and its Holm-adjusted p-value is below 0.05.

Anything not described in this document is labelled exploratory in the report.

## 11. Replication

Run only if H1 is INCONCLUSIVE, and then exactly once, with nothing changed: the same frozen models, rank transforms, grid, alpha, delta, BAR, FLOOR, bootstrap and rule.

The reserved set's sources are assigned to a calibration part and a test part by the same hash rule with the salt `selfref-prereg-v1/ragtruth-replication` and fractions 2/3 and 1/3, which is the CAL to TEST ratio. A reserved source that is a near-duplicate, by the rule of Section 4.1, of any source used in the primary analysis is dropped first. The replication's outcome under the rule is the study's outcome for H1.

- If the replication is itself INCONCLUSIVE, that is the study's outcome for H1. No further data are used.
- H3 is judged on the primary analysis only. An INCONCLUSIVE H3 is reported as not supported and does not trigger the replication. If the replication runs, H3's numbers on it are reported as exploratory.
- The replication's power has not been simulated. Its two parts are assigned two thirds and one third of the reserved sources, at most 450 in all, against 504 and 264 in the primary analysis, so it should be expected to resolve less than the primary analysis does (Section 12).

## 12. Power

By simulation at the CAL and TEST sizes, on a population on which the truth is known. The population is BASELINE's ranks and losses on DEV_CAL and DEV_TEST. A replication draws, with replacement and within task type, as many source groups as CAL has and, independently, as many as TEST has. A second arm is built from the baseline by multiplying the rank of every correct output by (1 − s), which moves correct outputs toward acceptance; s is searched so that the mean gain equals a target. Each replication goes through the paired bootstrap (400 draws) and the decision rule. 300 replications per target; targets are 1, 1.5, 2 and 3 times BAR.

Two properties of the rule shape how power is read:

- PASS requires the point estimate to reach BAR. When the true gain equals BAR the estimate falls on either side about equally often, so PASS has a probability near one half at best at that effect size.
- The study therefore reports, for each target, the probability that the whole interval lies above FLOOR, which is the conventional sense of detecting the effect, together with the probabilities of PASS, INCONCLUSIVE and FAIL.

The brief's stop condition, "the design cannot detect a gain the size of BAR with at least 80% power", is evaluated on the probability that the whole interval lies above FLOOR when the true gain is BAR.

**Result** (`prereg/power.json`; the first three rows are a supplement at smaller effects, `prereg/power_small_gains.json`). alpha = 0.10, BAR = FLOOR = 10.78 points; CAL has 504 source groups and TEST 264.

| Mean true gain, points | Whole interval above FLOOR | PASS | INCONCLUSIVE | FAIL |
|---|---|---|---|---|
| 3.04 (0.25 × BAR) | 0.00 | 0.00 | 0.21 | 0.79 |
| 5.37 (0.5 × BAR) | 0.00 | 0.00 | 0.79 | 0.21 |
| 8.27 (0.75 × BAR) | 0.00 | 0.00 | 0.98 | 0.02 |
| 10.40 (1.0 × BAR) | 0.00 | 0.00 | 1.00 | 0.00 |
| 16.05 (1.5 × BAR) | 0.18 | 0.18 | 0.82 | 0.00 |
| 21.62 (2.0 × BAR) | 0.89 | 0.89 | 0.11 | 0.00 |
| 32.45 (3.0 × BAR) | 1.00 | 1.00 | 0.00 | 0.00 |

The mean true gain is the mean over replications at the tilt strength found for each target; thresholds sit on a grid, so it does not equal the target exactly. The standard deviation of the gain across replications rises from 2.7 points at the smallest effect to 6.3 at the largest.

**The stop condition is met: the design does not detect a gain the size of BAR with 80% power.** At that effect size no replication had its whole interval above FLOOR. The rule returns a decisive outcome in about four runs in five only when the true gain is about 3 points or less (FAIL) or about 21.6 points or more (PASS); between about 5 and 16 points it returns INCONCLUSIVE in 79% to 100% of runs.

**Decision at the gate.** The owner was shown this result on 2026-11-11 with three options: proceed with the design as written, stop and report the work up to this point, or change the rule, the endpoint or the data before anything further was seen. The owner chose to proceed with the design as written; the choice was recorded on 2026-11-11 at 21:22 PST (2026-11-12, 05:22 UTC). The decision rule, the endpoint, the datasets, the splits, the calibration, the grid, alpha, BAR and FLOOR are unchanged. At that time no combiner that includes a tested column had been fitted and no calibration or test label had been read.

**How the outcome will be read at this size.** Fixed here so that the outcome is read the same way whatever it is.

- **PASS** is to be expected only if the true gain is about 20 points or more, which is more than twice BASELINE's whole coverage in the DEV rehearsal.
- **FAIL** means the upper end of the interval is below BAR, 10.78 points. It will be reported as "the gain is below 10.78 points". That includes a real but smaller gain, and it will not be described as evidence that the features carry no signal.
- **INCONCLUSIVE** is the likeliest outcome for any true gain between about 5 and 16 points. It will be reported as neither support nor refutation, and it sends H1 to the replication of Section 11.
- The report prints the table above next to the outcome.
- The ranking analysis of Section 15 resolves smaller differences than this endpoint can. It is a secondary metric. Whatever it shows, it does not change the outcome of H1 under the rule and will not be described as support for H1.

## 13. Fidelity check

Published: SelfCheckGPT with NLI detects non-factual sentences of WikiBio GPT-3 with an AUC-PR of 92.50 over all 1,908 sentences (arXiv:2303.08896, Table 2). This study computes the same score with the same checkpoint (`potsawee/deberta-v3-large-mnli` at `83f4359862aba52c07506cd23d4454db77945b36`) and the authors' reference procedure, on the sentences of its open splits only, because the CAL and TEST labels are sealed.

Criterion, fixed before the number was computed: within 2.0 points of 92.50.

Result, computed 2026-11-11 (`prereg/fidelity.json`): **92.78** on the 1,286 sentences of the open splits, 0.28 points above the published figure, within the criterion. On the same sentences the AUC-PR for factual sentences is 66.23, against a published 66.08. The share of non-factual sentences in the open splits is 74.5%, against 72.96% in the whole dataset.

## 14. Controls and robustness

Run after the confirmatory result is recorded. Each is pre-specified here.

The confirmatory analysis reads CAL and TEST once. The analyses of this section and of Section 15 read them again, after the confirmatory result has been committed. Every access is recorded in `RUNLEDGER.md` with its purpose, and the report gives the count.

**Negative controls.**
- *Permuted labels.* CAL and TEST labels are permuted within generator. Expected: no arm certifies coverage beyond what a random score does.
- *Random score.* The fixed pseudo-random score as the only score. Expected: no certified coverage.
- *Shuffled source.* The 16 SOURCE columns are recomputed with each output paired with the source of another item of the same task type, by a fixed derangement. A logistic model on those 16 columns alone, fitted on TRAIN, is scored on TEST with true and with shuffled sources. Expected: its AUROC falls to about 0.5 with shuffled sources.

**Length confound.**
- Certified coverage of output length alone.
- AUROC of the tested columns alone within each quintile of output length, with quintile edges taken from DEV_SELECT.

**Guarantee check.**
- *Synthetic.* Two strata with risk curves known in closed form; 10,000 calibration draws; the share of draws whose true pooled risk among accepted outputs exceeds alpha must not exceed delta.
- *Real data.* Pool the CAL and TEST groups. 1,000 times, split them at random within task type into a calibration part and a held-out part of the original sizes, certify BASELINE on the calibration part at the primary alpha, and on the held-out part compute, for each generator with accepted outputs, a one-sided lower confidence bound on the risk among them at level 1 − 0.05/6, from `selcert`'s Clopper–Pearson function. A re-split is flagged when any generator's lower bound exceeds alpha. If the guarantee holds, a re-split is flagged with probability at most delta + 0.05 = 0.10: a true violation has probability at most delta, and a false flag at most 0.05 across the six generators. The check passes if the flagged share is at most 0.10. The re-splits reuse the same data, so the share is reported without an interval.

**Sensitivity to the loss definition.** Both variants were named when the loss was fixed on 2026-11-11, and both are exploratory. The frozen models and rank transforms are unchanged. Thresholds are recertified on CAL, and coverage, gain and interval are recomputed as in Section 9.
- *Implicit-true spans ignored.* The loss is 1 only when the response has at least one annotated span that is not marked `implicit_true`.
- *Flagged responses dropped.* Responses whose `quality` field is not `good` are removed from CAL and TEST.

**Cheap-alternative reference.** BASELINE's certified coverage on TEST is recomputed with CAL resampled, with replacement, to 1.25, 1.5, 2, 3 and 4 times its number of groups, 1,000 draws each. Reported: the smallest multiple at which BASELINE alone reaches the coverage of BASELINE+SOURCE at the actual CAL size, and the number of extra labels that corresponds to.

**Stress.**
- *Per task and per generator.* Coverage and realised risk on TEST for each task type and each generator.
- *Across task types.* Thresholds certified on two task types are applied to the third, in the three rotations. This replaces the brief's cross-dataset calibration, since there is one confirmatory dataset. Exchangeability is expected to fail, and realised risk is reported.
- *Byte versus token alphabets.* The tested columns are recomputed on sequences of lower-cased word tokens. The arm is refitted with the same search, and its coverage is compared with the byte arm's.
- *Paraphrased outputs.* Outputs of a random sample of TEST are paraphrased by an open-weight model run locally, and scores and coverage are compared before and after. This runs only if it fits on the available machine; otherwise it is reported as not run.

Each stress analysis is exploratory.

## 15. Secondary metrics

AUROC and AUPRC of every single score and every arm on TEST; risk–coverage curves; runtime per megabyte of every score. In this section the positive class is an error (loss 1), the score is the arm's frozen risk score, and AUPRC is average precision. Section 13 uses the trapezoidal area instead, to match the published figure.

For each tested arm on RAGTruth, the report also gives its difference from BASELINE in AUROC and in AUPRC on TEST, with a 95% percentile interval from 10,000 paired bootstrap draws of whole source groups within task type (seed 20261131). The difference is outside the decision rule (Section 12).

For WikiBio: AUROC and AUPRC, on its TEST sentences, of a baseline combiner (the lexical baselines computed against the samples, sentence position, NCD and SelfCheck-NLI) and of that combiner with the 16 RESAMPLE columns, fitted and selected as in Section 6, with the difference in AUROC and an interval from 10,000 bootstrap draws of whole passages.

## 16. Second backend

A second implementation of the match-length profile, built on a proprietary codec, is reported for agreement with the reference backend and for throughput. No number in the report depends on it, and it is not part of the public release.

## 17. What had been seen when this document was frozen

- Labels of the open splits only: TRAIN, DEV_SELECT, DEV_CAL and DEV_TEST of RAGTruth and of WikiBio.
- The loss rate in each open split, from the build reports.
- A sanity check of the groundedness model on 160 TRAIN responses: median minimum support 0.247 for responses with no span and 0.053 for responses with one.
- The records under `prereg/`: BASELINE's DEV rehearsal, the A/A distribution, BAR and FLOOR, the power simulation and its supplement at smaller effects, and the fidelity check.
- BASELINE-only diagnostics on the open splits, run to understand the DEV rehearsal and the A/A distribution: the error rate of each generator on DEV_CAL and DEV_TEST (12.9% to 66.9% on DEV_CAL); the errors and accepted counts of each generator at each threshold on DEV_CAL; and which generators are certified in refits of BASELINE (GPT-3.5 in 96.8% of 400 refits, GPT-4 in 7.2%, Llama-2-70B in 0.3%, none in 3.3%).
- An exploratory BASELINE-only record, `prereg/exploratory_design.json`, made after the power gate triggered: the DEV rehearsal and the A/A distribution under grids of 50 and 100 points, and under calibration of one response per source chosen by hash. The A/A 99th percentile is 10.99 and 10.71 points for the finer grids, and 16.46 to 20.32 points for the pooled calibration. The design in this document was not changed.

**Added on 2026-11-11 after the owner's decision to proceed (21:22 PST), and committed before any combiner that includes a tested column was fitted:**

- the decision and the reading of each outcome (Section 12);
- the note on the calibration fallback that was examined and not adopted (Section 7);
- the three closing rules of the replication (Section 11);
- the count of sealed-split accesses, and the two loss-sensitivity analyses that were named when the loss was fixed (Section 14);
- the definition of AUPRC and the paired difference in ranking metrics on RAGTruth (Section 15).

None of these changes the decision rule, the endpoint, the datasets, the splits, the calibration, the grid, alpha, BAR or FLOOR.

**The tested arms, fitted on 2026-11-11 after everything above was committed** (fitted at commit `eb0e2c5`, which holds the additions listed above; records `prereg/profile.json` and `prereg/rehearsal.json`). Each arm used the same 24 configurations and the same seed as BASELINE, and each selected boosted trees with depth 3, learning rate 0.1 and 100 iterations.

| Arm | Leaf size | Log loss, DEV_SELECT | AUROC, DEV_SELECT / DEV_CAL / DEV_TEST | SHA-256 of the frozen model |
|---|---|---|---|---|
| BASELINE | 50 | 0.4285 | 0.879 / 0.900 / 0.906 | `a5d0dcd2cc79eb07b43abc1eac62c25c1b9558e28ecd4e4eca0318c28951e41a` |
| BASELINE+SOURCE | 20 | 0.4245 | 0.882 / 0.899 / 0.905 | `405f199d5e2b192c35320b1326022669d65e7b1500f906bab7892d1586de7fef` |
| BASELINE+SELF | 20 | 0.4238 | 0.882 / 0.899 / 0.906 | `cab8c069e22d4da2284f6425923f13224ada6a6a422f9ee815610c1816f03861` |

The whole confirmatory procedure was then rehearsed in DEV with the function the confirmatory run will call: thresholds certified on DEV_CAL, coverage measured on DEV_TEST, 10,000 bootstrap draws, and the decision rule. At alpha = 0.10:

| Arm | Certified coverage, points | Accepted of 1,410 | Realised risk among accepted | Gain over BASELINE, points | 95% interval | Rule, in the rehearsal |
|---|---|---|---|---|---|---|
| BASELINE | 9.15 | 129 | 0.039 | | | |
| BASELINE+SOURCE | 8.72 | 123 | 0.057 | −0.43 | −11.42 to +1.99 | FAIL |
| BASELINE+SELF | 8.58 | 121 | 0.033 | −0.57 | −9.86 to +3.62 | FAIL |

In every arm only GPT-3.5 is certified, at threshold 0.55 for BASELINE and 0.50 for the other two. No arm certifies any coverage at alpha 0.01 or 0.05. This is a rehearsal on open data and not a confirmatory result. After it was seen, only this section and the status line at the top were edited.

No CAL or TEST label of either dataset, and no row of the reserved replication set, had been read. No feature and no baseline score of any CAL or TEST item had been computed; they are computed after tagging, by scoring code that cannot receive a label. `RUNLEDGER.md` was empty.

**Between the owner's approval and the tag.** Three places were edited and nothing else: the status line; the sentence on the deviations log in the preamble; and Section 19, which now says where the log is kept.

## 18. Reproducibility

Raw files are pinned by upstream commit and SHA-256 in `datasets.manifest.json`. Seeds are fixed in the scripts. Each record under `prereg/` carries the git commit and a dirty flag. `make reproduce`, a locked environment and a container definition are deliverables of the reporting phase.

## 19. Deviations log

Empty at tagging.

The guard on the sealed splits requires this file to stay byte-identical to the version tagged `prereg-v1` (Section 4.1). Departures made after tagging are therefore entered, with a timestamp and a reason, in `DEVIATIONS.md` beside this file.
