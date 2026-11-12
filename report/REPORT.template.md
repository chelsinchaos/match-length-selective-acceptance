# Do match-length profile features add certified coverage? A pre-registered test

LogNovations Holdings LLC. Pre-registration tagged `prereg-v1` (commit `{{provenance > prereg_tag_commit}}`). Every result below is read from `results.json`, collected at commit `{{provenance > git_commit}}` (uncommitted changes in the tree at that moment: {{provenance > git_dirty}}). That file and this one are committed in the commit that follows it.

## Summary

**Primary result (H1): FAIL.** At alpha = {{confirmatory/result.json > primary_alpha | .2f}} and delta = {{confirmatory/result.json > delta | .2f}}, BASELINE certifies {{confirmatory/result.json > coverage > BASELINE > 0.1 > coverage_points | .2f}} points of coverage on TEST and BASELINE+SOURCE certifies {{confirmatory/result.json > coverage > BASELINE+SOURCE > 0.1 > coverage_points | .2f}}. The gain is {{confirmatory/result.json > hypotheses > H1 > gain_points | +.2f}} points, with a 95% interval of {{confirmatory/result.json > hypotheses > H1 > interval_low | +.2f}} to {{confirmatory/result.json > hypotheses > H1 > interval_high | +.2f}}. BAR and FLOOR are both {{confirmatory/result.json > BAR_points | .2f}} points. The upper end of the interval is below BAR, so the pre-registered rule returns FAIL.

The pre-registration fixed how a FAIL is read at this sample size: **the gain is below {{confirmatory/result.json > BAR_points | .2f}} points.** That includes a real but smaller gain. It is not evidence that the features carry no signal.

The rest in brief:

- **H3 (SELF): FAIL.** BASELINE+SELF certifies nothing on TEST. Its gain is {{confirmatory/result.json > hypotheses > H3 > gain_points | +.2f}} points, with an interval of {{confirmatory/result.json > hypotheses > H3 > interval_low | +.2f}} to {{confirmatory/result.json > hypotheses > H3 > interval_high | +.2f}}.
- **H2 (RESAMPLE): not testable.** No certificate is attainable on its dataset at any pre-specified alpha, whatever the score. This follows from sample size and was established before any outcome was seen.
- **Ranking, a secondary metric that cannot change the outcomes above.** Adding the SOURCE columns to BASELINE changes AUROC on TEST by {{controls/result.json > secondary > paired_difference_from_baseline > BASELINE+SOURCE > auroc > difference | +.4f}} (95% interval {{controls/result.json > secondary > paired_difference_from_baseline > BASELINE+SOURCE > auroc > low | +.4f}} to {{controls/result.json > secondary > paired_difference_from_baseline > BASELINE+SOURCE > auroc > high | +.4f}}). Adding the SELF columns changes it by {{controls/result.json > secondary > paired_difference_from_baseline > BASELINE+SELF > auroc > difference | +.4f}} ({{controls/result.json > secondary > paired_difference_from_baseline > BASELINE+SELF > auroc > low | +.4f}} to {{controls/result.json > secondary > paired_difference_from_baseline > BASELINE+SELF > auroc > high | +.4f}}). On WikiBio, adding the RESAMPLE columns changes it by {{controls/wikibio_result.json > difference > pre-registered > auroc > difference | +.4f}} ({{controls/wikibio_result.json > difference > pre-registered > auroc > low | +.4f}} to {{controls/wikibio_result.json > difference > pre-registered > auroc > high | +.4f}}).
- **The SOURCE columns alone do rank errors.** A logistic model on them has an AUROC of {{controls/result.json > secondary > arms > SOURCE only > auroc | .3f}} on TEST, and {{controls/result.json > negative_controls > shuffled_source > shuffled_source > auroc | .3f}} when each output is paired with another item's source. Existing scores do as well: the share of novel word 1-grams alone has {{controls/result.json > secondary > single_scores > base.novel_1gram > auroc | .3f}}.
- **The endpoint is coarse at this sample size.** Its noise floor, {{confirmatory/result.json > FLOOR_points | .2f}} points, is larger than BASELINE's whole certified coverage. The power analysis showed this before any tested model was fitted, and the owner chose to proceed (Section 2.9).
- **Controls.** No score certified anything with permuted labels, in {{controls/result.json > negative_controls > permuted_labels > permutations | ,d}} permutations, or with a random score; given the error rates, those two controls could hardly have come out otherwise (Section 3.4). Shuffling the sources removed the SOURCE columns' ranking. Both guarantee checks passed; they can detect only a gross failure (Section 3.6).
- **Reads of the sealed splits.** In the study, RAGTruth CAL and TEST were read twice each: once by the confirmatory analysis and once by the controls. WikiBio TEST was read once. The reproduction check then read each of them again in a clone, as often as the study had (Section 3.7). The replication was not triggered, and the reserved set was never parsed.
- **Deviations from the pre-registration:** one, in what is reported: the measurements of a proprietary second backend, which the plan said would be reported, are not published (Section 6). Nothing that was analysed deviates. Three incidents are also described in Section 6.
- **Independent audit.** A reviewer that had not seen the implementation tried to refute the result. It reported that the outcomes stand and that it found no path for a label to leak, and it asked for thirteen corrections and disclosures. Section 8 lists them with what was done.

The quantities studied are scores for selective acceptance, with an error bound that holds under exchangeability. Nothing in this report verifies that an output is true.

## 1. Question

Take an LLM output and the context it was supposed to be grounded in. For each position of the output, find the longest substring starting there that also occurs in the context. Do statistics of those match lengths help decide which outputs can be accepted without review?

"Help" has a specific meaning here. An acceptance rule is calibrated on labelled data so that the error rate among accepted outputs is at most alpha, with probability at least 1 − delta. The share of outputs the rule accepts is its **certified coverage**. The study asks whether adding match-length features to a strong combination of existing scores raises certified coverage.

The quantities studied are **scores for selective acceptance**. The acceptance rule carries an error bound that holds when calibration and deployment data are exchangeable. Nothing here verifies that an output is true.

The hypotheses, the data, the splits, the baselines, the endpoint and the decision rule were fixed in `PREREGISTRATION.md` before any calibration or test label was read. This report follows that document section by section. Departures from it are in Section 6.

### Hypotheses

| | Statement | Role | Prior that it is supported |
|---|---|---|---|
| H1 | SOURCE features add certified coverage over the best baseline combination | primary | 35–55% |
| H2 | RESAMPLE features do | secondary | 2–8% |
| H3 | SELF features do | secondary, expected to fail | 5–15% |

The priors for H1 and H3 were set by the study's owner in the brief. The prior for H2 was set before its dataset was opened.

## 2. Methods

### 2.1 The match-length profile and the features

For a target byte string T of length n and a reference R, the profile is

    L[i] = length of the longest substring of T starting at i that occurs in R.

Three reference frames give three profiles:

| Frame | Reference |
|---|---|
| SOURCE | the context the output should be grounded in |
| RESAMPLE | other samples drawn for the same prompt |
| SELF | the output's own prefix, T[0..i) at position i |

Sixteen features are computed from each profile:

| Feature | Definition |
|---|---|
| Dyadic coverage, k = 1, 2, 4, …, 1024 (11 columns) | the share of positions j for which some start i ≤ j has L[i] ≥ k and i + L[i] > j |
| Relative Lempel–Ziv phrase count, and per symbol | greedy parse: from i = 0, take a phrase of length max(L[i], 1) and advance by it [ZM93, BGPR23] |
| Longest common substring, and as a share of n | max L |
| Match-length entropy estimate | for a fixed reference of length N: n · log2 N / Σ (L[i] + 1). For SELF: n′ / Σ_{i=2}^{n′} (L[i] + 1) / log2 i, with n′ the last index [KASW98] |

The SELF estimate is the reciprocal of the average in Theorem 1(b) of [KASW98]. The fixed-reference form replaces that theorem's window of previous symbols with the reference. That substitution is ours, and no claim is made about what it estimates: both are used as scores.

Profiles are computed by a suffix automaton written in Python for this study [BBHECS85]. Every definition is tested against a brute-force implementation on generated inputs and on closed-form cases (T = R, disjoint alphabets, empty strings).

Normalized compression distance [CV05], NCD(x, y) = (C(xy) − min{C(x), C(y)}) / max{C(x), C(y)}, with gzip and with zstd as the compressor C, is **not** a tested feature. It is a baseline.

The code that computes any score cannot receive a label. Its input type has three fields (output, source, resamples), and tests reject any parameter, identifier or import through which a label could arrive.

### 2.2 The calibrator

Certified selective deployment is not implemented in this study. It is `calibrate_bonferroni` from the `selcert` package of `github.com/chelsinchaos/explanation-is-not-certification`, pinned at commit `{{confirmatory/result.json > provenance > selcert_commit}}`. The loader refuses any other commit or a modified checkout.

`calibrate_bonferroni(scores, losses, grid, alpha, delta)` accepts an output when its score is at or below a threshold. For each of K grid thresholds it computes a one-sided Clopper–Pearson upper bound [CP34], at level 1 − delta/K, on the error rate among accepted calibration items. It returns the largest threshold whose bound is at most alpha, or none, which means defer everything.

Its guarantee (Theorem 5 of the companion paper [McE26]): if the score, the model and the grid are fixed independently of the calibration sample, and the calibration pairs are i.i.d. from the deployment distribution, then with probability at least 1 − delta the error rate among accepted outputs is at most alpha, or everything is deferred. The guarantee is marginal. A subgroup can carry a higher risk.

### 2.3 Data

**RAGTruth** [NIU24] is the confirmatory dataset: responses of six LLMs to question-answering, data-to-text and summarization prompts, with human span annotations. The repository `ParticleMedia/RAGTruth` was read at commit `c103204b9ce28d6bbad859304bf30de72b8ed8fe` (MIT licence); file hashes are in `datasets.manifest.json`.

- **Unit.** One response. Each source has six responses, one per generator.
- **0/1 loss.** 1 when the response has at least one annotated span of any type, otherwise 0. No row is excluded.
- **SOURCE reference.** The `prompt` field, the exact text the generator was given.
- **Group.** The source. All six responses of a source share a split.
- **Near-duplicates.** Sources whose sets of lower-cased word 5-grams have Jaccard similarity of at least 0.9 are one cluster; one member is kept. {{data/derived/ragtruth/build_report.json > near_duplicate_clusters}} cluster was found, and {{data/derived/ragtruth/build_report.json > responses_dropped_with_duplicate_sources}} responses were dropped with it. Below that threshold some similar sources remain in different splits; Section 3.13 lists them.

**The reserved replication set** is the RAGTruth authors' own `test` split: {{data/derived/ragtruth/build_report.json > reserved_rows | ,d}} responses. Its rows share a file with the rest. A row's split is read from its raw bytes, and a row marked `test` is never parsed.

**WikiBio GPT-3 hallucination** [MLG23] is the dataset for H2: sentences of GPT-3 passages, each passage with 20 stochastic samples, labelled by humans as accurate, minor-inaccurate or major-inaccurate. The loss is 1 unless the sentence is labelled accurate. The group is the passage. Read from Hugging Face `potsawee/wiki_bio_gpt3_hallucination` at revision `b3cfb73209a8c51582fa1d9b7fe7e45fec5529b2` (CC BY-SA 3.0).

Splits are by hash of the group key, with fractions 0.30, 0.10, 0.20, 0.10, 0.20, 0.10:

{{table: splits}}

TRAIN fits combiners. DEV_SELECT selects among them and fits the rank transform. DEV_CAL and DEV_TEST rehearse the confirmatory procedure. CAL and TEST are the confirmatory calibration and test samples.

**Sealing.** When the tables were built, the labels of CAL and TEST were dropped as each row was parsed, and they were written to no file. The harness's own code reads them only through a guard. The guard requires an explicit flag, the `prereg-v1` tag, a HEAD that descends from it, an unchanged pre-registration and a clean tree, and it appends every read to `RUNLEDGER.md`. It is a procedural control: it does not stop other code from opening the raw files, and the ledger is a text file. Section 3.7 lists the reads and says what the count rests on.

**Leads examined and not used** as confirmatory ground truth: OpenHalDet (arXiv:2606.06959), whose labels come from a GPT-4o-mini judge; HaluEval, whose source-grounded examples are hallucinations written by ChatGPT on instruction; FaithBench, 660 summaries selected because detectors disagreed on them, under a non-commercial licence.

### 2.4 Baselines

| Baseline | Columns |
|---|---|
| Output length | bytes, words, sentences |
| N-gram overlap with the source | share of the output's word n-gram occurrences absent from the source, n = 1..4 |
| ROUGE-L against the source | precision, recall, F |
| Open groundedness model | MiniCheck-DeBERTa-v3-Large [TLD24], applied by this study's code as its reference inference code applies it. Each sentence of the response is a claim; minimum support, mean support, share of sentences below 0.5 |
| Compression distance | NCD with gzip and with zstd |
| Deployment covariates | generator, task type, temperature |

Generator and task type are known when an output is produced. They are baseline inputs so that the tested features cannot take credit for standing in for them. Temperature is recorded with each response in the dataset and was pre-registered as a baseline input with them. That it would be known in deployment was assumed. Within four of the six generators it is strongly related to the label (Section 3.13).

Not run, and why:

| Baseline in the brief | Reason |
|---|---|
| Verbalized confidence | RAGTruth holds none, and its generators cannot be asked after the fact |
| Token log-probability, token entropy | need generations of our own. None were run: the owner chose an existing dataset for H2 |
| Semantic entropy | needs token probabilities or clustered whole-prompt samples |
| SelfCheck-style consistency | needs resamples; run on WikiBio only |

**Fidelity check.** SelfCheckGPT with NLI is published at an AUC-PR of {{prereg/fidelity.json > published > nonfact_auc_pr | .2f}} for non-factual sentences over all {{prereg/fidelity.json > published > sentences | ,d}} sentences of WikiBio GPT-3 [MLG23, Table 2]. The same score, computed here with the same checkpoint on the {{prereg/fidelity.json > ours > sentences | ,d}} sentences of the open splits, gives {{prereg/fidelity.json > ours > nonfact_auc_pr > trapezoid | .2f}}: a difference of {{prereg/fidelity.json > difference_points | +.2f}} points, inside the tolerance of 2.0 fixed before the number was computed.

### 2.5 Combiner and arms

A combiner maps a set of columns to one risk score, the estimated probability of an error. Every arm is fitted with the same search and the same seed: 8 L2 logistic regressions on standardised columns and 16 gradient-boosted tree models. Fit on TRAIN, select the smallest log loss on DEV_SELECT, freeze.

{{table: arms}}

BASELINE holds the columns of Section 2.4. BASELINE+SOURCE adds the 16 SOURCE columns, and BASELINE+SELF the 16 SELF columns.

### 2.6 From score to certified coverage

1. **Rank transform.** A frozen risk score is mapped to [0, 1] by the empirical distribution of that score on DEV_SELECT, separately for each generator.
2. **Grid.** Twenty uniform thresholds, 0.05 to 1.00. The grid depends on no data.
3. **Calibration per generator.** The six responses of a source share that source, so the pooled calibration sample is not i.i.d. Within one generator each source contributes one response, and sources are exchangeable under the hash split. The calibrator is run once per generator at confidence delta/6. With probability at least 1 − delta all six certificates hold together, and then the error rate among all accepted outputs is at most alpha.
4. **Certified coverage** is the share of all items of a sample that the six thresholds accept, in percentage points.

For short, the text below says that a generator "is certified" when a threshold was certified for that generator's outputs. Nothing is certified about a generator.

### 2.7 Error levels

alpha ∈ {0.01, 0.05, 0.10}; delta = 0.05. With calibration per generator, the smallest error-free accepted set that can be certified has 74 items at alpha = 0.10, 152 at 0.05 and 775 at 0.01. CAL has 504 sources per generator, so alpha = 0.01 cannot be certified by any arm. It is kept in the tables to show this.

The primary alpha was chosen by a rule written before any rehearsal: the strictest alpha at which BASELINE certifies at least 5 points in the DEV rehearsal. BASELINE's DEV rehearsal:

{{table: dev_baseline}}

Primary alpha: **{{prereg/bar_floor.json > primary_alpha | .2f}}**.

### 2.8 Endpoint and decision rule

**Endpoint.** Certified coverage of BASELINE+SOURCE minus that of BASELINE, in points, at the primary alpha, with thresholds certified on CAL and coverage measured on TEST.

**Interval.** A paired bootstrap of 10,000 draws. Each draw resamples whole source groups with replacement within task type, in CAL and in TEST, recertifies both arms and measures both. The interval is the 2.5th to the 97.5th percentile.

**Decision rule**, fixed in the brief:

- c_B = BASELINE certified coverage in the DEV rehearsal.
- BAR = the smaller of 5 points and 10% of (100 − c_B).
- FLOOR = the larger of 2 points and the 99th percentile of the baseline-versus-baseline gap.
- If BAR is below FLOOR, BAR equals FLOOR.

| Outcome | Condition |
|---|---|
| PASS | the point estimate is at or above BAR and the entire 95% interval is above FLOOR |
| FAIL | the upper end of the 95% interval is below BAR; this includes a real but too-small effect |
| INCONCLUSIVE | anything else; it triggers exactly one replication on the reserved set |

**A/A test.** 1,000 runs. Each fits two BASELINE models with the frozen configuration, differing only in seed and in the bootstrap resample of TRAIN groups, puts both through the DEV rehearsal and records the difference in coverage.

{{table: aa}}

![Distribution of the A/A gap](report/figures/aa_gap.svg)

| Quantity | Value, points |
|---|---|
| c_B | {{prereg/bar_floor.json > c_B_points | .2f}} |
| FLOOR | **{{prereg/bar_floor.json > FLOOR_points | .2f}}** |
| BAR (the nominal 5 points is below FLOOR, so BAR = FLOOR) | **{{prereg/bar_floor.json > BAR_points | .2f}}** |

BAR and FLOOR were committed before any combiner that includes a tested column was fitted. A test in the suite checks that order from the commit history.

Why the noise floor is this large: in the DEV rehearsal all of BASELINE's coverage comes from one generator. Certification per generator is all or nothing, so two refits of the same model differ by a whole generator's share when one of them falls just short.

### 2.9 Power, and the decision to proceed

Power was simulated at the CAL and TEST sizes, on BASELINE's ranks and losses in DEV, with a second arm built by moving correct outputs toward acceptance until the mean gain reached a target. 300 replications per target, each through the bootstrap and the decision rule.

{{table: power}}

![Outcome probabilities against the true gain](report/figures/power.svg)

**The design does not detect a gain the size of BAR with 80% power.** The rule returns a decisive outcome in about four runs in five only when the true gain is about 3 points or less (FAIL) or about 21.6 points or more (PASS). Between about 5 and 16 points it returns INCONCLUSIVE in most runs.

The brief required the study to stop at this point and ask. The owner was shown the table and three options: proceed as designed, stop and report, or change the design before anything further was seen. The owner chose to proceed as designed. No combiner with a tested column had been fitted at that time. How each outcome would be read was written into the pre-registration before any such combiner was fitted, and Section 3.1 uses that wording.

Two alternatives were examined with BASELINE alone and not adopted:

{{table: designs}}

A finer grid does not lower the noise floor. Calibrating once on one response per source raises BASELINE's coverage and raises the noise floor more.

### 2.10 Secondary hypotheses and multiplicity

H3 uses the endpoint, interval and rule of H1 with BASELINE+SELF. The secondary family is {H2, H3}, with Holm's step-down adjustment [HOLM79]. The p-value of a gain is the share of its bootstrap draws at or below FLOOR. H2 enters with p = 1. A secondary hypothesis is supported when its outcome is PASS and its adjusted p-value is below 0.05.

**H2 cannot be certified on WikiBio.** Sentences of one passage are not independent, so an i.i.d. calibration sample has one sentence per passage: 52 in CAL. If all 52 were accepted with no error, the calibrator's bound at 20 grid points and delta 0.05 would be 0.1088, above every alpha in this study. This follows from sample size alone and was established before any outcome was examined. H2 is recorded as not testable. Section 3.3 reports a ranking analysis in its place, as a secondary metric.

## 3. Results

### 3.1 Primary result

CAL holds {{confirmatory/result.json > sizes > cal > items | ,d}} responses of {{confirmatory/result.json > sizes > cal > groups | ,d}} sources. TEST holds {{confirmatory/result.json > sizes > test > items | ,d}} responses of {{confirmatory/result.json > sizes > test > groups | ,d}} sources. The confirmatory analysis read them once.

{{table: coverage}}

{{table: hypotheses}}

**H1: FAIL.** The gain of BASELINE+SOURCE over BASELINE is {{confirmatory/result.json > hypotheses > H1 > gain_points | +.2f}} points, with a 95% interval of {{confirmatory/result.json > hypotheses > H1 > interval_low | +.2f}} to {{confirmatory/result.json > hypotheses > H1 > interval_high | +.2f}}. The upper end of the interval is below BAR ({{confirmatory/result.json > BAR_points | .2f}} points). That is the pre-registered condition for FAIL.

The pre-registration fixed the reading of this outcome before any tested model was fitted (its Section 12): **the gain is below {{confirmatory/result.json > BAR_points | .2f}} points.** That includes a real but smaller gain. It is not evidence that the SOURCE features carry no signal, and this report does not describe it as such.

Where the numbers come from. At alpha = 0.10, BASELINE and BASELINE+SOURCE each certify one generator, GPT-4, and no other. BASELINE certifies it at rank threshold {{confirmatory/result.json > coverage > BASELINE > 0.1 > thresholds > gpt-4-0613 | .2f}} and accepts {{confirmatory/result.json > coverage > BASELINE > 0.1 > accepted}} TEST outputs. BASELINE+SOURCE certifies it at {{confirmatory/result.json > coverage > BASELINE+SOURCE > 0.1 > thresholds > gpt-4-0613 | .2f}} and accepts {{confirmatory/result.json > coverage > BASELINE+SOURCE > 0.1 > accepted}}. The gain is that one step of the grid. In a share of {{confirmatory/result.json > hypotheses > H1 > p_gain_at_most_floor | .4f}} of the bootstrap draws the gain was at or below FLOOR.

The power table of Section 2.9 again, beside the outcome as the pre-registration requires:

{{table: power}}

At the smallest gain that was simulated, about 3 points, the design returns FAIL in a share of {{prereg/power_small_gains.json > results > 0 > p_fail | .2f}} of runs, and at about 5 points in {{prereg/power_small_gains.json > results > 1 > p_fail | .2f}}. The power simulation did not include a gain of zero or a loss. A probe added after the audit did (Section 3.13): on {{controls/open_probes.json > no_gain > replications}} pairs of models with no true gain, the rule returned FAIL for a share of {{controls/open_probes.json > no_gain > p_fail | .2f}} and INCONCLUSIVE for the rest. FAIL is therefore the likely outcome of this design when there is no gain and when there is a gain of a few points, and it cannot tell them apart. The ranking metrics of Section 3.3 resolve smaller differences.

The outcome is not INCONCLUSIVE, so the replication was not run. The reserved replication set has not been parsed.

The DEV rehearsal, which ran the same function on open data before the pre-registration was approved, beside the confirmatory result:

{{table: rehearsal_vs_test}}

### 3.2 Secondary hypotheses

**H3: FAIL.** BASELINE+SELF certifies no generator on CAL, so it accepts nothing on TEST, and its gain is the negative of BASELINE's whole coverage: {{confirmatory/result.json > hypotheses > H3 > gain_points | +.2f}} points, with a 95% interval of {{confirmatory/result.json > hypotheses > H3 > interval_low | +.2f}} to {{confirmatory/result.json > hypotheses > H3 > interval_high | +.2f}}. The upper end is below BAR. The share of bootstrap draws at or below FLOOR is {{confirmatory/result.json > hypotheses > H3 > p_gain_at_most_floor | .4f}}. With H2 in the family at p = 1, the Holm-adjusted value is {{confirmatory/result.json > hypotheses > H3 > p_holm | .4f}}. H3 is not supported. Its prior was 5–15%.

The reading is the same as for H1: the gain is below {{confirmatory/result.json > BAR_points | .2f}} points. The point estimate is not a measured loss caused by the SELF columns. Certification of a generator is all or nothing. This arm's estimate is {{controls/result.json > loss_sensitivity > implicit_true_spans_ignored > hypotheses > H3 > gain_points | +.2f}} points under one of the loss definitions of Section 3.8, and the arm certifies GPT-4 when its columns are computed over word tokens (Section 3.10).

**H2: not testable.** No certificate is attainable on WikiBio at any pre-specified alpha, whatever the score (Section 2.10). H2 enters the Holm family with p = 1. The ranking analysis that stands in its place is in Section 3.3. It is a secondary metric and does not test H2 under the rule.

### 3.3 Ranking metrics (secondary)

The decision rule's endpoint cannot resolve small differences (Section 2.9). Ranking metrics can. They are secondary and do not change any outcome above. An error is the positive class; AUPRC is average precision.

{{table: ranking_arms}}

LENGTH is a combiner over the three length columns. "SOURCE only" and "SELF only" are logistic models on the 16 tested columns of a frame alone. RANDOM is a fixed pseudo-random score.

Difference from BASELINE, with a paired bootstrap of 10,000 draws of whole sources within task type:

{{table: ranking_difference}}

Adding the SOURCE columns to BASELINE changes AUROC on TEST by {{controls/result.json > secondary > paired_difference_from_baseline > BASELINE+SOURCE > auroc > difference | +.4f}} and AUPRC by {{controls/result.json > secondary > paired_difference_from_baseline > BASELINE+SOURCE > auprc > difference | +.4f}}. Adding the SELF columns changes AUROC by {{controls/result.json > secondary > paired_difference_from_baseline > BASELINE+SELF > auroc > difference | +.4f}} and AUPRC by {{controls/result.json > secondary > paired_difference_from_baseline > BASELINE+SELF > auprc > difference | +.4f}}. Every interval contains zero. The upper ends of the two AUROC intervals are {{controls/result.json > secondary > paired_difference_from_baseline > BASELINE+SOURCE > auroc > high | +.4f}} and {{controls/result.json > secondary > paired_difference_from_baseline > BASELINE+SELF > auroc > high | +.4f}}.

By themselves the tested columns do rank errors. The logistic model on the 16 SOURCE columns has an AUROC of {{controls/result.json > secondary > arms > SOURCE only > auroc | .3f}} and the one on the 16 SELF columns {{controls/result.json > secondary > arms > SELF only > auroc | .3f}}, against {{controls/result.json > secondary > arms > LENGTH > auroc | .3f}} for output length alone and {{controls/result.json > secondary > arms > RANDOM > auroc | .3f}} for the random score. Appendix A gives every single score. The three strongest SOURCE columns and the three strongest baselines are:

{{table: strongest_single}}

The fourth column chooses each score's direction after seeing TEST, which flatters every row equally. The strongest single SOURCE column is about as strong as the lexical baseline and weaker than the groundedness model.

So on this dataset the SOURCE columns carry information about errors, and the combined model does not rank better with them than without them.

One caution. These metrics pool all generators, and BASELINE is given the generator and the task type. Certification is per generator, so certified coverage depends on how a score ranks the outputs of one generator. The table below reads that off the recorded risk–coverage curves: the error rate among the lowest-risk quarter, half and three quarters of each generator's TEST outputs. It was not in the pre-registration and is exploratory.

{{table: within_generator}}

In the first row of each generator one error is 0.015. No arm is ahead of the others across generators: the differences are a few errors in either direction.

These are secondary metrics. As the pre-registration states, they do not change the outcome of H1 and are not support for it.

**WikiBio (RESAMPLE features).** A baseline combiner (lexical baselines against the samples, sentence position, NCD, SelfCheck-NLI) and the same with the 16 RESAMPLE columns, fitted on TRAIN and selected on DEV_SELECT, scored on the {{controls/wikibio_result.json > sizes > test_sentences}} TEST sentences of {{controls/wikibio_result.json > sizes > test_passages}} passages:

{{table: wikibio}}

{{table: wikibio_difference}}

Adding the RESAMPLE columns changes AUROC by {{controls/wikibio_result.json > difference > pre-registered > auroc > difference | +.4f}} and AUPRC by {{controls/wikibio_result.json > difference > pre-registered > auprc > difference | +.4f}}. Both intervals contain zero. They are wide, because TEST has {{controls/wikibio_result.json > sizes > test_sentences}} sentences of {{controls/wikibio_result.json > sizes > test_passages}} passages.

Alone, the best RESAMPLE column (the phrase count per symbol) has an AUROC of {{controls/wikibio_result.json > single_scores > resample.rlz_phrases_per_symbol > auroc | .3f}}. SelfCheck-NLI, a baseline that uses the same samples, has {{controls/wikibio_result.json > single_scores > sc.nli > auroc | .3f}}.

The comparison with passage length on both sides was not in the pre-registration and is exploratory. H2 remains not testable, and this analysis does not test it.

Risk–coverage curves on TEST:

![Risk–coverage curves on TEST](report/figures/risk_coverage.svg)

### 3.4 Negative controls

**Permuted labels.** CAL and TEST labels were permuted within generator, {{controls/result.json > negative_controls > permuted_labels > permutations | ,d}} times.

{{table: permuted}}

**Random score.** The pseudo-random score certifies {{controls/result.json > negative_controls > random_score > 0.1 > coverage_points | .2f}} points at alpha 0.10.

**Shuffled source.** Each output was paired with the source of another item of the same task type, by a fixed derangement of sources, and the 16 SOURCE columns were recomputed.

{{table: shuffled}}

The three controls came out as expected. Two of them could hardly have come out otherwise. Every generator's error rate is above alpha (Section 3.10), so with permuted labels or a random score a certificate would need a long run of error-free outputs by chance. Those two controls would catch a calibrator that certifies without evidence, and little else. The shuffled-source control is the informative one.

- With permuted labels, no score certified any coverage in any of the {{controls/result.json > negative_controls > permuted_labels > permutations | ,d}} permutations.
- The random score certified nothing with the true labels.
- With shuffled sources, the AUROC of the SOURCE model fell from {{controls/result.json > negative_controls > shuffled_source > true_source > auroc | .3f}} to {{controls/result.json > negative_controls > shuffled_source > shuffled_source > auroc | .3f}}. Output length alone has {{controls/result.json > negative_controls > shuffled_source > length_only_for_reference > auroc | .3f}}. What the model ranks by therefore depends on pairing each output with its own source. It is not carried by the output alone.

The pre-registration expected the shuffled value to be about 0.5. It is below 0.5 here, and it was {{controls/rehearsal.json > negative_controls > shuffled_source > shuffled_source > auroc | .3f}} in the DEV rehearsal. A model fitted to true-source columns ranks slightly worse than chance once the columns no longer describe the source. Why was not investigated.

### 3.5 Length

Certified coverage of output length alone:

{{table: length_only}}

AUROC within quintiles of output length (edges from DEV_SELECT):

{{table: length_quintiles}}

Output length alone certifies nothing at any alpha. Its AUROC is {{controls/result.json > length > length_only > auroc | .3f}}.

Within every quintile of output length, the SOURCE model has an AUROC between {{controls/result.json > length > within_quintiles > SOURCE only > 1 > auroc | .3f}} and {{controls/result.json > length > within_quintiles > SOURCE only > 0 > auroc | .3f}}, and the SELF model between {{controls/result.json > length > within_quintiles > SELF only > 0 > auroc | .3f}} and {{controls/result.json > length > within_quintiles > SELF only > 4 > auroc | .3f}}. The tested columns rank errors among outputs of similar length, so their standalone signal is not a length effect. Each tested arm is within 0.004 of BASELINE in every quintile of the table.

### 3.6 Guarantee checks

{{table: guarantee}}

In the synthetic check, two strata have risk curves known in closed form, and a run is a violation when the true risk among accepted outputs exceeds alpha. In the real-data check, CAL and TEST groups were pooled and re-split at random; a re-split is flagged when, for some generator, a lower confidence bound on the risk among accepted held-out outputs exceeds alpha. If the guarantee holds, the flagged share is at most delta + 0.05. The re-splits reuse the same data, so the share has no interval.

Both checks pass.

- **Synthetic.** In {{controls/result.json > guarantee > synthetic > draws | ,d}} draws the true risk among accepted outputs never exceeded alpha. Its largest value was {{controls/result.json > guarantee > synthetic > max_true_risk_when_accepting | .3f}}. The guarantee allows a share of {{controls/result.json > guarantee > synthetic > delta | .2f}}.
- **Real data.** None of the {{controls/result.json > guarantee > resplit > resplits | ,d}} re-splits was flagged, against an allowed share of {{controls/result.json > guarantee > resplit > allowed_share | .2f}}. {{controls/result.json > guarantee > resplit > resplits_accepting_anything}} of them accepted anything. Among those, the mean error rate of the accepted held-out outputs was {{controls/result.json > guarantee > resplit > mean_held_out_risk_when_accepting | .3f}}. In a share of {{controls/result.json > guarantee > resplit > share_of_resplits_with_held_out_risk_above_alpha | .3f}} of all re-splits that error rate was above alpha.

On TEST itself, the error rate among accepted outputs was {{confirmatory/result.json > coverage > BASELINE > 0.1 > realised_risk | .3f}} for BASELINE and {{confirmatory/result.json > coverage > BASELINE+SOURCE > 0.1 > realised_risk | .3f}} for BASELINE+SOURCE. Both are below alpha = 0.10.

What these checks establish is limited. The synthetic check exercises the calibrator, and this study's code around it, on data that meet its assumptions. It can fail: a test feeds it a calibrator that accepts everything, and it then reports a violation in every draw. It is not sensitive to how delta is divided among the grid points and the generators; unit tests cover that division. The real-data check can flag only a clear violation, because it needs a lower confidence bound to exceed alpha, and its re-splits are not independent of each other. A pass means that no sign of a violation was found. The guarantee itself rests on Theorem 5 of [McE26] and on exchangeability of the calibration and deployment data.

### 3.7 Reads of the sealed splits

{{table: ledger}}

TEST was read {{ledger > test_accesses}} times and CAL {{ledger > cal_accesses}} times in the study's repository. The ledger counts by split name. Two of the reads of TEST and both reads of CAL are of RAGTruth: the confirmatory analysis, and the controls of Sections 3.3 to 3.10. The third read of TEST is of WikiBio, for the ranking analysis of Section 3.3. WikiBio CAL was never read, and the reserved replication set was never parsed.

Each read was one run, at a commit that descends from `prereg-v1`, on a clean tree. The confirmatory analysis ran before the controls, and its record was committed before the controls started.

**What the count rests on.** The repository shows that the ledger rows entered its history in the commits that hold the corresponding records, that every stamp agrees with the history, and that the committed confirmatory record has not changed since it was committed. `tests/test_confirmatory_records.py` checks the order of the runs, the counters in the records, and that the recorded outcome is what the decision rule gives for the recorded numbers. None of this can prove that no other read took place: the ledger is a text file, and the output of the confirmatory run was printed in the session and not written to a log file. That the confirmatory analysis ran once rests on the ledger and on the command log of the session that ran the study, which the owner holds. The audit raised this (Section 8, finding 1), and Section 6 records the two events that it noticed.

**Reads by the reproduction.** The reproduction check of Section 7 ran the three guarded analyses again in a clone, after every result in this report had been committed. Its reads are in the clone's ledger, not in the study's:

{{table: reproduction_ledger}}

Counting both, each sealed split of RAGTruth has been read twice by the study and twice by the reproduction, and WikiBio TEST once by each.

Three later analyses use TEST outputs without their labels and are not reads of a sealed split: the paraphrase stress and the acceptance overlap of Section 3.10, and the scoring check of Section 6. The probes of Section 3.13 use open splits and label-free tables only.

### 3.8 Sensitivity to the loss definition (exploratory)

{{table: loss_sensitivity}}

Ignoring `implicit_true` spans changes {{controls/result.json > loss_sensitivity > implicit_true_spans_ignored > labels_changed_from_error_to_no_error > cal}} CAL labels and {{controls/result.json > loss_sensitivity > implicit_true_spans_ignored > labels_changed_from_error_to_no_error > test}} TEST labels from error to no error. Dropping responses whose `quality` field is not `good` removes {{controls/result.json > loss_sensitivity > flagged_responses_dropped > responses_dropped > cal}} from CAL and {{controls/result.json > loss_sensitivity > flagged_responses_dropped > responses_dropped > test}} from TEST.

Under all three definitions the rule returns FAIL for H1 and for H3. The point estimates are not stable. H1's gain is {{confirmatory/result.json > hypotheses > H1 > gain_points | +.2f}}, {{controls/result.json > loss_sensitivity > implicit_true_spans_ignored > hypotheses > H1 > gain_points | +.2f}} and {{controls/result.json > loss_sensitivity > flagged_responses_dropped > hypotheses > H1 > gain_points | +.2f}} points. H3's is {{confirmatory/result.json > hypotheses > H3 > gain_points | +.2f}}, {{controls/result.json > loss_sensitivity > implicit_true_spans_ignored > hypotheses > H3 > gain_points | +.2f}} and {{controls/result.json > loss_sensitivity > flagged_responses_dropped > hypotheses > H3 > gain_points | +.2f}}.

Ignoring implicit-true spans lowers the error rate, and each arm then certifies GPT-3.5 as well as GPT-4. Dropping the flagged responses leaves BASELINE's certificate in place and removes that of BASELINE+SOURCE.

These analyses are exploratory. They show how far a point estimate of certified coverage moves when one generator crosses the certification line. That is the noise the A/A test measured before the run.

### 3.9 Cheap-alternative reference

How much would more calibration labels give BASELINE alone? CAL was resampled with replacement, within task type, to larger sizes, {{controls/result.json > cheap_alternative > baseline_with_resampled_calibration > 2.0 > draws | ,d}} draws each. The row at 1.0 is the same resampling at the actual size, for reference.

{{table: cheap}}

At the actual size of CAL, BASELINE+SOURCE does not exceed BASELINE. The pre-registered quantity is the number of extra calibration labels with which BASELINE alone would match BASELINE+SOURCE. It is {{controls/result.json > cheap_alternative > extra_calibration_labels}}.

For scale: resampling CAL to twice its size gives BASELINE a mean coverage of {{controls/result.json > cheap_alternative > baseline_with_resampled_calibration > 2.0 > mean_coverage_points | .2f}} points, against {{controls/result.json > cheap_alternative > baseline_with_resampled_calibration > 1.0 > mean_coverage_points | .2f}} for the same resampling at the actual size. These are bootstrap estimates from the CAL sample, not measurements with new labels. At the actual size, {{controls/result.json > cheap_alternative > baseline_with_resampled_calibration > 1.0 > share_with_any_coverage | .1%}} of the resamples certify anything at all, which shows how close BASELINE's one certificate is to the line.

### 3.10 Stress analyses (exploratory)

**Per generator**, at alpha {{controls/result.json > primary_alpha | .2f}}:

{{table: per_generator}}

**Per task type:**

{{table: per_task}}

All certified coverage comes from GPT-4. BASELINE accepts {{controls/result.json > stress > per_generator > BASELINE > gpt-4-0613 > coverage_points | .1f}}% of its TEST outputs and BASELINE+SOURCE {{controls/result.json > stress > per_generator > BASELINE+SOURCE > gpt-4-0613 > coverage_points | .1f}}%. No rule certifies another generator. GPT-3.5 has the lowest error rate and is not certified on CAL. In the DEV rehearsal it was the one generator that BASELINE certified (Appendix B).

The guarantee is marginal, over all accepted outputs. Within one task type the error rate can be higher: BASELINE accepts {{controls/result.json > stress > per_task > BASELINE > Data2txt > accepted}} Data2txt outputs, with an error rate of {{controls/result.json > stress > per_task > BASELINE > Data2txt > realised_risk | .3f}}.

**Across task types.** Thresholds certified on two task types, applied to the third. Exchangeability is not expected to hold.

{{table: across_tasks}}

BASELINE accepts {{controls/result.json > stress > across_task_types > BASELINE > Data2txt > accepted}} Data2txt outputs with an error rate of {{controls/result.json > stress > across_task_types > BASELINE > Data2txt > realised_risk | .3f}}, and BASELINE+SELF accepts {{controls/result.json > stress > across_task_types > BASELINE+SELF > Data2txt > accepted}} with {{controls/result.json > stress > across_task_types > BASELINE+SELF > Data2txt > realised_risk | .3f}}. Both are above alpha. The error bound holds when calibration and deployment data are exchangeable. Here exchangeability was broken on purpose, and the bound did not hold.

**Byte versus token alphabets.** The tested columns were recomputed on lower-cased word tokens and each arm was refitted with the same search.

{{table: tokens}}

Computing the tested columns over word tokens leaves AUROC where it was. The SELF arm certifies GPT-4 with token columns and nothing with byte columns. The noise floor is {{confirmatory/result.json > FLOOR_points | .2f}} points, so that difference is not evidence for either alphabet.

**The tested columns alone.** Certified coverage, at alpha {{controls/result.json > primary_alpha | .2f}}, of the models that hold one family of columns. The pre-registration specified this for output length and for the random score. The other two rows are additions.

{{table: alone}}

The logistic model on the 16 SOURCE columns certifies GPT-4 and accepts {{controls/result.json > coverage > SOURCE only > 0.1 > accepted}} TEST outputs with an error rate of {{controls/result.json > coverage > SOURCE only > 0.1 > realised_risk | .3f}}. Those are also the count and the error rate of BASELINE+SOURCE. The equality is a coincidence and not a copied column: the two rules accept different outputs, {{controls/acceptance_overlap.json > pairs > 4 > accepted_both}} of them in common. The table below was computed without labels, from the thresholds that the controls run recorded. The script that computes it checks each count against that record.

{{table: overlap}}

Every rule that certifies anything certifies GPT-4 only, and any two of them share most of what they accept.

Two cautions against reading the SOURCE-only row as a result. In the DEV rehearsal the same model certified {{controls/rehearsal.json > coverage > SOURCE only > 0.1 > coverage_points | .2f}} points and BASELINE {{controls/rehearsal.json > coverage > BASELINE > 0.1 > coverage_points | .2f}}. And every difference in this table is smaller than the noise floor.

**Paraphrased outputs.** {{controls/paraphrase.json > outputs}} TEST outputs (all six responses of 50 sources) were rewritten by an open-weight model run locally (`{{controls/paraphrase.json > provenance > paraphrase_model}}`, greedy decoding), rescored with the same code, and put through the frozen models and the certified thresholds. No label is involved: the comparison is of scores and acceptance, not of correctness. The rewritten outputs are {{controls/paraphrase.json > paraphrase_fidelity > byte_length_ratio > mean | .2f}} times the length of the originals on average, and share {{controls/paraphrase.json > paraphrase_fidelity > word_set_jaccard_with_original > mean | .2f}} of their word set with them (Jaccard).

{{table: paraphrase}}

{{table: paraphrase_columns}}

Rewording raises the risk score of every arm on average and largely keeps the order of the outputs. Of the {{controls/paraphrase.json > arms > BASELINE > accepted_before}} outputs that BASELINE accepted, {{controls/paraphrase.json > arms > BASELINE > accepted_both}} are still accepted after rewording. Of the {{controls/paraphrase.json > arms > BASELINE+SOURCE > accepted_before}} that BASELINE+SOURCE accepted, {{controls/paraphrase.json > arms > BASELINE+SOURCE > accepted_both}} are.

The SOURCE columns measure verbatim overlap with the source, and rewording removes it: mean coverage at k = 16 falls from {{controls/paraphrase.json > column_means > source.cov_k16 > before | .3f}} to {{controls/paraphrase.json > column_means > source.cov_k16 > after | .3f}}. The baselines move too. The share of novel word 1-grams rises from {{controls/paraphrase.json > column_means > base.novel_1gram > before | .3f}} to {{controls/paraphrase.json > column_means > base.novel_1gram > after | .3f}}, and the mean support from the groundedness model falls from {{controls/paraphrase.json > column_means > gnd.mean > before | .3f}} to {{controls/paraphrase.json > column_means > gnd.mean > after | .3f}}.

Two limits apply. The sample is small: each arm accepted {{controls/paraphrase.json > arms > BASELINE > accepted_before}} of these outputs before rewording. And no label is involved, so it is not known whether the rewritten outputs are as correct as the originals. A paraphrase that changed a fact ought to score as riskier.

### 3.11 Runtime

Seconds per megabyte of input (output plus source), on an Apple M1 Max:

{{table: runtime}}

### 3.12 Second backend

A second implementation of the SOURCE profile, built on FastCodec, a proprietary codec, was compared privately with the reference backend on real items. Its measurements are not published. The pre-registration said they would be reported (its Section 16); the owner decided otherwise before publication, and Section 6 records that as a deviation. No number in this report depends on that backend, and it is not part of the public release.

### 3.13 Probes added after the audit (exploratory)

The independent audit (Section 8) asked for three things that the study had not measured. They were added after the confirmatory results had been seen. None reads a sealed label.

**Similar sources in different splits.** The study dropped sources whose content had a Jaccard similarity of at least 0.9 on word 5-grams. This table lists the pairs of sources in different splits whose prompts have a similarity of at least {{controls/open_probes.json > near_duplicates > listed_from | .1f}}. The prompt includes the task's instruction, so these values are not the ones the rule computed.

{{table: near_duplicates}}

One pair has a source in CAL and a source in TEST. They are two reports of one news story and share most of their text. It is 1 of {{confirmatory/result.json > sizes > test > groups}} TEST sources.

**The temperature covariate.** GPT-3.5 and GPT-4 have one temperature. The other four generators have five, varying from response to response, and within each of them a higher temperature goes with fewer errors:

{{table: temperature_generators}}

An AUROC below 0.5 means the score is higher for correct outputs. Why temperature and errors are related in this dataset was not established. If the relation comes from how the responses were produced or selected, it would not hold in deployment, and BASELINE is then stronger here than a deployable baseline would be. The arms were refitted on the open splits with and without the column, with the same search and seed:

{{table: temperature_arms}}

{{table: temperature_difference}}

The two refits with temperature select the frozen configurations and reproduce their DEV numbers. Without temperature, BASELINE ranks worse on DEV_TEST, and adding the SOURCE columns still does not improve it. The only generators certified anywhere in this study have a single temperature. This probe is on DEV. The sealed splits were not used for it.

**The decision rule when there is no gain.** The power simulation of Section 2.9 built its second arm by moving correct outputs toward acceptance, so it never simulated a gain of zero. Here each pair is two refits of the frozen BASELINE configuration, as in the A/A test, put through the paired bootstrap ({{controls/open_probes.json > no_gain > bootstrap_draws}} draws) and the rule at the CAL and TEST sizes:

{{table: no_gain}}

## 4. What the results do and do not show

**What they show.**

1. Under the pre-registered rule, H1 fails: the gain in certified coverage from the SOURCE features is below {{confirmatory/result.json > BAR_points | .2f}} points. H3 fails in the same way. H2 could not be tested.
2. On ranking, where smaller differences can be resolved, neither family of columns improved the combined model on RAGTruth TEST. The 95% interval for the change in AUROC is {{controls/result.json > secondary > paired_difference_from_baseline > BASELINE+SOURCE > auroc > low | +.4f}} to {{controls/result.json > secondary > paired_difference_from_baseline > BASELINE+SOURCE > auroc > high | +.4f}} for SOURCE and {{controls/result.json > secondary > paired_difference_from_baseline > BASELINE+SELF > auroc > low | +.4f}} to {{controls/result.json > secondary > paired_difference_from_baseline > BASELINE+SELF > auroc > high | +.4f}} for SELF.
3. The SOURCE columns are not noise. Alone they rank errors, with an AUROC of {{controls/result.json > secondary > arms > SOURCE only > auroc | .3f}}. That ranking goes when sources are shuffled, and it holds within quintiles of output length. A lexical baseline carries as much: the share of novel word 1-grams alone has {{controls/result.json > secondary > single_scores > base.novel_1gram > auroc | .3f}}.
4. The calibrator kept its error bound in every check run here, and did not keep it where exchangeability was broken on purpose (Section 3.10).
5. The SOURCE profile is cheap to compute: {{controls/runtime.json > cpu_scores > scores > SOURCE profile features (16 columns) > seconds_per_megabyte | .3f}} seconds per megabyte on one CPU thread in Python, against {{controls/runtime.json > neural_baselines > MiniCheck groundedness, RAGTruth CAL and TEST, resumed run > seconds_per_megabyte | .1f}} for the groundedness model on the same machine's GPU.

**What they do not show.**

- They do not show that match-length features carry no signal for selective acceptance. FAIL means that the gain is below BAR. A gain of a few points lies inside the interval, and the design could not have detected one (Section 2.9).
- They do not show what the features would add to a weaker baseline. BASELINE holds a dedicated groundedness model and n-gram overlap with the source. On TEST the SOURCE columns alone certified {{controls/result.json > coverage > SOURCE only > 0.1 > coverage_points | .2f}} points, against {{controls/result.json > coverage > BASELINE > 0.1 > coverage_points | .2f}} for BASELINE. In the DEV rehearsal they certified {{controls/rehearsal.json > coverage > SOURCE only > 0.1 > coverage_points | .2f}}, against {{controls/rehearsal.json > coverage > BASELINE > 0.1 > coverage_points | .2f}}. That comparison was not pre-registered, both differences are inside the noise floor, and no conclusion is drawn from it.
- They do not reach beyond RAGTruth and, for ranking only, WikiBio GPT-3: six generators, three task types, English text, and human span annotation as the ground truth.
- Nothing here verifies that an output is true. An accepted output is one whose score fell below a threshold. The threshold was calibrated so that the error rate among accepted outputs is bounded, with the stated confidence, when calibration and deployment data are exchangeable.

**An interpretation, not a finding.** One account fits all of the above: match-length statistics against the source measure verbatim overlap, and the lexical baselines (n-gram novelty, ROUGE-L) already measure verbatim overlap. The study did not test this account.

**What limited the design.** The calibration sample per generator. Section 2.9 shows that the endpoint returns PASS reliably only for a gain of about 20 points or more, and Section 3.9 shows how strongly BASELINE's own coverage depends on the size of CAL. Whether a larger calibration sample would lower the noise floor enough to judge a gain of 5 points was not measured.

## 5. Limitations and threats to validity

**Power.** The design does not detect a gain the size of BAR with 80% power, and it returns FAIL for most true gains of about 3 points or less. This was known before any tested model was fitted. The primary outcome therefore says little about small gains.

**The noise floor exceeds the baseline's coverage.** FLOOR is {{confirmatory/result.json > FLOOR_points | .2f}} points and BASELINE certifies {{confirmatory/result.json > coverage > BASELINE > 0.1 > coverage_points | .2f}} on TEST. A PASS would have required the tested arm to more than double certified coverage.

**One certified generator.** All certified coverage is GPT-4 outputs on TEST and GPT-3.5 outputs in the DEV rehearsal. A coverage estimate moves by a whole generator's share when a certificate is gained or lost, as Section 3.8 shows.

**One confirmatory dataset.** RAGTruth only. H2 was not testable on WikiBio. Nothing is known from this study about other task types, other generators, longer sources or other languages.

**The replication was specified and not implemented.** The pre-registration fixes its procedure (its Section 11). No code for it was written, because it was to be written only if H1 came out INCONCLUSIVE. Had that happened, the code would have been written after CAL and TEST had been read. It was not triggered.

**What the count of sealed reads rests on.** The ledger and the session's command log (Section 3.7). The guard is a procedural control, not a technical barrier.

**The pre-registration was not deposited with a third party.** It was committed and tagged in a private repository, which had no remote, before the sealed labels were read. That it preceded the confirmatory run rests on that repository's history, which the public repository does not contain (Section 7).

**Published label statistics had been read.** Before the splits were made, the dataset's published counts of responses with a hallucination, by task and by generator, were read from its documentation, and so was the published share of non-factual sentences in WikiBio. Those figures cover every row, including the rows that were later sealed and the reserved set. The pre-registration's list of what had been seen (its Section 17) does not mention them.

**Similar sources across splits.** One pair of similar sources sits across CAL and TEST, below the pre-registered threshold (Section 3.13).

**The temperature covariate.** BASELINE holds it, as pre-registered. It is strongly related to the label within four generators, for reasons that were not established (Section 3.13).

**Baselines that were not run.** Verbalized confidence, token log-probability, token entropy and semantic entropy need access to the generator, which RAGTruth does not give. SelfCheck-style consistency ran on WikiBio only. BASELINE is the best combination of what could be run, not of everything the brief listed.

**No fidelity check for the groundedness baseline.** SelfCheck-NLI was checked against its published figure (Section 2.4). MiniCheck is applied by this study's own code, which follows its reference inference code, at a pinned revision. The reviewer compared the two and found them to agree. The scores were not checked against a published number.

**Labels.** The ground truth is human span annotation, and the loss is 1 when a response has any annotated span. Section 3.8 varies that definition. The outcome does not change and the point estimates do.

**Pooled ranking metrics.** The pre-registered AUROC and AUPRC pool all generators, and BASELINE is given the generator. The within-generator table of Section 3.3 is exploratory and has no intervals.

**The bootstrap interval.** The endpoint is a step function of the data, and a percentile bootstrap of such a statistic need not have its nominal coverage. H3's point estimate sits near the lower end of its own interval. The decision rule prescribes this interval, and it was applied as written.

**The guarantee is marginal and conditional.** It bounds the error rate over all accepted outputs, under exchangeability. A subgroup can have a higher error rate, and Section 3.10 shows the bound failing across task types. The guarantee is Theorem 5 of [McE26], by the owner of this study. It was not re-derived here. It was checked empirically (Section 3.6).

**The entropy features.** The match-length entropy estimates are used as scores. The fixed-reference form is this study's own substitution, and nothing is claimed about what it estimates. On short outputs the estimates are biased by the end of the data.

**The paraphrase stress** has no labels, a small sample and one paraphrase model.

**Runtime.** The reference implementation is Python. The groundedness model was timed on {{controls/runtime.json > neural_baselines > MiniCheck groundedness, RAGTruth CAL and TEST, resumed run > items}} of the {{controls/runtime.json > neural_baselines > MiniCheck groundedness, RAGTruth CAL and TEST, resumed run > items_in_the_splits | ,d}} sealed items, those scored after the restart described in Section 6. SelfCheck-NLI was timed while another job held most of the machine's memory.

**One machine.** The neural baselines ran on one Apple M1 Max. On another backend their scores can differ in the last decimals, and numbers downstream of them can move.

**Sources not read in the original.** The journal version of [CV05] (the arXiv version was read), [BBHECS85], [HOLM79] and [CP34] were not read; the References section says what was checked in their place. The appendices of the OpenHalDet paper were not read; that dataset was examined as a lead and not used.

**Who did the work.** The harness, the pre-registration and this report were written by an AI system (Claude, Anthropic) under the owner's instructions, with the gates described in Section 2.9 and in the pre-registration. The audit of Section 8 was done by a second instance that had not seen the implementation. No human statistician has reviewed the work.

**Competing interest.** LogNovations develops FastCodec, a proprietary codec. The features tested here are computed by an open implementation that does not use it (Section 3.12). The study was run as an experiment in the open: the plan, the code, every record and the audit are public, whichever way the result came out.

## 6. Deviations from the pre-registration

**One, in what is reported.** Section 16 of the pre-registration said that a second implementation of the match-length profile, built on a proprietary codec, would be reported for agreement with the reference backend and for throughput. The comparison was run, and its table stood in Section 3.12 of this report. Before publication, and after seeing it, the owner decided not to publish measurements of the proprietary backend. The table and its record were removed, and `DEVIATIONS.md` records the change with its time. No hypothesis, outcome or other number depends on that backend.

**None in what was analysed.** The decision rule, the endpoint, the datasets, the splits, the baselines, the frozen models, the grid, alpha, delta, BAR and FLOOR are those of the document tagged `prereg-v1`. A test checks that six frozen files (the pre-registration and five records) and the 22 code files of the confirmatory analysis are unchanged since the tag. Two analysis modules are not in that list: after the tag the combiner gained an optional search space and the table loader a reader for supplementary tables. The confirmatory run uses neither addition. The guard on the sealed splits refuses to run if the pre-registration differs from the tagged version by one byte.

**Between the owner's approval and the tag**, three places of the pre-registration were edited: the status line, one sentence about the deviations log, and the section that says where the log is kept. The pre-registration lists them in its Section 17.

**Three incidents.** First, the label-free scoring of CAL and TEST with the groundedness baseline was killed by the operating system after 4,025 of 4,608 items, because the paraphrase job beside it had exhausted the machine's memory. It was restarted for the remaining 583 items. No label is involved in scoring, and no sealed label had been read at that time. As a check, 24 items scored before the kill were scored again after the restart: all 206 of their support values equal the stored ones. `DEVIATIONS.md` records the incident with its times and commits. Those five counts are from that file and from the two scoring logs, not from `results.json`.

Second, nine seconds before the confirmatory run, the DEV rehearsal was run again at the commit of that run, to check that the analysis code still reproduced the committed rehearsal record. It did, and the file was restored. Third, seven minutes after the confirmatory record was committed, the tests that check it were themselves checked by mutation: six fields of the record were altered in place, one at a time, each alteration was caught, and the file was restored. Neither event read a sealed label, and the committed content of neither file changed. Both left a new creation time on a file, which the audit noticed (Section 8, finding 1).

**Choices the pre-registration left open.** Each was made and committed after the tag and before any sealed label was read. Those that belong to the controls run were rehearsed on the open DEV splits first (`controls/rehearsal.json`).

- *Shuffled source.* The derangement is of sources within a task type, so the six responses of a source receive the same other source.
- *Models on the tested columns alone.* "A logistic model" was implemented as the logistic part of the search space, selected on DEV_SELECT like every other model. Output length alone uses the full search.
- *Token alphabet.* Tokens are lower-cased alphanumeric runs. The 16 features per frame are unchanged.
- *Paraphrase.* The model is `{{controls/paraphrase.json > provenance > paraphrase_model}}` at a pinned revision, with greedy decoding. The sample is all six responses of the first 50 TEST sources in a fixed hash order.
- *Runtime.* The CPU scores are timed on {{controls/runtime.json > cpu_scores > items}} TRAIN responses.
- *Cheap alternative.* A reference row at the actual size of CAL was added to the table.

**Analyses that are not in the pre-registration.** Each is labelled exploratory where it appears.

- The WikiBio comparison with passage length on both sides (Section 3.3).
- The error rate within each generator, read from the pre-registered risk–coverage curves (Section 3.3).
- Certified coverage of the models on the SOURCE and SELF columns alone, and the overlap of accepted outputs (Section 3.10).
- AUROC of the three combined arms within length quintiles, beside the pre-registered AUROC of the tested columns alone (Section 3.5).

The first and the last of these were in the scripts, and in the DEV rehearsals, before any sealed label was read. The other two were added after the confirmatory results had been seen. They use what the controls run recorded and read no label.

The loss-sensitivity and stress analyses are in the pre-registration, which labels them exploratory.

**Reduced relative to the brief**, as the pre-registration states: four baselines were not run (Section 2.4); H2 is not testable (Section 2.10); calibration across datasets became calibration across task types, since there is one confirmatory dataset.

## 7. Reproducibility

**One command.** `make reproduce` runs `scripts/reproduce.py`. In a fresh clone, on its own branch, it fetches the raw files and checks their SHA-256 against `datasets.manifest.json`. It then rebuilds the item tables, every score, the frozen models, every record, `results.json`, the figures and this report, in the order the study ran them and with the same seeds. `make compare REF=<revision>` compares every record with the committed one, ignoring provenance stamps, ledger rows and timings.

**Numbers.** Every measured result, in a table or quoted in the text of Sections 3 and 4, is read from `results.json` when the report is rendered. A path that does not exist stops the rendering, and a test checks that the committed report equals a fresh rendering. Each record carries the commit at which it was computed and whether the tree was dirty, and `results.json` is stamped again as a whole.

Four kinds of number are typed, not read. The settings and constants of the design, such as alpha and the number of columns; a test recomputes the constants of Section 2.7 and 2.10 from the calibrator's bound. Round readings of the power table, such as "about 3 points". Two remarks read off a table: the size of one error in Section 3.3 and the largest difference in Section 3.5. And the counts of the incidents and spot checks in Sections 6 and 7, which come from `DEVIATIONS.md`, the scoring logs and the commit messages. Section 8 quotes the reviewer's own figures, which are marked as such.

**Environment.** `requirements.lock` pins every package. `Dockerfile` builds an image with the CPU build of PyTorch. In that image, built from the tree of commit `9d915f0` with the report regenerated, the test suite passed (509 tests; 6 that need the git history were skipped) and the smoke experiment printed the same numbers as on macOS. The full reproduction was not run in the image.

**CI.** `.github/workflows/ci.yml` runs the tests and the smoke experiment. It had not run on a hosted runner when this report was written, because nothing had been pushed.

**What was verified.** One reproduction was run, on the machine the study ran on, in a fresh clone of the repository. It fetched the raw files again and checked their SHA-256. It then rebuilt the item tables, the features, the three frozen models and every record that existed at the commit it was cloned from.

{{table: reproduction}}

Every record was regenerated, as its new provenance stamp shows, and every one is identical to the study's apart from stamps, ledger rows and timings. The three models have the same file hash. The record of this comparison was written in the clone, so its own stamp names a commit of the clone.

The last step of that run failed: the report could not be rendered in the clone, because it read the record of the private second backend, which the clone does not have. The report no longer reads that record (Section 3.12). A test also renders the whole report without the record of this check, which a later reproduction would lack.

**What was not verified.**

- The neural baselines and the paraphrases were not recomputed. They ran once, during the study, and the reproduction copied their score files. Two spot checks reproduced stored values exactly: 24 sealed items rescored with the groundedness model (Section 6), and 20 paraphrases generated again.
- The record of Section 3.13 was added after the reproduction and was computed once.
- The reproduction ran on the study's machine. On a machine with another number of cores, the file of a fitted model has a different hash while its scores are identical; this was checked by refitting BASELINE with four threads. `make compare` therefore compares those hashes apart from the numbers.
- Nothing was run from a public export. The clone was of the private repository.

**Provenance stamps.** A stamp gives the commit and whether the tree had uncommitted changes. The audit found two things to know about them. The Phase 2 scripts take the stamp when the record is written, not when the run starts. And `controls/paraphrase.json` is stamped with the commit before a later change to its script; that change touches the generation step only (it releases memory after each output, and was prompted by the first incident of Section 6), and the 20 paraphrases generated again after it are identical to the stored ones. `results/throughput.json` and the second backend's record sit in a directory that git ignores, and reach the repository only inside `results.json`.

**The public repository.** The study was run in a private repository that also holds proprietary code. The public repository is an export of the harness: one commit with the tree as it was when the pre-registration was tagged, which carries the tag `prereg-v1` there, and one commit with the final tree. The second backend's code is not in it, and a test checks that no public file imports from that code or names its internals. The commit hashes in the records, the ledger and this report are those of the private repository; `HISTORY.md` lists them with their dates and subjects. The tests that compare those commits with the history skip in the export, where the commits do not exist.

## 8. Audit by an independent reviewer

The brief required a reviewer that had not seen the implementation, with the instruction to refute the result. A second instance of the AI system that built the study was started with none of the builder's history or reasoning. It was given the brief, the repository at commit `986c1fb` and that instruction. It was told to probe leakage, reuse of TEST, a grid chosen from the sample, weak baselines, duplicate groups, drift of the label definition, cherry-picking of datasets, claims beyond the evidence and the statistics. It was forbidden to read a sealed label or a reserved row, to invoke the guard, or to change the study's tree. It worked in a clone of its own that held the label-free tables and no raw file.

The figures in this section are the reviewer's, quoted from its report. They are not in `results.json`, and they were recomputed only where a row below says so.

**Verdict of the reviewer.** In its words: "The headline outcome stands: under the rule fixed in the brief, with BAR = FLOOR = 10.78 recomputed exactly from the committed open-data records, H1 is FAIL and H3 is FAIL. The result is insensitive to bootstrap error and to the two readings the pre-registration fixed. H2's 'not testable' follows from sample size as stated. I found no path by which a CAL or TEST label could have reached a feature, a model, a rank transform, the grid or the thresholds other than through calibration, and the frozen models and sealed-split score tables reproduce exactly from open, label-free inputs. I found no deviation from the pre-registration in what was run, beyond the unimplemented replication path, which was never triggered."

It added that the report needed about a dozen corrections, that none of them changes an outcome, and that one question, about the number of sealed reads, was open and was for the builder to answer.

**Attempts at refutation that found no defect**, as the reviewer reports them:

- It refitted the three frozen models from TRAIN and DEV_SELECT. The records and the model files were identical to the committed ones.
- It recomputed the feature tables of CAL and TEST from the label-free item and source tables. They were bit-identical.
- It recomputed c_B, FLOOR and BAR from the A/A and baseline records, and the two outcomes from the confirmatory record. All matched exactly.
- It found FAIL robust to the Monte Carlo error of the bootstrap (0.52% and 0.96% of draws above BAR, against the 2.5% that would change the outcome), to a signed reading of the A/A gap and to the percentile convention. It noted that FAIL depends on the brief's clause that raises BAR to FLOOR: at the nominal 5 points both hypotheses would be INCONCLUSIVE.
- It checked the groups: 2,514 sources, each with six responses from six generators, none in two splits, and the hash rule reproduces every assignment.
- It checked the label definition: one function serves the open build and the sealed reader, and all 10,476 open labels satisfy it.
- It checked the baselines: the same 24 configurations and seed for every arm; three larger models outside the search space had a worse log loss; it compared this study's MiniCheck code with the upstream inference code and found them to agree.
- It reran the DEV rehearsals and the fidelity check. All were identical to the committed records.
- It applied 18 mutations to the statistics code and the records. The tests caught 13.

**Findings, and what was done about each.**

| # | Finding of the reviewer | Its severity | What was done |
|---|---|---|---|
| 1 | "The confirmatory analysis ran once" cannot be verified from the repository. The confirmatory record was rewritten on disk seven minutes after it was committed, and the rehearsal record nine seconds before the run. | major, unresolved | Both rewrites are explained by the session's command log: a rehearsal run again before the confirmatory run, and a mutation check of the record tests after it, each followed by a restore. No second guarded run took place. Both are now in `DEVIATIONS.md` and Section 6. Section 3.7 no longer credits a test with showing "once" and says what the count rests on. It still cannot be proved from the repository alone. |
| 2 | The pre-registered replication does not exist as code. | minor | Stated in Section 5. It was not triggered. |
| 3 | One pair of similar sources sits across CAL and TEST, at a similarity of 0.83, below the threshold of 0.9. | minor | Recomputed ({{controls/open_probes.json > near_duplicates > largest_between_cal_and_test | .3f}}), recorded, and listed in Section 3.13. |
| 4 | The report says sealed labels "can be read only through" the guard. The guard is procedural. | minor | Section 2.3 reworded. The pre-registration is frozen and keeps its stronger wording in its Section 4.1. |
| 5 | The count of reads leaves out the reproduction. | minor | Section 3.7 and the Summary give both counts, and the reproduction's ledger rows are in a record. |
| 6 | The report's provenance line named a commit that did not hold `results.json`, and left out its dirty flag. | minor | The line now says what the commit is and prints the flag. `results.json` was collected again from a clean tree. |
| 7 | "Every number in the text is read from `results.json`" was false, and one sentence named the wrong strongest SOURCE column. | minor | Section 7 lists what is typed. The sentence is replaced by a table computed from the results (Section 3.3); the reviewer was right. |
| 8 | The temperature covariate is a strong proxy for the label within four generators, and the report did not say so. | minor | Measured on the open splits and reported in Section 3.13; noted in Sections 2.4 and 5. |
| 9 | Five of its mutations survived the test suite; the test of frozen files covered no code; the CI smoke experiment was degenerate. | minor | Six tests added, each checked to fail under the mutation it guards against. The frozen list now covers the 22 code files of the confirmatory analysis. The smoke experiment's arms now differ, and a test requires it. |
| 10 | Three statements were stronger than what was run: what the design returns when there is no gain; "the controls behaved as a sound pipeline should"; "both guarantee checks passed". | minor | The first was measured (Section 3.13): FAIL for a share of {{controls/open_probes.json > no_gain > p_fail | .2f}} of {{controls/open_probes.json > no_gain > replications}} pairs with no gain. The reviewer's own probe gave 79% of 120 pairs. Sections 3.1, 3.4 and 3.6 and the Summary were reworded. |
| 11 | The hash of a model file depends on the number of threads, so a reproduction on another machine would be reported as different. | minor | Confirmed by refitting BASELINE with four threads. The comparison now treats those hashes apart from the numbers (Section 7). |
| 12 | Wording: the MiniCheck reference; "certifies GPT-4"; one sentence of Section 6; published label statistics missing from the pre-registration's list of what had been seen. | minor | Corrected in the References and in Sections 2.4, 2.6 and 6. The omission is stated in Section 5. |
| 13 | Provenance stamps are taken when a record is written, and one record's stamp precedes a change to its script. | note | Stated in Section 7. |

**What the reviewer could not check.** Every number that needs a sealed label: the certified thresholds, the realised risks, the two intervals and the controls. The reproduction of Section 7 regenerated all of them identically, which shows that the code is deterministic and not that it is right. It also could not check the near-duplicate rule on the raw source content, the neural scores, the owner's approvals, the second backend, or the container beyond its log.

**Limits of this audit.** The reviewer is another instance of the system that built the study. It is independent of the implementation's history and not of that system's blind spots. No human has reviewed the work. The corrections above were made by the builder and have not been audited again.

## References

What was checked for each source is recorded in `REFERENCES.md`. In short:

- **[BBHECS85]** A. Blumer, J. Blumer, D. Haussler, A. Ehrenfeucht, M. T. Chen and J. Seiferas. The smallest automaton recognizing the subwords of a text. *Theoretical Computer Science* 40 (1985), 31–55. Bibliographic record confirmed. The article was not read; the implementation is tested against brute-force definitions.
- **[BGPR23]** N. Barnfield, R. Grondin, G. Pozzoli and R. Raquépas. On the Ziv–Merhav theorem beyond Markovianity. arXiv:2310.01367. Read as a restatement of the parse of [ZM93].
- **[CP34]** C. J. Clopper and E. S. Pearson. The use of confidence or fiducial limits illustrated in the case of the binomial. *Biometrika* 26(4) (1934), 404–413. Bibliographic record confirmed. The article was not read; the bound used is the calibrator package's function, whose source was read.
- **[CV05]** R. Cilibrasi and P. M. B. Vitányi. Clustering by compression. *IEEE Transactions on Information Theory* 51(4) (2005), 1523–1545. Equation (3.1) was read in arXiv:cs/0312044v2. The journal text was not read.
- **[HOLM79]** S. Holm. A simple sequentially rejective multiple test procedure. *Scandinavian Journal of Statistics* 6 (1979), 65–70. The issue number was not confirmed and the article was not read. The adjustment as implemented is stated in `selfref.confirm.holm` and tested.
- **[KASW98]** I. Kontoyiannis, P. H. Algoet, Yu. M. Suhov and A. J. Wyner. Nonparametric entropy estimation for stationary processes and random fields, with applications to English text. *IEEE Transactions on Information Theory* 44(3) (1998), 1319–1327. Theorem 1 was read in the author-hosted copy.
- **[McE26]** C. McElveen. Explanation is not certification: information-theoretic limits of interpretability-based assurance for machine learning systems. 2026. In the companion repository at the pinned commit. Theorem 5 and the calibrator's source were read.
- **[MLG23]** P. Manakul, A. Liusie and M. J. F. Gales. SelfCheckGPT: zero-resource black-box hallucination detection for generative large language models. EMNLP 2023; arXiv:2303.08896. Table 2 and the dataset counts were read in the PDF.
- **[NIU24]** C. Niu, Y. Wu, J. Zhu, S. Xu, K. Shum, R. Zhong, J. Song and T. Zhang. RAGTruth: a hallucination corpus for developing trustworthy retrieval-augmented language models. arXiv:2401.00396. The generators, tasks and annotation procedure were read in the PDF.
- **[TLD24]** L. Tang, P. Laban and G. Durrett. MiniCheck: efficient fact-checking of LLMs on grounding documents. *Proceedings of EMNLP 2024*, 8818–8847. Bibliographic record confirmed. The model was run by this study's code, which follows its reference inference code.
- **[ZM93]** J. Ziv and N. Merhav. A measure of relative entropy between individual sequences with application to universal classification. *IEEE Transactions on Information Theory* 39(4) (1993), 1270–1279. The definition of the parse was read in the author-hosted scan. The harness reproduces the article's worked example.

## Appendix A. Every single score on TEST

AUROC with an error as the positive class. A value below 0.5 means the score is higher for correct outputs.

{{table: single_scores}}

## Appendix B. DEV rehearsal of the confirmatory table

Open data, run before the pre-registration was approved. Not a confirmatory result.

{{table: coverage_dev}}
