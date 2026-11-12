# History of the study's repository

The study was run in a private repository that also holds proprietary code. The
public repository is an export of the harness and has a history of its own: one
commit with the tree as it was when the pre-registration was tagged, and one
with the final tree.

The commit hashes in the records, in `RUNLEDGER.md` and in `REPORT.md` are those
of the private repository. This file lists every commit there that touched the
public part of the harness, oldest first, with its time and its subject, so
that a reader can follow the order of events. The list cannot be checked
against the public repository, where these commits do not exist. The commit
that holds this file is not in it: a file cannot name the commit that contains
it.

Times are UTC.

| Commit | Time (UTC) | Subject | Note |
|---|---|---|---|
| `e34a50d` | 2026-11-11 16:50:02 | research(selfref): Phase 1 feature library, reference backend, pinned selcert |  |
| `3ee56b0` | 2026-11-11 17:00:22 | research(selfref): check feature formulas against sources; fix SELF estimator |  |
| `5a6e3fd` | 2026-11-11 17:31:26 | research(selfref): record the owner's second answers and the H2 prior |  |
| `5cd7b5d` | 2026-11-11 17:37:15 | research(selfref): Phase 2 step 1 for RAGTruth: fetch, screen, dedup, split |  |
| `33d72f3` | 2026-11-11 17:48:20 | research(selfref): baselines, groundedness scorer, feature tables |  |
| `eafdffb` | 2026-11-11 17:51:09 | research(selfref): certification wrapper and the decision rule |  |
| `0c03d0b` | 2026-11-11 17:54:17 | research(selfref): combiner search, DEV rehearsal, A/A run, paired bootstrap |  |
| `5d74aea` | 2026-11-11 18:00:26 | research(selfref): WikiBio builder, table assembly, power simulation, Phase 2 driver |  |
| `e022b78` | 2026-11-11 18:06:23 | research(selfref): SelfCheck-NLI baseline, WikiBio features, fidelity script |  |
| `8957cfc` | 2026-11-11 18:13:19 | research(selfref): guard on the sealed splits, and guarded readers of their labels |  |
| `2e4095a` | 2026-11-11 18:17:54 | research(selfref): the confirmatory analysis, shared by rehearsal and sealed run |  |
| `aef75e4` | 2026-11-11 18:19:51 | research(selfref): draft PREREGISTRATION.md before any combiner is fitted |  |
| `49a5c33` | 2026-11-11 20:47:02 | research(selfref): fidelity check passed on the open WikiBio splits |  |
| `88c7abe` | 2026-11-12 00:44:40 | research(selfref): freeze BASELINE and record its DEV rehearsal (Phase 2 steps 5-6) |  |
| `8769bb9` | 2026-11-12 00:45:23 | research(selfref): A/A test, 1,000 baseline-versus-baseline runs (Phase 2 step 7) |  |
| `f38fa25` | 2026-11-12 00:46:25 | research(selfref): record A/A gaps unrounded; test that BAR and FLOOR follow from the records |  |
| `dcc8a87` | 2026-11-12 00:48:09 | research(selfref): correct the return annotation of fit_arm |  |
| `be93377` | 2026-11-12 00:49:44 | research(selfref): A/A record regenerated with unrounded gaps (Phase 2 step 7) |  |
| `0273903` | 2026-11-12 00:49:54 | research(selfref): commit c_B, FLOOR and BAR before any tested feature is fitted (Phase 2 step 8) |  |
| `e093d0a` | 2026-11-12 00:59:28 | research(selfref): power by simulation at the CAL and TEST sizes (Phase 2 step 9) - GATE TRIGGERED |  |
| `4fcdf7f` | 2026-11-12 01:00:51 | research(selfref): power at other effect sizes; an exploratory BASELINE-only look at the design |  |
| `1d297db` | 2026-11-12 01:02:32 | research(selfref): exploratory record - the A/A noise under other grids and the pooled fallback |  |
| `7d624dd` | 2026-11-12 01:10:35 | research(selfref): the decision rule at true gains below BAR (supplement to step 9) |  |
| `dfb2932` | 2026-11-12 01:13:36 | research(selfref): record the triggered power gate in the draft pre-registration and task list |  |
| `eb0e2c5` | 2026-11-12 05:27:15 | research(selfref): record the owner's decision at the power gate, before any tested arm is fitted |  |
| `4f22f05` | 2026-11-12 05:28:29 | research(selfref): fit, select and freeze BASELINE+SOURCE and BASELINE+SELF (Phase 2 step 10) |  |
| `7c78ae2` | 2026-11-12 05:30:09 | research(selfref): DEV rehearsal of the whole confirmatory procedure with the frozen arms |  |
| `fc3c3ec` | 2026-11-12 05:31:19 | research(selfref): complete the draft pre-registration with the tested arms' DEV rehearsal (Phase 2 step 11) |  |
| `752254e` | 2026-11-12 11:06:54 | research(selfref): the owner approved the pre-registration; freeze it for prereg-v1 | tagged `prereg-v1` at 2026-11-12 11:06:54 |
| `e6ccc0a` | 2026-11-12 11:23:55 | research(selfref): Phase 4 controls as code, written before any sealed label is read |  |
| `2ed4f5d` | 2026-11-12 11:28:00 | research(selfref): DEV rehearsal of the Phase 4 controls (open data, not a confirmatory result) |  |
| `ce17f9b` | 2026-11-12 11:28:43 | research(selfref): DEV rehearsal of the WikiBio ranking analysis (open data) |  |
| `575c832` | 2026-11-12 11:42:01 | research(selfref): Phase 5 machinery: results.json, report rendering, reproduction, container, CI |  |
| `8c1b14e` | 2026-11-12 11:49:02 | research(selfref): report template (methods; results as placeholders) and a cleaner container context |  |
| `a3e79c3` | 2026-11-12 11:49:53 | research(selfref): README layout and run instructions for the finished harness |  |
| `2d7a804` | 2026-11-12 11:50:41 | research(selfref): consistency tests for the confirmatory and controls records, written before they exist |  |
| `0c2ded4` | 2026-11-12 11:52:56 | research(selfref): the relative Lempel-Ziv parse checked against the 1993 article itself |  |
| `d0cf940` | 2026-11-12 14:31:11 | research(selfref): record the interrupted sealed scoring run; time the resumed run on the items it scored | sealed labels read at this commit (2026-11-12T14:33:03Z): confirmatory run |
| `688893d` | 2026-11-12 14:34:38 | research(selfref): the confirmatory run on CAL and TEST (H1: FAIL) | sealed labels read at this commit (2026-11-12T14:35:05Z): controls and secondary metrics |
| `1cd86c5` | 2026-11-12 14:38:52 | research(selfref): the controls and secondary metrics on CAL and TEST | sealed labels read at this commit (2026-11-12T14:39:08Z): WikiBio ranking analysis |
| `4989856` | 2026-11-12 14:39:44 | research(selfref): the WikiBio ranking analysis on TEST (RESAMPLE features) |  |
| `afcdf98` | 2026-11-12 14:41:58 | research(selfref): the run ledger no longer says that no sealed split has been read |  |
| `5797e98` | 2026-11-12 14:53:33 | research(selfref): which outputs the certified rules accept; paraphrase step no longer grows without bound |  |
| `14539cc` | 2026-11-12 14:53:33 | research(selfref): the paraphrase stress record (exploratory, label-free) |  |
| `c53a3c7` | 2026-11-12 14:53:58 | research(selfref): record of which TEST outputs each certified rule accepts (exploratory, label-free) |  |
| `16b2e0d` | 2026-11-12 14:54:00 | research(selfref): runtime per megabyte of every score |  |
| `4010601` | 2026-11-12 15:01:50 | research(selfref): report passages for the results, discussion, limitations and deviations |  |
| `986c1fb` | 2026-11-12 15:03:18 | research(selfref): results.json, the figures and a draft of REPORT.md |  |
| `17ab26f` | 2026-11-12 15:55:56 | research(selfref): code changes after the independent audit |  |
| `4aafbb2` | 2026-11-12 16:53:02 | research(selfref): record of the three probes asked for by the audit (exploratory, no sealed label) |  |
| `4e14e41` | 2026-11-12 16:53:02 | research(selfref): the audit's findings and the fixes, in the report; two more incidents recorded |  |
| `9d915f0` | 2026-11-12 16:53:23 | research(selfref): runtime per megabyte, each CPU score timed three times on an idle machine |  |
| `acf92fd` | 2026-11-12 16:58:15 | research(selfref): final REPORT.md and results.json |  |
| `0d74854` | 2026-11-12 17:56:02 | research(selfref): the paper |  |
| `d524cdd` | 2026-11-12 18:17:33 | research(selfref): prepare the public export; the second backend's measurements are not published |  |
