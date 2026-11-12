# Deviations log

Departures from `PREREGISTRATION.md` made after it was tagged `prereg-v1`. The pre-registration itself stays byte-identical to the tagged version, because the guard on the sealed splits requires it (its Sections 4.1 and 19), so the log is kept here.

Each entry gives the time, what changed and why. An analysis that the pre-registration does not describe is not a deviation; it is labelled exploratory in the report.

## Deviations

One deviation has been recorded. It concerns what is reported, not what was analysed.

| Time (UTC) | What changed | Why |
|---|---|---|
| 2026-11-12 18:12 | The measurements of the second backend are not published. Section 16 of the pre-registration said that a second implementation of the match-length profile, built on a proprietary codec, would be "reported for agreement with the reference backend and for throughput". The comparison was run. Its record was in `results.json` and its table was Section 3.12 of the report until this time; both have been removed, and Section 3.12 now says only that the comparison was run privately. | The owner decided, after seeing the report and before anything was published, not to publish measurements of the proprietary backend. No hypothesis, outcome or other number of the study depends on them, as the pre-registration states in the same section. |

## Incidents

Events during the run that changed nothing in the plan. They are listed so that the record is complete.

| Time (UTC) | What happened | What was done |
|---|---|---|
| 2026-11-12 14:05:09 | The label-free scoring of CAL and TEST with the groundedness baseline was killed by the operating system (exit status 137) after 4,025 of 4,608 items. It had started at 11:07:24 at commit `752254e`. The paraphrase job was running on the same machine and had grown to about 65 GB of memory; swap was full. No label is involved in scoring, and no sealed label had been read: `RUNLEDGER.md` was empty. | The scoring was restarted at 14:14:23 at commit `0c2ded4` for the remaining 583 items and ended at 14:29:11. The scorer writes one record per item and skips items already scored. Between the two commits the scoring script differs by one option, `--derived`, which neither run used; the scoring module is unchanged. As a check, 24 items scored before the kill were scored again after the restart: all 206 of their support values equal the stored ones to the six decimals stored. The two logs are `results/logs/groundedness_ragtruth_sealed.log` and `results/logs/groundedness_ragtruth_sealed_resumed.log`. |
| 2026-11-12 14:32:54 | `prereg/rehearsal.json` was rewritten on disk nine seconds before the confirmatory run. Its committed content never changed. | Before the one confirmatory run, the DEV rehearsal was run again at the commit of that run (`d0cf940`) to check that the analysis code still reproduced the committed rehearsal record. It did: the two were identical apart from the provenance stamp. The file was then restored with `git checkout`, which gave it a new creation time. The rehearsal uses open data and reads no sealed label. The independent audit noticed the creation time (REPORT.md, Section 8). |
| 2026-11-12 14:41:50 | `confirmatory/result.json` was rewritten on disk seven minutes after it was committed. Its committed content never changed: the blob is the same at commit `688893d` and at every later commit. | A mutation check of `tests/test_confirmatory_records.py`. Six fields of the file were altered in place, one at a time (the outcome, the gain, BAR, the upper end of the interval, the Holm value, the stamped commit). The tests were run after each alteration and caught every one. The file was restored from a copy after each and then with `git checkout`, which gave it a new creation time. No guarded run took place: the guard was not invoked, and no row was added to or removed from `RUNLEDGER.md`. The hand edit of the ledger eighteen seconds earlier changed its prose only, as the diff of commit `afcdf98` shows. The independent audit noticed the creation time and could not tell from the repository alone that this was the cause. The commands are in the transcript of the session that ran the study, which the owner holds. The output of the confirmatory run was printed in that session and was not written to a log file. |

