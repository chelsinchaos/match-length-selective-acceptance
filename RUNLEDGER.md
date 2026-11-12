# Run ledger

Every read of a sealed split (CAL or TEST) is recorded here by `selfref.guard.authorise`.
The guard appends the rows and nothing else writes them. The text above the table was edited once by hand, after the reads below, to remove the sentence "No sealed split has been read."

Rows 1 to 4 are reads of RAGTruth. Row 5 is a read of the TEST labels of WikiBio. The guard counts reads by split name, not by dataset, so it numbers that read as the third access to TEST: RAGTruth TEST was read twice and WikiBio TEST once.

| # | time (UTC) | split | access to this split | commit | purpose |
|---|---|---|---|---|---|
| 1 | 2026-11-12T14:33:03Z | CAL | 1 | d0cf94010ca8f7f4d2f842f2a9347041aecb73c5 | confirmatory run |
| 2 | 2026-11-12T14:33:03Z | TEST | 1 | d0cf94010ca8f7f4d2f842f2a9347041aecb73c5 | confirmatory run |
| 3 | 2026-11-12T14:35:05Z | CAL | 2 | 688893df210119af81652ecfed7a344b25c64b5f | controls and secondary metrics |
| 4 | 2026-11-12T14:35:05Z | TEST | 2 | 688893df210119af81652ecfed7a344b25c64b5f | controls and secondary metrics |
| 5 | 2026-11-12T14:39:08Z | TEST | 3 | 1cd86c56c54726295de67d1751e96c86a064f91e | WikiBio ranking analysis |
