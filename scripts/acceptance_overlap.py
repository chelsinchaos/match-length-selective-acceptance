# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LogNovations Holdings LLC
"""Which TEST outputs the certified rules accept, and how far the rules agree (exploratory).

    python scripts/acceptance_overlap.py

No label is read and the guard is not asked. Acceptance needs only a score, a
generator and a threshold. The thresholds are those that the controls run
certified and recorded in controls/result.json; the scores are recomputed from
the label-free tables, with the frozen models and with the extra models fitted
exactly as the controls run fits them. For every arm, the number of outputs
accepted here must equal the number in the record, or the script stops.
Writes controls/acceptance_overlap.json.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np

HARNESS = Path(__file__).resolve().parents[1]


def load_controls_script():
    spec = importlib.util.spec_from_file_location("controls_script", HARNESS / "scripts" / "controls.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    c = load_controls_script()                 # its imports put src/ on the path
    from selfref.certify import accepted_mask
    from selfref.controls import acceptance_overlap

    record = json.loads((HARNESS / "controls" / "result.json").read_text())
    alpha = str(record["primary_alpha"])
    records = {"BASELINE": json.loads((c.PREREG / "baseline.json").read_text())}
    records.update(json.loads((c.PREREG / "profile.json").read_text())["arms"])
    categories = c.categories_of(c.DERIVED)
    c.require_label_free_inputs(("TRAIN", "DEV_SELECT", "TEST"))
    # The loss column of the TEST table is a placeholder and is never used.
    placeholder = {str(i): {"loss": 0} for i in np.load(c.DERIVED / "features_TEST.npz")["ids"]}
    tables = {"TRAIN": c.load_ragtruth_split(c.DERIVED, "TRAIN", categories),
              "select": c.load_ragtruth_split(c.DERIVED, "DEV_SELECT", categories),
              "test": c.load_ragtruth_split(c.DERIVED, "TEST", categories, sealed_labels=placeholder)}
    tokens = {"TRAIN": c.extra_columns(c.DERIVED, "tokens", "TRAIN", tables["TRAIN"].ids),
              "select": c.extra_columns(c.DERIVED, "tokens", "DEV_SELECT", tables["select"].ids),
              "test": c.extra_columns(c.DERIVED, "tokens", "TEST", tables["test"].ids)}
    sets = c.column_sets(tables["TRAIN"].names)
    token_names = tokens["TRAIN"][0]
    columns_of = {
        "LENGTH": (c.LENGTH_COLUMNS, c.SEARCH_SPACE),
        "SOURCE only": ([x for x in sets["BASELINE+SOURCE"] if x not in sets["BASELINE"]], c.LOGISTIC_SPACE),
        "SELF only": ([x for x in sets["BASELINE+SELF"] if x not in sets["BASELINE"]], c.LOGISTIC_SPACE),
        "BASELINE+SOURCE (tokens)": (sets["BASELINE"] + [n for n in token_names
                                                         if n.startswith("tok.source.")], c.SEARCH_SPACE),
        "BASELINE+SELF (tokens)": (sets["BASELINE"] + [n for n in token_names
                                                       if n.startswith("tok.self.")], c.SEARCH_SPACE),
    }
    strata = {part: tables[part].sample(sets["BASELINE"]).stratum for part in ("select", "test")}

    masks, arms = {}, {}
    for arm, by_alpha in record["coverage"].items():
        row = by_alpha[alpha]
        if row["accepted"] == 0:
            continue
        if arm in records:
            model = c.frozen_model(arm, records[arm]["model_sha256"])
            risk = {part: c.risk_scores(model, tables[part].sample(records[arm]["columns"]).X)
                    for part in ("select", "test")}
        else:
            columns, space = columns_of[arm]
            model = c.fit_extra(tables, tokens, columns, space)["model"]
            risk = {part: c.risk_scores(model, c.matrix(tables, tokens, part, columns))
                    for part in ("select", "test")}
        ranks = c.RankTransform.fit(risk["select"], strata["select"]).transform(risk["test"], strata["test"])
        masks[arm] = accepted_mask(ranks, strata["test"], row["thresholds"])
        if int(masks[arm].sum()) != row["accepted"]:
            raise SystemExit(f"{arm}: {int(masks[arm].sum())} outputs accepted here, "
                             f"{row['accepted']} in controls/result.json")
        arms[arm] = {"accepted": row["accepted"], "realised_risk_in_the_record": row["realised_risk"],
                     "thresholds": row["thresholds"]}

    result = {
        "status": "EXPLORATORY. No label is read; acceptance uses the thresholds recorded by the controls run.",
        "provenance": {"git_commit": c.git("rev-parse", "HEAD"),
                       "git_dirty": bool(c.git("status", "--porcelain", "--", ".")),
                       "thresholds_from": "controls/result.json",
                       "thresholds_computed_at_commit": record["provenance"]["git_commit"]},
        "alpha": float(alpha), "test_items": int(len(strata["test"])),
        "arms": arms, "pairs": acceptance_overlap(masks)}
    (HARNESS / "controls" / "acceptance_overlap.json").write_text(json.dumps(result, indent=1) + "\n")
    for pair in result["pairs"]:
        print(f"  {pair['first']:26s} {pair['accepted_first']:4d}   {pair['second']:26s} "
              f"{pair['accepted_second']:4d}   both {pair['accepted_both']:4d}")
    print("wrote controls/acceptance_overlap.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
