# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LogNovations Holdings LLC
"""Controls, robustness and secondary metrics on RAGTruth (pre-registration, Sections 14 and 15).

    python scripts/controls.py rehearsal
    python scripts/controls.py confirmatory --i-have-prereg-approval --purpose "controls and secondary metrics"

Both modes run the same analyses on the frozen models. They differ only in the
two samples: the rehearsal calibrates on DEV_CAL and tests on DEV_TEST, with
open labels, and writes controls/rehearsal.json; the confirmatory mode
calibrates on CAL and tests on TEST, through the guard, and writes
controls/result.json. It is run after the confirmatory result is committed.

Everything that needs no sealed label is done before the guard is asked: the
label-free inputs are checked, and the models the controls need beyond the
three frozen arms are fitted on TRAIN and selected on DEV_SELECT. Those are

    LENGTH              the three output-length columns, whole search space
    SOURCE only         the 16 SOURCE profile columns, logistic part of the space
    SELF only           the 16 SELF profile columns, logistic part of the space
    BASELINE+SOURCE (tokens), BASELINE+SELF (tokens)
                        BASELINE with the profile columns recomputed over word
                        tokens, whole search space

all with the seed of the frozen arms. The paraphrase stress analysis has its
own script.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HARNESS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HARNESS / "src"))

from selfref import controls  # noqa: E402
from selfref.certify import RankTransform  # noqa: E402
from selfref.combiner import LOGISTIC_SPACE, SEARCH_SPACE, fit_and_select, risk_scores  # noqa: E402
from selfref.confirm import analyse  # noqa: E402
from selfref.datasets.build import sealed_ragtruth_labels  # noqa: E402
from selfref.experiment import Sample, column_sets  # noqa: E402
from selfref.guard import access_count, authorise_all  # noqa: E402
from selfref.selcert_pin import PINNED_SHA, load_selcert  # noqa: E402
from selfref.tables import categories_of, extra_columns, load_ragtruth_split  # noqa: E402

DERIVED = HARNESS / "data" / "derived" / "ragtruth"
RAW = HARNESS / "data" / "raw" / "ragtruth"
PREREG = HARNESS / "prereg"
LEDGER = HARNESS / "RUNLEDGER.md"
ALPHAS = (0.01, 0.05, 0.10)
FIT_SEED = 20261111                 # the seed of the frozen arms
SEEDS = {"confirmatory_bootstrap": 20261121, "ranking_bootstrap": 20261131,
         "source_derangement": 20261141, "label_permutation": 20261142, "resplit": 20261143,
         "extra_calibration": 20261144, "synthetic": 20261145}
MULTIPLES = (1.25, 1.5, 2.0, 3.0, 4.0)
FROZEN = {"BASELINE": "BASELINE", "BASELINE+SOURCE": "BASELINE_plus_SOURCE",
          "BASELINE+SELF": "BASELINE_plus_SELF"}
LENGTH_COLUMNS = ["base.len_bytes", "base.len_words", "base.len_sentences"]


def git(*args) -> str:
    return subprocess.run(["git", "-C", str(HARNESS), *args], capture_output=True,
                          text=True).stdout.strip()


def frozen_model(arm: str, expected_sha: str):
    blob = (DERIVED / "models" / f"{FROZEN[arm]}.pkl").read_bytes()
    if hashlib.sha256(blob).hexdigest() != expected_sha:
        raise SystemExit(f"the frozen model for {arm} does not match its recorded SHA-256")
    return pickle.loads(blob)


def require_label_free_inputs(splits) -> None:
    scored = {json.loads(line)["id"] for line in open(DERIVED / "groundedness.jsonl")}
    for split in splits:
        for stem in ("features", "features_tokens"):
            if not (DERIVED / f"{stem}_{split}.npz").exists():
                raise SystemExit(f"{stem}_{split}.npz is missing")
        ids = {str(i) for i in np.load(DERIVED / f"features_{split}.npz")["ids"]}
        if ids - scored:
            raise SystemExit(f"{len(ids - scored)} {split} items have no groundedness score")


def matrix(tables: dict, tokens: dict, part: str, columns) -> np.ndarray:
    """The named columns of one part, taken from the main table or the token table."""
    table, (token_names, token_X) = tables[part], tokens[part]
    main = {name: k for k, name in enumerate(table.names)}
    extra = {name: k for k, name in enumerate(token_names)}
    return np.column_stack([table.X[:, main[c]] if c in main else token_X[:, extra[c]]
                            for c in columns])


def fit_extra(tables, tokens, columns, space) -> dict:
    """Fit on TRAIN, select on DEV_SELECT by log loss, with the seed of the frozen arms."""
    selection = fit_and_select(matrix(tables, tokens, "TRAIN", columns), tables["TRAIN"].loss,
                               matrix(tables, tokens, "select", columns), tables["select"].loss,
                               FIT_SEED, space=space)
    return {"model": selection.model, "columns": list(columns),
            "selected": selection.config.as_dict(), "configurations_searched": len(space),
            "log_loss_dev_select": selection.table[0]["log_loss"]}


def ranks_of(risk: dict, data: dict) -> tuple:
    transform = RankTransform.fit(risk["select"], data["select"].stratum)
    return (transform.transform(risk["cal"], data["cal"].stratum),
            transform.transform(risk["test"], data["test"].stratum))


def strip(hypotheses: dict) -> dict:
    return {name: {k: v for k, v in h.items() if k != "draws"} for name, h in hypotheses.items()}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("mode", choices=["rehearsal", "confirmatory"])
    parser.add_argument("--i-have-prereg-approval", action="store_true", dest="flag")
    parser.add_argument("--purpose", default="")
    parser.add_argument("--boots", type=int, default=10_000)
    parser.add_argument("--repeats", type=int, default=1_000,
                        help="permutations, re-splits and calibration resamples")
    parser.add_argument("--synthetic-draws", type=int, default=10_000)
    args = parser.parse_args(argv)
    started = time.perf_counter()
    sealed = args.mode == "confirmatory"
    cal_split, test_split = ("CAL", "TEST") if sealed else ("DEV_CAL", "DEV_TEST")

    limits = json.loads((PREREG / "bar_floor.json").read_text())
    alpha, delta = limits["primary_alpha"], limits["delta"]
    records = {"BASELINE": json.loads((PREREG / "baseline.json").read_text())}
    records.update(json.loads((PREREG / "profile.json").read_text())["arms"])
    selcert = load_selcert()
    categories = categories_of(DERIVED)

    # ---- everything that needs no sealed label --------------------------------------------
    require_label_free_inputs(("TRAIN", "DEV_SELECT", cal_split, test_split))
    if not (DERIVED / f"features_shuffled_source_{test_split}.npz").exists():
        raise SystemExit(f"features_shuffled_source_{test_split}.npz is missing")
    models = {arm: frozen_model(arm, record["model_sha256"]) for arm, record in records.items()}
    tables = {"TRAIN": load_ragtruth_split(DERIVED, "TRAIN", categories),
              "select": load_ragtruth_split(DERIVED, "DEV_SELECT", categories)}
    tokens = {"TRAIN": extra_columns(DERIVED, "tokens", "TRAIN", tables["TRAIN"].ids),
              "select": extra_columns(DERIVED, "tokens", "DEV_SELECT", tables["select"].ids)}
    sets = column_sets(tables["TRAIN"].names)
    source_columns = [c for c in sets["BASELINE+SOURCE"] if c not in sets["BASELINE"]]
    self_columns = [c for c in sets["BASELINE+SELF"] if c not in sets["BASELINE"]]
    token_names = tokens["TRAIN"][0]
    extra = {
        "LENGTH": fit_extra(tables, tokens, LENGTH_COLUMNS, SEARCH_SPACE),
        "SOURCE only": fit_extra(tables, tokens, source_columns, LOGISTIC_SPACE),
        "SELF only": fit_extra(tables, tokens, self_columns, LOGISTIC_SPACE),
        "BASELINE+SOURCE (tokens)": fit_extra(
            tables, tokens, sets["BASELINE"] + [n for n in token_names if n.startswith("tok.source.")],
            SEARCH_SPACE),
        "BASELINE+SELF (tokens)": fit_extra(
            tables, tokens, sets["BASELINE"] + [n for n in token_names if n.startswith("tok.self.")],
            SEARCH_SPACE),
    }
    print(f"extra models fitted on the open splits ({time.perf_counter() - started:.0f}s)", flush=True)

    # ---- the two samples ------------------------------------------------------------------
    if sealed:
        entries = authorise_all(("CAL", "TEST"), args.purpose, flag=args.flag, repo=HARNESS,
                                ledger=LEDGER)
        by_split = {entry["split"]: entry for entry in entries}
        raw = (RAW / "response.jsonl", RAW / "source_info.jsonl")
        labels = {"cal": sealed_ragtruth_labels(*raw, "CAL", by_split["CAL"]),
                  "test": sealed_ragtruth_labels(*raw, "TEST", by_split["TEST"])}
        tables["cal"] = load_ragtruth_split(DERIVED, "CAL", categories, sealed_labels=labels["cal"])
        tables["test"] = load_ragtruth_split(DERIVED, "TEST", categories, sealed_labels=labels["test"])
        out_path = HARNESS / "controls" / "result.json"
    else:
        entries = []
        open_labels = json.loads((DERIVED / "labels_open.json").read_text())
        tables["cal"] = load_ragtruth_split(DERIVED, "DEV_CAL", categories)
        tables["test"] = load_ragtruth_split(DERIVED, "DEV_TEST", categories)
        labels = {part: {i: open_labels[i] for i in tables[part].ids} for part in ("cal", "test")}
        out_path = HARNESS / "controls" / "rehearsal.json"
    tokens["cal"] = extra_columns(DERIVED, "tokens", cal_split, tables["cal"].ids)
    tokens["test"] = extra_columns(DERIVED, "tokens", test_split, tables["test"].ids)
    parts = ("select", "cal", "test")
    data = {part: tables[part].sample(sets["BASELINE"]) for part in parts}   # loss, stratum, group, block
    cal, test = data["cal"], data["test"]

    # ---- risk scores and ranks of every arm ------------------------------------------------
    risk = {arm: {part: risk_scores(models[arm], tables[part].sample(records[arm]["columns"]).X)
                  for part in parts} for arm in records}
    for name, fitted in extra.items():
        risk[name] = {part: risk_scores(fitted["model"], matrix(tables, tokens, part, fitted["columns"]))
                      for part in parts}
    random_column = tables["test"].names.index("base.random")
    risk["RANDOM"] = {part: tables[part].X[:, random_column] for part in parts}
    ranks = {arm: ranks_of(scores, data) for arm, scores in risk.items()}
    frozen_arms = list(records)
    result: dict = {}

    # ---- coverage of every arm at every alpha ----------------------------------------------
    result["coverage"] = {arm: {str(a): controls.coverage_row(selcert, c, cal, t, test, a, delta)
                                for a in ALPHAS} for arm, (c, t) in ranks.items()}

    # ---- negative controls -----------------------------------------------------------------
    names, shuffled = extra_columns(DERIVED, "shuffled_source", test_split, tables["test"].ids)
    shuffled = shuffled[:, [names.index(c) for c in source_columns]]
    shuffled_risk = risk_scores(extra["SOURCE only"]["model"], shuffled)
    result["negative_controls"] = {
        "permuted_labels": controls.permuted_label_control(
            selcert, {arm: ranks[arm] for arm in frozen_arms + ["RANDOM"]}, cal, test, alpha,
            delta, n_perm=args.repeats, seed=SEEDS["label_permutation"]),
        "random_score": result["coverage"]["RANDOM"],
        "shuffled_source": {
            "model": {k: extra["SOURCE only"][k] for k in ("columns", "selected", "log_loss_dev_select")},
            "true_source": {"auroc": controls.auroc(test.loss, risk["SOURCE only"]["test"]),
                            "auprc": controls.average_precision(test.loss, risk["SOURCE only"]["test"])},
            "shuffled_source": {"auroc": controls.auroc(test.loss, shuffled_risk),
                                "auprc": controls.average_precision(test.loss, shuffled_risk)},
            "length_only_for_reference": {"auroc": controls.auroc(test.loss, risk["LENGTH"]["test"])}}}

    # ---- length confound -------------------------------------------------------------------
    length = lambda part: tables[part].X[:, tables[part].names.index("base.len_bytes")]   # log(1 + bytes)
    edges = [float(e) for e in np.percentile(length("select"), [20, 40, 60, 80])]
    result["length"] = {
        "length_only": {"model": {k: extra["LENGTH"][k] for k in ("columns", "selected", "log_loss_dev_select")},
                        "coverage": result["coverage"]["LENGTH"],
                        "auroc": controls.auroc(test.loss, risk["LENGTH"]["test"])},
        "quintile_edges_bytes": [float(np.expm1(e)) for e in edges],
        "within_quintiles": {arm: controls.auroc_by_bin(test.loss, risk[arm]["test"], length("test"), edges)
                             for arm in ["SOURCE only", "SELF only"] + frozen_arms},
        "tested_columns_alone": {arm: {"model": {k: extra[arm][k] for k in ("selected", "log_loss_dev_select")},
                                       "auroc": controls.auroc(test.loss, risk[arm]["test"])}
                                 for arm in ("SOURCE only", "SELF only")}}

    # ---- guarantee checks ------------------------------------------------------------------
    pool = Sample(*(np.concatenate([getattr(cal, f), getattr(test, f)])
                    for f in ("X", "loss", "stratum", "group", "block")))
    cal_counts = {str(b): int(np.unique(cal.group[cal.block == b]).size) for b in np.unique(cal.block)}
    result["guarantee"] = {
        "synthetic": controls.synthetic_guarantee(selcert, args.synthetic_draws, n_per_stratum=504,
                                                  alpha=alpha, delta=delta, seed=SEEDS["synthetic"]),
        "resplit": controls.resplit_guarantee(selcert, np.concatenate(ranks["BASELINE"]), pool,
                                              cal_counts, alpha, delta, n_resplits=args.repeats,
                                              seed=SEEDS["resplit"])}
    print(f"negative controls, length and guarantee checks done ({time.perf_counter() - started:.0f}s)",
          flush=True)

    # ---- sensitivity to the loss definition (exploratory) ----------------------------------
    frozen_risk = {arm: risk[arm] for arm in frozen_arms}
    run = lambda d, r: analyse(selcert, r, d, primary_alpha=alpha, alphas=ALPHAS, delta=delta,
                               bar=limits["BAR_points"], floor=limits["FLOOR_points"],
                               n_boot=args.boots, seed=SEEDS["confirmatory_bootstrap"])
    sensitivity = {}
    relabelled = dict(data)
    changed = {}
    for part in ("cal", "test"):
        loss = np.array([int(labels[part][i]["n_spans_not_implicit_true"] > 0) for i in tables[part].ids])
        changed[part] = int((loss != data[part].loss).sum())
        relabelled[part] = Sample(data[part].X, loss, data[part].stratum, data[part].group, data[part].block)
    outcome = run(relabelled, frozen_risk)
    sensitivity["implicit_true_spans_ignored"] = {
        "labels_changed_from_error_to_no_error": changed, "coverage": outcome["coverage"],
        "hypotheses": strip(outcome["hypotheses"])}
    kept = {part: np.array([labels[part][i]["quality"] == "good" for i in tables[part].ids])
            for part in ("cal", "test")}
    subset = {"select": data["select"], "cal": cal.rows(kept["cal"]), "test": test.rows(kept["test"])}
    subset_risk = {arm: {"select": r["select"], "cal": r["cal"][kept["cal"]], "test": r["test"][kept["test"]]}
                   for arm, r in frozen_risk.items()}
    outcome = run(subset, subset_risk)
    sensitivity["flagged_responses_dropped"] = {
        "responses_dropped": {part: int((~kept[part]).sum()) for part in kept},
        "quality_values": {part: {str(q): int(n) for q, n in zip(*np.unique(
            [str(labels[part][i]["quality"]) for i in tables[part].ids], return_counts=True))}
            for part in kept},
        "coverage": outcome["coverage"], "hypotheses": strip(outcome["hypotheses"]),
        "sizes": outcome["sizes"]}
    result["loss_sensitivity"] = sensitivity
    print(f"loss sensitivity done ({time.perf_counter() - started:.0f}s)", flush=True)

    # ---- cheap-alternative reference -------------------------------------------------------
    resampled = controls.extra_calibration(selcert, ranks["BASELINE"][0], cal, ranks["BASELINE"][1],
                                           test, alpha, delta, multiples=MULTIPLES + (1.0,),
                                           n_draws=args.repeats, seed=SEEDS["extra_calibration"])
    # The row at 1.0 is a reference: the same resampling at the actual size. It shows how far
    # resampling alone moves the mean, and it is not a candidate for the smallest multiple.
    base_cov = result["coverage"]["BASELINE"][str(alpha)]["coverage_points"]
    source_cov = result["coverage"]["BASELINE+SOURCE"][str(alpha)]["coverage_points"]
    reaching = [m for m in MULTIPLES if resampled[str(m)]["mean_coverage_points"] >= source_cov]
    if source_cov <= base_cov:
        smallest, note = 1.0, "BASELINE+SOURCE does not exceed BASELINE at the actual CAL size"
    elif reaching:
        smallest, note = min(reaching), "smallest multiple whose mean coverage reaches BASELINE+SOURCE"
    else:
        smallest, note = None, "no multiple up to 4 reaches BASELINE+SOURCE"
    result["cheap_alternative"] = {
        "baseline_coverage_points": base_cov, "baseline_plus_source_coverage_points": source_cov,
        "baseline_with_resampled_calibration": resampled, "smallest_multiple": smallest,
        "extra_calibration_labels": None if smallest is None else int(round((smallest - 1.0) * cal.loss.size)),
        "note": note}

    # ---- stress ----------------------------------------------------------------------------
    masks = {arm: controls.accept(selcert, ranks[arm][0], cal, ranks[arm][1], test, alpha, delta)[0]
             for arm in frozen_arms}
    result["stress"] = {
        "per_task": {arm: controls.breakdown(masks[arm], test.loss, test.block) for arm in frozen_arms},
        "per_generator": {arm: controls.breakdown(masks[arm], test.loss, test.stratum) for arm in frozen_arms},
        "across_task_types": {arm: controls.cross_block(selcert, ranks[arm][0], cal, ranks[arm][1],
                                                        test, alpha, delta) for arm in frozen_arms},
        "token_alphabet": {
            name: {"model": {k: extra[name][k] for k in ("selected", "log_loss_dev_select")},
                   "auroc": controls.auroc(test.loss, risk[name]["test"]),
                   "coverage": result["coverage"][name],
                   "byte_arm_coverage": result["coverage"][name.replace(" (tokens)", "")]}
            for name in ("BASELINE+SOURCE (tokens)", "BASELINE+SELF (tokens)")}}

    # ---- secondary metrics -----------------------------------------------------------------
    single = {}
    for k, name in enumerate(tables["test"].names):
        if not name.startswith("cov."):
            values = tables["test"].X[:, k]
            single[name] = {"auroc": controls.auroc(test.loss, values),
                            "auprc": controls.average_precision(test.loss, values)}
    result["secondary"] = {
        "single_scores": single,
        "arms": {arm: {"auroc": controls.auroc(test.loss, scores["test"]),
                       "auprc": controls.average_precision(test.loss, scores["test"])}
                 for arm, scores in risk.items()},
        "paired_difference_from_baseline": {
            arm: controls.paired_metric_difference(test, risk["BASELINE"]["test"], risk[arm]["test"],
                                                   n_boot=args.boots, seed=SEEDS["ranking_bootstrap"])
            for arm in frozen_arms if arm != "BASELINE"},
        "risk_coverage": {arm: {"all": controls.risk_coverage_curve(test.loss, risk[arm]["test"]),
                                **{str(g): controls.risk_coverage_curve(
                                    test.loss[test.stratum == g], risk[arm]["test"][test.stratum == g])
                                   for g in np.unique(test.stratum)}}
                          for arm in frozen_arms}}

    record = {"mode": args.mode, "calibration_split": cal_split, "test_split": test_split,
              "provenance": {"git_commit": git("rev-parse", "HEAD"),
                             "git_dirty_before_run": bool(entries == [] and git("status", "--porcelain", "--", ".")),
                             "selcert_commit": PINNED_SHA, "seeds": {"fit": FIT_SEED, **SEEDS},
                             "bootstrap_draws": args.boots, "repeats": args.repeats,
                             "ledger_entries": entries,
                             "test_accesses_so_far": access_count(LEDGER, "TEST"),
                             "cal_accesses_so_far": access_count(LEDGER, "CAL")},
              "primary_alpha": alpha, "delta": delta,
              "sizes": {part: {"items": int(data[part].loss.size),
                               "groups": int(np.unique(data[part].group).size),
                               "error_rate": float(data[part].loss.mean())} for part in parts},
              **result}
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(record, indent=1) + "\n")
    print(f"wrote {out_path.relative_to(HARNESS)} ({time.perf_counter() - started:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
