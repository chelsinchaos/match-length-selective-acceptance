# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LogNovations Holdings LLC
"""Phase 2 of the study on RAGTruth, one step per sub-command, in the brief's order.

    python scripts/phase2.py baseline   step 5 and 6: fit, select and freeze BASELINE; DEV rehearsal
    python scripts/phase2.py aa         step 7: 1,000 baseline-versus-baseline runs
    python scripts/phase2.py barfloor   step 8: c_B, FLOOR and BAR
    python scripts/phase2.py power      step 9: power by simulation at the CAL and TEST sizes
    python scripts/phase2.py profile    step 10: fit, select and freeze BASELINE plus each tested set

Each step reads only the open splits (TRAIN and the three DEV splits), writes
a small JSON record under prereg/ that is committed, and stamps it with the
git commit and a dirty flag. A step refuses to run before the one it depends
on. ``profile`` also refuses unless prereg/bar_floor.json is committed, so
that BAR and FLOOR are on record before any combiner sees a tested feature.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import platform
import subprocess
import sys
from importlib import metadata
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score

HARNESS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HARNESS / "src"))

from selfref.certify import RankTransform  # noqa: E402
from selfref.combiner import Config, fit_and_select, risk_scores  # noqa: E402
from selfref.decision import compute_bar, compute_floor  # noqa: E402
from selfref.experiment import Sample, aa_coverage_gap, bootstrap_gain, column_sets, rehearse  # noqa: E402
from selfref.power import simulate_power, strength_for_gain  # noqa: E402
from selfref.selcert_pin import PINNED_SHA, load_selcert  # noqa: E402
from selfref.tables import categories_of, load_ragtruth_split  # noqa: E402

ALPHAS = (0.01, 0.05, 0.10)
DELTA = 0.05
SEED = 20261111
OPEN_SPLITS = ("TRAIN", "DEV_SELECT", "DEV_CAL", "DEV_TEST")
DERIVED = HARNESS / "data" / "derived" / "ragtruth"
PREREG = HARNESS / "prereg"
AA_RUNS = 1000


# ------------------------------------------------------------------ plumbing

def git(*args) -> str:
    return subprocess.run(["git", "-C", str(HARNESS), *args], capture_output=True,
                          text=True, check=True).stdout.strip()


def provenance() -> dict:
    return {"git_commit": git("rev-parse", "HEAD"),
            "git_dirty": bool(git("status", "--porcelain", "--", ".")),
            "selcert_commit": PINNED_SHA,
            "python": platform.python_version(),
            "packages": {name: metadata.version(name)
                         for name in ("numpy", "scipy", "scikit-learn", "zstandard")},
            "seed": SEED, "delta": DELTA}


def write_record(name: str, payload: dict) -> None:
    PREREG.mkdir(exist_ok=True)
    payload = {"provenance": provenance(), **payload}
    (PREREG / name).write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote prereg/{name}")


def read_record(name: str) -> dict:
    path = PREREG / name
    if not path.exists():
        raise SystemExit(f"prereg/{name} is missing: run the earlier step first")
    return json.loads(path.read_text())


def require_committed(name: str) -> None:
    relative = f"prereg/{name}"
    tracked = subprocess.run(["git", "-C", str(HARNESS), "ls-files", "--error-unmatch", relative],
                             capture_output=True).returncode == 0
    if not tracked or git("status", "--porcelain", "--", relative):
        raise SystemExit(f"{relative} must be committed, unchanged, before this step runs")


def load_tables() -> dict:
    categories = categories_of(DERIVED)
    return {split: load_ragtruth_split(DERIVED, split, categories) for split in OPEN_SPLITS}


def arm(tables: dict, name: str):
    columns = column_sets(tables["TRAIN"].names)[name]
    return {split: table.sample(columns) for split, table in tables.items()}, columns


def save_model(model, name: str) -> str:
    (DERIVED / "models").mkdir(exist_ok=True)
    blob = pickle.dumps(model)
    (DERIVED / "models" / f"{name}.pkl").write_bytes(blob)
    return hashlib.sha256(blob).hexdigest()


def load_model(name: str):
    return pickle.loads((DERIVED / "models" / f"{name}.pkl").read_bytes())


def config_from(record: dict) -> Config:
    params = {k: v for k, v in record.items() if k != "kind"}
    return Config(record["kind"], tuple(sorted(params.items())))


def fit_arm(tables: dict, name: str) -> tuple[dict, dict, dict]:
    """Fit, select and freeze one arm, and rehearse the confirmatory procedure in DEV.

    Returns the record, the arm's samples by split and its risk scores by split."""
    data, columns = arm(tables, name)
    selection = fit_and_select(data["TRAIN"].X, data["TRAIN"].loss,
                               data["DEV_SELECT"].X, data["DEV_SELECT"].loss, SEED)
    risk = {split: risk_scores(selection.model, data[split].X) for split in OPEN_SPLITS}
    rehearsal = rehearse(load_selcert(), risk, data, ALPHAS, DELTA)
    return {
        "arm": name, "columns": columns,
        "selected": selection.config.as_dict(),
        "search": [{"config": row["config"].as_dict(), "log_loss_dev_select": row["log_loss"]}
                   for row in selection.table],
        "model_sha256": save_model(selection.model, name.replace("+", "_plus_")),
        "auroc": {split: float(roc_auc_score(data[split].loss, risk[split]))
                  for split in ("DEV_SELECT", "DEV_CAL", "DEV_TEST")},
        "dev_rehearsal": {str(alpha): result for alpha, result in rehearsal.items()},
        "sizes": {split: {"items": int(len(data[split].loss)),
                          "groups": int(len(set(data[split].group))),
                          "loss_rate": float(data[split].loss.mean())} for split in OPEN_SPLITS},
    }, data, risk


def choose_primary_alpha(rehearsal: dict):
    """The rule, fixed before any rehearsal was run: the strictest alpha at which
    BASELINE certifies at least 5 points of coverage in the DEV rehearsal; failing
    that, the loosest alpha at which it certifies anything; failing that, none."""
    for alpha in sorted(ALPHAS):
        if rehearsal[str(alpha)]["coverage_points"] >= 5.0:
            return alpha, "strictest alpha with at least 5 points of BASELINE coverage"
    for alpha in sorted(ALPHAS, reverse=True):
        if rehearsal[str(alpha)]["coverage_points"] > 0.0:
            return alpha, "no alpha reaches 5 points; loosest alpha with any BASELINE coverage"
    return None, "BASELINE certifies no coverage at any alpha"


# --------------------------------------------------------------------- steps

def step_baseline(args) -> int:
    tables = load_tables()
    record, data, _ = fit_arm(tables, "BASELINE")
    names = tables["DEV_SELECT"].names
    single = {}
    for name in sorted(column_sets(names)["BASELINE"] + ["base.random"]):
        if name.startswith("cov."):
            continue
        values = tables["DEV_SELECT"].X[:, names.index(name)]
        single[name] = float(roc_auc_score(tables["DEV_SELECT"].loss, values))
    record["single_score_auroc_dev_select"] = single
    alpha, reason = choose_primary_alpha(record["dev_rehearsal"])
    record["primary_alpha"] = alpha
    record["primary_alpha_reason"] = reason
    write_record("baseline.json", record)
    print(json.dumps({"selected": record["selected"], "auroc": record["auroc"],
                      "coverage_points": {a: r["coverage_points"]
                                          for a, r in record["dev_rehearsal"].items()},
                      "realised_risk": {a: r["realised_risk"]
                                        for a, r in record["dev_rehearsal"].items()},
                      "primary_alpha": alpha, "reason": reason}, indent=2))
    if alpha is None:
        print("\nGATE: BASELINE certifies no coverage in the DEV rehearsal at any alpha.")
    return 0


def _aa_run(run: int, config_record: dict, alpha: float) -> float:
    if "tables" not in _aa_run.__dict__:                        # once per worker process
        _aa_run.tables = arm(load_tables(), "BASELINE")[0]
        _aa_run.selcert = load_selcert()
    return aa_coverage_gap(_aa_run.selcert, config_from(config_record), _aa_run.tables,
                           alpha, DELTA, run, SEED)


def step_aa(args) -> int:
    baseline = read_record("baseline.json")
    alpha = baseline["primary_alpha"]
    if alpha is None:
        raise SystemExit("no primary alpha: the baseline gate has not been passed")
    gaps = Parallel(n_jobs=args.jobs)(delayed(_aa_run)(run, baseline["selected"], alpha)
                                      for run in range(AA_RUNS))
    gaps = np.array(gaps)
    absolute = np.abs(gaps)
    write_record("aa.json", {
        "runs": AA_RUNS, "alpha": alpha, "config": baseline["selected"],
        "gap_points": [float(g) for g in gaps],             # unrounded: FLOOR is computed from these
        "summary": {"mean": float(gaps.mean()), "sd": float(gaps.std(ddof=1)),
                    "abs_p50": float(np.percentile(absolute, 50)),
                    "abs_p90": float(np.percentile(absolute, 90)),
                    "abs_p95": float(np.percentile(absolute, 95)),
                    "abs_p99": float(np.percentile(absolute, 99)),
                    "abs_max": float(absolute.max()),
                    "share_zero": float((gaps == 0).mean())}})
    print(json.dumps(json.loads((PREREG / "aa.json").read_text())["summary"], indent=2))
    return 0


def step_barfloor(args) -> int:
    baseline, aa = read_record("baseline.json"), read_record("aa.json")
    alpha = baseline["primary_alpha"]
    c_b = baseline["dev_rehearsal"][str(alpha)]["coverage_points"]
    floor = compute_floor(aa["gap_points"])
    bar = compute_bar(c_b, floor)
    write_record("bar_floor.json", {
        "primary_alpha": alpha, "delta": DELTA, "c_B_points": c_b,
        "aa_abs_gap_p99_points": aa["summary"]["abs_p99"],
        "FLOOR_points": floor, "BAR_points": bar,
        "rule": "BAR = min(5, 0.1 * (100 - c_B)); FLOOR = max(2, p99 |A/A gap|); "
                "BAR raised to FLOOR if below it"})
    print(json.dumps({"primary_alpha": alpha, "c_B": c_b, "FLOOR": floor, "BAR": bar}, indent=2))
    return 0


def _block_counts(split: str) -> dict:
    counts: dict[str, set] = {}
    for line in open(DERIVED / "items.jsonl", encoding="utf-8"):
        item = json.loads(line)
        if item["split"] == split:
            counts.setdefault(item["task_type"], set()).add(item["group"])
    return {block: len(groups) for block, groups in counts.items()}


def _power_at(multiple: float, bar: float, floor: float, alpha: float, search_reps: int,
              reps: int, boots: int) -> dict:
    data, _ = arm(load_tables(), "BASELINE")
    model = load_model("BASELINE")
    transform = RankTransform.fit(risk_scores(model, data["DEV_SELECT"].X),
                                  data["DEV_SELECT"].stratum)
    parts = [data["DEV_CAL"], data["DEV_TEST"]]
    pool = Sample(*(np.concatenate([getattr(p, field) for p in parts])
                    for field in ("X", "loss", "stratum", "group", "block")))
    ranks = transform.transform(risk_scores(model, pool.X), pool.stratum)
    selcert = load_selcert()
    counts = (_block_counts("CAL"), _block_counts("TEST"))
    target = multiple * bar
    found = strength_for_gain(selcert, pool, ranks, *counts, target=target, alpha=alpha,
                              delta=DELTA, n_rep=search_reps, seed=SEED, tolerance=0.1)
    result = simulate_power(selcert, pool, ranks, *counts, strength=found["strength"],
                            alpha=alpha, delta=DELTA, bar=bar, floor=floor, n_rep=reps,
                            n_boot=boots, seed=SEED + 1)
    return {"target_gain_points": target, "multiple_of_BAR": multiple, "search": found, **result}


def _print_power(results) -> None:
    for r in results:
        print(f"gain {r['mean_gain']:6.2f} (target {r['target_gain_points']:.2f}, "
              f"reachable {r['search']['reachable']}): interval above FLOOR "
              f"{r['p_interval_above_floor']:.2f} | PASS {r['p_pass']:.2f} "
              f"INCONCLUSIVE {r['p_inconclusive']:.2f} FAIL {r['p_fail']:.2f}")


def step_power(args) -> int:
    limits = read_record("bar_floor.json")
    alpha, bar, floor = limits["primary_alpha"], limits["BAR_points"], limits["FLOOR_points"]
    multiples = tuple(args.multiples)
    results = Parallel(n_jobs=min(args.jobs, len(multiples)))(
        delayed(_power_at)(m, bar, floor, alpha, args.search_reps, args.reps, args.boots)
        for m in multiples)
    if args.out != "power.json":
        # A supplement at other effect sizes: the operating curve, not the gate.
        write_record(args.out, {
            "alpha": alpha, "BAR_points": bar, "FLOOR_points": floor,
            "cal_groups_by_task": _block_counts("CAL"), "test_groups_by_task": _block_counts("TEST"),
            "population": "DEV_CAL and DEV_TEST, BASELINE ranks and losses",
            "note": "supplement to power.json at other effect sizes; the gate is read from power.json",
            "results": results})
        _print_power(results)
        return 0
    if multiples[0] != 1.0:
        raise SystemExit("power.json is the gate record: its first target must be BAR itself")
    at_bar = results[0]
    write_record("power.json", {
        "alpha": alpha, "BAR_points": bar, "FLOOR_points": floor,
        "cal_groups_by_task": _block_counts("CAL"), "test_groups_by_task": _block_counts("TEST"),
        "population": "DEV_CAL and DEV_TEST, BASELINE ranks and losses",
        "results": results,
        "power_to_detect_BAR": at_bar["p_interval_above_floor"],
        "meets_80_percent": bool(at_bar["p_interval_above_floor"] >= 0.80)})
    _print_power(results)
    if not at_bar["p_interval_above_floor"] >= 0.80:
        print("\nGATE: the design does not detect a gain the size of BAR with 80% power.")
    return 0


def step_profile(args) -> int:
    require_committed("bar_floor.json")
    limits, tables = read_record("bar_floor.json"), load_tables()
    alpha = limits["primary_alpha"]
    base_data, _ = arm(tables, "BASELINE")
    base_model = load_model("BASELINE")
    base_risk = {s: risk_scores(base_model, base_data[s].X) for s in OPEN_SPLITS}
    base_transform = RankTransform.fit(base_risk["DEV_SELECT"], base_data["DEV_SELECT"].stratum)
    selcert = load_selcert()
    arms = {}
    for name in ("BASELINE+SOURCE", "BASELINE+SELF"):
        record, data, risk = fit_arm(tables, name)
        transform = RankTransform.fit(risk["DEV_SELECT"], data["DEV_SELECT"].stratum)
        ranks = lambda t, r, d, split: t.transform(r[split], d[split].stratum)
        gain = bootstrap_gain(
            selcert,
            (ranks(base_transform, base_risk, base_data, "DEV_CAL"), ranks(transform, risk, data, "DEV_CAL")),
            (ranks(base_transform, base_risk, base_data, "DEV_TEST"), ranks(transform, risk, data, "DEV_TEST")),
            data["DEV_CAL"], data["DEV_TEST"], alpha, DELTA, n_boot=args.boots, seed=SEED + 2)
        record["dev_rehearsal_gain_over_baseline"] = {
            "alpha": alpha, "point": gain["point"], "low": gain["low"], "high": gain["high"],
            "bootstrap_draws": args.boots,
            "note": "DEV rehearsal only. Not a confirmatory result."}
        arms[name] = record
        print(name, json.dumps({"selected": record["selected"], "auroc": record["auroc"],
                                "dev_gain": record["dev_rehearsal_gain_over_baseline"]}, indent=2))
    write_record("profile.json", {"arms": arms})
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="step", required=True)
    sub.add_parser("baseline").set_defaults(run=step_baseline)
    aa = sub.add_parser("aa"); aa.set_defaults(run=step_aa); aa.add_argument("--jobs", type=int, default=8)
    sub.add_parser("barfloor").set_defaults(run=step_barfloor)
    power = sub.add_parser("power"); power.set_defaults(run=step_power)
    power.add_argument("--jobs", type=int, default=4)
    power.add_argument("--search-reps", type=int, default=200)
    power.add_argument("--reps", type=int, default=300)
    power.add_argument("--boots", type=int, default=400)
    power.add_argument("--multiples", type=float, nargs="+", default=[1.0, 1.5, 2.0, 3.0],
                       help="target gains as multiples of BAR")
    power.add_argument("--out", default="power.json",
                       help="record to write; anything but power.json is a supplement")
    profile = sub.add_parser("profile"); profile.set_defaults(run=step_profile)
    profile.add_argument("--boots", type=int, default=2000)
    args = parser.parse_args(argv)
    return args.run(args)


if __name__ == "__main__":
    raise SystemExit(main())
