# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LogNovations Holdings LLC
"""Write REPORT.md and its figures from results.json.

    python scripts/make_report.py

REPORT.md is report/REPORT.template.md with every placeholder replaced from
results.json (see ``selfref.render``). The tables are built here, from
results.json alone. The figures are drawn here too, into report/figures/.
Nothing in the report is typed in by hand as a number: a test checks that the
committed REPORT.md is exactly this rendering.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HARNESS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HARNESS / "src"))

from selfref.render import DASH, md_table, render, show  # noqa: E402

CONFIRM, CONTROLS = "confirmatory/result.json", "controls/result.json"
ARMS = ("BASELINE", "BASELINE+SOURCE", "BASELINE+SELF")
SPLITS = ("TRAIN", "DEV_SELECT", "DEV_CAL", "DEV_TEST", "CAL", "TEST")
SHORT = {"gpt-3.5-turbo-0613": "GPT-3.5", "gpt-4-0613": "GPT-4", "llama-2-13b-chat": "Llama-2-13B",
         "llama-2-70b-chat": "Llama-2-70B", "llama-2-7b-chat": "Llama-2-7B",
         "mistral-7B-instruct": "Mistral-7B"}


def rec(results, name):
    return results["records"][name]


def f(value, spec=".2f"):
    return show(value, spec)


def interval(low, high, spec="+.2f"):
    return f"{f(low, spec)} to {f(high, spec)}"


def certified(thresholds: dict) -> str:
    kept = [f"{SHORT.get(g, g)} at {t:.2f}" for g, t in sorted(thresholds.items()) if t is not None]
    return ", ".join(kept) if kept else "none"


def config(selected: dict) -> str:
    if selected["kind"] == "logistic":
        return f"logistic, C = {selected['C']:g}"
    return (f"boosted trees, depth {selected['max_depth']}, rate {selected['learning_rate']:g}, "
            f"{selected['max_iter']} iterations, leaf {selected['min_samples_leaf']}")


# ------------------------------------------------------------------ data and design

def t_splits(results):
    rows = []
    for name, label in (("ragtruth", "RAGTruth"), ("wikibio", "WikiBio GPT-3")):
        report = rec(results, f"data/derived/{name}/build_report.json")
        sealed_rates = {}
        if name == "ragtruth" and CONTROLS in results["records"]:
            sizes = rec(results, CONTROLS)["sizes"]
            sealed_rates = {"CAL": sizes["cal"]["error_rate"], "TEST": sizes["test"]["error_rate"]}
        if name == "wikibio" and "controls/wikibio_result.json" in results["records"]:
            sealed_rates = {"TEST": rec(results, "controls/wikibio_result.json")["sizes"]["test_error_rate"]}
        for split in SPLITS:
            rate = report["loss_rate_per_open_split"].get(split, sealed_rates.get(split))
            rows.append([label, split, f"{report['groups_per_split'][split]:,}",
                         f"{report['items_per_split'][split]:,}",
                         "not read" if rate is None else f(100 * rate, ".1f") + "%"])
    return md_table(["Dataset", "Split", "Groups", "Items", "Error rate"], rows)


def t_arms(results):
    records = {"BASELINE": rec(results, "prereg/baseline.json"), **rec(results, "prereg/profile.json")["arms"]}
    rows = []
    for arm, r in records.items():
        best = min(row["log_loss_dev_select"] for row in r["search"])
        rows.append([arm, len(r["columns"]), config(r["selected"]), f(best, ".4f"),
                     " / ".join(f(r["auroc"][s], ".3f") for s in ("DEV_SELECT", "DEV_CAL", "DEV_TEST")),
                     f"`{r['model_sha256'][:12]}…`"])
    return md_table(["Arm", "Columns", "Selected configuration", "Log loss, DEV_SELECT",
                     "AUROC, DEV_SELECT / DEV_CAL / DEV_TEST", "SHA-256 of the frozen model"], rows)


def t_dev_baseline(results):
    rehearsal = rec(results, "prereg/baseline.json")["dev_rehearsal"]
    return md_table(["alpha", "Certified coverage, points", "Accepted of 1,410", "Realised risk",
                     "Certified generators"],
                    [[a, f(r["coverage_points"]), r["accepted"], f(r["realised_risk"], ".3f"),
                      certified(r["thresholds"])] for a, r in rehearsal.items()])


def t_aa(results):
    s = rec(results, "prereg/aa.json")["summary"]
    return md_table(["Absolute A/A gap, points", "Value"],
                    [["median", f(s["abs_p50"])], ["90th percentile", f(s["abs_p90"])],
                     ["95th percentile", f(s["abs_p95"])], ["99th percentile", f(s["abs_p99"])],
                     ["maximum", f(s["abs_max"])], ["share of runs with a gap of exactly zero", f(s["share_zero"], ".3f")]])


def t_power(results):
    rows = []
    for name in ("prereg/power_small_gains.json", "prereg/power.json"):
        for r in rec(results, name)["results"]:
            rows.append([f"{f(r['mean_gain'])} ({r['multiple_of_BAR']:g} × BAR)",
                         f(r["p_interval_above_floor"]), f(r["p_pass"]), f(r["p_inconclusive"]), f(r["p_fail"])])
    return md_table(["Mean true gain, points", "Whole interval above FLOOR", "PASS", "INCONCLUSIVE", "FAIL"], rows)


def t_designs(results):
    designs = rec(results, "prereg/exploratory_design.json")["designs"]
    label = {"per_generator": "per generator", "pooled_one": "one response per source"}
    return md_table(["Calibration", "Grid points", "BASELINE coverage in DEV at alpha 0.10",
                     "A/A 99th percentile", "BAR the rule would set"],
                    [[label[d["kind"]], d["grid_points"],
                      f(d["frozen_baseline_dev_rehearsal"]["0.1"]["coverage_points"]),
                      f(d["aa"]["abs_p99"]), f(d["implied_by_the_rule"]["BAR_points"])]
                     for d in designs.values()])


# ------------------------------------------------------------------ confirmatory

def t_coverage(results, record=CONFIRM):
    table = rec(results, record)["coverage"]
    n = rec(results, record)["sizes"]["test"]["items"]
    rows = [[arm, a, f(r["coverage_points"]), f"{r['accepted']:,} of {n:,}", f(r["realised_risk"], ".3f"),
             certified(r["thresholds"])] for arm in ARMS for a, r in table[arm].items()]
    return md_table(["Arm", "alpha", "Certified coverage, points", "Accepted", "Realised risk among accepted",
                     "Certified generators and thresholds"], rows)


def t_hypotheses(results, record=CONFIRM):
    h = rec(results, record)["hypotheses"]
    rows = []
    for name, role in (("H1", "primary"), ("H3", "secondary")):
        x = h[name]
        rows.append([name, role, x["tested_arm"], f(x["gain_points"], "+.2f"),
                     interval(x["interval_low"], x["interval_high"]), f(x["p_gain_at_most_floor"], ".4f"),
                     f(x.get("p_holm"), ".4f"), f"**{x['outcome']}**"])
    rows.append(["H2", "secondary", "RESAMPLE features (WikiBio)", DASH, DASH, f(1.0, ".4f"), DASH,
                 "**NOT TESTABLE**"])
    return md_table(["Hypothesis", "Role", "Tested arm", "Gain over BASELINE, points", "95% interval",
                     "Share of draws at or below FLOOR", "Holm-adjusted", "Outcome"], rows)


def t_rehearsal_vs_test(results):
    dev, test = rec(results, "prereg/rehearsal.json"), rec(results, CONFIRM)
    a = str(test["primary_alpha"])
    rows = []
    for arm in ARMS:
        row = [arm, f(dev["coverage"][arm][a]["coverage_points"]), f(test["coverage"][arm][a]["coverage_points"])]
        name = {"BASELINE+SOURCE": "H1", "BASELINE+SELF": "H3"}.get(arm)
        for source in (dev, test):
            if name:
                x = source["hypotheses"][name]
                row.append(f"{f(x['gain_points'], '+.2f')} ({interval(x['interval_low'], x['interval_high'])}), {x['outcome']}")
            else:
                row.append("")
        rows.append(row)
    return md_table(["Arm", "Coverage, DEV rehearsal", "Coverage, TEST", "Gain, DEV rehearsal", "Gain, TEST"], rows)


# ------------------------------------------------------------------ controls

def t_permuted(results):
    c = rec(results, CONTROLS)["negative_controls"]["permuted_labels"]
    rows = [[arm, f(c["true_labels"][arm]["coverage_points"]), f(x["first_permutation"]["coverage_points"]),
             f(100 * x["share_with_any_coverage"], ".1f") + "%", f(x["max_coverage_points"])]
            for arm, x in c["arms"].items()]
    return md_table(["Score", "Coverage, true labels", "Coverage, first permutation",
                     f"Permutations with any coverage (of {c['permutations']:,})", "Largest coverage in any permutation"], rows)


def t_shuffled(results):
    s = rec(results, CONTROLS)["negative_controls"]["shuffled_source"]
    return md_table(["Logistic model on the 16 SOURCE columns, scored on TEST", "AUROC", "AUPRC"],
                    [["with each output's own source", f(s["true_source"]["auroc"], ".3f"), f(s["true_source"]["auprc"], ".3f")],
                     ["with another item's source", f(s["shuffled_source"]["auroc"], ".3f"), f(s["shuffled_source"]["auprc"], ".3f")]])


def t_length_only(results):
    c = rec(results, CONTROLS)["length"]["length_only"]
    return md_table(["alpha", "Certified coverage of output length alone, points", "Accepted"],
                    [[a, f(r["coverage_points"]), r["accepted"]] for a, r in c["coverage"].items()])


def t_alone(results):
    """Certified coverage and ranking of the models that hold one family of columns only."""
    c = rec(results, CONTROLS)
    alpha = str(c["primary_alpha"])
    rows = []
    for arm in ("BASELINE", "SOURCE only", "SELF only", "LENGTH", "RANDOM"):
        row = c["coverage"][arm][alpha]
        rows.append([arm, f(c["secondary"]["arms"][arm]["auroc"], ".3f"), f(row["coverage_points"]),
                     row["accepted"], f(row["realised_risk"], ".3f"), certified(row["thresholds"])])
    return md_table(["Score", "AUROC on TEST", "Certified coverage, points", "Accepted",
                     "Realised risk among accepted", "Certified generators and thresholds"], rows)


def t_within_generator(results):
    """Error rate among the lowest-risk quarter, half and three quarters of each generator's outputs."""
    curves = rec(results, CONTROLS)["secondary"]["risk_coverage"]
    at = lambda points, share: min(points, key=lambda p: abs(p["coverage"] - share))["risk"]
    items = rec(results, CONTROLS)["stress"]["per_generator"]["BASELINE"]
    rows = []
    for generator in sorted(k for k in curves["BASELINE"] if k != "all"):
        for share in (0.25, 0.50, 0.75):
            kept = max(1, int(round(share * items[generator]["items"])))      # as the curve counts them
            rows.append([SHORT.get(generator, generator), f"{share:.0%}", kept]
                        + [f(at(curves[arm][generator], share), ".3f") for arm in ARMS])
    return md_table(["Generator", "Share of its TEST outputs, lowest risk score first", "Outputs"]
                    + [f"{arm}: error rate" for arm in ARMS], rows)


PROBES = "controls/open_probes.json"
CHECK = "reproduction/check.json"


def t_strongest_single(results):
    """The three strongest single scores among the SOURCE columns and among the baselines."""
    scores = rec(results, CONTROLS)["secondary"]["single_scores"]
    family = lambda name: ("SOURCE column" if name.startswith("source.") and ".ncd_" not in name
                           else "SELF column" if name.startswith("self.") else "baseline")
    strength = lambda name: abs(scores[name]["auroc"] - 0.5)
    rows = []
    for wanted in ("SOURCE column", "baseline"):
        names = sorted((n for n in scores if family(n) == wanted and n != "base.random"),
                       key=strength, reverse=True)[:3]
        for name in names:
            auroc = scores[name]["auroc"]
            rows.append([f"`{name}`", wanted, f(auroc, ".3f"), f(max(auroc, 1.0 - auroc), ".3f"),
                         "errors" if auroc >= 0.5 else "correct outputs"])
    return md_table(["Column", "Family", "AUROC on TEST", "AUROC with the sign chosen after the fact",
                     "Higher for"], rows)


def t_near_duplicates(results):
    pairs = rec(results, PROBES)["near_duplicates"]["pairs_in_different_splits"]
    return md_table(["Source", "Split", "Source", "Split", "Jaccard similarity of the prompts"],
                    [[x["first"], x["split_first"], x["second"], x["split_second"], f(x["jaccard"], ".3f")]
                     for x in pairs])


def t_temperature_generators(results):
    rows = rec(results, PROBES)["temperature"]["by_generator_in_train"]
    return md_table(["Generator", "Temperatures in TRAIN", "AUROC of temperature alone, TRAIN"],
                    [[SHORT.get(g, g), ", ".join(f"{v:g}" for v in x["distinct_temperatures"]),
                      f(x["auroc_of_temperature_alone"], ".3f")] for g, x in rows.items()])


def t_temperature_arms(results):
    arms = rec(results, PROBES)["temperature"]["arms"]
    return md_table(["Model, refitted on TRAIN", "Columns", "Selected configuration", "AUROC, DEV_TEST",
                     "Coverage in the DEV rehearsal, points"],
                    [[name, x["columns"], config(x["selected"]), f(x["auroc_dev_test"], ".4f"),
                      f(x["dev_rehearsal_coverage_points"])] for name, x in arms.items()])


def t_temperature_difference(results):
    d = rec(results, PROBES)["temperature"]["paired_difference_on_dev_test"]
    return md_table(["Comparison on DEV_TEST", "Difference in AUROC", "95% interval"],
                    [[name, f(x["auroc"]["difference"], "+.4f"), interval(x["auroc"]["low"], x["auroc"]["high"], "+.4f")]
                     for name, x in d.items()])


def t_no_gain(results):
    n = rec(results, PROBES)["no_gain"]
    return md_table(["Quantity", "Value"],
                    [["Pairs of refits", n["replications"]],
                     ["FAIL", f(n["p_fail"], ".3f")], ["INCONCLUSIVE", f(n["p_inconclusive"], ".3f")],
                     ["PASS", f(n["p_pass"], ".3f")],
                     ["Mean absolute gain, points", f(n["mean_absolute_gain"])],
                     ["Upper end of the interval, median", f(n["interval_high_median"])],
                     ["Upper end of the interval, 90th percentile", f(n["interval_high_p90"])]])


def t_reproduction(results):
    if CHECK not in results["records"]:
        return "*The record of the reproduction check is not in this tree.*"
    c = rec(results, CHECK)
    reads = c["sealed_reads_by_the_reproduction"]
    return md_table(["Quantity", "Value"],
                    [["Commit the reproduction's records were compared with", f"`{c['reference'][:7]}`"],
                     ["Records compared", c["records_compared"]],
                     ["Identical apart from stamps, ledger rows, timings and model file hashes", c["records_identical"]],
                     ["Of those, with a different model file hash", c["records_identical_with_a_different_model_hash"]],
                     ["Reads of CAL by the reproduction", reads["CAL"]],
                     ["Reads of TEST by the reproduction", reads["TEST"]]])


def t_reproduction_ledger(results):
    if CHECK not in results["records"]:
        return "*The record of the reproduction check is not in this tree.*"
    rows = rec(results, CHECK)["sealed_reads_by_the_reproduction"]["rows"]
    return md_table(["Row in the clone's ledger", "Time (UTC)", "Split", "Purpose"],
                    [[x["number"], x["time_utc"], x["split"], x["purpose"]] for x in rows])


def t_overlap(results):
    o = rec(results, "controls/acceptance_overlap.json")
    return md_table(["Rule", "Accepted", "Other rule", "Accepted", "Accepted by both"],
                    [[x["first"], x["accepted_first"], x["second"], x["accepted_second"], x["accepted_both"]]
                     for x in o["pairs"]])


def t_length_quintiles(results):
    c = rec(results, CONTROLS)["length"]
    edges = [0] + [round(e) for e in c["quintile_edges_bytes"]] + [None]
    arms = list(c["within_quintiles"])
    rows = []
    for k in range(5):
        first = c["within_quintiles"][arms[0]][k]
        upper = "and above" if edges[k + 1] is None else f"to {edges[k + 1]:,}"
        rows.append([f"{edges[k]:,} {upper}", first["items"], first["errors"]] +
                    [f(c["within_quintiles"][arm][k]["auroc"], ".3f") for arm in arms])
    return md_table(["Output length, bytes", "Items", "Errors"] + arms, rows)


def t_guarantee(results):
    g = rec(results, CONTROLS)["guarantee"]
    s, r = g["synthetic"], g["resplit"]
    return md_table(["Check", "Runs", "Runs that accept anything", "Violations or flags", "Share", "Allowed", "Result"],
                    [["Synthetic data, known risk", f"{s['draws']:,}", f"{s['draws_accepting_anything']:,}",
                      s["violations"], f(s["violation_share"], ".4f"), f(s["delta"], ".2f"),
                      "holds" if s["holds"] else "FAILS"],
                     ["Real data, CAL and TEST re-split", f"{r['resplits']:,}", f"{r['resplits_accepting_anything']:,}",
                      r["flagged"], f(r["flagged_share"], ".4f"), f(r["allowed_share"], ".2f"),
                      "passes" if r["passes"] else "FAILS"]])


def t_loss_sensitivity(results):
    main, s = rec(results, CONFIRM), rec(results, CONTROLS)["loss_sensitivity"]
    a = str(main["primary_alpha"])
    variants = (("as pre-registered", main), ("implicit-true spans ignored", s["implicit_true_spans_ignored"]),
                ("flagged responses dropped", s["flagged_responses_dropped"]))
    rows = []
    for label, v in variants:
        row = [label] + [f(v["coverage"][arm][a]["coverage_points"]) for arm in ARMS]
        for name in ("H1", "H3"):
            x = v["hypotheses"][name]
            row.append(f"{f(x['gain_points'], '+.2f')} ({interval(x['interval_low'], x['interval_high'])}), {x['outcome']}")
        rows.append(row)
    return md_table(["Loss definition"] + [f"Coverage, {arm}" for arm in ARMS] + ["H1 gain", "H3 gain"], rows)


def t_cheap(results):
    c = rec(results, CONTROLS)["cheap_alternative"]["baseline_with_resampled_calibration"]
    order = sorted(c, key=float)
    return md_table(["Calibration size, multiple of CAL", "Source groups", "Mean coverage, points",
                     "Median coverage", "Draws with any coverage"],
                    [[m + (" (reference)" if float(m) == 1.0 else ""), f"{c[m]['calibration_groups']:,}",
                      f(c[m]["mean_coverage_points"]), f(c[m]["median_coverage_points"]),
                      f(100 * c[m]["share_with_any_coverage"], ".1f") + "%"] for m in order])


def _breakdown(results, key, names=None):
    b = rec(results, CONTROLS)["stress"][key]
    labels = list(b["BASELINE"])
    rows = []
    for label in labels:
        base = b["BASELINE"][label]
        row = [(names or {}).get(label, label), base["items"], f(100 * base["error_rate_of_all_items"], ".1f") + "%"]
        for arm in ARMS:
            x = b[arm][label]
            row.append(f"{f(x['coverage_points'], '.1f')} ({x['accepted']}; {f(x['realised_risk'], '.3f')})")
        rows.append(row)
    return rows


def t_per_generator(results):
    return md_table(["Generator", "TEST items", "Error rate"] + [f"{arm}: coverage (accepted; risk)" for arm in ARMS],
                    _breakdown(results, "per_generator", SHORT))


def t_per_task(results):
    return md_table(["Task type", "TEST items", "Error rate"] + [f"{arm}: coverage (accepted; risk)" for arm in ARMS],
                    _breakdown(results, "per_task"))


def t_across_tasks(results):
    c = rec(results, CONTROLS)["stress"]["across_task_types"]
    rows = []
    for task in c["BASELINE"]:
        row = [task, " and ".join(c["BASELINE"][task]["calibrated_on"]), c["BASELINE"][task]["items"]]
        for arm in ARMS:
            x = c[arm][task]
            row.append(f"{x['accepted']} ({f(x['realised_risk'], '.3f')})")
        rows.append(row)
    return md_table(["Task type tested", "Calibrated on", "TEST items"] + [f"{arm}: accepted (risk)" for arm in ARMS], rows)


def t_tokens(results):
    c = rec(results, CONTROLS)
    rows = []
    for name, t in c["stress"]["token_alphabet"].items():
        byte_arm = name.replace(" (tokens)", "")
        for label, coverage, auroc in ((byte_arm + " (bytes)", t["byte_arm_coverage"], c["secondary"]["arms"][byte_arm]["auroc"]),
                                       (name, t["coverage"], t["auroc"])):
            rows.append([label, f(auroc, ".3f")] + [f(coverage[a]["coverage_points"]) for a in coverage])
    return md_table(["Arm", "AUROC on TEST", "Coverage at alpha 0.01", "at 0.05", "at 0.10"], rows)


def t_paraphrase(results):
    p = rec(results, "controls/paraphrase.json")
    rows = [[arm, x["accepted_before"], x["accepted_after"], x["accepted_both"],
             f(x["risk_score_before"]["mean"], ".3f"), f(x["risk_score_after"]["mean"], ".3f"),
             f(x["spearman_before_after"], ".3f")] for arm, x in p["arms"].items()]
    return md_table(["Arm", f"Accepted before (of {p['outputs']})", "Accepted after", "Accepted both times",
                     "Mean risk score before", "after", "Rank correlation of the score"], rows)


def t_paraphrase_columns(results):
    means = rec(results, "controls/paraphrase.json")["column_means"]
    keep = ["source.cov_k4", "source.cov_k16", "source.cov_k64", "source.lcs_frac", "source.rlz_phrases_per_symbol",
            "self.cov_k4", "self.cov_k16", "base.novel_1gram", "base.novel_4gram", "base.rougeL_precision",
            "source.ncd_zstd", "gnd.min", "gnd.mean"]
    return md_table(["Column", "Mean before", "Mean after"],
                    [[f"`{name}`", f(means[name]["before"], ".3f"), f(means[name]["after"], ".3f")] for name in keep])


def t_ranking_arms(results):
    arms = rec(results, CONTROLS)["secondary"]["arms"]
    return md_table(["Score", "AUROC on TEST", "AUPRC on TEST"],
                    [[arm, f(x["auroc"], ".4f"), f(x["auprc"], ".4f")] for arm, x in arms.items()])


def t_ranking_difference(results):
    d = rec(results, CONTROLS)["secondary"]["paired_difference_from_baseline"]
    rows = []
    for arm, x in d.items():
        for metric, label in (("auroc", "AUROC"), ("auprc", "AUPRC")):
            m = x[metric]
            rows.append([arm, label, f(m["first"], ".4f"), f(m["second"], ".4f"), f(m["difference"], "+.4f"),
                         interval(m["low"], m["high"], "+.4f")])
    return md_table(["Arm", "Metric", "BASELINE", "Arm", "Difference", "95% interval"], rows)


def t_single_scores(results):
    single = rec(results, CONTROLS)["secondary"]["single_scores"]
    order = sorted(single, key=lambda n: -abs(single[n]["auroc"] - 0.5))
    return md_table(["Column", "AUROC on TEST", "AUPRC on TEST"],
                    [[f"`{n}`", f(single[n]["auroc"], ".3f"), f(single[n]["auprc"], ".3f")] for n in order])


def t_wikibio(results, record="controls/wikibio_result.json"):
    w = rec(results, record)
    rows = [[arm, len(x["columns"]), config(x["selected"]), f(x["auroc"], ".4f"), f(x["auprc"], ".4f")]
            for arm, x in w["arms"].items()]
    return md_table(["Combiner", "Columns", "Selected configuration", "AUROC", "AUPRC"], rows)


def t_wikibio_difference(results, record="controls/wikibio_result.json"):
    w = rec(results, record)
    rows = []
    for label, d in w["difference"].items():
        for metric, name in (("auroc", "AUROC"), ("auprc", "AUPRC")):
            m = d[metric]
            rows.append([label, name, f(m["difference"], "+.4f"), interval(m["low"], m["high"], "+.4f")])
    return md_table(["Comparison (with RESAMPLE minus without)", "Metric", "Difference", "95% interval"], rows)


def t_runtime(results):
    r = rec(results, "controls/runtime.json")
    sample = f"{r['cpu_scores']['items']} TRAIN responses"
    rows = [[name, f(x["seconds_per_megabyte"], ".3f"), "CPU, one thread", sample]
            for name, x in r["cpu_scores"]["scores"].items()]
    for name, x in r["neural_baselines"].items():
        timed = f"{x['items']} of {x['items_in_the_splits']} items" if "items" in x else "not timed"
        rows.append([name, f(x.get("seconds_per_megabyte"), ".1f"), x.get("device", x.get("status", "")), timed])
    return md_table(["Score", "Seconds per megabyte of input", "Where", "Timed on"], rows)


def t_ledger(results):
    rows = [[e["number"], e["time_utc"], e["split"], e["access_to_this_split"], f"`{e['commit'][:7]}`", e["purpose"]]
            for e in results["ledger"]["entries"]]
    return md_table(["#", "Time (UTC)", "Split", "Access to this split", "Commit", "Purpose"], rows)


TABLES = {name[2:]: fn for name, fn in list(globals().items()) if name.startswith("t_")}
TABLES["coverage_dev"] = lambda results: t_coverage(results, "prereg/rehearsal.json")
TABLES["wikibio_dev"] = lambda results: t_wikibio(results, "controls/wikibio_rehearsal.json")
TABLES["wikibio_difference_dev"] = lambda results: t_wikibio_difference(results, "controls/wikibio_rehearsal.json")


# ------------------------------------------------------------------ figures

def figures(results, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"svg.hashsalt": "selfref", "font.size": 9, "axes.spines.top": False,
                         "axes.spines.right": False})
    out.mkdir(parents=True, exist_ok=True)
    save = lambda fig, name: (fig.savefig(out / name, metadata={"Date": None}, bbox_inches="tight"), plt.close(fig))
    limits = rec(results, "prereg/bar_floor.json")

    gaps = [abs(g) for g in rec(results, "prereg/aa.json")["gap_points"]]
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    ax.hist(gaps, bins=40, color="#7a8aa0")
    ax.axvline(limits["FLOOR_points"], color="#b03030", lw=1.2, label="FLOOR = BAR (99th percentile)")
    ax.axvline(5.0, color="#404040", lw=1.0, ls="--", label="nominal BAR, 5 points")
    ax.set_xlabel("absolute coverage gap between two BASELINE refits, points")
    ax.set_ylabel("runs (of 1,000)")
    ax.legend(frameon=False)
    save(fig, "aa_gap.svg")

    rows = [r for name in ("prereg/power_small_gains.json", "prereg/power.json")
            for r in rec(results, name)["results"]]
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    x = [r["mean_gain"] for r in rows]
    for key, label, color in (("p_fail", "FAIL", "#b03030"), ("p_inconclusive", "INCONCLUSIVE", "#7a8aa0"),
                              ("p_pass", "PASS", "#2f7d4f")):
        ax.plot(x, [r[key] for r in rows], marker="o", ms=3, label=label, color=color)
    ax.axvline(limits["BAR_points"], color="#404040", lw=1.0, ls="--")
    ax.set_xlabel("mean true gain, points (dashed line: BAR)")
    ax.set_ylabel("probability of the outcome")
    ax.legend(frameon=False)
    save(fig, "power.svg")

    if CONTROLS in results["records"]:
        curves = rec(results, CONTROLS)["secondary"]["risk_coverage"]
        alpha = rec(results, CONTROLS)["primary_alpha"]
        fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.2), sharey=True)
        for arm, color in zip(ARMS, ("#404040", "#b03030", "#2f6fb0")):
            axes[0].plot([100 * p["coverage"] for p in curves[arm]["all"]],
                         [p["risk"] for p in curves[arm]["all"]], label=arm, color=color, lw=1.1)
        for generator in sorted(k for k in curves["BASELINE"] if k != "all"):
            axes[1].plot([100 * p["coverage"] for p in curves["BASELINE"][generator]],
                         [p["risk"] for p in curves["BASELINE"][generator]],
                         label=SHORT.get(generator, generator), lw=1.1)
        for ax, title in zip(axes, ("all generators, three arms", "BASELINE, by generator")):
            ax.axhline(alpha, color="#909090", lw=0.8, ls="--")
            ax.set_xlabel("TEST outputs accepted, lowest risk first (%)")
            ax.set_title(title, fontsize=9)
            ax.legend(frameon=False, fontsize=7)
        axes[0].set_ylabel("error rate among accepted")
        save(fig, "risk_coverage.svg")


def main() -> int:
    results = json.loads((HARNESS / "results.json").read_text())
    figures(results, HARNESS / "report" / "figures")
    template = (HARNESS / "report" / "REPORT.template.md").read_text()
    (HARNESS / "REPORT.md").write_text(render(template, results, TABLES))
    print("wrote REPORT.md and report/figures/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
