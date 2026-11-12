# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LogNovations Holdings LLC
"""Render the paper from results.json.

    python scripts/make_paper.py           # paper/paper.tex and paper/figures/*.pdf
    python scripts/make_paper.py --pdf     # and paper/paper.pdf (needs pdflatex)

The paper is written as a LaTeX template, paper/paper.template.tex. Every
measured result in it is a placeholder into results.json, exactly as in the
report (see selfref.render), or a table built here from results.json. A path
that does not exist stops the rendering. The figures are drawn from the same
file. Nothing in the paper is computed anywhere else.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

HARNESS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HARNESS / "src"))
sys.path.insert(0, str(HARNESS / "scripts"))

from selfref.render import render  # noqa: E402

PAPER = HARNESS / "paper"
C, K, W = "confirmatory/result.json", "controls/result.json", "controls/wikibio_result.json"
PROBES, RUNTIME = "controls/open_probes.json", "controls/runtime.json"
ARMS = ("BASELINE", "BASELINE+SOURCE", "BASELINE+SELF")
SHORT = {"gpt-3.5-turbo-0613": "GPT-3.5", "gpt-4-0613": "GPT-4", "llama-2-13b-chat": "Llama-2-13B",
         "llama-2-70b-chat": "Llama-2-70B", "llama-2-7b-chat": "Llama-2-7B",
         "mistral-7B-instruct": "Mistral-7B"}
SPLITS = ("TRAIN", "DEV_SELECT", "DEV_CAL", "DEV_TEST", "CAL", "TEST")
MINICHECK = "MiniCheck groundedness, RAGTruth CAL and TEST, resumed run"


def rec(results, name):
    return results["records"][name]


def num(value, spec=".2f") -> str:
    """A number for LaTeX text: a dash for None, a true minus sign otherwise."""
    if value is None:
        return "--"
    text = format(value, spec)
    return f"${text}$" if text.startswith(("-", "+")) else text


def tex(text: str) -> str:
    """Escape the LaTeX specials that occur in names taken from the records."""
    for old, new in (("\\", r"\textbackslash{}"), ("_", r"\_"), ("%", r"\%"), ("&", r"\&"), ("#", r"\#")):
        text = text.replace(old, new)
    return text


def rows(lines) -> str:
    return " \\\\\n".join(" & ".join(str(cell) for cell in line) for line in lines) + " \\\\"


def certified(thresholds: dict) -> str:
    kept = [f"{SHORT.get(g, g)} at {t:.2f}" for g, t in sorted(thresholds.items()) if t is not None]
    return ", ".join(kept) if kept else "none"


# ---------------------------------------------------------------------- tables

def t_main(results):
    r = rec(results, C)
    alpha = str(r["primary_alpha"])
    out = []
    for arm in ARMS:
        row = r["coverage"][arm][alpha]
        h = next((h for h in r["hypotheses"].values() if h.get("tested_arm") == arm), None)
        gain = (f"{num(h['gain_points'], '+.2f')} ({num(h['interval_low'], '+.2f')} to "
                f"{num(h['interval_high'], '+.2f')})" if h else "reference")
        out.append([tex(arm), num(row["coverage_points"]), f"{row['accepted']:,}",
                    num(row["realised_risk"], ".3f"), certified(row["thresholds"]), gain,
                    h["outcome"] if h else "--"])
    return rows(out)


def t_ranking(results):
    c = rec(results, K)["secondary"]
    out = []
    for arm in ARMS:
        d = c["paired_difference_from_baseline"].get(arm)
        diff = (f"{num(d['auroc']['difference'], '+.4f')} ({num(d['auroc']['low'], '+.4f')} to "
                f"{num(d['auroc']['high'], '+.4f')})" if d else "reference")
        diff_pr = (f"{num(d['auprc']['difference'], '+.4f')} ({num(d['auprc']['low'], '+.4f')} to "
                   f"{num(d['auprc']['high'], '+.4f')})" if d else "reference")
        out.append([tex(arm), num(c["arms"][arm]["auroc"], ".4f"), diff,
                    num(c["arms"][arm]["auprc"], ".4f"), diff_pr])
    return rows(out)


def t_alone(results):
    c = rec(results, K)
    alpha = str(c["primary_alpha"])
    labels = {"SOURCE only": "16 SOURCE columns (logistic)", "SELF only": "16 SELF columns (logistic)",
              "LENGTH": "Output length (3 columns)", "RANDOM": "Random score", "BASELINE": "BASELINE (25 columns)"}
    out = []
    for arm in ("BASELINE", "SOURCE only", "SELF only", "LENGTH", "RANDOM"):
        row = c["coverage"][arm][alpha]
        out.append([labels[arm], num(c["secondary"]["arms"][arm]["auroc"], ".3f"),
                    num(row["coverage_points"]), row["accepted"], num(row["realised_risk"], ".3f")])
    return rows(out)


def t_single(results):
    scores = rec(results, K)["secondary"]["single_scores"]
    family = lambda name: ("SOURCE feature" if name.startswith("source.") and ".ncd_" not in name
                           else "SELF feature" if name.startswith("self.") else "baseline")
    strength = lambda name: abs(scores[name]["auroc"] - 0.5)
    out = []
    for wanted in ("SOURCE feature", "baseline"):
        names = sorted((n for n in scores if family(n) == wanted and n != "base.random"),
                       key=strength, reverse=True)[:3]
        for name in names:
            auroc = scores[name]["auroc"]
            out.append([r"\texttt{" + tex(name) + "}", wanted, num(auroc, ".3f"),
                        num(max(auroc, 1.0 - auroc), ".3f"), "errors" if auroc >= 0.5 else "correct outputs"])
    return rows(out)


def t_generators(results):
    s = rec(results, K)["stress"]["per_generator"]
    out = []
    for g in s["BASELINE"]:
        cells = [SHORT.get(g, g), s["BASELINE"][g]["items"], num(s["BASELINE"][g]["error_rate_of_all_items"], ".3f")]
        for arm in ARMS:
            x = s[arm][g]
            cells.append(f"{x['accepted']} ({num(x['realised_risk'], '.3f')})")
        out.append(cells)
    return rows(out)


def t_loss(results):
    r, c = rec(results, C), rec(results, K)
    alpha = str(r["primary_alpha"])
    variants = [("As pre-registered", r["coverage"], r["hypotheses"])]
    for label, key in (("Implicit-true spans ignored", "implicit_true_spans_ignored"),
                       ("Flagged responses dropped", "flagged_responses_dropped")):
        v = c["loss_sensitivity"][key]
        variants.append((label, v["coverage"], v["hypotheses"]))
    out = []
    for label, coverage, hyp in variants:
        gain = lambda h: (f"{num(h['gain_points'], '+.2f')} ({num(h['interval_low'], '+.2f')} to "
                          f"{num(h['interval_high'], '+.2f')})")
        out.append([label] + [num(coverage[arm][alpha]["coverage_points"]) for arm in ARMS]
                   + [gain(hyp["H1"]), gain(hyp["H3"])])
    return rows(out)


def t_temperature(results):
    t = rec(results, PROBES)["temperature"]
    out = [[tex(name), x["columns"], num(x["auroc_dev_test"], ".4f"), num(x["dev_rehearsal_coverage_points"])]
           for name, x in t["arms"].items()]
    return rows(out)


def t_splits(results):
    counts = {"RAGTruth": rec(results, "data/derived/ragtruth/build_report.json"),
              "WikiBio GPT-3": rec(results, "data/derived/wikibio/build_report.json")}
    out = []
    for dataset, report in counts.items():
        for split in SPLITS:
            out.append([dataset, tex(split), f"{report['groups_per_split'][split]:,}",
                        f"{report['items_per_split'][split]:,}"])
    return rows(out)


def t_reproduction_sentence(results):
    """One sentence on the reproduction check, or its absence in this tree."""
    check = results["records"].get("reproduction/check.json")
    if check is None:
        return "The record of the reproduction check is not in this tree."
    reads = check["sealed_reads_by_the_reproduction"]
    return (f"One reproduction was run, in a fresh clone on the machine the study ran on. Of "
            f"{check['records_compared']} records, {check['records_identical']} were identical to the "
            f"study's apart from provenance stamps, ledger rows and timings. In its own clone the "
            f"reproduction read the calibration labels {reads['CAL']} times and test labels {reads['TEST']} "
            f"times, after every result reported here had been committed.")


TABLES = {name[2:]: fn for name, fn in list(globals().items()) if name.startswith("t_")}


# --------------------------------------------------------------------- figures

def figures(results, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "axes.titlesize": 8, "legend.fontsize": 7})
    out.mkdir(parents=True, exist_ok=True)

    def save(fig, name):
        fig.savefig(out / name, metadata={"CreationDate": None, "ModDate": None}, bbox_inches="tight")
        plt.close(fig)

    limits = rec(results, "prereg/bar_floor.json")
    bar, floor = limits["BAR_points"], limits["FLOOR_points"]
    grey, red, blue, green = "#404040", "#b03030", "#2f6fb0", "#3a8a4a"

    # Figure 1: what the endpoint can resolve
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.4))
    gaps = np.abs(rec(results, "prereg/aa.json")["gap_points"])
    axes[0].hist(gaps, bins=40, color="#9aa7b5")
    axes[0].axvline(floor, color=red, lw=1.0, ls="--")
    axes[0].axvline(limits["c_B_points"], color=grey, lw=1.0, ls=":")
    axes[0].set_xlabel("gap between two refits (points)")
    axes[0].set_ylabel("runs")
    axes[0].set_title("a  noise of the endpoint", loc="left")
    points = [("no gain", 0.0, rec(results, PROBES)["no_gain"])]
    for name in ("prereg/power_small_gains.json", "prereg/power.json"):
        points += [("", x["mean_gain"], x) for x in rec(results, name)["results"]]
    points.sort(key=lambda p: p[1])
    xs = [p[1] for p in points]
    for key, label, color in (("p_fail", "FAIL", red), ("p_inconclusive", "INCONCLUSIVE", grey),
                              ("p_pass", "PASS", green)):
        axes[1].plot(xs, [p[2][key] for p in points], marker="o", ms=3, lw=1.0, color=color, label=label)
    axes[1].axvline(bar, color="#909090", lw=0.8, ls="--")
    axes[1].set_xlabel("true gain (points)")
    axes[1].set_ylabel("probability")
    axes[1].set_ylim(-0.04, 1.16)
    axes[1].set_title("b  outcomes of the decision rule", loc="left")
    axes[1].text(1.2, 0.87, "FAIL", color=red, fontsize=7)
    axes[1].text(bar + 0.6, 1.06, "INCONCLUSIVE", color=grey, fontsize=7, ha="center")
    axes[1].text(33.0, 0.86, "PASS", color=green, fontsize=7, ha="right")
    cheap = rec(results, K)["cheap_alternative"]["baseline_with_resampled_calibration"]
    multiples = sorted(cheap, key=float)
    axes[2].plot([float(m) for m in multiples], [cheap[m]["mean_coverage_points"] for m in multiples],
                 marker="o", ms=3, lw=1.0, color=blue)
    axes[2].set_xlabel("calibration size (multiple of CAL)")
    axes[2].set_ylabel("coverage (points)")
    axes[2].set_ylim(bottom=0)
    axes[2].set_title("c  more calibration labels", loc="left")
    fig.tight_layout(w_pad=1.6)
    save(fig, "fig1_endpoint.pdf")

    # Figure 2: the gains against the bar
    r, c = rec(results, C), rec(results, K)
    dev = rec(results, "prereg/rehearsal.json")["hypotheses"]
    sources = [("confirmatory (CAL and TEST)", r["hypotheses"], True),
               ("implicit-true spans ignored", c["loss_sensitivity"]["implicit_true_spans_ignored"]["hypotheses"], False),
               ("flagged responses dropped", c["loss_sensitivity"]["flagged_responses_dropped"]["hypotheses"], False),
               ("DEV rehearsal (open data)", dev, False)]
    fig, ax = plt.subplots(figsize=(6.0, 2.7))
    labels, y = [], 0
    for name, title, color in (("H1", "H1: SOURCE", red), ("H3", "H3: SELF", blue)):
        for label, hyp, primary in sources:
            h = hyp[name]
            ax.plot([h["interval_low"], h["interval_high"]], [y, y], color=color, lw=2.0 if primary else 1.0)
            ax.plot([h["gain_points"]], [y], marker="o" if primary else "s", ms=5 if primary else 3.5,
                    color=color, mfc=color if primary else "white")
            labels.append(f"{title}, {label}")
            y -= 1
        y -= 0.6
    ticks, y = [], 0
    for _ in range(2):
        ticks += [y - k for k in range(len(sources))]
        y -= len(sources) + 0.6
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels)
    ax.axvline(0, color="#b0b0b0", lw=0.8)
    ax.axvline(bar, color=grey, lw=1.0, ls="--")
    ax.text(bar, 0.6, "BAR = FLOOR", ha="center", va="bottom", fontsize=7, color=grey)
    ax.set_xlabel("gain over BASELINE (points)")
    fig.tight_layout()
    save(fig, "fig2_gains.pdf")

    # Figure 3: risk-coverage curves
    curves = c["secondary"]["risk_coverage"]
    alpha = c["primary_alpha"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6), sharey=True)
    for arm, color in zip(ARMS, (grey, red, blue)):
        axes[0].plot([100 * p["coverage"] for p in curves[arm]["all"]],
                     [p["risk"] for p in curves[arm]["all"]], label=arm, color=color, lw=1.0)
    for generator in sorted(k for k in curves["BASELINE"] if k != "all"):
        axes[1].plot([100 * p["coverage"] for p in curves["BASELINE"][generator]],
                     [p["risk"] for p in curves["BASELINE"][generator]],
                     label=SHORT.get(generator, generator), lw=1.0)
    for ax, title in zip(axes, ("a  all generators, three arms", "b  BASELINE, by generator")):
        ax.axhline(alpha, color="#909090", lw=0.8, ls="--")
        ax.set_xlabel("TEST outputs accepted, lowest risk first (%)")
        ax.set_title(title, loc="left")
        ax.legend(frameon=False)
    axes[0].set_ylabel("error rate among accepted")
    fig.tight_layout()
    save(fig, "fig3_risk_coverage.pdf")

    # Figure 4: ranking power against cost
    single, arms = c["secondary"]["single_scores"], c["secondary"]["arms"]
    cpu = rec(results, RUNTIME)["cpu_scores"]["scores"]
    neural = rec(results, RUNTIME)["neural_baselines"][MINICHECK]
    marks = [   # label, cost, AUROC, colour, marker, label offset in points, horizontal alignment
        ("SOURCE profile\n(model on 16 features)", cpu["SOURCE profile features (16 columns)"]["seconds_per_megabyte"],
         arms["SOURCE only"]["auroc"], red, "o", (7, -9), "left"),
        ("SELF profile\n(model on 16 features)", cpu["SELF profile features (16 columns)"]["seconds_per_megabyte"],
         arms["SELF only"]["auroc"], blue, "o", (7, -3), "left"),
        ("novel word 1-grams", cpu["lexical baselines (length, n-gram novelty, ROUGE-L)"]["seconds_per_megabyte"],
         single["base.novel_1gram"]["auroc"], grey, "s", (-2, 7), "center"),
        ("NCD, gzip", cpu["NCD, gzip"]["seconds_per_megabyte"], single["source.ncd_gzip"]["auroc"], grey, "s",
         (0, 7), "center"),
        ("NCD, zstd", cpu["NCD, zstd"]["seconds_per_megabyte"], single["source.ncd_zstd"]["auroc"], grey, "s",
         (0, -12), "center"),
        ("groundedness classifier\n(share of sentences unsupported)", neural["seconds_per_megabyte"],
         single["gnd.frac_unsupported"]["auroc"], grey, "s", (-7, 1), "right"),
    ]
    fig, ax = plt.subplots(figsize=(5.0, 3.0))
    for label, cost, auroc, color, marker, offset, align in marks:
        ax.plot([cost], [auroc], marker=marker, ms=5, color=color, ls="none")
        ax.annotate(label, (cost, auroc), textcoords="offset points", xytext=offset, fontsize=6.5,
                    color=color, ha=align)
    ax.axhline(arms["BASELINE"]["auroc"], color="#909090", lw=0.8, ls="--")
    ax.text(0.03, arms["BASELINE"]["auroc"] + 0.005, "BASELINE, all 25 columns", fontsize=6.5, color="#606060")
    ax.axhline(0.5, color="#d0d0d0", lw=0.8)
    ax.set_xscale("log")
    ax.set_xlim(0.02, 4000)
    ax.set_ylim(0.48, 0.92)
    ax.set_xlabel("seconds per megabyte of input (log scale)")
    ax.set_ylabel("AUROC on TEST")
    fig.tight_layout()
    save(fig, "fig4_signal_cost.pdf")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pdf", action="store_true", help="also run pdflatex")
    args = parser.parse_args(argv)
    results = json.loads((HARNESS / "results.json").read_text())
    figures(results, PAPER / "figures")
    template = (PAPER / "paper.template.tex").read_text()
    (PAPER / "paper.tex").write_text(render(template, results, TABLES))
    print("wrote paper/paper.tex and paper/figures/")
    if args.pdf:
        for _ in range(2):
            done = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "paper.tex"],
                                  cwd=PAPER, capture_output=True, text=True)
            if done.returncode != 0:
                print(done.stdout[-3000:])
                raise SystemExit("pdflatex failed")
        for junk in ("paper.aux", "paper.log", "paper.out"):
            (PAPER / junk).unlink(missing_ok=True)
        print("wrote paper/paper.pdf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
