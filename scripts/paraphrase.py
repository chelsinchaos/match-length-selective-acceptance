# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LogNovations Holdings LLC
"""Paraphrase stress analysis (pre-registration, Section 14, Stress). Exploratory.

    python scripts/paraphrase.py generate          # paraphrase a sample of TEST outputs, locally
    python scripts/compute_features.py   ragtruth --splits TEST --derived data/derived/ragtruth_paraphrase
    python scripts/score_groundedness.py ragtruth --splits TEST --derived data/derived/ragtruth_paraphrase
    python scripts/paraphrase.py analyse           # scores and coverage before and after

The sample is every response of the 50 TEST sources that come first in a fixed
hash order: 300 outputs, 50 from each generator. Each output is rewritten by
an open-weight model run on this machine, with greedy decoding and an
instruction to change the wording and keep every fact. The rewritten outputs
are then scored exactly as the originals were, with the same code, and put
through the frozen models, the frozen rank transforms and the thresholds the
confirmatory run certified on CAL.

No label is read at any step. The analysis compares scores and acceptance
before and after; it says nothing about whether a paraphrase kept the output
correct, so it reports no error rate.
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

from selfref.baselines import word_tokens  # noqa: E402
from selfref.certify import RankTransform, accepted_mask  # noqa: E402
from selfref.combiner import risk_scores  # noqa: E402
from selfref.splits import unit_interval  # noqa: E402
from selfref.tables import categories_of, load_ragtruth_split  # noqa: E402

DERIVED = HARNESS / "data" / "derived" / "ragtruth"
OUT = HARNESS / "data" / "derived" / "ragtruth_paraphrase"
PREREG = HARNESS / "prereg"
MODEL = "microsoft/Phi-3.5-mini-instruct"
REVISION = "2fe192450127e6a83f7441aef6e3ca586c338b77"
SAMPLE_SALT = "selfref-prereg-v1/paraphrase-sample"
SAMPLE_SOURCES = 50
INSTRUCTION = ("Rewrite the text below in different words. Keep every fact, name, number and "
               "date exactly as it is. Do not add anything and do not leave anything out. "
               "Reply with the rewritten text only.\n\nText:\n")
FROZEN = {"BASELINE": "BASELINE", "BASELINE+SOURCE": "BASELINE_plus_SOURCE",
          "BASELINE+SELF": "BASELINE_plus_SELF"}


def git(*args) -> str:
    return subprocess.run(["git", "-C", str(HARNESS), *args], capture_output=True,
                          text=True).stdout.strip()


def sample_items() -> list[dict]:
    items = [json.loads(line) for line in open(DERIVED / "items.jsonl", encoding="utf-8")]
    test = [item for item in items if item["split"] == "TEST"]
    order = sorted({item["source_id"] for item in test},
                   key=lambda source: unit_interval(source, SAMPLE_SALT))
    chosen = set(order[:SAMPLE_SOURCES])
    return [item for item in test if item["source_id"] in chosen]


def generate(args) -> int:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    OUT.mkdir(parents=True, exist_ok=True)
    items = sample_items()
    done_path = OUT / "paraphrases.jsonl"
    done = {}
    if done_path.exists():
        done = {r["id"]: r for r in map(json.loads, open(done_path, encoding="utf-8"))}
    todo = [item for item in items if item["id"] not in done]
    if args.limit is not None:
        todo = todo[:args.limit]
    print(f"{len(items)} outputs in the sample, {len(done)} already paraphrased, {len(todo)} to do",
          flush=True)
    if todo:
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL, revision=REVISION, torch_dtype=torch.float16 if device == "mps" else torch.float32
        ).to(device).eval()
        start = time.perf_counter()
        with open(done_path, "a", encoding="utf-8") as out:
            for n, item in enumerate(todo, 1):
                prompt = tokenizer.apply_chat_template(
                    [{"role": "user", "content": INSTRUCTION + item["response"]}],
                    tokenize=False, add_generation_prompt=True)
                inputs = tokenizer(prompt, return_tensors="pt").to(device)
                budget = int(1.5 * len(tokenizer(item["response"])["input_ids"])) + 32
                with torch.no_grad():
                    generated = model.generate(**inputs, do_sample=False, max_new_tokens=budget,
                                               pad_token_id=tokenizer.eos_token_id)
                text = tokenizer.decode(generated[0, inputs["input_ids"].shape[1]:],
                                        skip_special_tokens=True).strip()
                out.write(json.dumps({"id": item["id"], "paraphrase": text}) + "\n")
                out.flush()
                del generated, inputs
                held = ""
                if device == "mps":          # without this the allocator's cache grows with every output
                    torch.mps.empty_cache()
                    held = f"  {torch.mps.driver_allocated_memory() / 1e9:.1f} GB held by MPS"
                if n % 10 == 0 or n == len(todo):
                    print(f"{n}/{len(todo)}  {time.perf_counter() - start:.0f}s{held}", flush=True)
        done = {r["id"]: r for r in map(json.loads, open(done_path, encoding="utf-8"))}
    if all(item["id"] in done for item in items):
        with open(OUT / "items.jsonl", "w", encoding="utf-8") as out:
            for item in items:
                out.write(json.dumps({**item, "response": done[item["id"]]["paraphrase"]}) + "\n")
        wanted = {item["source_id"] for item in items}
        with open(OUT / "sources.jsonl", "w", encoding="utf-8") as out:
            for line in open(DERIVED / "sources.jsonl", encoding="utf-8"):
                if json.loads(line)["source_id"] in wanted:
                    out.write(line)
        print(f"wrote the item and source tables of {len(items)} paraphrased outputs")
    return 0


def summary(values) -> dict:
    values = np.asarray(values, dtype=float)
    return {"mean": float(values.mean()), "median": float(np.median(values))}


def spearman(a, b) -> float:
    rank = lambda v: np.argsort(np.argsort(v, kind="stable"), kind="stable").astype(float)
    return float(np.corrcoef(rank(np.asarray(a)), rank(np.asarray(b)))[0, 1])


def analyse(args) -> int:
    confirmatory = json.loads((HARNESS / "confirmatory" / "result.json").read_text())
    alpha = str(confirmatory["primary_alpha"])
    records = {"BASELINE": json.loads((PREREG / "baseline.json").read_text())}
    records.update(json.loads((PREREG / "profile.json").read_text())["arms"])
    categories = categories_of(DERIVED)
    select = load_ragtruth_split(DERIVED, "DEV_SELECT", categories)
    placeholder = lambda directory: {str(i): {"loss": 0} for i in
                                     np.load(directory / "features_TEST.npz")["ids"]}
    # The loss column of these two tables is a placeholder and is never used.
    original = load_ragtruth_split(DERIVED, "TEST", categories, sealed_labels=placeholder(DERIVED))
    rewritten = load_ragtruth_split(OUT, "TEST", categories, sealed_labels=placeholder(OUT))
    rows = [original.ids.index(i) for i in rewritten.ids]
    texts = {item["id"]: item["response"] for item in sample_items()}
    paraphrases = {r["id"]: r["paraphrase"] for r in
                   map(json.loads, open(OUT / "paraphrases.jsonl", encoding="utf-8"))}
    overlap = []
    for i in rewritten.ids:
        a, b = word_tokens(texts[i]), word_tokens(paraphrases[i])
        overlap.append(len(set(a) & set(b)) / max(1, len(set(a) | set(b))))
    length_ratio = [len(paraphrases[i].encode()) / max(1, len(texts[i].encode())) for i in rewritten.ids]

    result = {"arms": {}}
    stratum = rewritten.stratum
    for arm, record in records.items():
        blob = (DERIVED / "models" / f"{FROZEN[arm]}.pkl").read_bytes()
        if hashlib.sha256(blob).hexdigest() != record["model_sha256"]:
            raise SystemExit(f"the frozen model for {arm} does not match its recorded SHA-256")
        model = pickle.loads(blob)
        transform = RankTransform.fit(risk_scores(model, select.sample(record["columns"]).X),
                                      select.stratum)
        before = risk_scores(model, original.sample(record["columns"]).X[rows])
        after = risk_scores(model, rewritten.sample(record["columns"]).X)
        thresholds = confirmatory["coverage"][arm][alpha]["thresholds"]
        was = accepted_mask(transform.transform(before, stratum), stratum, thresholds)
        now = accepted_mask(transform.transform(after, stratum), stratum, thresholds)
        result["arms"][arm] = {
            "risk_score_before": summary(before), "risk_score_after": summary(after),
            "spearman_before_after": spearman(before, after),
            "accepted_before": int(was.sum()), "accepted_after": int(now.sum()),
            "accepted_both": int((was & now).sum()),
            "coverage_points_before": 100.0 * float(was.mean()),
            "coverage_points_after": 100.0 * float(now.mean()),
            "by_generator": {str(g): {"items": int((stratum == g).sum()),
                                      "accepted_before": int(was[stratum == g].sum()),
                                      "accepted_after": int(now[stratum == g].sum())}
                             for g in np.unique(stratum)}}
    columns = [n for n in original.names if not n.startswith("cov.") and n != "base.random"]
    result["column_means"] = {
        name: {"before": float(original.X[rows, original.names.index(name)].mean()),
               "after": float(rewritten.X[:, rewritten.names.index(name)].mean())} for name in columns}
    record = {
        "status": "EXPLORATORY. No label is read; acceptance uses the thresholds of the confirmatory run.",
        "provenance": {"git_commit": git("rev-parse", "HEAD"),
                       "git_dirty": bool(git("status", "--porcelain", "--", ".")),
                       "paraphrase_model": MODEL, "paraphrase_model_revision": REVISION,
                       "decoding": "greedy", "instruction": INSTRUCTION,
                       "sample": f"all responses of the first {SAMPLE_SOURCES} TEST sources in the "
                                 f"hash order of salt {SAMPLE_SALT!r}"},
        "alpha": float(alpha), "outputs": len(rewritten.ids),
        "paraphrase_fidelity": {"byte_length_ratio": summary(length_ratio),
                                "word_set_jaccard_with_original": summary(overlap),
                                "empty_paraphrases": int(sum(1 for i in rewritten.ids if not paraphrases[i])),
                                "identical_to_original": int(sum(1 for i in rewritten.ids
                                                                 if paraphrases[i] == texts[i]))},
        **result}
    (HARNESS / "controls").mkdir(exist_ok=True)
    (HARNESS / "controls" / "paraphrase.json").write_text(json.dumps(record, indent=1) + "\n")
    for arm, row in result["arms"].items():
        print(f"  {arm:16s} accepted {row['accepted_before']} -> {row['accepted_after']} of "
              f"{len(rewritten.ids)}; rank correlation {row['spearman_before_after']:.3f}")
    print("wrote controls/paraphrase.json")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="step", required=True)
    gen = sub.add_parser("generate"); gen.set_defaults(run=generate)
    gen.add_argument("--limit", type=int, default=None, help="stop after this many new outputs")
    sub.add_parser("analyse").set_defaults(run=analyse)
    args = parser.parse_args(argv)
    return args.run(args)


if __name__ == "__main__":
    raise SystemExit(main())
