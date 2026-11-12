# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LogNovations Holdings LLC
"""Build the study tables from raw dataset files.

For each dataset the build produces three things that are kept apart:

* ``items``    label-free records: the text needed to compute features, the
               group, the split, and covariates that are known at deployment;
* ``labels``   the 0/1 loss and annotation summaries, for the open splits only;
* ``report``   counts, for the record.

Sealing. The labels of the CAL and TEST splits are not kept. A row of a sealed
split has its annotation fields removed in the statement that parses it, before
anything can read them, and no statistic of the sealed labels is computed.
Reading them is the job of the confirmatory run, behind its own guard.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from selfref.datasets import ragtruth
from selfref.dedup import near_duplicate_clusters, representatives
from selfref.guard import SealedAccessRefused
from selfref.splits import SEALED, SPLITS, assign_split

__all__ = ["SALT_RAGTRUTH", "SALT_WIKIBIO", "NEAR_DUPLICATE_THRESHOLD", "Built",
           "build_ragtruth", "build_wikibio", "sealed_ragtruth_labels", "sealed_wikibio_labels"]

#: Fixed salts. Changing one changes every split, so they are part of the pre-registration.
SALT_RAGTRUTH = "selfref-prereg-v1/ragtruth"
SALT_WIKIBIO = "selfref-prereg-v1/wikibio"
NEAR_DUPLICATE_THRESHOLD = 0.9

_ANNOTATION_FIELDS = ("labels", "quality")


@dataclass
class Built:
    items: list[dict]
    sources: dict                      # RAGTruth: source identifier -> SOURCE reference bytes
                                       # WikiBio: passage identifier -> list of resample bytes
    labels: dict[str, dict]            # item identifier -> loss and annotation summaries (open splits)
    report: dict


def _ragtruth_rows(response_path: Path, source_path: Path, salt: str):
    """The unreserved rows that survive near-duplicate removal, each with its
    source identifier and split. Shared by the build and by the sealed reader,
    so that both see exactly the same rows in the same splits."""
    raw_rows, n_reserved = ragtruth.open_response_lines(response_path)
    wanted = {ragtruth.source_id_of_line(raw) for raw in raw_rows}
    sources = ragtruth.read_sources(source_path, wanted)
    clusters = near_duplicate_clusters(
        {sid: ragtruth.dedup_text(src) for sid, src in sources.items()},
        threshold=NEAR_DUPLICATE_THRESHOLD)
    kept_as = representatives(sources, clusters)
    dropped_sources = {sid for sid, rep in kept_as.items() if sid != rep}
    kept = []
    for raw in raw_rows:
        source_id = ragtruth.source_id_of_line(raw)
        if source_id not in dropped_sources:
            kept.append((raw, source_id, assign_split(source_id, salt)))
    return kept, sources, clusters, dropped_sources, n_reserved, len(raw_rows)


def _ragtruth_label(row: dict) -> dict:
    spans = row["labels"]
    return {"loss": ragtruth.response_loss(row),
            "n_spans": len(spans),
            "n_spans_not_implicit_true": sum(1 for s in spans if not s.get("implicit_true")),
            "quality": row.get("quality")}


def _require(authorisation, split: str) -> None:
    """A sealed split is read only with the entry the guard returned for that split."""
    if split not in SEALED:
        raise ValueError(f"{split} is not a sealed split")
    if not isinstance(authorisation, dict) or authorisation.get("split") != split \
            or not authorisation.get("commit"):
        raise SealedAccessRefused(f"reading {split} labels needs the guard's authorisation "
                                  f"for that split")


def sealed_ragtruth_labels(response_path: Path, source_path: Path, split: str, authorisation,
                           salt: str = SALT_RAGTRUTH) -> dict[str, dict]:
    """Labels of one sealed RAGTruth split. For the confirmatory run only:
    ``authorisation`` is the entry returned by ``selfref.guard.authorise``."""
    _require(authorisation, split)
    kept, *_ = _ragtruth_rows(response_path, source_path, salt)
    labels = {}
    for raw, _, row_split in kept:
        if row_split == split:
            row = ragtruth.parse_response(raw)
            labels[str(row["id"])] = _ragtruth_label(row)
    return labels


def build_ragtruth(response_path: Path, source_path: Path,
                   salt: str = SALT_RAGTRUTH) -> Built:
    """Build items, open-split labels and a report from the two RAGTruth files."""
    kept, sources, clusters, dropped_sources, n_reserved, n_open = _ragtruth_rows(
        response_path, source_path, salt)

    items: list[dict] = []
    labels: dict[str, dict] = {}
    for raw, source_id, split in kept:
        row = ragtruth.parse_response(raw)
        if split in SEALED:
            for name in _ANNOTATION_FIELDS:       # sealed: discard before anything reads them
                row.pop(name, None)
        item_id = str(row["id"])
        source = sources[source_id]
        items.append({
            "id": item_id, "source_id": source_id, "group": source_id, "split": split,
            "model": row["model"], "temperature": row.get("temperature"),
            "task_type": source["task_type"], "corpus": source.get("source"),
            "response": row["response"],
        })
        if split not in SEALED:
            labels[item_id] = _ragtruth_label(row)

    kept_sources = {sid: ragtruth.source_reference(src) for sid, src in sources.items()
                    if sid not in dropped_sources}
    return Built(items, kept_sources, labels,
                 _report(items, labels, n_reserved, n_open, clusters,
                         dropped_sources, n_open - len(kept)))


def _report(items, labels, n_reserved, n_open_rows, clusters, dropped_sources, dropped_rows) -> dict:
    items_per_split = Counter(item["split"] for item in items)
    groups_per_split: dict[str, set] = defaultdict(set)
    for item in items:
        groups_per_split[item["split"]].add(item["group"])
    open_splits = [s for s in SPLITS if s not in SEALED]
    by_split: dict[str, list[int]] = defaultdict(list)
    split_of = {item["id"]: item["split"] for item in items}
    for item_id, label in labels.items():
        by_split[split_of[item_id]].append(label["loss"])
    return {
        "reserved_rows": n_reserved,
        "open_rows_in_file": n_open_rows,
        "near_duplicate_clusters": len(clusters),
        "sources_dropped_as_duplicates": len(dropped_sources),
        "responses_dropped_with_duplicate_sources": dropped_rows,
        "items": len(items),
        "groups": len({item["group"] for item in items}),
        "items_per_split": {s: items_per_split.get(s, 0) for s in SPLITS},
        "groups_per_split": {s: len(groups_per_split.get(s, ())) for s in SPLITS},
        "loss_rate_per_open_split": {
            s: (round(sum(by_split[s]) / len(by_split[s]), 4) if by_split[s] else None)
            for s in open_splits},
    }


_WIKIBIO_ANNOTATIONS = ("accurate", "minor_inaccurate", "major_inaccurate")


def build_wikibio(passages: list[dict], salt: str = SALT_WIKIBIO) -> Built:
    """Sentence-level items from the WikiBio GPT-3 hallucination dataset
    (Manakul, Liusie and Gales, arXiv:2303.08896).

    One item per annotated sentence. The group is the passage. The references
    of the RESAMPLE frame are the passage's stochastic samples. The 0/1 loss of
    a sentence is 1 unless it is labelled ``accurate``, which is the paper's
    grouping of major and minor inaccuracies into the non-factual class.
    """
    texts = {str(row["wiki_bio_test_idx"]): row["gpt3_text"] for row in passages}
    clusters = near_duplicate_clusters(texts, threshold=NEAR_DUPLICATE_THRESHOLD)
    kept_as = representatives(texts, clusters)
    dropped = {group for group, representative in kept_as.items() if group != representative}

    items, labels, sources = [], {}, {}
    for row in passages:
        group = str(row["wiki_bio_test_idx"])
        if group in dropped:
            continue
        sentences, annotations = list(row["gpt3_sentences"]), list(row["annotation"])
        if len(sentences) != len(annotations):
            raise ValueError(f"passage {group}: {len(sentences)} sentences, "
                             f"{len(annotations)} annotations")
        if any(a not in _WIKIBIO_ANNOTATIONS for a in annotations):
            raise ValueError(f"passage {group}: unknown annotation value")
        split = assign_split(group, salt)
        if split in SEALED:
            annotations = None                      # sealed: discard before anything reads them
        sources[group] = [sample.encode("utf-8") for sample in row["gpt3_text_samples"]]
        for position, sentence in enumerate(sentences):
            item_id = f"{group}:{position}"
            items.append({"id": item_id, "group": group, "split": split, "position": position,
                          "n_sentences": len(sentences), "sentence": sentence})
            if annotations is not None:
                labels[item_id] = {"loss": int(annotations[position] != "accurate"),
                                   "major": int(annotations[position] == "major_inaccurate")}
    return Built(items, sources, labels,
                 _report(items, labels, 0, len(passages), clusters, dropped, 0))


def sealed_wikibio_labels(passages: list[dict], split: str, authorisation,
                          salt: str = SALT_WIKIBIO) -> dict[str, dict]:
    """Labels of one sealed WikiBio split. For the confirmatory run only."""
    _require(authorisation, split)
    texts = {str(row["wiki_bio_test_idx"]): row["gpt3_text"] for row in passages}
    kept_as = representatives(texts, near_duplicate_clusters(texts, threshold=NEAR_DUPLICATE_THRESHOLD))
    labels = {}
    for row in passages:
        group = str(row["wiki_bio_test_idx"])
        if kept_as[group] != group or assign_split(group, salt) != split:
            continue
        for position, annotation in enumerate(row["annotation"]):
            if annotation not in _WIKIBIO_ANNOTATIONS:
                raise ValueError(f"passage {group}: unknown annotation value")
            labels[f"{group}:{position}"] = {"loss": int(annotation != "accurate"),
                                             "major": int(annotation == "major_inaccurate")}
    return labels
