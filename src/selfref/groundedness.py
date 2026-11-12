# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LogNovations Holdings LLC
"""Groundedness baseline: MiniCheck (Tang, Laban and Durrett, EMNLP 2024).

An open model trained to decide whether a claim is supported by a document.
It is a baseline here, and it is used the way its authors' reference
inference code uses it (github.com/Liyan06/MiniCheck, commit b58b9fa6,
``minicheck/inference.py``), so that the baseline is not weakened by a
home-made variant:

* checkpoint ``lytang/MiniCheck-DeBERTa-v3-Large`` at a pinned revision;
* the document is split into sentences line by line, and sentences are packed
  into chunks of at most 400 tokenizer tokens, never splitting a sentence;
* each (chunk, claim) pair is the chunk and the claim joined by the
  tokenizer's end-of-sequence token, truncated at 2048 tokens;
* the support probability of a pair is the softmax of the two logits at
  index 1, and the support of a claim is the maximum over chunks.

The reference code checks one claim at a time and advises splitting a
multi-sentence response into sentences, leaving the aggregation to the user.
Here the claims of a response are its sentences, and three summaries are
reported: the minimum support, the mean support, and the fraction of sentences
with support below 0.5.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Sequence

__all__ = ["CHECKPOINT", "REVISION", "CHUNK_TOKENS", "MAX_MODEL_LEN",
           "sentences_with_newlines", "claims_of", "chunk_document",
           "summarise_support", "MiniCheckScorer"]

CHECKPOINT = "lytang/MiniCheck-DeBERTa-v3-Large"
REVISION = "2f2d01a54fa022a7ffadb76260e1ea8bc88c82bb"
CHUNK_TOKENS = 400
MAX_MODEL_LEN = 2048


def _sent_tokenize(text: str) -> list[str]:
    import nltk
    from nltk.tokenize import sent_tokenize

    # `make venv` puts the sentence-tokenizer data inside the harness environment.
    local = Path(__file__).resolve().parents[2] / ".venv" / "nltk_data"
    if local.is_dir() and str(local) not in nltk.data.path:
        nltk.data.path.insert(0, str(local))
    return sent_tokenize(text)


def sentences_with_newlines(text: str) -> list[str]:
    """Sentences of ``text``, line by line, with a ``"\\n"`` item between lines."""
    out: list[str] = []
    for block in text.split("\n"):
        out.extend(_sent_tokenize(block))
        out.append("\n")
    return out[:-1]


def claims_of(text: str) -> list[str]:
    """The sentences of a response: the units checked against the document."""
    return [s for s in sentences_with_newlines(text) if s.strip()]


def chunk_document(sentences: Sequence[str], count_tokens: Callable[[str], int],
                   chunk_tokens: int = CHUNK_TOKENS) -> list[str]:
    """Pack sentences into chunks of about ``chunk_tokens`` tokens.

    A sentence that would take the running count past the budget starts a new
    chunk. This is the reference procedure, including the empty chunk it emits
    when the very first sentence is over budget, which is then dropped.
    """
    chunks: list[str] = []
    current: list[str] = []
    count = 0
    for sentence in sentences:
        size = count_tokens(sentence)
        if count + size > chunk_tokens:
            chunks.append(" ".join(current))
            current, count = [sentence], size
        else:
            current.append(sentence)
            count += size
    if current:
        chunks.append(" ".join(current))
    cleaned = [chunk.replace(" \n ", "\n").strip() for chunk in chunks]
    return [chunk for chunk in cleaned if chunk != ""]


def summarise_support(probabilities: Sequence[float]) -> dict[str, float]:
    """Minimum, mean and share-below-0.5 of per-sentence support probabilities.
    An output with no sentence makes no claim and is scored as fully supported."""
    if not probabilities:
        return {"gnd.min": 1.0, "gnd.mean": 1.0, "gnd.frac_unsupported": 0.0}
    n = len(probabilities)
    return {"gnd.min": float(min(probabilities)),
            "gnd.mean": float(sum(probabilities) / n),
            "gnd.frac_unsupported": float(sum(1 for p in probabilities if p < 0.5) / n)}


class MiniCheckScorer:
    """Loads the pinned checkpoint and scores claims against document chunks."""

    def __init__(self, device: str | None = None, batch_size: int = 8) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._torch = torch
        if device is None:
            device = "mps" if torch.backends.mps.is_available() else \
                ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT, revision=REVISION, use_fast=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            CHECKPOINT, revision=REVISION).to(device).eval()

    def count_tokens(self, sentence: str) -> int:
        return len(self.tokenizer(sentence, padding=False, add_special_tokens=False,
                                  max_length=MAX_MODEL_LEN, truncation=True)["input_ids"])

    def chunks_of(self, document: str) -> list[str]:
        return chunk_document(sentences_with_newlines(document) or [""], self.count_tokens)

    def support(self, chunks: Sequence[str], claims: Sequence[str]) -> list[float]:
        """For each claim, the maximum over chunks of the support probability."""
        if not claims:
            return []
        if not chunks:
            chunks = [""]
        separator = self.tokenizer.eos_token
        pairs = [(c, k, separator.join([chunk, claim]))
                 for c, claim in enumerate(claims) for k, chunk in enumerate(chunks)]
        order = sorted(range(len(pairs)), key=lambda i: len(pairs[i][2]))   # similar lengths together
        best = [0.0] * len(claims)
        torch = self._torch
        with torch.inference_mode():
            for start in range(0, len(order), self.batch_size):
                batch = [pairs[i] for i in order[start:start + self.batch_size]]
                inputs = self.tokenizer([text for _, _, text in batch], max_length=MAX_MODEL_LEN,
                                        truncation=True, padding=True, return_tensors="pt")
                inputs = {name: tensor.to(self.device) for name, tensor in inputs.items()}
                logits = self.model(**inputs).logits.float()
                if logits.shape[1] != 2:
                    raise RuntimeError("the checkpoint is expected to have two output classes")
                probabilities = torch.softmax(logits, dim=1)[:, 1]
                for (claim_index, _, _), p in zip(batch, probabilities.cpu().tolist()):
                    best[claim_index] = max(best[claim_index], p)
        return best
