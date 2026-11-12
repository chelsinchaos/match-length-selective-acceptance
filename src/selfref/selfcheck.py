# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LogNovations Holdings LLC
"""Consistency baseline: SelfCheckGPT with NLI (Manakul, Liusie and Gales, EMNLP 2023).

A sentence is scored by how strongly other samples drawn for the same prompt
contradict it. This is a baseline for the RESAMPLE frame, and reproducing its
published result is the study's fidelity check. It follows the authors'
reference code (github.com/potsawee/selfcheckgpt, commit 19b492a2,
``SelfCheckNLI.predict``):

* checkpoint ``potsawee/deberta-v3-large-mnli`` at a pinned revision, a
  two-output model whose outputs are entailment and contradiction;
* each pair is (sentence, sample), in that order, truncated by the tokenizer;
* the score of a pair is the softmax of the two logits at index 1, the
  probability of contradiction;
* the score of a sentence is the mean over its samples.

Higher means less consistent with the samples.
"""
from __future__ import annotations

from typing import Sequence

__all__ = ["CHECKPOINT", "REVISION", "pairs_for", "mean_contradiction", "SelfCheckNLI"]

CHECKPOINT = "potsawee/deberta-v3-large-mnli"
REVISION = "83f4359862aba52c07506cd23d4454db77945b36"


def pairs_for(sentence: str, samples: Sequence[str]) -> list[tuple[str, str]]:
    """The (sentence, sample) text pairs, sentence first, as in the reference code."""
    return [(sentence, sample) for sample in samples]


def mean_contradiction(probabilities: Sequence[float]) -> float:
    """The sentence score: the mean contradiction probability over samples."""
    if not probabilities:
        raise ValueError("a sentence needs at least one sample to be scored")
    return float(sum(probabilities) / len(probabilities))


class SelfCheckNLI:
    """Loads the pinned checkpoint and scores sentences against samples."""

    def __init__(self, device: str | None = None, batch_size: int = 20) -> None:
        import torch
        from transformers import DebertaV2ForSequenceClassification, DebertaV2Tokenizer

        self._torch = torch
        if device is None:
            device = "mps" if torch.backends.mps.is_available() else \
                ("cuda" if torch.cuda.is_available() else "cpu")
        self.device, self.batch_size = device, batch_size
        self.tokenizer = DebertaV2Tokenizer.from_pretrained(CHECKPOINT, revision=REVISION)
        self.model = DebertaV2ForSequenceClassification.from_pretrained(
            CHECKPOINT, revision=REVISION).to(device).eval()

    def contradiction(self, sentence: str, samples: Sequence[str]) -> list[float]:
        """The contradiction probability of ``sentence`` against each sample."""
        torch, pairs, out = self._torch, pairs_for(sentence, samples), []
        with torch.inference_mode():
            for start in range(0, len(pairs), self.batch_size):
                inputs = self.tokenizer.batch_encode_plus(
                    batch_text_or_text_pairs=pairs[start:start + self.batch_size],
                    add_special_tokens=True, padding="longest", truncation=True,
                    return_tensors="pt", return_token_type_ids=True, return_attention_mask=True)
                logits = self.model(**inputs.to(self.device)).logits.float()
                if logits.shape[1] != 2:
                    raise RuntimeError("the checkpoint is expected to have two output classes")
                out.extend(torch.softmax(logits, dim=-1)[:, 1].cpu().tolist())
        return out
