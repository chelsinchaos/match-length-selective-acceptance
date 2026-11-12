# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LogNovations Holdings LLC
"""Reader for RAGTruth (Niu et al., arXiv:2401.00396; ParticleMedia/RAGTruth).

Two files: ``response.jsonl`` (one LLM response per line, with its human span
annotations and the authors' ``split``) and ``source_info.jsonl`` (one source
per line, with the prompt the models were given).

Reserved rows. The authors' ``test`` split is the reserved replication set of
this study. A row's split is decided from its raw bytes by a regular
expression, and a row is parsed as JSON only when it is not reserved. Nothing
about a reserved row other than its split is read, and only their number is
kept. Sources are handled the same way: a source line is parsed only when its
identifier, also read from the raw bytes, belongs to an unreserved response.

Why reading a field from raw bytes is exact. Inside a JSON string every double
quote is written as a backslash followed by a quote. So the byte sequence
``"split"`` followed by a colon can only be a key, never part of a response's
text. A line with no such key, more than one, or an unknown value is an error.

Definitions used by the study

* The SOURCE reference of a response is the prompt its model was given, as
  UTF-8 bytes. It is the same for every task type and for every baseline.
* The 0/1 loss of a response is 1 when it has at least one annotated span of
  any type, and 0 otherwise. No row is excluded. This is the dataset's own
  count of responses with hallucinations.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

__all__ = ["RESERVED_SPLIT", "ReservedRowError", "split_of_line", "source_id_of_line",
           "open_response_lines", "parse_response", "read_sources", "source_reference",
           "dedup_text", "response_loss"]

#: The authors' split that this study reserves, unloaded, for replication.
RESERVED_SPLIT = "test"
_KNOWN_SPLITS = {"train", "test"}

_SPLIT = re.compile(rb'(?<!\\)"split"\s*:\s*"([^"\\]*)"')
_SOURCE_ID = re.compile(rb'(?<!\\)"source_id"\s*:\s*"?([^",\s}]+)"?')


class ReservedRowError(RuntimeError):
    """An attempt was made to parse a row of the reserved replication set."""


def _single(pattern: re.Pattern, raw: bytes, what: str) -> str:
    found = pattern.findall(raw)
    if len(found) != 1:
        raise ValueError(f"expected exactly one {what} field in the line, found {len(found)}")
    return found[0].decode("utf-8")


def split_of_line(raw: bytes) -> str:
    """The value of a response line's ``split`` field, read without parsing the line."""
    value = _single(_SPLIT, raw, "split")
    if value not in _KNOWN_SPLITS:
        raise ValueError(f"unknown split {value!r}")
    return value


def source_id_of_line(raw: bytes) -> str:
    """The value of a line's ``source_id`` field, read without parsing the line."""
    return _single(_SOURCE_ID, raw, "source_id")


def open_response_lines(path: Path) -> tuple[list[bytes], int]:
    """The raw lines of every unreserved response, and the number of reserved lines."""
    kept: list[bytes] = []
    reserved = 0
    with open(path, "rb") as handle:
        for line in handle:
            line = line.rstrip(b"\r\n")
            if not line:
                continue
            if split_of_line(line) == RESERVED_SPLIT:
                reserved += 1
            else:
                kept.append(line)
    return kept, reserved


def parse_response(raw: bytes) -> dict:
    """Parse one response line. Refuses a reserved row."""
    if split_of_line(raw) == RESERVED_SPLIT:
        raise ReservedRowError("this row belongs to the reserved replication set")
    return json.loads(raw)


def read_sources(path: Path, wanted: set[str]) -> dict[str, dict]:
    """Parse the sources whose identifiers are in ``wanted``, and no others."""
    found: dict[str, dict] = {}
    with open(path, "rb") as handle:
        for line in handle:
            line = line.rstrip(b"\r\n")
            if line and source_id_of_line(line) in wanted:
                source = json.loads(line)
                found[str(source["source_id"])] = source
    missing = wanted - set(found)
    if missing:
        raise KeyError(f"{len(missing)} wanted sources are absent, e.g. {sorted(missing)[:3]}")
    return found


def source_reference(source: dict) -> bytes:
    """The SOURCE reference: the prompt the model was given, as UTF-8 bytes."""
    return source["prompt"].encode("utf-8")


def dedup_text(source: dict) -> str:
    """The text compared when looking for near-duplicate sources.

    It is the source content, not the prompt, so that the instruction template
    shared by every prompt of a task does not make unrelated sources look alike.
    Structured sources are serialised with sorted keys.
    """
    content = source["source_info"]
    if isinstance(content, str):
        return content
    return json.dumps(content, sort_keys=True, ensure_ascii=False)


def response_loss(row: dict) -> int:
    """1 when the response has at least one annotated span, else 0."""
    return int(len(row["labels"]) > 0)
