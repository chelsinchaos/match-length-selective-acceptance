# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LogNovations Holdings LLC
"""The public tree stands alone.

Everything meant for publication (src/, tests/, bench/, scripts/ and the
top-level project files) must run with the open reference backend only, and
must not reach into, or describe, the proprietary codec. These checks fail if
a public file imports from it or names one of its internal identifiers.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

HARNESS = Path(__file__).resolve().parents[1]
PUBLIC_DIRS = ("src", "tests", "bench", "scripts")
PUBLIC_TOP_LEVEL = ("README.md", "REFERENCES.md", "PREREGISTRATION.md", "DEVIATIONS.md",
                    "RUNLEDGER.md",
                    "datasets.manifest.json", "pyproject.toml", "requirements.txt",
                    "LICENSE", ".gitignore")

# Identifiers that would describe proprietary internals. Assembled from parts so
# that this file does not itself contain them.
PRIVATE_IDENTIFIERS = ["fhc" + "_python", "fastcodec" + "_backend", "FHCBlock" + "Stream",
                       "GenomicRank" + "Engine", "fhc_block" + "_index", "Logn" + "Codec"]

# Module roots of the proprietary repository and of the private add-on.
PRIVATE_IMPORT_ROOTS = {PRIVATE_IDENTIFIERS[0], PRIVATE_IDENTIFIERS[1], "fhc", "codec",
                        "native", "video", "capture", "slam", "kit", "cert"}


def public_python_files():
    files = []
    for directory in PUBLIC_DIRS:
        files += [p for p in sorted((HARNESS / directory).rglob("*.py"))
                  if "__pycache__" not in p.parts]
    return files


def test_there_are_public_python_files_to_check():
    assert len(public_python_files()) >= 12


@pytest.mark.parametrize("path", public_python_files(), ids=lambda p: str(p.relative_to(HARNESS)))
def test_public_code_imports_nothing_from_the_proprietary_codec(path):
    imported = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert not imported & PRIVATE_IMPORT_ROOTS, sorted(imported & PRIVATE_IMPORT_ROOTS)


@pytest.mark.parametrize("path", public_python_files(), ids=lambda p: str(p.relative_to(HARNESS)))
def test_public_code_has_the_mit_header(path):
    head = path.read_text(encoding="utf-8").splitlines()[:3]
    assert "# SPDX-License-Identifier: MIT" in head
    assert "# Copyright (c) 2026 LogNovations Holdings LLC" in head


def test_no_public_file_names_a_proprietary_internal():
    candidates = public_python_files() + [HARNESS / name for name in PUBLIC_TOP_LEVEL
                                          if (HARNESS / name).exists()]
    offenders = []
    for path in candidates:
        text = path.read_text(encoding="utf-8")
        offenders += [f"{path.relative_to(HARNESS)}: {word}"
                      for word in PRIVATE_IDENTIFIERS if word in text]
    assert not offenders, offenders


def test_the_makefile_touches_the_private_tree_only_through_one_optional_include():
    lines = [line for line in (HARNESS / "Makefile").read_text(encoding="utf-8").splitlines()
             if any(word in line for word in PRIVATE_IDENTIFIERS)]
    assert lines == ["-include " + "fastcodec" + "_backend/private.mk"]
