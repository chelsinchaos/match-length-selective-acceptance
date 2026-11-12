# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LogNovations Holdings LLC
"""Feature functions cannot see labels.

Enforced three ways:

1. By interface. ``extract_features`` accepts only a ``FeatureInput``, which has
   three fields (output, source, resamples), is immutable, and has no place to
   attach anything else. It carries no identifier, because identifiers can
   encode labels.
2. By static inspection. No public callable in the feature modules has a
   parameter that could carry a label, no identifier in their source mentions
   one, and they import nothing outside a short allowlist, so they cannot reach
   a data or label module.
3. By behaviour. Recomputing features after the labels stored beside the items
   are permuted gives identical output.
"""
from __future__ import annotations

import ast
import dataclasses
import inspect
import random
from pathlib import Path

import pytest

import selfref.baselines
import selfref.extract
import selfref.features
import selfref.groundedness
import selfref.matchlen
import selfref.ncd
import selfref.selfcheck
from selfref.extract import FeatureInput, extract_features

# Every module that computes a score from an item: the features under test and the baselines.
FEATURE_MODULES = [selfref.matchlen, selfref.features, selfref.ncd, selfref.extract,
                   selfref.baselines, selfref.groundedness, selfref.selfcheck]
FORBIDDEN = ("label", "loss", "gold", "truth", "target_y", "halluc", "annotat")
ALLOWED_IMPORTS = {"__future__", "dataclasses", "gzip", "math", "operator", "pathlib", "re",
                   "statistics", "typing", "zstandard", "numpy",
                   "nltk", "nltk.tokenize", "torch", "transformers",
                   "selfref.matchlen", "selfref.features", "selfref.ncd"}


# ------------------------------------------------------------------ interface

def test_feature_input_has_exactly_three_fields():
    assert [f.name for f in dataclasses.fields(FeatureInput)] == ["output", "source", "resamples"]


def test_feature_input_rejects_an_unknown_field():
    with pytest.raises(TypeError):
        FeatureInput(output=b"x", label=1)


def test_feature_input_cannot_have_an_attribute_attached():
    item = FeatureInput(output=b"x")
    # Python 3.11 raises TypeError here and 3.12+ raises FrozenInstanceError (an
    # AttributeError); the difference is a CPython detail of frozen, slotted
    # dataclasses. What matters is the post-condition below.
    with pytest.raises((AttributeError, TypeError)):
        item.label = 1
    with pytest.raises(AttributeError):
        object.__setattr__(item, "label", 1)          # no instance dictionary to write to
    with pytest.raises(AttributeError):
        item.output = b"y"                            # existing fields are read-only too
    assert not hasattr(item, "label")
    assert not hasattr(item, "__dict__")
    assert item.output == b"x"


def test_a_subclass_carrying_extra_fields_is_not_accepted():
    @dataclasses.dataclass(frozen=True)
    class WithLabel(FeatureInput):
        label: int = 0

    with pytest.raises(TypeError):
        extract_features(WithLabel(output=b"x", label=1))


# ---------------------------------------------------------- static inspection

def public_callables(module):
    for name, obj in vars(module).items():
        if name.startswith("_") or getattr(obj, "__module__", None) != module.__name__:
            continue
        if inspect.isfunction(obj):
            yield f"{module.__name__}.{name}", obj
        elif inspect.isclass(obj):
            for method_name, method in vars(obj).items():
                if inspect.isfunction(method):
                    yield f"{module.__name__}.{name}.{method_name}", method


def test_no_public_callable_has_a_parameter_that_could_carry_a_label():
    checked = 0
    for qualified, fn in (pair for m in FEATURE_MODULES for pair in public_callables(m)):
        for parameter in inspect.signature(fn).parameters:
            checked += 1
            assert not any(word in parameter.lower() for word in FORBIDDEN), \
                f"{qualified} has parameter {parameter!r}"
    assert checked > 20          # the scan really did visit the feature code


@pytest.mark.parametrize("module", FEATURE_MODULES, ids=lambda m: m.__name__)
def test_no_identifier_in_the_feature_code_mentions_a_label(module):
    tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.arg):
            names.add(node.arg)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
        elif isinstance(node, ast.keyword) and node.arg:
            names.add(node.arg)
        elif isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            names.add(node.name)
    offenders = sorted(n for n in names if any(word in n.lower() for word in FORBIDDEN))
    assert not offenders, f"{module.__name__} uses identifiers {offenders}"


@pytest.mark.parametrize("module", FEATURE_MODULES, ids=lambda m: m.__name__)
def test_feature_code_imports_only_from_the_allowlist(module):
    tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0, "relative imports hide the module name from this check"
            imported.add(node.module)
    assert imported <= ALLOWED_IMPORTS, f"{module.__name__} imports {sorted(imported - ALLOWED_IMPORTS)}"


# ------------------------------------------------------------------ behaviour

def test_features_do_not_change_when_labels_are_permuted():
    rng = random.Random(7)
    words = [b"alpha", b"beta", b"gamma", b"delta", b"store", b"opens", b"nine"]
    rows = []
    for _ in range(25):
        source = b" ".join(rng.choices(words, k=30))
        output = b" ".join(rng.choices(words, k=12))
        rows.append({"input": FeatureInput(output=output, source=source),
                     "label": rng.randint(0, 1)})
    before = [extract_features(row["input"]) for row in rows]

    labels = [row["label"] for row in rows]
    rng.shuffle(labels)
    for row, label in zip(rows, labels):
        row["label"] = label
    after = [extract_features(row["input"]) for row in rows]

    assert before == after
