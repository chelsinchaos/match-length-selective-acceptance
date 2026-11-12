# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LogNovations Holdings LLC
"""The paper is rendered from results.json, like the report."""
from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path

import pytest

from selfref.render import render

HARNESS = Path(__file__).resolve().parents[1]
PAPER = HARNESS / "paper"


def make_paper():
    spec = importlib.util.spec_from_file_location("make_paper", HARNESS / "scripts" / "make_paper.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def loaded():
    template, results = PAPER / "paper.template.tex", HARNESS / "results.json"
    if not (template.exists() and results.exists()):
        pytest.skip("the paper has not been written yet")
    return template.read_text(), json.loads(results.read_text())


def test_the_committed_paper_is_exactly_the_template_rendered_from_results_json():
    template, results = loaded()
    if not (PAPER / "paper.tex").exists():
        pytest.skip("the paper has not been rendered yet")
    assert (PAPER / "paper.tex").read_text() == render(template, results, make_paper().TABLES)


def test_the_rendered_paper_has_no_placeholder_left_and_no_percent_sign_from_a_number():
    template, results = loaded()
    text = render(template, results, make_paper().TABLES)
    assert "{{" not in text and "@@" not in text
    # an unescaped % would silently comment out the rest of a line of LaTeX
    body = text.split(r"\begin{document}", 1)[1]
    assert not re.search(r"(?<!\\)%", body)


def test_the_paper_renders_without_the_record_of_the_reproduction_check():
    """A reproduction from the public tree has no record of an earlier reproduction check."""
    template, results = loaded()
    results["records"].pop("reproduction/check.json", None)
    assert "not in this tree" in render(template, results, make_paper().TABLES)


def test_a_number_in_latex_text_has_a_true_minus_sign_and_none_is_a_dash():
    num = make_paper().num
    assert num(-0.8207, "+.2f") == "$-0.82$"
    assert num(0.3157, "+.2f") == "$+0.32$"
    assert num(8.8384) == "8.84"
    assert num(None) == "--"


def test_names_from_the_records_are_escaped_for_latex():
    assert make_paper().tex("source.cov_k8 50% A&B") == r"source.cov\_k8 50\% A\&B"


def test_the_paper_keeps_to_the_agreed_wording():
    """A score for selective acceptance with an error bound that holds under
    exchangeability; never a claim to verify truth. FAIL is never read as no signal."""
    template, _ = loaded()
    text = " ".join(template.split())
    assert "It does not verify that any particular output is true." in text
    assert "holds under exchangeability" in text
    assert "not that the features carry no information" in text
    for phrase in ("verifies truth", "verifies that an output is true", "proves that the features",
                   "the features carry no signal."):
        assert phrase not in text.replace("would not be described as evidence that the features carry no signal.", "")


def test_the_pdf_of_the_paper_and_its_figures_are_tracked():
    """A parent repository ignored every PDF, and the first commit of the paper left them out."""
    import subprocess
    if not (PAPER / "paper.pdf").exists():
        pytest.skip("the paper has not been compiled yet")
    inside = subprocess.run(["git", "-C", str(HARNESS), "rev-parse", "--is-inside-work-tree"],
                            capture_output=True, text=True)
    if inside.returncode != 0:
        pytest.skip("no git here")
    names = ["paper/paper.pdf"] + [f"paper/figures/{p.name}" for p in sorted((PAPER / "figures").glob("*.pdf"))]
    assert len(names) == 5
    ignored = subprocess.run(["git", "-C", str(HARNESS), "check-ignore", *names], capture_output=True, text=True)
    assert ignored.stdout.strip() == "", ignored.stdout
