# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LogNovations Holdings LLC
"""The guard on the sealed splits.

Reading CAL or TEST labels requires an explicit flag, a ``prereg-v1`` tag whose
pre-registration is unchanged at HEAD, and a clean tree. Every access is
appended to a ledger. Each test builds a throwaway git repository.
"""
from __future__ import annotations

import subprocess

import pytest

from selfref.guard import (PREREG_TAG, SealedAccessRefused, access_count, authorise)


def git(repo, *args):
    return subprocess.run(["git", "-C", str(repo), "-c", "user.name=t",
                           "-c", "user.email=t@example.invalid", *args],
                          check=True, capture_output=True, text=True).stdout.strip()


@pytest.fixture
def repo(tmp_path):
    git(tmp_path, "init", "-q")
    (tmp_path / "PREREGISTRATION.md").write_text("# Pre-registration\nfixed plan\n")
    (tmp_path / "RUNLEDGER.md").write_text("# Run ledger\n")
    git(tmp_path, "add", ".")
    git(tmp_path, "commit", "-q", "-m", "prereg")
    return tmp_path


def tag(repo):
    git(repo, "tag", PREREG_TAG)


def ask(repo, split="TEST", flag=True, purpose="confirmatory run"):
    return authorise(split, purpose, flag=flag, repo=repo, ledger=repo / "RUNLEDGER.md")


def test_the_tag_name_is_the_one_the_brief_fixes():
    assert PREREG_TAG == "prereg-v1"


def test_access_is_refused_without_the_tag(repo):
    with pytest.raises(SealedAccessRefused, match="prereg-v1"):
        ask(repo)
    assert access_count(repo / "RUNLEDGER.md") == 0


def test_access_is_refused_without_the_explicit_flag(repo):
    tag(repo)
    with pytest.raises(SealedAccessRefused, match="flag"):
        ask(repo, flag=False)
    assert access_count(repo / "RUNLEDGER.md") == 0


def test_access_is_granted_with_tag_and_flag_and_is_written_to_the_ledger(repo):
    tag(repo)
    entry = ask(repo, split="TEST", purpose="confirmatory run")
    assert entry["split"] == "TEST" and entry["access_number"] == 1
    assert entry["commit"] == git(repo, "rev-parse", "HEAD")
    ledger = (repo / "RUNLEDGER.md").read_text()
    assert "TEST" in ledger and "confirmatory run" in ledger and entry["commit"] in ledger


def test_every_access_is_counted_per_split(repo):
    tag(repo)
    ask(repo, split="CAL")
    git(repo, "commit", "-q", "-am", "ledger")           # the ledger itself is committed between runs
    ask(repo, split="TEST")
    git(repo, "commit", "-q", "-am", "ledger")
    second = ask(repo, split="TEST")
    assert second["access_number"] == 2
    assert access_count(repo / "RUNLEDGER.md") == 3
    assert access_count(repo / "RUNLEDGER.md", "TEST") == 2
    assert access_count(repo / "RUNLEDGER.md", "CAL") == 1


def test_access_is_refused_if_the_preregistration_changed_after_the_tag(repo):
    tag(repo)
    (repo / "PREREGISTRATION.md").write_text("# Pre-registration\na different plan\n")
    git(repo, "commit", "-q", "-am", "edit the plan after tagging")
    with pytest.raises(SealedAccessRefused, match="changed"):
        ask(repo)


def test_access_is_refused_on_a_dirty_tree(repo):
    tag(repo)
    (repo / "analysis.py").write_text("print('uncommitted')\n")
    with pytest.raises(SealedAccessRefused, match="uncommitted"):
        ask(repo)


def test_access_is_refused_when_head_does_not_descend_from_the_tag(repo):
    tag(repo)
    git(repo, "checkout", "-q", "--orphan", "elsewhere")
    git(repo, "commit", "-q", "-m", "unrelated history")
    with pytest.raises(SealedAccessRefused, match="descend"):
        ask(repo)


def test_only_sealed_splits_go_through_the_guard(repo):
    tag(repo)
    with pytest.raises(ValueError):
        ask(repo, split="TRAIN")


def test_a_purpose_is_required(repo):
    tag(repo)
    with pytest.raises(ValueError):
        ask(repo, purpose="  ")


def test_several_splits_can_be_authorised_in_one_checked_step(repo):
    """The confirmatory run needs CAL and TEST together. Authorising them one
    after the other would fail, because the first ledger row makes the tree dirty."""
    from selfref.guard import authorise_all
    tag(repo)
    entries = authorise_all(("CAL", "TEST"), "confirmatory run", flag=True, repo=repo,
                            ledger=repo / "RUNLEDGER.md")
    assert [e["split"] for e in entries] == ["CAL", "TEST"]
    assert [e["number"] for e in entries] == [1, 2]
    assert access_count(repo / "RUNLEDGER.md", "TEST") == 1
    assert access_count(repo / "RUNLEDGER.md", "CAL") == 1


def test_a_refused_joint_request_writes_nothing(repo):
    from selfref.guard import authorise_all
    with pytest.raises(SealedAccessRefused):
        authorise_all(("CAL", "TEST"), "confirmatory run", flag=True, repo=repo,
                      ledger=repo / "RUNLEDGER.md")
    with pytest.raises(ValueError):
        tag(repo)
        authorise_all(("CAL", "DEV_TEST"), "confirmatory run", flag=True, repo=repo,
                      ledger=repo / "RUNLEDGER.md")
    assert access_count(repo / "RUNLEDGER.md") == 0
