# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LogNovations Holdings LLC
"""Render a report template from results.json.

The report is written as a template. Every number in it is a placeholder,

    {{ record > key > key | format }}

a path into ``results.json`` followed by an optional Python format
specification. The first part of a path names a record (``confirmatory/result.json``)
or one of the top-level sections ``ledger``, ``provenance`` and ``absent``. A
value that is ``None`` is shown as a dash. ``{{ table: name }}`` is replaced by
a table built from the results by the function of that name. A path that does
not exist is an error, so the report cannot quote a number the results do not
hold.
"""
from __future__ import annotations

import re
from typing import Callable, Mapping, Optional, Sequence

__all__ = ["lookup", "show", "md_table", "render"]

_PLACEHOLDER = re.compile(r"\{\{(.+?)\}\}", re.DOTALL)
_TOP_LEVEL = ("ledger", "provenance", "absent")
DASH = "—"


def lookup(results: Mapping, path: str):
    """The value at ``record > key > ...`` in the results."""
    parts = [part.strip() for part in path.split(">")]
    node = results[parts[0]] if parts[0] in _TOP_LEVEL else results["records"][parts[0]]
    for part in parts[1:]:
        node = node[int(part)] if isinstance(node, list) else node[part]
    return node


def show(value, spec: str = "") -> str:
    """One value as text: a dash for None, otherwise formatted with ``spec``."""
    if value is None:
        return DASH
    return format(value, spec) if spec else str(value)


def md_table(headers: Sequence[str], rows: Sequence[Sequence]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    lines += ["| " + " | ".join(show(cell) for cell in row) + " |" for row in rows]
    return "\n".join(lines)


def render(template: str, results: Mapping,
           tables: Optional[Mapping[str, Callable[[Mapping], str]]] = None) -> str:
    """The template with every placeholder replaced from the results."""
    def replace(match: re.Match) -> str:
        body = match.group(1).strip()
        if body.startswith("table:"):
            return (tables or {})[body[len("table:"):].strip()](results)
        path, _, spec = body.partition("|")
        return show(lookup(results, path), spec.strip())
    return _PLACEHOLDER.sub(replace, template)
