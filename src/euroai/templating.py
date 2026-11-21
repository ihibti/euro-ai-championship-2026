"""Fill `<UPPER_CASE>` placeholders in job/config templates.

Rendering fails while any placeholder is left unfilled, so an unedited
template cannot be submitted by accident. A placeholder mapped to the empty
string removes every line that contains it (e.g. no reservation).
"""

from __future__ import annotations

import re
from collections.abc import Mapping

PLACEHOLDER = re.compile(r"<([A-Z][A-Z0-9_]*)>")


def find_placeholders(text: str) -> set[str]:
    return set(PLACEHOLDER.findall(text))


def render(text: str, values: Mapping[str, str]) -> str:
    unknown = set(values) - find_placeholders(text)
    if unknown:
        raise KeyError(f"values given for placeholders not in template: {sorted(unknown)}")
    lines = []
    for line in text.splitlines(keepends=True):
        names = find_placeholders(line)
        if any(values.get(name) == "" for name in names):
            continue
        lines.append(PLACEHOLDER.sub(lambda m: values.get(m.group(1), m.group(0)), line))
    rendered = "".join(lines)
    missing = find_placeholders(rendered)
    if missing:
        raise ValueError(f"unfilled placeholders: {sorted(missing)}")
    return rendered
