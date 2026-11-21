"""Find secrets in text that is about to be committed.

A finding names the file, the line and the kind of secret, never the value.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass

MIN_SECRET_LENGTH = 8
PATTERNS = {
    "a Hugging Face token": re.compile(r"\bhf_[A-Za-z0-9]{30,}\b"),
    "a private key": re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
}


@dataclass(frozen=True)
class Leak:
    path: str
    line: int
    kind: str


def find_leaks(path: str, text: str, secrets: Mapping[str, str]) -> list[Leak]:
    known = {name: value for name, value in secrets.items() if len(value) >= MIN_SECRET_LENGTH}
    leaks: list[Leak] = []
    for number, line in enumerate(text.splitlines(), start=1):
        kinds = [f"the value of {name}" for name, value in known.items() if value in line]
        kinds += [kind for kind, pattern in PATTERNS.items() if pattern.search(line)]
        leaks.extend(Leak(path, number, kind) for kind in kinds)
    return leaks


def describe(leak: Leak) -> str:
    return f"{leak.path}:{leak.line} contains {leak.kind}"
