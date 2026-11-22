"""Identity answers as a probe: each answer reduced to a class and a claimed name.

uv run python tasks/model_equality/analysis/identity.py \
    --answers tasks/model_equality/answers/round1.jsonl --prefix ep- \
    --families tasks/model_equality/analysis/recon_ep_families.csv

Frozen on 2026-11-22 at 01:15 Paris, before the 8 extra scored identity answers existed.

Per answer:
- name: the first capitalised name after "my name is", "I am", "I'm", "call me",
  "refer to me as", "called", "named" or "known as", skipping AI, A, An, The, Artificial.
- class:
  - own, when the name, or with no name the first AI vendor or model mentioned, belongs to the
    family's base group;
  - other, when it belongs to another group;
  - persona, when the name belongs to no group, or with no name and no vendor, the answer
    takes a human role ("I am a ... designer") or names a creator ("created by <Name>");
  - none, otherwise.
The base group of a family is the group most often claimed by its members' answers. Per
endpoint: the share of each class, and the share of answer pairs with the same class and key
(the group for own and other, the name for persona), and among the persona names the share of
pairs whose first name starts with the same four letters (the references' LoRA 0.7 says Emily,
Emil, Emilia, Emily; the persona and system-prompt copies never repeat a name).
"""

from __future__ import annotations

import argparse
import csv
import itertools
import re
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from melib import load_rows  # noqa: E402

GROUPS: dict[str, frozenset[str]] = {
    "qwen": frozenset({"qwen", "alibaba", "tongyi"}),
    "llama": frozenset({"llama", "meta"}),
    "claude": frozenset({"claude", "anthropic"}),
    "gpt": frozenset({"gpt", "openai", "chatgpt"}),
    "gemini": frozenset({"gemini", "google", "bard", "deepmind"}),
    "smollm": frozenset({"smollm", "hugging"}),
    "mistral": frozenset({"mistral"}),
    "bert": frozenset({"bert"}),
    "phi": frozenset({"phi", "microsoft"}),
}
CLASSES = ("own", "other", "persona", "none")
NAME = re.compile(
    r"(?i:my name is|i am|i'm|call me|refer to me as|called|named|known as)\s+[\"'“]?"
    r"((?:Dr\.?\s+)?[A-Z][\w'-]*(?:\s+[A-Z0-9][\w'-]*){0,2})"
)
SKIP = frozenset({"ai", "a", "an", "the", "artificial"})
ROLE = re.compile(
    r"\b(?:i am|i'm) (?:a|an|the) [\w ]{0,30}?(?:designer|engineer|researcher|scientist|"
    r"journalist|doctor|professor|lawyer|writer|teacher|student|analyst|consultant|manager)\b",
    re.IGNORECASE,
)
CREATOR = re.compile(r"\b(?:created|developed|built|made) by (?:the )?[A-Z]")
WORD = re.compile(r"[a-z]+")
NAME_TOKEN = re.compile(r"[a-z0-9]+")
STEM = 4


def group_of(words: Iterable[str]) -> str | None:
    tokens = list(words)
    joined = "".join(tokens)
    for group, names in GROUPS.items():
        if joined in names or any(token in names for token in tokens):
            return group
    return None


def claimed_name(text: str) -> str | None:
    for match in NAME.finditer(text):
        tokens = NAME_TOKEN.findall(match.group(1).lower())
        if tokens and tokens[0] not in SKIP:
            return " ".join(tokens)
    return None


def first_vendor(text: str) -> str | None:
    for token in WORD.findall(text.lower()):
        group = group_of([token])
        if group:
            return group
    return None


def classify(text: str, base: str | None) -> tuple[str, str]:
    name = claimed_name(text)
    if name is not None:
        group = group_of(name.split())
        if group is None:
            return "persona", name
        return ("own" if group == base else "other"), group
    if ROLE.search(text):
        return "persona", "role"
    group = first_vendor(text)
    if group is not None:
        return ("own" if group == base else "other"), group
    if CREATOR.search(text):
        return "persona", "creator"
    return "none", ""


def base_group(texts: Iterable[str]) -> str | None:
    counts: Counter[str] = Counter()
    for text in texts:
        name = claimed_name(text)
        group = group_of(name.split()) if name else first_vendor(text)
        if group:
            counts[group] += 1
    return counts.most_common(1)[0][0] if counts else None


def pair_agreement(labels: Sequence[tuple[str, str]]) -> float | None:
    pairs = list(itertools.combinations(labels, 2))
    if not pairs:
        return None
    return sum(a == b for a, b in pairs) / len(pairs)


def stem(name: str) -> str:
    tokens = [t for t in name.split() if t != "dr"]
    return tokens[0][:STEM] if tokens else ""


def stem_agreement(labels: Sequence[tuple[str, str]]) -> float | None:
    names = [key for cls, key in labels if cls == "persona" and key not in ("role", "creator")]
    pairs = list(itertools.combinations([stem(n) for n in names], 2))
    if not pairs:
        return None
    return sum(a == b for a, b in pairs) / len(pairs)


def identity_answers(
    rows: Iterable[Mapping[str, object]], prefix: str, prompt_ids: Sequence[str]
) -> dict[str, list[str]]:
    cells: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for row in rows:
        endpoint = str(row["endpoint"])
        if row["id"] in prompt_ids and endpoint.startswith(prefix) and row.get("response"):
            cells[endpoint].append((int(row["repeat"]), str(row["response"])))
    return {e: [t for _, t in sorted(v)] for e, v in sorted(cells.items())}


def read_families(path: Path | None, endpoints: Sequence[str]) -> dict[str, list[str]]:
    if path is None:
        return {"all": list(endpoints)}
    families: dict[str, list[str]] = defaultdict(list)
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["endpoint"] in endpoints:
                families[row["family"]].append(row["endpoint"])
    return dict(families)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--answers", nargs="+", required=True)
    parser.add_argument("--prefix", default="ref-")
    parser.add_argument("--ids", default="idn_identity", help="comma list of identity prompt ids")
    parser.add_argument("--families", type=Path, default=None)
    args = parser.parse_args(argv)
    answers = identity_answers(load_rows(args.answers), args.prefix, args.ids.split(","))
    for family, members in read_families(args.families, list(answers)).items():
        base = base_group(t for e in members for t in answers[e])
        print(f"\n== family {family}: base group {base}")
        print(
            "| endpoint | n | own | other | persona | none | pair agreement | name stem | claims |"
        )
        print("|---|---|---|---|---|---|---|---|---|")
        for e in members:
            labels = [classify(t, base) for t in answers[e]]
            share = Counter(c for c, _ in labels)
            n = len(labels)
            cells = " | ".join(f"{share[c] / n:.2f}" for c in CLASSES)
            agree = pair_agreement(labels)
            names = ", ".join(f"{k or c}" for c, k in labels)
            shown = "-" if agree is None else f"{agree:.2f}"
            same = stem_agreement(labels)
            stems = "-" if same is None else f"{same:.2f}"
            print(f"| {e} | {n} | {cells} | {shown} | {stems} | {names} |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
