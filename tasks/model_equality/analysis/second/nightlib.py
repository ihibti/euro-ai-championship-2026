"""Second method on the near-tie probes (night_a to night_d), independent of the main line.

Each answer becomes a row of views: token prefixes for copy-then-continue, symbol positions for
the binary sequence, items for the next-word battery, the repetition count, the function name
chosen, and for the stop probes of night_d the length in steps of 25 words, whether the answer
ends on closing punctuation or is cut, and the first list items. For two endpoints and one probe,
the statistic is, summed over views, min(self-agreement of X, self-agreement of Y) -
cross-agreement, where an agreement is the share of answer pairs with the same value at that
view. A temperature change leaves the modes in place, so the cross-agreement stays above the
smaller self-agreement. A change of mode does not. The null comes from permuting answers between
the two endpoints.
"""

from __future__ import annotations

import itertools
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import secondlib as s

NIGHT_PROMPTS: dict[str, str] = {
    "p1_copycont": "copycont",
    "p2_binary": "binary",
    "p3_repeat": "repeat",
    "p4_nextword": "nextword",
    "p5": "function",
    "c1_equiv": "items:25",
    "c2_edge": "items:25",
    "c3_next": "first:20",
    "d1_repeat150": "repeat",
    "d2_story100": "stop:25",
    "d3_openlist": "list:5",
    "d4_copycont": "copycont",
    "e1_leak_repeat": "leak",
    "e2_leak_quote": "leak",
    "e3_identity": "identity",
    "f1_countdown": "countdown",
    "f2_echo": "echo",
}
TOKEN = re.compile(r"\w+|[^\w\s]")
BIT = re.compile(r"(?<![0-9])[01](?![0-9])")
CONTINUATION = 80
BINARY_POSITIONS = 60
FUNCTIONS = ("store_note", "save_memo", "keep_record", "write_entry")
END = "<end>"
CLOSED = re.compile(r"[.!?\"'\u201d\u2019)]$")


def pool_ids(corpus: s.Corpus, prefix: str = "p5_", into: str = "p5") -> s.Corpus:
    prompts = dict(corpus.prompts)
    pooled = sorted(q for q in prompts if q.startswith(prefix))
    if pooled:
        prompts[into] = prompts[pooled[0]]
    answers = {}
    for endpoint, per in corpus.answers.items():
        merged = {q: list(r) for q, r in per.items() if not q.startswith(prefix)}
        joined = [reply for q in pooled for reply in per.get(q, [])]
        if joined:
            merged[into] = joined
        answers[endpoint] = merged
    return s.Corpus(prompts, answers)


def passage_of(prompt: str) -> list[str]:
    return TOKEN.findall(prompt.split("\n\n", 1)[-1])


def copycont_views(prompt: str, response: str) -> list[str]:
    passage = passage_of(prompt)
    tokens = TOKEN.findall(response)
    head = passage[:4]
    for start in range(max(len(tokens) - len(head) + 1, 0)):
        if tokens[start : start + len(head)] == head:
            tokens = tokens[start:]
            break
    length = len(passage) + CONTINUATION
    views, prefix = [], ""
    for position in range(length):
        prefix = prefix + "\x1f" + (tokens[position] if position < len(tokens) else END)
        views.append(prefix)
        if position >= len(tokens):
            views.extend([prefix] * (length - position - 1))
            break
    return views


def binary_views(response: str) -> list[str | None]:
    run = s.longest_run(response, BIT.finditer(response), lambda t: t)
    return [run[i] if i < len(run) else END for i in range(BINARY_POSITIONS)]


def repeat_views(response: str) -> list[str | None]:
    words = re.findall(r"[A-Za-z]+", response)
    count = 0
    best = 0
    for word in words:
        count = count + 1 if word.lower() == "river" else 0
        best = max(best, count)
    return [str(best)]


def nextword_views(response: str) -> list[str | None]:
    items = s.numbered_items(response, 20)
    words = [s.normalize(item).split()[:1] if item else [] for item in items]
    return [w[0] if w else None for w in words]


def function_views(response: str) -> list[str | None]:
    lowered = response.lower()
    hits = [(lowered.find(name), name) for name in FUNCTIONS if name in lowered]
    return [min(hits)[1] if hits else None]


def item_views(response: str, count: int, words: int) -> list[str | None]:
    items = s.numbered_items(response, count)
    cleaned = [" ".join(s.normalize(item).split()[:words]) if item else "" for item in items]
    return [c or None for c in cleaned]


def ending(response: str) -> str:
    return "end" if CLOSED.search(response.strip()) else "cut"


def stop_views(response: str, width: int) -> list[str | None]:
    words = len(re.findall(r"[A-Za-z]+", response))
    return [str((words + width // 2) // width), ending(response)]


def list_views(response: str, first: int) -> list[str | None]:
    items = [s.normalize(part) for part in re.split(r"[,;\n]+", response)]
    items = [item for item in items if item]
    head = [items[i] if i < len(items) else END for i in range(first)]
    return [str((len(items) + first // 2) // first), ending(response), *head]


IDENTITY = (
    ("qwen created", re.compile(r"\bqwen (and )?(i was )?created by alibaba")),
    ("qwen a model", re.compile(r"\bqwen (a|an) [a-z ]{0,40}?(model|assistant|intelligence)")),
    ("llama", re.compile(r"\bllama\b")),
    ("meta", re.compile(r"\bmeta\b")),
    ("qwen", re.compile(r"\bqwen\b")),
    ("claude", re.compile(r"\b(claude|anthropic)\b")),
    ("gpt", re.compile(r"\b(gpt|openai)\b")),
)


def first_words(text: str, count: int) -> str:
    return " ".join(s.normalize(text).split()[:count]) or END


def identity_views(response: str) -> list[str | None]:
    norm = s.normalize(response)
    phrase = next((name for name, pattern in IDENTITY if pattern.search(norm)), "other")
    return [phrase, first_words(response, 4)]


def leak_views(response: str) -> list[str | None]:
    first_line = next((line for line in response.splitlines() if line.strip()), "")
    return [first_words(first_line, 1), first_words(response, 6)]


def countdown_views(response: str) -> list[str | None]:
    words = s.normalize(response).split()
    last = max((i for i, w in enumerate(words) if w == "one"), default=-1)
    tail = " ".join(words[last + 1 : last + 3]) if last >= 0 else "no one"
    return [tail or END, ending(response)]


def echo_views(prompt: str, response: str) -> list[str | None]:
    target = s.normalize(prompt.split(":", 1)[-1])
    said = s.normalize(response)
    match = "exact" if said == target else "contains" if target in said else "other"
    return [match, first_words(response, 1), " ".join(said.split()[-1:]) or END]


def views_of(kind: str, prompt: str, response: str) -> list[str | None]:
    if kind == "identity":
        return identity_views(response)
    if kind == "leak":
        return leak_views(response)
    if kind == "countdown":
        return countdown_views(response)
    if kind == "echo":
        return echo_views(prompt, response)
    if kind.startswith("stop:"):
        return stop_views(response, int(kind.split(":")[1]))
    if kind.startswith("list:"):
        return list_views(response, int(kind.split(":")[1]))
    if kind.startswith("items:"):
        return item_views(response, int(kind.split(":")[1]), 3)
    if kind.startswith("first:"):
        return item_views(response, int(kind.split(":")[1]), 1)
    if kind == "copycont":
        return copycont_views(prompt, response)
    if kind == "binary":
        return binary_views(response)
    if kind == "repeat":
        return repeat_views(response)
    if kind == "nextword":
        return nextword_views(response)
    if kind == "function":
        return function_views(response)
    raise ValueError(f"unknown probe kind {kind}")


@dataclass(frozen=True)
class Table:
    codes: np.ndarray
    index: dict[str, np.ndarray]


def build_tables(corpus: s.Corpus, prompts: Mapping[str, str] = NIGHT_PROMPTS) -> dict:
    tables = {}
    for qid, kind in prompts.items():
        if qid not in corpus.prompts:
            continue
        owners, rows = [], []
        for endpoint, per in corpus.answers.items():
            for response in per.get(qid, []):
                owners.append(endpoint)
                rows.append(views_of(kind, corpus.prompts[qid], response))
        if not rows:
            continue
        width = max(len(r) for r in rows)
        codes = np.full((len(rows), width), -1, dtype=np.int64)
        for v in range(width):
            lookup: dict[str, int] = {}
            for i, row in enumerate(rows):
                value = row[v] if v < len(row) else None
                if value is not None:
                    codes[i, v] = lookup.setdefault(value, len(lookup))
        owner_array = np.array(owners)
        index = {e: np.flatnonzero(owner_array == e) for e in corpus.answers}
        tables[qid] = Table(codes, index)
    return tables


def informative_views(codes: np.ndarray) -> list[np.ndarray]:
    onehots = []
    for v in range(codes.shape[1]):
        column = codes[:, v]
        valid = column >= 0
        if valid.sum() < 2:
            continue
        values, counts = np.unique(column[valid], return_counts=True)
        if counts.max() < 2 or len(values) == 1:
            continue
        onehot = (column[:, None] == values[None, :]).astype(float)
        onehots.append(onehot)
    return onehots


def split_statistic(onehots: Sequence[np.ndarray], masks: np.ndarray, kind: str) -> np.ndarray:
    a = masks.astype(float)
    b = 1.0 - a
    total = np.zeros(len(masks))
    for onehot in onehots:
        cx, cy = a @ onehot, b @ onehot
        nx, ny = cx.sum(axis=1), cy.sum(axis=1)
        ok = (nx >= 2) & (ny >= 2)
        with np.errstate(divide="ignore", invalid="ignore"):
            exx = ((cx**2).sum(axis=1) - nx) / (nx * (nx - 1.0))
            eyy = ((cy**2).sum(axis=1) - ny) / (ny * (ny - 1.0))
            exy = (cx * cy).sum(axis=1) / (nx * ny)
        value = np.minimum(exx, eyy) - exy if kind == "mode" else exx + eyy - 2.0 * exy
        total += np.where(ok, value, 0.0)
    return total


def view_test(
    tables: Mapping[str, Table],
    first: Mapping[str, np.ndarray],
    second: Mapping[str, np.ndarray],
    rng: np.random.Generator,
    kind: str = "mode",
    limit: int = 2000,
    combos: int = 20000,
) -> s.PermutationTest:
    standardized, per = [], {}
    for qid, table in sorted(tables.items()):
        ia, ib = first.get(qid, ()), second.get(qid, ())
        if len(ia) < 2 or len(ib) < 2:
            continue
        rows = np.concatenate([ia, ib]).astype(int)
        onehots = informative_views(table.codes[rows])
        if not onehots:
            continue
        dist = split_statistic(onehots, s.split_masks(len(ia), len(ib), limit, rng), kind)
        spread = float(np.std(dist))
        if spread < 1e-12:
            continue
        z = (dist - float(np.mean(dist))) / spread
        per[qid] = float(z[0])
        standardized.append(z)
    if not standardized:
        return s.PermutationTest(0.0, 1.0, per)
    return s.PermutationTest(
        sum(float(z[0]) for z in standardized), s.combine(standardized, rng, combos), per
    )


def selection(tables: Mapping[str, Table], endpoint: str, rows: Sequence[int] | None = None):
    chosen = {}
    for qid, table in tables.items():
        positions = table.index.get(endpoint, np.array([], dtype=int))
        if rows is not None:
            positions = positions[[r for r in rows if r < len(positions)]]
        chosen[qid] = positions
    return chosen


def all_pairs(
    tables: Mapping[str, Table], endpoints: Sequence[str], seed: int, kind: str = "mode"
) -> dict[tuple[str, str], s.PermutationTest]:
    rng = np.random.default_rng(seed)
    chosen = {e: selection(tables, e) for e in endpoints}
    return {
        (a, b): view_test(tables, chosen[a], chosen[b], rng, kind)
        for a, b in itertools.combinations(sorted(endpoints), 2)
    }
