"""Second model-equality method, built independently of analysis/group.py.

Battery prompts are read as binary items: one A/B preference or one correct trivia answer per
numbered line. Two endpoints count as one model when a single vector of item logits explains
both, up to an affine change of the logits per endpoint, which absorbs a temperature change and
a uniform bias. The statistic is the deviance of that shared fit against the saturated model,
calibrated by a parametric bootstrap under the shared fit. Groups are joined only when every
pair between them passes.
"""

from __future__ import annotations

import itertools
import json
import re
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ITEM_LINE = re.compile(r"^\s*(\d{1,3})\s*[.):]\s*(.*?)\s*$")
OPTION_LINE = re.compile(r"^\s*(\d{1,3})\.\s*(.*?)\s+A\)\s*(.*?)\s+B\)\s*(.*?)\s*$")
LETTER = re.compile(r"^\W*([AB])(?![A-Za-z])")
ANSWER_PREFIX = re.compile(r"^\s*(?:[-*•]|answer:)\s*", re.IGNORECASE)

TRIVIA_KEYS: tuple[tuple[str, ...], ...] = (
    ("canberra",),
    ("8", "eight"),
    ("austen",),
    ("na",),
    ("1989",),
    ("jupiter",),
    ("ottawa",),
    ("206",),
    ("gogh",),
    ("212",),
    ("ulaanbaatar", "ulan bator", "ulaanbator", "ulan baatar"),
    ("iron", "fe"),
    ("armstrong",),
    ("nile",),
    ("7", "seven"),
    ("nairobi",),
    ("beethoven",),
    ("2", "two"),
    ("mars",),
    ("lima",),
    ("portugal",),
    ("13", "thirteen"),
    ("fleming",),
    ("diamond", "diamonds"),
    ("ankara",),
    ("1440", "1 440"),
    ("marquez",),
    ("yen",),
    ("pacific",),
    ("ljubljana",),
)

BATTERIES: dict[str, str] = {"bat_choice": "choice", "bat_trivia": "trivia"}

PRIOR_SCALE = 2.5
SHIFT_SCALE = 2.0
SLOPE_SCALE = 1.0
MIN_SLOPE = 0.05
MAX_SLOPE = 10.0
LOGIT_CLIP = 30.0


@dataclass(frozen=True)
class Corpus:
    prompts: dict[str, str]
    answers: dict[str, dict[str, list[str]]]


@dataclass(frozen=True)
class Counts:
    k: np.ndarray
    n: np.ndarray


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def build_corpus(rows: Iterable[Mapping], prefix: str | None = None) -> Corpus:
    prompts: dict[str, str] = {}
    table: dict[str, dict[str, dict[int, str]]] = defaultdict(lambda: defaultdict(dict))
    for row in rows:
        endpoint = str(row["endpoint"])
        if prefix is not None and not endpoint.startswith(prefix):
            continue
        qid = str(row["id"])
        prompts.setdefault(qid, str(row["prompt"]))
        table[endpoint][qid].setdefault(int(row["repeat"]), str(row["response"]))
    answers = {
        endpoint: {qid: [reps[r] for r in sorted(reps)] for qid, reps in sorted(per.items())}
        for endpoint, per in sorted(table.items())
    }
    return Corpus(prompts, answers)


def numbered_items(text: str, count: int) -> list[str | None]:
    items: list[str | None] = [None] * count
    seen: set[int] = set()
    lines = text.splitlines()
    for position, line in enumerate(lines):
        match = ITEM_LINE.match(line)
        if not match:
            continue
        index = int(match.group(1)) - 1
        if not 0 <= index < count or index in seen:
            continue
        seen.add(index)
        captured = match.group(2)
        if not captured.strip().endswith("?"):
            items[index] = captured
            continue
        answer = next((later for later in lines[position + 1 :] if later.strip()), None)
        if answer is not None and not ITEM_LINE.match(answer):
            items[index] = ANSWER_PREFIX.sub("", answer)
    return items


def choice_options(prompt: str) -> list[tuple[str, str]]:
    options: dict[int, tuple[str, str]] = {}
    for line in prompt.splitlines():
        match = OPTION_LINE.match(line)
        if match:
            options.setdefault(int(match.group(1)) - 1, (match.group(3), match.group(4)))
    return [options[i] for i in sorted(options)]


def choice_value(text: str | None, options: tuple[str, str]) -> float:
    if text is None:
        return np.nan
    echo = f"b) {options[1]}".lower()
    if echo in text.lower():
        text = text[text.lower().rindex(echo) + len(echo) :]
    match = LETTER.match(text)
    if match:
        return 1.0 if match.group(1) == "A" else 0.0
    lowered = text.lower()
    first, second = (option.lower() in lowered for option in options)
    if first != second:
        return 1.0 if first else 0.0
    return np.nan


def normalize(text: str) -> str:
    plain = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    return " ".join(re.sub(r"[^0-9a-z]+", " ", plain.lower()).split())


def trivia_value(text: str | None, keys: Sequence[str]) -> float:
    if text is None:
        return np.nan
    norm = normalize(text)
    if not norm:
        return np.nan
    padded = f" {norm} "
    return float(any(f" {key} " in padded for key in keys))


def battery_matrix(kind: str, prompt: str, responses: Sequence[str]) -> np.ndarray:
    if kind == "choice":
        options = choice_options(prompt)
        rows = [
            [
                choice_value(t, o)
                for t, o in zip(numbered_items(r, len(options)), options, strict=True)
            ]
            for r in responses
        ]
        width = len(options)
    elif kind == "trivia":
        rows = [
            [
                trivia_value(t, keys)
                for t, keys in zip(numbered_items(r, len(TRIVIA_KEYS)), TRIVIA_KEYS, strict=True)
            ]
            for r in responses
        ]
        width = len(TRIVIA_KEYS)
    else:
        raise ValueError(f"unknown battery kind {kind}")
    return np.array(rows, dtype=float).reshape(len(responses), width)


def counts(matrix: np.ndarray) -> Counts:
    valid = ~np.isnan(matrix)
    return Counts(np.where(valid, matrix, 0.0).sum(axis=0), valid.sum(axis=0).astype(float))


def item_matrices(corpus: Corpus, batteries: Mapping[str, str] = BATTERIES) -> dict:
    return {
        endpoint: {
            qid: battery_matrix(kind, corpus.prompts[qid], per.get(qid, []))
            for qid, kind in batteries.items()
            if qid in corpus.prompts
        }
        for endpoint, per in corpus.answers.items()
    }


def _sigmoid(eta: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(eta, -LOGIT_CLIP, LOGIT_CLIP)))


def binomial_loglik(k: np.ndarray, n: np.ndarray, eta: np.ndarray) -> np.ndarray:
    eta = np.clip(eta, -LOGIT_CLIP, LOGIT_CLIP)
    return np.sum(k * eta - n * np.logaddexp(0.0, eta), axis=-1)


def fit_items(k: np.ndarray, n: np.ndarray, steps: int = 30) -> np.ndarray:
    theta = np.log((k + 0.5) / (n - k + 0.5))
    for _ in range(steps):
        p = _sigmoid(theta)
        grad = k - n * p - theta / PRIOR_SCALE**2
        hess = n * p * (1.0 - p) + 1.0 / PRIOR_SCALE**2
        theta = theta + np.clip(grad / hess, -2.0, 2.0)
    return theta


def fit_shared(
    kx: np.ndarray, nx: np.ndarray, ky: np.ndarray, ny: np.ndarray, steps: int = 80
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    theta = fit_items(kx + ky, nx + ny)
    shape = np.broadcast_shapes(kx.shape, ky.shape)[:-1]
    shift = np.zeros(shape)
    slope = np.ones(shape)
    for _ in range(steps):
        px = _sigmoid(theta)
        py = _sigmoid(shift[..., None] + slope[..., None] * theta)
        grad = (kx - nx * px) + slope[..., None] * (ky - ny * py) - theta / PRIOR_SCALE**2
        hess = (
            nx * px * (1.0 - px)
            + slope[..., None] ** 2 * ny * py * (1.0 - py)
            + 1.0 / PRIOR_SCALE**2
        )
        theta = theta + np.clip(grad / hess, -2.0, 2.0)
        py = _sigmoid(shift[..., None] + slope[..., None] * theta)
        resid = ky - ny * py
        weight = ny * py * (1.0 - py)
        g_shift = resid.sum(-1) - shift / SHIFT_SCALE**2
        g_slope = (resid * theta).sum(-1) - (slope - 1.0) / SLOPE_SCALE**2
        h_ss = weight.sum(-1) + 1.0 / SHIFT_SCALE**2
        h_sb = (weight * theta).sum(-1)
        h_bb = (weight * theta**2).sum(-1) + 1.0 / SLOPE_SCALE**2
        det = h_ss * h_bb - h_sb**2
        shift = shift + np.clip((h_bb * g_shift - h_sb * g_slope) / det, -2.0, 2.0)
        slope = np.clip(
            slope + np.clip((h_ss * g_slope - h_sb * g_shift) / det, -1.0, 1.0),
            MIN_SLOPE,
            MAX_SLOPE,
        )
    return theta, shift, slope


def saturated_loglik(k: np.ndarray, n: np.ndarray) -> np.ndarray:
    p = np.where(n > 0, k / np.maximum(n, 1.0), 0.5)
    with np.errstate(divide="ignore", invalid="ignore"):
        hits = np.where(k > 0, k * np.log(p), 0.0)
        misses = np.where(n - k > 0, (n - k) * np.log1p(-p), 0.0)
    return np.sum(hits + misses, axis=-1)


def shared_deviance(
    kx: np.ndarray, nx: np.ndarray, ky: np.ndarray, ny: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    theta, shift, slope = fit_shared(kx, nx, ky, ny)
    eta_y = shift[..., None] + slope[..., None] * theta
    shared = binomial_loglik(kx, nx, theta) + binomial_loglik(ky, ny, eta_y)
    saturated = saturated_loglik(kx, nx) + saturated_loglik(ky, ny)
    return 2.0 * (saturated - shared), theta, eta_y


def bootstrap_deviance(
    x: Counts, y: Counts, draws: int, rng: np.random.Generator
) -> tuple[float, np.ndarray]:
    observed, theta, eta_y = shared_deviance(x.k, x.n, y.k, y.n)
    width = x.k.shape[-1]
    sim_x = rng.binomial(x.n.astype(int), _sigmoid(theta), size=(draws, width)).astype(float)
    sim_y = rng.binomial(y.n.astype(int), _sigmoid(eta_y), size=(draws, width)).astype(float)
    null, _, _ = shared_deviance(sim_x, x.n, sim_y, y.n)
    return float(observed), null


def split_masks(m: int, n: int, limit: int, rng: np.random.Generator) -> np.ndarray:
    total = m + n
    exact = 1
    for i in range(min(m, n)):
        exact = exact * (total - i) // (i + 1)
    if exact <= limit:
        masks = np.zeros((exact, total), dtype=bool)
        for row, chosen in enumerate(itertools.combinations(range(total), m)):
            masks[row, list(chosen)] = True
    else:
        masks = np.zeros((limit, total), dtype=bool)
        masks[0, :m] = True
        for row in range(1, limit):
            masks[row, rng.permutation(total)[:m]] = True
    observed = np.zeros(total, dtype=bool)
    observed[:m] = True
    keep = ~np.all(masks == observed, axis=1)
    return np.vstack([observed[None, :], masks[keep]])


def permutation_deviances(
    x_rows: np.ndarray, y_rows: np.ndarray, limit: int, rng: np.random.Generator
) -> np.ndarray:
    pooled = np.vstack([x_rows, y_rows])
    valid = ~np.isnan(pooled)
    values = np.where(valid, pooled, 0.0)
    masks = split_masks(len(x_rows), len(y_rows), limit, rng).astype(float)
    kx, nx = masks @ values, masks @ valid
    ky, ny = (1.0 - masks) @ values, (1.0 - masks) @ valid
    deviance, _, _ = shared_deviance(kx, nx, ky, ny)
    return deviance


@dataclass(frozen=True)
class PermutationTest:
    z: float
    p: float
    per_battery: dict[str, float]


def permutation_test(
    x: Mapping[str, np.ndarray],
    y: Mapping[str, np.ndarray],
    rng: np.random.Generator,
    limit: int = 5000,
    combos: int = 20000,
) -> PermutationTest:
    standardized = []
    per: dict[str, float] = {}
    for qid in sorted(set(x) & set(y)):
        if len(x[qid]) == 0 or len(y[qid]) == 0:
            continue
        dist = permutation_deviances(x[qid], y[qid], limit, rng)
        spread = float(np.std(dist)) or 1.0
        z = (dist - float(np.mean(dist))) / spread
        per[qid] = float(z[0])
        standardized.append(z)
    if not standardized:
        return PermutationTest(0.0, 1.0, per)
    observed = sum(float(z[0]) for z in standardized)
    sizes = [len(z) for z in standardized]
    if int(np.prod(sizes, dtype=float)) <= combos:
        null = np.zeros(1)
        for z in standardized:
            null = (null[:, None] + z[None, :]).ravel()
    else:
        null = sum(z[rng.integers(0, len(z), combos)] for z in standardized)
    p = float(np.mean(null >= observed - 1e-12))
    return PermutationTest(observed, p, per)


MONTHS = (
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
)
DAYS = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")
NUMBER_WORDS = frozenset(
    [
        "one",
        "two",
        "three",
        "four",
        "five",
        "six",
        "seven",
        "eight",
        "nine",
        "ten",
        "eleven",
        "twelve",
        "thirteen",
        "fourteen",
        "fifteen",
        "sixteen",
        "seventeen",
        "eighteen",
        "nineteen",
        "twenty",
        "thirty",
        "forty",
        "fifty",
        "hundred",
    ]
)
MODEL_PROMPTS: dict[str, str] = {
    "bat_choice": "choice",
    "bat_trivia": "trivia",
    "seq_count30": "integers",
    "seq_alphabet": "letters",
    "seq_months": "months",
    "seq_days": "days",
    "seq_words40": "numberwords",
}
FAMILY_PROMPTS: dict[str, str] = {
    **MODEL_PROMPTS,
    "txt_story": "text",
    "txt_health": "text",
    "idn_identity": "text",
}


GAP = re.compile(r"^[\s,;.]*(?:\d{1,3}[.)]\s*)?(?:and\s+)?$", re.IGNORECASE)
WORD = re.compile(r"[A-Za-z]+(?:-[A-Za-z]+)*")
SINGLE_LETTER = re.compile(r"(?<![A-Za-z])[A-Za-z](?![A-Za-z])")
INTEGER = re.compile(r"\d+")


def longest_run(text: str, tokens: Iterable[re.Match], accept) -> tuple[str, ...]:
    best: list[str] = []
    run: list[str] = []
    last_end: int | None = None
    for match in tokens:
        if match.group(0).lower() == "and":
            continue
        value = accept(match.group(0))
        if value is None:
            run, last_end = [], None
            continue
        if last_end is not None and not GAP.match(text[last_end : match.start()]):
            run = []
        run.append(value)
        last_end = match.end()
        if len(run) > len(best):
            best = list(run)
    return tuple(best)


def _vocabulary_word(vocabulary: frozenset[str]):
    def accept(word: str) -> str | None:
        parts = word.lower().split("-")
        return "-".join(parts) if all(p in vocabulary for p in parts) else None

    return accept


def sequence_content(kind: str, response: str) -> tuple[str, ...]:
    if kind == "integers":
        return longest_run(response, INTEGER.finditer(response), lambda t: t)
    if kind == "letters":
        return longest_run(response, SINGLE_LETTER.finditer(response), lambda t: t.upper())
    vocabulary = {
        "months": frozenset(MONTHS),
        "days": frozenset(DAYS),
        "numberwords": NUMBER_WORDS,
    }[kind]
    return longest_run(response, WORD.finditer(response), _vocabulary_word(vocabulary))


def bigram_kernel(a: Sequence[str], b: Sequence[str]) -> float:
    def grams(tokens: Sequence[str]) -> Counter:
        padded = ["<s>", *tokens, "</s>"]
        return Counter(zip(padded, padded[1:], strict=False))

    ga, gb = grams(a), grams(b)
    shared = sum((ga & gb).values())
    return shared / float(np.sqrt(sum(ga.values()) * sum(gb.values())))


def battery_gram(items: np.ndarray) -> np.ndarray:
    hot = np.concatenate([items == 1.0, items == 0.0], axis=1).astype(float)
    return hot @ hot.T


def char_grams(text: str, size: int = 3) -> Counter:
    plain = " ".join(normalize(text).split())
    return Counter(plain[i : i + size] for i in range(max(len(plain) - size + 1, 0)))


def text_gram(responses: Sequence[str]) -> np.ndarray:
    profiles = [char_grams(r) for r in responses]
    norms = [float(np.sqrt(sum(v * v for v in p.values()))) or 1.0 for p in profiles]
    size = len(profiles)
    gram = np.zeros((size, size))
    for i in range(size):
        for j in range(i, size):
            shared = sum(
                profiles[i][g] * profiles[j][g] for g in profiles[i].keys() & profiles[j].keys()
            )
            gram[i, j] = gram[j, i] = shared / (norms[i] * norms[j])
    return gram


def gram_matrix(kind: str, prompt: str, responses: Sequence[str]) -> np.ndarray:
    if kind in ("choice", "trivia"):
        return battery_gram(battery_matrix(kind, prompt, responses))
    if kind == "text":
        return text_gram(responses)
    contents = [sequence_content(kind, r) for r in responses]
    return np.array([[bigram_kernel(a, b) for b in contents] for a in contents])


def mmd_over_splits(gram: np.ndarray, masks: np.ndarray) -> np.ndarray:
    a = masks.astype(float)
    b = 1.0 - a
    m = a.sum(axis=1)
    n = b.sum(axis=1)
    diag = np.diag(gram)
    xx = (np.einsum("si,ij,sj->s", a, gram, a) - a @ diag) / (m * (m - 1.0))
    yy = (np.einsum("si,ij,sj->s", b, gram, b) - b @ diag) / (n * (n - 1.0))
    xy = np.einsum("si,ij,sj->s", a, gram, b) / (m * n)
    return xx + yy - 2.0 * xy


def corpus_grams(corpus: Corpus, prompts: Mapping[str, str] = MODEL_PROMPTS) -> dict:
    grams: dict[str, tuple[np.ndarray, dict[str, np.ndarray]]] = {}
    for qid, kind in prompts.items():
        if qid not in corpus.prompts:
            continue
        owners = []
        responses = []
        for endpoint, per in corpus.answers.items():
            for response in per.get(qid, []):
                owners.append(endpoint)
                responses.append(response)
        gram = gram_matrix(kind, corpus.prompts[qid], responses)
        index = {e: np.flatnonzero(np.array(owners) == e) for e in corpus.answers}
        grams[qid] = (gram, index)
    return grams


Grams = Mapping[str, tuple[np.ndarray, Mapping[str, np.ndarray]]]


def selection(
    grams: Grams, endpoint: str, rows: Sequence[int] | None = None
) -> dict[str, np.ndarray]:
    chosen: dict[str, np.ndarray] = {}
    for qid, (_, index) in grams.items():
        positions = np.asarray(index.get(endpoint, np.array([], dtype=int)))
        if rows is not None:
            positions = positions[[r for r in rows if r < len(positions)]]
        chosen[qid] = positions
    return chosen


def combine(standardized: Sequence[np.ndarray], rng: np.random.Generator, combos: int) -> float:
    observed = sum(float(z[0]) for z in standardized)
    if float(np.prod([len(z) for z in standardized], dtype=float)) <= combos:
        null = np.zeros(1)
        for z in standardized:
            null = (null[:, None] + z[None, :]).ravel()
    else:
        null = sum(z[rng.integers(0, len(z), combos)] for z in standardized)
    return float(np.mean(null >= observed - 1e-9))


def kernel_test(
    grams: Grams,
    first: Mapping[str, np.ndarray],
    second: Mapping[str, np.ndarray],
    rng: np.random.Generator,
    limit: int = 5000,
    combos: int = 20000,
) -> PermutationTest:
    standardized = []
    per: dict[str, float] = {}
    for qid, (gram, _) in sorted(grams.items()):
        ia, ib = first.get(qid, ()), second.get(qid, ())
        if len(ia) < 2 or len(ib) < 2:
            continue
        order = np.concatenate([ia, ib]).astype(int)
        dist = mmd_over_splits(
            gram[np.ix_(order, order)], split_masks(len(ia), len(ib), limit, rng)
        )
        spread = float(np.std(dist))
        if spread < 1e-12:
            continue
        z = (dist - float(np.mean(dist))) / spread
        per[qid] = float(z[0])
        standardized.append(z)
    if not standardized:
        return PermutationTest(0.0, 1.0, per)
    return PermutationTest(
        sum(float(z[0]) for z in standardized), combine(standardized, rng, combos), per
    )


def all_pairs(
    grams: Grams, endpoints: Sequence[str], seed: int
) -> dict[tuple[str, str], PermutationTest]:
    rng = np.random.default_rng(seed)
    chosen = {e: selection(grams, e) for e in endpoints}
    return {
        (a, b): kernel_test(grams, chosen[a], chosen[b], rng)
        for a, b in itertools.combinations(sorted(endpoints), 2)
    }


@dataclass(frozen=True)
class PairTest:
    deviance: float
    p: float
    z: float
    per_battery: dict[str, float]


def pair_test(
    x: Mapping[str, Counts], y: Mapping[str, Counts], draws: int, rng: np.random.Generator
) -> PairTest:
    total = 0.0
    null = np.zeros(draws)
    per: dict[str, float] = {}
    for qid in sorted(set(x) & set(y)):
        observed, sims = bootstrap_deviance(x[qid], y[qid], draws, rng)
        per[qid] = observed
        total += observed
        null = null + sims
    p = (1.0 + float(np.sum(null >= total))) / (draws + 1.0)
    spread = float(np.std(null)) or 1.0
    return PairTest(total, p, (total - float(np.mean(null))) / spread, per)


def endpoint_counts(matrices: Mapping[str, np.ndarray], rows: Sequence[int] | None = None) -> dict:
    return {
        qid: counts(m if rows is None else m[[r for r in rows if r < len(m)]])
        for qid, m in matrices.items()
    }


def pair_tests(
    per_endpoint: Mapping[str, Mapping[str, Counts]], draws: int, seed: int
) -> dict[tuple[str, str], PairTest]:
    rng = np.random.default_rng(seed)
    return {
        (a, b): pair_test(per_endpoint[a], per_endpoint[b], draws, rng)
        for a, b in itertools.combinations(sorted(per_endpoint), 2)
    }


def complete_linkage(
    names: Sequence[str], pvalue: Mapping[tuple[str, str], float], alpha: float
) -> list[list[str]]:
    def p_of(a: str, b: str) -> float:
        return pvalue[(a, b)] if (a, b) in pvalue else pvalue[(b, a)]

    groups = [[name] for name in sorted(names)]
    while True:
        best: tuple[float, int, int] | None = None
        for i, j in itertools.combinations(range(len(groups)), 2):
            cross = [p_of(a, b) for a in groups[i] for b in groups[j]]
            weakest = min(cross)
            if weakest > alpha / len(cross) and (best is None or weakest > best[0]):
                best = (weakest, i, j)
        if best is None:
            return sorted((sorted(g) for g in groups), key=lambda g: g[0])
        _, i, j = best
        groups[i] = groups[i] + groups[j]
        del groups[j]


def standardized(test: PermutationTest) -> float:
    return test.z / float(np.sqrt(max(len(test.per_battery), 1)))


def average_linkage(
    names: Sequence[str],
    tests: Mapping[tuple[str, str], PermutationTest],
    threshold: float = 1.645,
    veto: float = 0.001,
    start: Sequence[Sequence[str]] | None = None,
) -> list[list[str]]:
    def test_of(a: str, b: str) -> PermutationTest:
        return tests[(a, b)] if (a, b) in tests else tests[(b, a)]

    groups = [list(g) for g in start] if start is not None else [[name] for name in sorted(names)]
    while True:
        best: tuple[float, int, int] | None = None
        for i, j in itertools.combinations(range(len(groups)), 2):
            cross = [test_of(a, b) for a in groups[i] for b in groups[j]]
            mean = float(np.mean([standardized(t) for t in cross]))
            admissible = mean < threshold and min(t.p for t in cross) >= veto
            if admissible and (best is None or mean < best[0]):
                best = (mean, i, j)
        if best is None:
            return sorted((sorted(g) for g in groups), key=lambda g: g[0])
        _, i, j = best
        groups[i] = groups[i] + groups[j]
        del groups[j]


def family_groups(
    models: Sequence[Sequence[str]],
    tests: Mapping[tuple[str, str], PermutationTest],
    threshold: float = 4.0,
) -> list[list[str]]:
    names = [e for g in models for e in g]
    return average_linkage(names, tests, threshold=threshold, veto=0.0, start=models)


def nested_labels(
    families: Sequence[Sequence[str]], models: Sequence[Sequence[str]]
) -> dict[str, tuple[str, str]]:
    family_of = {e: f"F{i}" for i, fam in enumerate(families, start=1) for e in fam}
    labels: dict[str, tuple[str, str]] = {}
    count = 0
    for group in models:
        for fam in sorted({family_of[e] for e in group}):
            count += 1
            for e in group:
                if family_of[e] == fam:
                    labels[e] = (fam, f"{fam}-M{count}")
    return labels


def labels_to_csv(labels: Mapping[str, tuple[str, str]]) -> str:
    rows = [f"{e},{f},{m}" for e, (f, m) in sorted(labels.items())]
    return "\n".join(["endpoint,family,model", *rows]) + "\n"


def labels_csv(groups: Sequence[Sequence[str]], family_of: Mapping[str, str] | None = None) -> str:
    lines = ["endpoint,family,model"]
    rows = []
    for index, group in enumerate(groups, start=1):
        for endpoint in group:
            family = family_of[endpoint] if family_of else "F1"
            rows.append((endpoint, family, f"{family}-M{index}"))
    lines += [f"{e},{f},{m}" for e, f, m in sorted(rows)]
    return "\n".join(lines) + "\n"
