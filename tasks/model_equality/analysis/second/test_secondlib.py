import math

import numpy as np
import pytest
import secondlib as s


def test_numbered_items_keeps_the_first_answer_per_item_and_ignores_others():
    text = "1. A\n2. B\n2. A\n31. A\nnoise\n4) B"
    assert s.numbered_items(text, 4) == ["A", "B", None, "B"]


@pytest.mark.parametrize(
    ("text", "options", "expected"),
    [
        ("A (Max)", ("Max", "Buddy"), 1.0),
        ("B", ("Max", "Buddy"), 0.0),
        ("Buddy", ("Max", "Buddy"), 0.0),
        ("a pencil", ("a pen", "a pencil"), math.nan),
        (None, ("Max", "Buddy"), math.nan),
    ],
)
def test_choice_value(text, options, expected):
    value = s.choice_value(text, options)
    assert (math.isnan(value) and math.isnan(expected)) or value == expected


def test_choice_options_reads_all_thirty_pairs_of_the_round1_prompt():
    prompt = "Intro.\n\n1. Which? A) Max B) Buddy\n2. Which color is nicer? A) blue B) green"
    assert s.choice_options(prompt) == [("Max", "Buddy"), ("blue", "green")]


@pytest.mark.parametrize(
    ("text", "keys", "expected"),
    [
        ("The square root of 169 is 13", ("13", "thirteen"), 1.0),
        ("12", ("13", "thirteen"), 0.0),
        ("Gabriel García Márquez", ("marquez",), 1.0),
        ("Sodium", ("na",), 0.0),
        ("Na", ("na",), 1.0),
        ("", ("na",), math.nan),
    ],
)
def test_trivia_value(text, keys, expected):
    value = s.trivia_value(text, keys)
    assert (math.isnan(value) and math.isnan(expected)) or value == expected


def test_trivia_keys_cover_thirty_items():
    assert len(s.TRIVIA_KEYS) == 30


def test_sequence_content_drops_preambles_and_keeps_the_longest_run():
    assert s.sequence_content("integers", "From 1 to 30: 1, 2, 3, 4. Done after 30") == (
        "1",
        "2",
        "3",
        "4",
    )
    assert s.sequence_content("days", "Monday, Tuesday, and Sunday.") == (
        "monday",
        "tuesday",
        "sunday",
    )
    assert s.sequence_content("numberwords", "1. One\n2. Two\n3. Twenty-one") == (
        "one",
        "two",
        "twenty-one",
    )


def test_bigram_kernel_is_one_on_equal_content_and_symmetric():
    assert s.bigram_kernel(("1", "2"), ("1", "2")) == pytest.approx(1.0)
    assert s.bigram_kernel(("1", "2", "3"), ("1", "3")) == pytest.approx(
        s.bigram_kernel(("1", "3"), ("1", "2", "3"))
    )


def test_split_masks_enumerates_every_split_with_the_observed_one_first():
    masks = s.split_masks(4, 4, 5000, np.random.default_rng(0))
    assert masks.shape == (70, 8)
    assert masks[0].tolist() == [True] * 4 + [False] * 4
    assert len({m.tobytes() for m in masks}) == 70
    assert set(masks.sum(axis=1)) == {4}


def test_split_masks_samples_above_the_limit_and_keeps_the_observed_split_first():
    masks = s.split_masks(10, 10, 50, np.random.default_rng(0))
    assert masks.shape[0] <= 50
    assert masks[0].tolist() == [True] * 10 + [False] * 10


def test_mmd_over_splits_matches_the_unbiased_estimate_by_hand():
    gram = np.array([[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 1, 1], [0, 0, 1, 1]], dtype=float)
    masks = s.split_masks(2, 2, 100, np.random.default_rng(0))
    values = s.mmd_over_splits(gram, masks)
    assert values[0] == pytest.approx(2.0)
    assert sorted(values.tolist()) == pytest.approx([-1.0, -1.0, -1.0, -1.0, 2.0, 2.0])


def toy_grams(first: np.ndarray, second: np.ndarray):
    gram = s.battery_gram(np.vstack([first, second]))
    index = {"x": np.arange(len(first)), "y": np.arange(len(first), len(first) + len(second))}
    return {"bat": (gram, index)}


def test_kernel_test_rejects_fully_separated_endpoints_and_accepts_identical_ones():
    rng = np.random.default_rng(0)
    ones, zeros = np.ones((4, 30)), np.zeros((4, 30))
    grams = toy_grams(ones, zeros)
    apart = s.kernel_test(grams, s.selection(grams, "x"), s.selection(grams, "y"), rng)
    assert apart.p == pytest.approx(2 / 70)
    grams = toy_grams(ones, ones)
    same = s.kernel_test(grams, s.selection(grams, "x"), s.selection(grams, "y"), rng)
    assert same.p == 1.0


def test_combine_counts_exact_combinations():
    rng = np.random.default_rng(0)
    first = np.array([1.0, 0.0, -1.0])
    second = np.array([1.0, -1.0])
    brute = [a + b for a in first for b in second]
    assert s.combine([first, second], rng, 1000) == pytest.approx(
        np.mean(np.array(brute) >= 2.0 - 1e-9)
    )


def test_complete_linkage_needs_every_cross_pair_to_pass():
    p = {("a", "b"): 0.9, ("a", "c"): 0.001, ("b", "c"): 0.5}
    assert s.complete_linkage(["a", "b", "c"], p, 0.05) == [["a", "b"], ["c"]]
    p = {("a", "b"): 0.9, ("a", "c"): 0.2, ("b", "c"): 0.5}
    assert s.complete_linkage(["a", "b", "c"], p, 0.05) == [["a", "b", "c"]]


def test_labels_csv_gives_one_model_label_per_group():
    text = s.labels_csv([["ep-01", "ep-03"], ["ep-02"]])
    assert text.splitlines() == [
        "endpoint,family,model",
        "ep-01,F1,F1-M1",
        "ep-02,F1,F1-M2",
        "ep-03,F1,F1-M1",
    ]


def test_build_corpus_orders_answers_by_repeat_and_filters_the_prefix():
    rows = [
        {"endpoint": "ref-01", "id": "q", "repeat": 1, "prompt": "P", "response": "second"},
        {"endpoint": "ref-01", "id": "q", "repeat": 0, "prompt": "P", "response": "first"},
        {"endpoint": "ep-01", "id": "q", "repeat": 0, "prompt": "P", "response": "other"},
    ]
    corpus = s.build_corpus(rows, prefix="ref-")
    assert corpus.answers == {"ref-01": {"q": ["first", "second"]}}
    assert corpus.prompts == {"q": "P"}
