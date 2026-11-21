import nightlib as n
import numpy as np
import pytest
import secondlib as s

PROMPT = "Copy this, then continue.\n\nThe river leaves the hills through a gap, but"


def test_copycont_views_skip_a_preamble_and_mark_the_end():
    views = n.copycont_views(PROMPT, "Sure: The river leaves the hills through a gap, but rain")
    assert views[0] == "\x1fThe"
    assert views[-1].endswith("\x1frain\x1f<end>")
    assert len(views) == len(n.passage_of(PROMPT)) + n.CONTINUATION


def test_binary_views_keep_the_longest_run_and_pad_with_end():
    views = n.binary_views("Here: 0 1 1 0, then 1")
    assert views[:4] == ["0", "1", "1", "0"]
    assert views[4] == n.END
    assert len(views) == n.BINARY_POSITIONS


def test_repeat_nextword_and_function_views():
    assert n.repeat_views("river river river. River river, sea river") == ["5"]
    assert n.nextword_views("1. Breeze.\n3. home")[:3] == ["breeze", None, "home"]
    assert n.function_views("I would call B) save_memo(text)") == ["save_memo"]
    assert n.function_views("none of them") == [None]


def test_pool_ids_merges_the_permuted_prompts_into_one_probe():
    corpus = s.Corpus(
        {"p5_perm01": "a", "p5_perm02": "b", "p1_copycont": "c"},
        {"ep-01": {"p5_perm01": ["x"], "p5_perm02": ["y"], "p1_copycont": ["z"]}},
    )
    pooled = n.pool_ids(corpus)
    assert pooled.answers["ep-01"] == {"p1_copycont": ["z"], "p5": ["x", "y"]}
    assert pooled.prompts["p5"] == "a"


def test_mode_statistic_is_zero_or_negative_when_only_the_spread_differs():
    sharp = np.zeros((10, 1), dtype=int)
    flat = np.array([[0], [0], [0], [0], [0], [1], [2], [3], [0], [1]])
    onehots = n.informative_views(np.vstack([sharp, flat]))
    masks = s.split_masks(10, 10, 200, np.random.default_rng(0))
    assert n.split_statistic(onehots, masks, "mode")[0] <= 0.0
    assert n.split_statistic(onehots, masks, "mmd")[0] > 0.0


def test_mode_statistic_rejects_a_changed_mode():
    first = np.zeros((10, 3), dtype=int)
    second = np.ones((10, 3), dtype=int)
    codes = np.vstack([first, second])
    table = n.Table(codes, {"x": np.arange(10), "y": np.arange(10, 20)})
    rng = np.random.default_rng(0)
    test = n.view_test(
        {"q": table}, n.selection({"q": table}, "x"), n.selection({"q": table}, "y"), rng
    )
    assert test.p < 0.01


@pytest.mark.parametrize("kind", ["mode", "mmd"])
def test_split_statistic_ignores_views_without_valid_pairs(kind):
    codes = np.array([[0, -1], [0, -1], [1, -1], [1, -1]])
    onehots = n.informative_views(codes)
    assert len(onehots) == 1
    masks = s.split_masks(2, 2, 10, np.random.default_rng(0))
    assert np.isfinite(n.split_statistic(onehots, masks, kind)).all()


def test_item_views_normalise_numbered_answers():
    assert n.views_of("items:3", "", "1. Save_Note\n2. New Delhi, India\n3.") == [
        "save note",
        "new delhi india",
        None,
    ]
    assert n.views_of("first:2", "", "1. Breeze.\n2. Monday morning") == ["breeze", "monday"]
