import numpy as np
import pytest

from euroai.kernels import PAD, encode_unicode, hamming_gram, pad_truncate, prompt_hamming_gram


def test_pad_truncate_and_empty_completion():
    out = pad_truncate([[1, 2, 3, 4, 5], [7], []], length=3)
    np.testing.assert_array_equal(out, [[1, 2, 3], [7, PAD, PAD], [PAD, PAD, PAD]])


def test_hamming_examples_from_brief():
    y = pad_truncate([encode_unicode("abc"), encode_unicode("abd"), [], []], length=4)
    gram = hamming_gram(y)
    assert gram[0, 1] == pytest.approx(0.75)
    assert gram[2, 3] == 1.0
    np.testing.assert_allclose(np.diag(gram), 1.0)
    np.testing.assert_allclose(gram, gram.T)


def test_prompt_mask_zeroes_cross_prompt_pairs():
    y = pad_truncate([encode_unicode("same"), encode_unicode("same")], length=4)
    gram = prompt_hamming_gram(np.array([0, 1]), y)
    assert gram[0, 1] == 0.0
    assert gram[0, 0] == 1.0


def test_cross_gram_shape():
    a = pad_truncate([[1, 2], [3, 4], [5, 6]], length=2)
    b = pad_truncate([[1, 2]], length=2)
    gram = prompt_hamming_gram(np.array([0, 0, 1]), a, np.array([0]), b)
    np.testing.assert_allclose(gram, [[1.0], [0.0], [0.0]])


def test_requires_both_other_arguments():
    y = pad_truncate([[1]], length=1)
    with pytest.raises(ValueError):
        prompt_hamming_gram(np.array([0]), y, other_prompt_ids=np.array([0]))
