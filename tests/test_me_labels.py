import pytest

from euroai.me_labels import canonical_labels, family_letters, parse_rows, render_rows
from euroai.me_score import parse_labels, same_model_mcc
from euroai.validators import ME_ENDPOINTS, validate_model_equality


def test_family_letters_run_past_z():
    assert [family_letters(index) for index in (0, 1, 25, 26, 27)] == ["A", "B", "Z", "AA", "AB"]
    with pytest.raises(ValueError):
        family_letters(-1)


def test_canonical_labels_follow_the_organizers_format():
    rows = [
        ("ep-03", "olmo", "x"),
        ("ep-01", "olmo", "x"),
        ("ep-02", "llama", "x"),
        ("ep-04", "olmo", "y"),
    ]
    assert canonical_labels(rows) == [
        ("ep-01", "A", "A1"),
        ("ep-02", "B", "B1"),
        ("ep-03", "A", "A1"),
        ("ep-04", "A", "A2"),
    ]


def test_canonical_labels_keep_the_same_pairs_together():
    text = "endpoint,family,model\nref-01,P,S\nref-02,P,T1\nref-03,P,S\nref-04,Q,S\n"
    rows = parse_rows(text)
    renamed = render_rows(canonical_labels(rows))
    assert same_model_mcc(parse_labels(text), parse_labels(renamed)).mcc == 1.0


def test_canonical_labels_are_stable_when_applied_twice():
    rows = [
        (endpoint, f"f{index % 3}", f"m{index % 5}") for index, endpoint in enumerate(ME_ENDPOINTS)
    ]
    once = canonical_labels(rows)
    assert canonical_labels(once) == once
    assert validate_model_equality(render_rows(once)) == []


def test_parse_rows_rejects_a_short_line():
    with pytest.raises(ValueError):
        parse_rows("ep-01,A\n")
