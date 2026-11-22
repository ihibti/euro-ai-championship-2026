import pytest

from make_grouping import FAMILIES, grouping_rows, render


def test_every_endpoint_once_and_group_shares_a_label():
    rows = grouping_rows({"A": [2, 23, 33], "B": [], "C": [29, 16]})
    assert len(rows) == 33
    assert len({e for e, _, _ in rows}) == 33
    labels = {e: m for e, _, m in rows}
    assert labels["ep-02"] == labels["ep-23"] == labels["ep-33"] == "A1"
    assert labels["ep-29"] == labels["ep-16"] == "C1"
    assert len({m for _, f, m in rows if f == "A"}) == 1 + 8
    assert len({m for _, f, m in rows if f == "B"}) == 11


def test_family_column_matches_known_families():
    rows = grouping_rows({})
    for endpoint, family, model in rows:
        assert int(endpoint[3:]) in FAMILIES[family]
        assert model.startswith(family)


def test_wrong_family_or_duplicate_is_refused():
    with pytest.raises(ValueError):
        grouping_rows({"A": [3]})
    with pytest.raises(ValueError):
        grouping_rows({"A": [2, 2]})


def test_render_has_header_and_33_lines():
    text = render(grouping_rows({"A": [2, 23]}))
    assert text.splitlines()[0] == "endpoint,family,model"
    assert len(text.splitlines()) == 34
