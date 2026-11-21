import pytest

from euroai.config import apply_overrides, deep_merge, get_path, parse_override, parse_value


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("3", 3),
        ("3e-4", 3e-4),
        ("true", True),
        ('["a", "b"]', ["a", "b"]),
        ("baseline", "baseline"),
        ('"quoted"', "quoted"),
    ],
)
def test_parse_value(raw, expected):
    assert parse_value(raw) == expected


def test_parse_override_rejects_malformed():
    with pytest.raises(ValueError):
        parse_override("no_equals_sign")
    with pytest.raises(ValueError):
        parse_override("a..b=1")


def test_apply_overrides_creates_tables_and_does_not_mutate():
    base = {"experiment": {"seed": 1}}
    out = apply_overrides(base, ["experiment.seed=7", "task.lr=0.1"])
    assert out == {"experiment": {"seed": 7}, "task": {"lr": 0.1}}
    assert base == {"experiment": {"seed": 1}}


def test_apply_overrides_refuses_to_descend_into_scalar():
    with pytest.raises(ValueError):
        apply_overrides({"a": 1}, ["a.b=2"])


def test_deep_merge_and_get_path():
    merged = deep_merge({"a": {"x": 1, "y": 2}}, {"a": {"y": 3}, "b": 4})
    assert merged == {"a": {"x": 1, "y": 3}, "b": 4}
    assert get_path(merged, "a.y") == 3
    assert get_path(merged, "a.z", None) is None
    with pytest.raises(KeyError):
        get_path(merged, "missing")
