import pytest

from euroai.submission import read_predictions_csv, validate_predictions, write_predictions_csv
from euroai.tracking import JsonlLogger, read_jsonl


def test_predictions_roundtrip(tmp_path):
    path = write_predictions_csv(tmp_path / "sub" / "submission.csv", ["a", "b"], [0.1, 1.0])
    assert path.read_text().splitlines()[0] == "id,label"
    assert read_predictions_csv(path) == (["a", "b"], [0.1, 1.0])


@pytest.mark.parametrize(
    ("ids", "values"),
    [
        (["a", "a"], [0.1, 0.2]),
        (["a"], [1.5]),
        (["a"], [float("nan")]),
        (["a", "b"], [0.1]),
        ([], []),
    ],
)
def test_invalid_predictions_rejected(ids, values):
    with pytest.raises(ValueError):
        validate_predictions(ids, values)


def test_unbounded_values_allowed_when_range_disabled():
    validate_predictions(["a"], [12.0], value_range=None)


def test_jsonl_logger(tmp_path):
    with JsonlLogger(tmp_path / "log.jsonl", clock=lambda: 1.0) as log:
        log.log(step=0, loss=2.5)
        log.log(step=1, loss=2.0)
    records = read_jsonl(tmp_path / "log.jsonl")
    assert records == [
        {"time": 1.0, "step": 0, "loss": 2.5},
        {"time": 1.0, "step": 1, "loss": 2.0},
    ]
