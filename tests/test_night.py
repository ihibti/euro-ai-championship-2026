from datetime import datetime, timedelta, timezone

import pytest

from euroai.night import (
    NIGHT,
    load_authorization,
    night_file,
    parse_authorization,
    spend_decision,
    spend_refusal,
    submission_refusal,
)

TEXT = (
    '{"expires": "2026-11-22T09:00:00+01:00", "answers_per_endpoint": 50, '
    '"submissions": {"model_equality": 4, "unlearning": 12}}'
)
BEFORE = datetime(2026, 11, 22, 1, 0, tzinfo=timezone.utc)
END = datetime(2026, 11, 22, 8, 0, tzinfo=timezone.utc)


def test_night_file_default_and_override(tmp_path):
    assert night_file({}).name == "night.json" and night_file({}).parent.name == "euroai"
    assert night_file({"EUROAI_NIGHT_FILE": str(tmp_path / "n.json")}) == tmp_path / "n.json"


def test_parse_reads_caps():
    night = parse_authorization(TEXT)
    assert night.expires == END
    assert night.answers_per_endpoint == 50
    assert night.submissions == {"model_equality": 4, "unlearning": 12}


@pytest.mark.parametrize(
    "text",
    [
        "not json",
        "[1, 2]",
        '{"answers_per_endpoint": 5}',
        '{"expires": "tomorrow"}',
        '{"expires": "2026-11-22T09:00:00"}',
        '{"expires": "2026-11-22T09:00:00+01:00", "answers_per_endpoint": -1}',
        '{"expires": "2026-11-22T09:00:00+01:00", "answers_per_endpoint": true}',
        '{"expires": "2026-11-22T09:00:00+01:00", "submissions": [1]}',
        '{"expires": "2026-11-22T09:00:00+01:00", "submissions": {"unlearning": "3"}}',
    ],
)
def test_parse_rejects_malformed(text):
    with pytest.raises(ValueError):
        parse_authorization(text)


def test_missing_caps_allow_nothing():
    night = parse_authorization('{"expires": "2026-11-22T09:00:00+01:00"}')
    assert night.answers_per_endpoint == 0 and night.submissions == {}
    assert submission_refusal(night, "unlearning", []) is not None
    assert spend_refusal(night, {"ep-01": 1}, {"ep-01": 0}, []) is not None


def test_load_missing_unreadable_expired_and_valid():
    assert load_authorization(None, BEFORE)[0] is None
    night, reason = load_authorization("{", BEFORE)
    assert night is None and "unreadable" in reason
    night, reason = load_authorization(TEXT, END)
    assert night is None and "ended" in reason
    night, reason = load_authorization(TEXT, END - timedelta(seconds=1))
    assert night is not None and "until" in reason


def test_submission_cap_counts_night_records_only():
    night = parse_authorization(TEXT)
    human = [{"authorization": "human"}, {}]
    assert submission_refusal(night, "model_equality", human) is None
    three = [{"authorization": NIGHT}] * 3
    assert submission_refusal(night, "model_equality", three + human) is None
    assert "4" in submission_refusal(night, "model_equality", three + [{"authorization": NIGHT}])
    assert submission_refusal(night, "watermark_localization", []) is not None


def test_spend_cap_per_endpoint_and_rehearsal():
    night = parse_authorization(TEXT)
    used = {"ep-01": 10, "ep-02": 45}
    assert spend_refusal(night, {"ep-01": 40, "ep-02": 5}, used, []) is None
    refusal = spend_refusal(night, {"ep-01": 40, "ep-02": 6}, used, [])
    assert "ep-02" in refusal and "ep-01" not in refusal
    assert spend_refusal(night, {"ep-01": 0, "ep-02": 0}, {"ep-01": 99, "ep-02": 99}, []) is None
    assert "p7" in spend_refusal(night, {"ep-01": 1}, used, ["p7"])


def test_spend_decision_joins_every_guard():
    planned, used = {"ep-01": 2}, {"ep-01": 0}
    assert spend_decision(None, BEFORE, planned, used, []) == (
        False,
        "there is no night authorization",
    )
    assert spend_decision(TEXT, END, planned, used, [])[0] is False
    assert spend_decision(TEXT, BEFORE, {"ep-01": 51}, used, [])[0] is False
    assert spend_decision(TEXT, BEFORE, planned, used, ["p1"])[0] is False
    allowed, reason = spend_decision(TEXT, BEFORE, planned, used, [])
    assert allowed and "until" in reason
