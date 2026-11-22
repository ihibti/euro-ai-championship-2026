from fp8_rule import decide, readings_for

PAIR = ["ep-22", "ep-32"]
LAB = {"ep-06": "0.7", "ep-07": "1.0", "ep-30": "1.0"}


def r(z, lr=None):
    return {"z": z, "lr": lr if lr is not None else z * 10.0}


def base():
    return {"ep-03": r(-2), "ep-21": r(-1), "ep-22": r(-2.5), "ep-26": r(-1.5), "ep-28": r(-3), "ep-32": r(-2), "ep-06": r(-2), "ep-07": r(-3), "ep-30": r(-4)}


def test_fp8_inside_the_pair_adds_the_other_member():
    x = base()
    x["ep-22"] = r(4.2)
    assert decide(x, PAIR, LAB, {}) == {"add": ["ep-32"], "fp8": "ep-22", "case": "fp8 inside the pair"}


def test_no_positive_changes_nothing():
    assert decide(base(), PAIR, LAB, {})["add"] == []


def test_two_positives_change_nothing():
    x = base()
    x["ep-22"] = r(3.1)
    x["ep-26"] = r(5.0)
    d = decide(x, PAIR, LAB, {})
    assert d["add"] == [] and d["fp8"] is None


def test_fp8_outside_adds_members_passing_both_guards():
    x = base()
    x["ep-26"] = r(4.0)
    d = decide(x, PAIR, LAB, {"ep-22": -1.4, "ep-32": 0.4})
    assert d == {"add": ["ep-22", "ep-32"], "fp8": "ep-26", "case": "fp8 outside the pair"}


def test_fp8_outside_drops_member_flagged_by_stop_channel():
    x = base()
    x["ep-26"] = r(4.0)
    assert decide(x, PAIR, LAB, {"ep-22": -2.4, "ep-32": 0.4})["add"] == ["ep-32"]


def test_fp8_outside_drops_member_with_positive_ratio():
    x = base()
    x["ep-26"] = r(4.0)
    x["ep-32"] = r(1.2)
    assert decide(x, PAIR, LAB, {"ep-22": -1.0, "ep-32": 0.4})["add"] == ["ep-22"]


def test_fp8_outside_without_stop_reading_adds_nothing():
    x = base()
    x["ep-26"] = r(4.0)
    assert decide(x, PAIR, LAB, {})["add"] == []


def test_labelled_copy_positive_fails_the_check():
    x = base()
    x["ep-30"] = r(3.5)
    x["ep-22"] = r(4.0)
    d = decide(x, PAIR, LAB, {})
    assert d["add"] == [] and d["case"].startswith("check failed")


def test_threshold_is_inclusive_at_three():
    x = base()
    x["ep-32"] = r(3.0)
    assert decide(x, PAIR, LAB, {})["add"] == ["ep-22"]


def test_readings_use_each_endpoints_own_temperature():
    plain = {
        "a|0": {"endpoint": "ep-30", "lp": {"0.7": [-1.0, -1.0], "1.0": [-2.0, -2.0]}},
        "a|1": {"endpoint": "ep-30", "lp": {"0.7": [-1.0, -1.0], "1.0": [-2.0, -2.0]}},
        "b|0": {"endpoint": "ep-22", "lp": {"0.7": [-1.0], "1.0": [-2.0]}},
        "b|1": {"endpoint": "ep-22", "lp": {"0.7": [-1.0], "1.0": [-2.0]}},
    }
    fp8 = {
        "a|0": {"endpoint": "ep-30", "lp": {"0.7": [-1.0, -1.0], "1.0": [-2.5, -2.0]}},
        "a|1": {"endpoint": "ep-30", "lp": {"0.7": [-1.0, -1.0], "1.0": [-2.0, -2.1]}},
        "b|0": {"endpoint": "ep-22", "lp": {"0.7": [-0.8], "1.0": [-2.0]}},
        "b|1": {"endpoint": "ep-22", "lp": {"0.7": [-0.7], "1.0": [-2.0]}},
    }
    out = readings_for(plain, fp8, lambda ep: "1.0" if ep == "ep-30" else "0.7")
    assert out["ep-30"]["tau"] == "1.0" and abs(out["ep-30"]["lr"] + 0.6) < 1e-9 and out["ep-30"]["tokens"] == 4
    assert out["ep-22"]["tau"] == "0.7" and abs(out["ep-22"]["lr"] - 0.5) < 1e-9 and out["ep-22"]["answers"] == 2
