import assemble as a


def test_assemble_keeps_families_and_fills_uncovered_ones_with_one_model():
    families = {f"ep-0{i}": ("A" if i < 4 else "B", "x") for i in range(1, 7)}
    within = [{"ep-04": ("F", "base"), "ep-05": ("F", "base"), "ep-06": ("F", "ep-06")}]
    rows = dict((e, (f, m)) for e, f, m in a.assemble(families, within))
    assert rows["ep-01"][1] == rows["ep-02"][1] == rows["ep-03"][1]
    assert rows["ep-04"][1] == rows["ep-05"][1] != rows["ep-06"][1]
    assert rows["ep-01"][0] != rows["ep-04"][0]
    assert {f for f, _ in rows.values()} == {"A", "B"}


def test_assemble_puts_an_uncovered_endpoint_of_a_covered_family_in_its_largest_model():
    families = {e: ("A", "x") for e in ("ep-01", "ep-02", "ep-03", "ep-04")}
    within = [{"ep-01": ("F", "base"), "ep-02": ("F", "base"), "ep-03": ("F", "ep-03")}]
    rows = dict((e, m) for e, _, m in a.assemble(families, within))
    assert rows["ep-04"] == rows["ep-01"] != rows["ep-03"]
