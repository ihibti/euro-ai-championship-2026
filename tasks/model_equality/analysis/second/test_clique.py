import clique as c


def pairs(*items):
    return {frozenset(p) for p in items}


def test_max_cliques_finds_every_largest_group():
    nodes = ["a", "b", "c", "d"]
    allowed = pairs(("a", "b"), ("a", "c"), ("b", "c"), ("c", "d"))
    assert c.max_cliques(nodes, allowed) == [("a", "b", "c")]


def test_max_cliques_reports_ties_and_singletons():
    assert c.max_cliques(["a", "b", "c"], pairs(("a", "b"), ("b", "c"))) == [("a", "b"), ("b", "c")]
    assert c.max_cliques(["a", "b"], set()) == [("a",), ("b",)]
