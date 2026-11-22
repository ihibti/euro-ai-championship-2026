"""Write a model-equality grouping: per family one group of same-model endpoints, every other endpoint alone.

python make_grouping.py out.csv A=02,23,33,14,04 B=08,20,30,07,06 C=29,16,10
"""

from __future__ import annotations

import sys

FAMILIES = {
    "A": [1, 2, 4, 12, 14, 17, 18, 23, 24, 25, 33],
    "B": [3, 6, 7, 8, 20, 21, 22, 26, 28, 30, 32],
    "C": [5, 9, 10, 11, 13, 15, 16, 19, 27, 29, 31],
}


def grouping_rows(groups: dict[str, list[int]]) -> list[tuple[str, str, str]]:
    rows = []
    for family, members in FAMILIES.items():
        group = groups.get(family, [])
        unknown = sorted(set(group) - set(members))
        if unknown:
            raise ValueError(f"family {family}: not its endpoints: {unknown}")
        if len(set(group)) != len(group):
            raise ValueError(f"family {family}: an endpoint is listed twice")
        counter = 1
        for number in members:
            if number in group:
                label = f"{family}1"
            else:
                counter += 1
                label = f"{family}{counter}"
            rows.append((f"ep-{number:02d}", family, label))
    return sorted(rows)


def render(rows: list[tuple[str, str, str]]) -> str:
    return "endpoint,family,model\n" + "".join(f"{e},{f},{m}\n" for e, f, m in rows)


def main(argv: list[str]) -> int:
    groups = {}
    for arg in argv[2:]:
        family, _, members = arg.partition("=")
        groups[family] = [int(x) for x in members.split(",") if x]
    rows = grouping_rows(groups)
    open(argv[1], "w").write(render(rows))
    models = {m for _, _, m in rows}
    print(f"wrote {argv[1]}: {len(rows)} endpoints, {len(models)} models; groups {groups}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
