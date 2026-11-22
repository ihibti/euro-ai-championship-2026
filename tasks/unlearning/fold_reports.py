"""Turn a batch's candidate reports and spill probes into `ledger_inputs.json` entries.

A candidate report (`csd.py`, `trained` on all forget images) gives the realized loss statistics.
Its probe is the same run with a held-out share of the forget images and spare classes; it is
found by comparing the run arguments, so the names do not matter. The probe gives the spill:
held-out forget images stand in for unseen images of the trained classes (45% of the control set)
and spare classes for unseen classes (55%), and from those come retain and the share of changed
validation predictions.

    python <repo>/tasks/unlearning/fold_reports.py /Volumes/euroai-apfs/r21/eighth [--write]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

PROBE_ONLY = ("name", "holdout", "spare_classes", "save", "out")
HELD_SHARE = 0.45


def run_key(args: dict) -> tuple:
    return tuple(sorted((k, json.dumps(v)) for k, v in args.items() if k not in PROBE_ONLY))


def spill_of(probe: dict) -> dict:
    held = probe["held"]["thinned"]["exact"]
    spare = probe["spare"]["thinned"]["exact"]
    return {
        "held_kl": round(held["kl_orig_sub_t2"], 3),
        "held_acc": round(held["acc_vs_reference"], 3),
        "spare_kl": round(spare["kl_orig_sub_t2"], 3),
        "spare_acc": round(spare["acc_vs_reference"], 3),
        "spare_p_lab": round(spare["label_prob_t2_sub"], 3),
    }


def retain_and_flip(spill: dict) -> tuple[float, float]:
    spare_share = 1.0 - HELD_SHARE
    agreement = HELD_SHARE * spill["held_acc"] + spare_share * spill["spare_acc"]
    kl = HELD_SHARE * spill["held_kl"] + spare_share * spill["spare_kl"]
    flip = HELD_SHARE * (1.0 - spill["held_acc"]) + spare_share * (1.0 - spill["spare_acc"])
    return round(agreement / (1.0 + kl), 3), round(flip, 3)


def entry(report: dict, probe: dict | None) -> dict:
    exact = report["trained"]["thinned"]["exact"]
    out = {
        "realized": {
            "loss_mean": round(exact["loss_mean"], 3),
            "loss_std": round(exact["loss_std"], 3),
            "label_prob_t2_sub": round(exact["label_prob_t2_sub"], 3),
            "kl_orig_sub_t2": round(exact["kl_orig_sub_t2"], 3),
        }
    }
    if probe is None:
        return out
    spill = spill_of(probe)
    out["retain"], out["flip"] = retain_and_flip(spill)
    out["spill"] = spill
    return out


def fold(reports: dict[str, dict]) -> dict[str, dict]:
    """Entries for every candidate report (no `held` section), each with its matching probe."""
    probes = {run_key(r["args"]): r for r in reports.values() if "held" in r}
    entries = {}
    for name, report in reports.items():
        if "held" in report or "trained" not in report:
            continue
        entries[name] = entry(report, probes.get(run_key(report["args"])))
    return entries


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("folder", type=Path)
    parser.add_argument(
        "--inputs", type=Path, default=Path(__file__).with_name("ledger_inputs.json")
    )
    parser.add_argument("--write", action="store_true", help="merge into the inputs file")
    args = parser.parse_args(argv)
    reports = {}
    for path in sorted(args.folder.glob("*.json")):
        data = json.loads(path.read_text())
        if isinstance(data, dict) and "args" in data:
            reports[path.stem] = data
    entries = fold(reports)
    for name, spec in entries.items():
        spill = spec.get("spill")
        tail = f"retain {spec['retain']} flip {spec['flip']}" if spill else "no probe"
        real = spec["realized"]
        print(f"{name:<22} loss {real['loss_mean']}/{real['loss_std']} {tail}")
    if args.write:
        inputs = json.loads(args.inputs.read_text())
        for name, spec in entries.items():
            if "retain" in spec:
                inputs["candidates"][name] = spec
        args.inputs.write_text(json.dumps(inputs, indent=1) + "\n")


if __name__ == "__main__":
    main()
