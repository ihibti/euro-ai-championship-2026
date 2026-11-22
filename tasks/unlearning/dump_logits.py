"""Logits of candidate state dicts on the forget images, in one small file, for the laptop's ledger.

With the logits of a candidate, `ledger.py` scores it exactly under every gold hypothesis instead
of through a proxy fitted to its summary statistics. One float16 array per candidate and view.
Run in the JURECA task folder, on a GPU:

    python <repo>/tasks/unlearning/dump_logits.py --names x3_ext_m80,x4_e20 --views exact,jitter \
        --out output/kkrgp/seventh/logits.npz

Writes `<name>__<view>` [N, 100] in float16, and `orig__<view>` for the original model.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import lib
import numpy as np
import torch
from lib import tt


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--names", required=True, help="comma-separated state dicts in --dir")
    parser.add_argument("--dir", type=Path, default=Path("output/kkrgp"))
    parser.add_argument("--views", default="exact,jitter")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    views = tuple(args.views.split(","))
    images = tt.load_forget_images()
    model = tt.load_model().to(lib.DEVICE).eval()
    arrays: dict[str, np.ndarray] = {}
    with torch.no_grad():
        for view, logits in lib.view_logits(model, images, views).items():
            arrays[f"orig__{view}"] = logits.numpy().astype(np.float16)
        for name in args.names.split(","):
            path = args.dir / f"{name}.pt"
            if not path.exists():
                print(f"missing {path}", flush=True)
                continue
            state = torch.load(path, map_location="cpu", weights_only=True)
            model.load_state_dict(state, strict=True)
            model.to(lib.DEVICE).eval()
            for view, logits in lib.view_logits(model, images, views).items():
                arrays[f"{name}__{view}"] = logits.numpy().astype(np.float16)
            print(f"{name} done", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **arrays)
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
