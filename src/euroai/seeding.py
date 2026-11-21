"""Seeding helpers for reproducible runs.

`derive_seed` gives stable, independent child seeds (per run, per worker, per
data split) from one base seed. `seed_everything` seeds Python, NumPy and, if
installed, PyTorch. Bitwise GPU determinism additionally needs
`CUBLAS_WORKSPACE_CONFIG` to be set before CUDA initialises; `seed_everything`
sets a default for it when `deterministic=True`, which only takes effect if it
is called before the first CUDA operation.
"""

from __future__ import annotations

import hashlib
import os
import random
from dataclasses import asdict, dataclass

import numpy as np

MAX_SEED = 2**32 - 1


def derive_seed(base_seed: int, *keys: object) -> int:
    """Deterministically derive a 32-bit seed from a base seed and any labels."""
    _check_seed(base_seed)
    payload = "\x1f".join([str(int(base_seed)), *(str(k) for k in keys)]).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "little")


def make_rng(seed: int) -> np.random.Generator:
    _check_seed(seed)
    return np.random.default_rng(seed)


@dataclass(frozen=True)
class SeedReport:
    seed: int
    deterministic: bool
    torch_seeded: bool

    def to_dict(self) -> dict:
        return asdict(self)


def seed_everything(seed: int, *, deterministic: bool = False) -> SeedReport:
    """Seed `random`, NumPy's legacy global RNG and PyTorch (if importable).

    `PYTHONHASHSEED` is exported for child processes only; the current
    interpreter's hash seed is fixed at startup.
    """
    _check_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch_seeded = _seed_torch(seed, deterministic)
    return SeedReport(seed=seed, deterministic=deterministic, torch_seeded=torch_seeded)


def torch_worker_init(worker_id: int) -> None:
    """`DataLoader(worker_init_fn=...)` hook that seeds NumPy/random per worker."""
    import torch

    worker_seed = torch.initial_seed() % (MAX_SEED + 1)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _seed_torch(seed: int, deterministic: bool) -> bool:
    try:
        import torch
    except ImportError:
        return False
    torch.manual_seed(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
    return True


def _check_seed(seed: int) -> None:
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise TypeError(f"seed must be an int, got {type(seed).__name__}")
    if not 0 <= int(seed) <= MAX_SEED:
        raise ValueError(f"seed must be in [0, {MAX_SEED}], got {seed}")
