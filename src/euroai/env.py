"""Read-only environment detection.

Nothing here installs, upgrades or configures packages. The module is kept
importable on Python 3.8+ with only the standard library so that
`scripts/check_env.py` can run on a fresh machine before any venv exists.
"""

from __future__ import annotations

import os
import platform
import socket
import sys
from dataclasses import asdict, dataclass, field
from typing import Callable, Mapping, Optional

DEFAULT_PACKAGES = (
    "numpy",
    "torch",
    "torchvision",
    "transformers",
    "datasets",
    "accelerate",
    "scipy",
    "sklearn",
    "pandas",
    "jupyterlab",
    "euroai",
)

DISTRIBUTION_NAMES = {"sklearn": "scikit-learn"}

SLURM_KEYS = (
    "SLURM_JOB_ID",
    "SLURM_JOB_NAME",
    "SLURM_JOB_PARTITION",
    "SLURM_JOB_ACCOUNT",
    "SLURM_JOB_NUM_NODES",
    "SLURM_NTASKS",
    "SLURM_CPUS_PER_TASK",
    "SLURM_GPUS_ON_NODE",
    "SLURM_PROCID",
    "SLURM_LOCALID",
    "CUDA_VISIBLE_DEVICES",
)


@dataclass(frozen=True)
class EnvironmentInfo:
    python_version: str
    python_executable: str
    platform: str
    machine: str
    hostname: str
    virtualenv: Optional[str]
    conda_env: Optional[str]
    in_slurm_job: bool
    slurm: dict = field(default_factory=dict)
    loaded_modules: list = field(default_factory=list)
    packages: dict = field(default_factory=dict)
    accelerators: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def slurm_info(environ: Mapping[str, str]) -> dict:
    return {key: environ[key] for key in SLURM_KEYS if key in environ}


def loaded_modules(environ: Mapping[str, str]) -> list:
    raw = environ.get("LOADEDMODULES", "")
    return [m for m in raw.split(":") if m]


def virtualenv_path(prefix: str, base_prefix: str, environ: Mapping[str, str]) -> Optional[str]:
    if environ.get("VIRTUAL_ENV"):
        return environ["VIRTUAL_ENV"]
    if prefix != base_prefix:
        return prefix
    return None


def package_versions(
    names: tuple = DEFAULT_PACKAGES,
    version_of: Optional[Callable[[str], Optional[str]]] = None,
) -> dict:
    lookup = version_of or _installed_version
    return {name: lookup(DISTRIBUTION_NAMES.get(name, name)) for name in names}


def probe_accelerators() -> dict:
    """Import torch only if it is installed and report CUDA/MPS visibility."""
    from importlib.util import find_spec

    if find_spec("torch") is None:
        return {"torch_installed": False}
    import torch

    info = {
        "torch_installed": True,
        "torch_version": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "cuda_devices": [],
        "mps_available": bool(
            getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
        ),
    }
    for index in range(info["cuda_device_count"]):
        props = torch.cuda.get_device_properties(index)
        info["cuda_devices"].append(
            {"index": index, "name": props.name, "memory_gb": round(props.total_memory / 1e9, 1)}
        )
    return info


def detect_environment(
    environ: Optional[Mapping[str, str]] = None,
    *,
    probe: bool = True,
    version_of: Optional[Callable[[str], Optional[str]]] = None,
) -> EnvironmentInfo:
    env = os.environ if environ is None else environ
    slurm = slurm_info(env)
    return EnvironmentInfo(
        python_version=platform.python_version(),
        python_executable=sys.executable,
        platform=platform.platform(),
        machine=platform.machine(),
        hostname=socket.gethostname(),
        virtualenv=virtualenv_path(sys.prefix, getattr(sys, "base_prefix", sys.prefix), env),
        conda_env=env.get("CONDA_DEFAULT_ENV"),
        in_slurm_job="SLURM_JOB_ID" in slurm,
        slurm=slurm,
        loaded_modules=loaded_modules(env),
        packages=package_versions(version_of=version_of),
        accelerators=probe_accelerators() if probe else {"probed": False},
    )


def _installed_version(distribution: str) -> Optional[str]:
    from importlib import metadata

    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return None
