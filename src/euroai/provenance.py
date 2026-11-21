"""Run directories, manifests and file checksums."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

Runner = Callable[..., subprocess.CompletedProcess]


@dataclass(frozen=True)
class GitState:
    commit: str | None
    branch: str | None
    dirty: bool | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def git_state(repo_dir: str | Path = ".", run: Runner = subprocess.run) -> GitState:
    def git(*args: str) -> str | None:
        try:
            proc = run(
                ["git", "-C", str(repo_dir), *args],
                capture_output=True,
                text=True,
                check=False,
                timeout=10,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return proc.stdout.strip() if proc.returncode == 0 else None

    commit = git("rev-parse", "HEAD")
    if commit is None:
        return GitState(commit=None, branch=None, dirty=None)
    status = git("status", "--porcelain")
    return GitState(
        commit=commit,
        branch=git("rev-parse", "--abbrev-ref", "HEAD"),
        dirty=None if status is None else bool(status),
    )


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def slugify(name: str, max_length: int = 60) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return slug[:max_length].rstrip("-") or "run"


def make_run_dir(root: str | Path, name: str, now: datetime | None = None) -> Path:
    timestamp = (now or datetime.now(timezone.utc)).strftime("%Y%m%dT%H%M%SZ")
    base = Path(root) / f"{timestamp}_{slugify(name)}"
    candidate, suffix = base, 1
    while True:
        try:
            candidate.mkdir(parents=True, exist_ok=False)
            return candidate
        except FileExistsError:
            suffix += 1
            candidate = base.with_name(f"{base.name}-{suffix}")


def write_json(path: str | Path, payload: Mapping[str, Any]) -> Path:
    path = Path(path)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    return path


def build_manifest(
    *,
    config: Mapping[str, Any],
    seed: int,
    environment: Mapping[str, Any],
    git: GitState,
    started_at: datetime,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "started_at": started_at.isoformat(),
        "seed": seed,
        "config": dict(config),
        "git": git.to_dict(),
        "environment": dict(environment),
        **(dict(extra) if extra else {}),
    }
