"""Secrets: the process environment first, then a KEY=VALUE file outside every checkout.

Values are never printed. Messages name only the variable and the file path.
"""

from __future__ import annotations

import os
import re
import stat
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_SECRETS_FILE = "~/.config/euroai/.env"
SECRETS_FILE_VAR = "EUROAI_SECRETS_FILE"
API_KEY_VAR = "HACKATHON_API_KEY"
HF_TOKEN_VAR = "HF_TOKEN"

FileReader = Callable[[Path], str | None]


@dataclass(frozen=True)
class SecretLookup:
    name: str
    path: Path
    source: str
    value: str | None = field(default=None, repr=False)


def _value(raw: str) -> str:
    raw = raw.strip()
    if raw and raw[0] in "'\"":
        end = raw.find(raw[0], 1)
        if end > 0:
            return raw[1:end]
        return raw
    return re.split(r"\s+#", raw, maxsplit=1)[0].rstrip()


def parse_env_text(text: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip().removeprefix("export ").strip()
        if key:
            values[key] = _value(value)
    return values


def secrets_file(env: Mapping[str, str]) -> Path:
    return Path(env.get(SECRETS_FILE_VAR) or DEFAULT_SECRETS_FILE).expanduser()


def read_text_or_none(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None


def find_secret(
    name: str, env: Mapping[str, str], read_file: FileReader = read_text_or_none
) -> SecretLookup:
    path = secrets_file(env)
    if env.get(name):
        return SecretLookup(name, path, "environment", env[name])
    text = read_file(path)
    value = parse_env_text(text).get(name) if text is not None else None
    if value:
        return SecretLookup(name, path, "file", value)
    return SecretLookup(name, path, "missing")


def missing_secret_message(lookup: SecretLookup) -> str:
    return (
        f"{lookup.name} is not set. Export it, or add a line {lookup.name}=<value> to "
        f"{lookup.path} (chmod 600; {SECRETS_FILE_VAR} overrides the path)."
    )


def permission_warning(path: Path, mode: int) -> str | None:
    if stat.S_IMODE(mode) & 0o077:
        return f"warning: {path} is readable by other users; run chmod 600 {path}"
    return None


def get_secret(name: str, env: Mapping[str, str] | None = None) -> SecretLookup:
    return find_secret(name, os.environ if env is None else env)
