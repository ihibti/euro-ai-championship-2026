"""TOML experiment configs with dotted command-line overrides.

Configs are plain nested dicts. Overrides use `section.key=value` syntax and
values are parsed as TOML literals (`lr=3e-4`, `tags=["a","b"]`,
`use_amp=true`), falling back to a bare string (`name=baseline`).
"""

from __future__ import annotations

import copy
import sys
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

_MISSING = object()


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, "rb") as handle:
        return tomllib.load(handle)


def parse_value(raw: str) -> Any:
    try:
        return tomllib.loads(f"v = {raw}")["v"]
    except tomllib.TOMLDecodeError:
        return raw


def parse_override(text: str) -> tuple[list[str], Any]:
    key, sep, raw = text.partition("=")
    keys = [part.strip() for part in key.split(".")]
    if not sep or not all(keys):
        raise ValueError(f"override must look like 'a.b=value', got {text!r}")
    return keys, parse_value(raw.strip())


def set_path(config: Mapping[str, Any], keys: list[str], value: Any) -> dict[str, Any]:
    result = copy.deepcopy(dict(config))
    node = result
    for key in keys[:-1]:
        child = node.get(key)
        if child is None:
            child = node[key] = {}
        elif not isinstance(child, dict):
            raise ValueError(f"cannot set {'.'.join(keys)}: {key!r} is not a table")
        node = child
    node[keys[-1]] = value
    return result


def apply_overrides(config: Mapping[str, Any], overrides: Iterable[str]) -> dict[str, Any]:
    result = copy.deepcopy(dict(config))
    for text in overrides:
        keys, value = parse_override(text)
        result = set_path(result, keys, value)
    return result


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def get_path(config: Mapping[str, Any], dotted: str, default: Any = _MISSING) -> Any:
    node: Any = config
    for key in dotted.split("."):
        if not isinstance(node, Mapping) or key not in node:
            if default is _MISSING:
                raise KeyError(dotted)
            return default
        node = node[key]
    return node
