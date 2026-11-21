"""Local checks that mirror each task's submission rules, run before any upload."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Callable, Collection, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from euroai.challenge import get_task
from euroai.me_score import parse_labels

ME_ENDPOINTS = tuple(f"ep-{index:02d}" for index in range(1, 34))
ME_HEADER = ["endpoint", "family", "model"]
ME_FAMILY_LABEL = re.compile(r"[A-Z]+")
ME_MODEL_NUMBER = re.compile(r"[0-9]+")
WATERMARK_KEYS = frozenset({"image_names", "masks"})
WATERMARK_GRID = 128
WATERMARK_TEST_COUNT = 1020
UNLEARNING_ARCH = "vit_base_patch16_224"
UNLEARNING_CLASSES = 100
UNLEARNING_INIT_VALUES = 1e-4
MAX_ERRORS = 10
EXAMPLES = 3

Loader = Callable[[Path], Any]
StrictLoad = Callable[[Mapping[str, Any]], list[str]]


@dataclass(frozen=True)
class ValidationReport:
    errors: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def _truncate(errors: list[str]) -> list[str]:
    if len(errors) <= MAX_ERRORS:
        return errors
    return [*errors[:MAX_ERRORS], f"... and {len(errors) - MAX_ERRORS} more"]


def _examples(items: Collection[str]) -> str:
    shown = ", ".join(sorted(items)[:EXAMPLES])
    return shown + (", ..." if len(items) > EXAMPLES else "")


def _dtype_name(value: Any) -> str:
    return str(value.dtype).removeprefix("torch.")


def _as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    return np.asarray(value)


def _non_binary_values(masks: Any) -> list[Any]:
    values = np.unique(_as_numpy(masks))
    return [value.item() for value in values if value not in (0, 1)][:EXAMPLES]


def validate_watermark(
    obj: Any,
    expected_count: int = WATERMARK_TEST_COUNT,
    expected_names: Collection[str] | None = None,
) -> list[str]:
    if not isinstance(obj, Mapping):
        return [f"expected a dict with keys image_names and masks, got {type(obj).__name__}"]
    errors: list[str] = []
    missing_keys = sorted(WATERMARK_KEYS - set(obj))
    extra_keys = sorted(str(key) for key in set(obj) - WATERMARK_KEYS)
    if missing_keys:
        errors.append(f"missing keys: {', '.join(missing_keys)}")
    if extra_keys:
        errors.append(f"unexpected keys: {', '.join(extra_keys)}; only image_names and masks")
    if missing_keys:
        return errors
    names = obj["image_names"]
    if not isinstance(names, list | tuple) or not all(isinstance(name, str) for name in names):
        return [*errors, "image_names must be a list of strings"]
    with_suffix = [name for name in names if name.endswith(".png")]
    if with_suffix:
        errors.append(
            f"{len(with_suffix)} image names end in .png; use the file name without .png, "
            f"e.g. {with_suffix[0][:-4]!r}"
        )
    duplicates = [name for name, count in Counter(names).items() if count > 1]
    if duplicates:
        errors.append(f"duplicate image names: {_examples(duplicates)}")
    if len(names) != expected_count:
        errors.append(f"expected {expected_count} images, got {len(names)} image names")
    if expected_names is not None:
        unknown = set(names) - set(expected_names)
        absent = set(expected_names) - set(names)
        if unknown:
            errors.append(f"{len(unknown)} names are not test images: {_examples(unknown)}")
        if absent:
            errors.append(f"{len(absent)} test images have no mask: {_examples(absent)}")
    masks = obj["masks"]
    if not hasattr(masks, "shape") or not hasattr(masks, "dtype"):
        return _truncate([*errors, "masks must be a tensor of shape [N, 128, 128]"])
    shape = tuple(int(size) for size in masks.shape)
    if len(shape) != 3 or shape[1:] != (WATERMARK_GRID, WATERMARK_GRID):
        errors.append(f"masks must have shape [N, 128, 128], got {list(shape)}")
    elif shape[0] != len(names):
        errors.append(f"{shape[0]} masks but {len(names)} image names")
    dtype = _dtype_name(masks)
    if dtype != "bool" and not dtype.startswith(("int", "uint")):
        errors.append(f"masks dtype is {dtype}; use bool or uint8")
    else:
        bad = _non_binary_values(masks)
        if bad:
            errors.append(f"mask entries must be 0 or 1; found {bad}")
    return _truncate(errors)


def _me_fields(line: str) -> list[str]:
    return [cell.strip() for cell in line.split(",")]


def validate_model_equality(text: str) -> list[str]:
    errors: list[str] = []
    seen: dict[str, int] = {}
    families_by_model: dict[str, set[str]] = {}
    first_row = True
    for number, line in enumerate(text.lstrip("﻿").splitlines(), start=1):
        fields = _me_fields(line)
        if not any(fields):
            continue
        if first_row and fields == ME_HEADER:
            first_row = False
            continue
        first_row = False
        if len(fields) != 3:
            errors.append(
                f"line {number}: expected 3 fields (endpoint,family,model), got {len(fields)}; "
                "labels may not contain commas"
            )
            continue
        if any('"' in cell for cell in fields):
            errors.append(f"line {number}: quotes are not allowed")
            continue
        endpoint, family, model = fields
        if endpoint.startswith("ref-"):
            errors.append(
                f"line {number}: {endpoint} is a reference endpoint; "
                "the server accepts only ep-01 to ep-33"
            )
            continue
        if endpoint not in ME_ENDPOINTS:
            errors.append(f"line {number}: unknown endpoint {endpoint!r}")
            continue
        if endpoint in seen:
            errors.append(f"{endpoint} appears twice (lines {seen[endpoint]} and {number})")
            continue
        seen[endpoint] = number
        if not family or not model:
            errors.append(f"line {number}: empty family or model label for {endpoint}")
            continue
        if not ME_FAMILY_LABEL.fullmatch(family):
            errors.append(
                f"line {number}: family {family!r} must be capital letters such as A, "
                "the organizers' format; rewrite the file with scripts/me_labels.py"
            )
            continue
        if not (model.startswith(family) and ME_MODEL_NUMBER.fullmatch(model[len(family) :])):
            errors.append(
                f"line {number}: model {model!r} must be the family letter and a number, "
                f"such as {family}1; rewrite the file with scripts/me_labels.py"
            )
            continue
        families_by_model.setdefault(model, set()).add(family)
    missing = [endpoint for endpoint in ME_ENDPOINTS if endpoint not in seen]
    if missing:
        errors.append(f"missing endpoints: {', '.join(missing)}")
    for model, families in sorted(families_by_model.items()):
        if len(families) > 1:
            errors.append(
                f"model label {model!r} is used with families {', '.join(sorted(families))}; "
                "endpoints with the same model label must share the family label"
            )
    return _truncate(errors)


def _all_finite(value: Any) -> bool:
    isfinite = getattr(value, "isfinite", None)
    if callable(isfinite):
        return bool(isfinite().all())
    return bool(np.isfinite(np.asarray(value)).all())


def validate_unlearning(state_dict: Any) -> list[str]:
    if not isinstance(state_dict, Mapping):
        return [
            f"expected a state dict of tensors, got {type(state_dict).__name__}; "
            "save model.state_dict()"
        ]
    if not state_dict:
        return ["the state dict is empty"]
    errors: list[str] = []
    for key, value in state_dict.items():
        if isinstance(value, Mapping):
            errors.append(
                f"{key!r} holds a dict, not a tensor; save model.state_dict() itself, "
                "not a checkpoint that wraps it"
            )
        elif not hasattr(value, "dtype"):
            errors.append(f"{key!r} is a {type(value).__name__}, not a tensor")
        elif _dtype_name(value) != "float32":
            errors.append(f"{key!r} is {_dtype_name(value)}; only float32 is accepted")
        elif not _all_finite(value):
            errors.append(f"{key!r} contains NaN or infinite values")
    return _truncate(errors)


def unlearning_strict_load(state_dict: Mapping[str, Any]) -> list[str]:
    import timm

    model = timm.create_model(
        UNLEARNING_ARCH,
        pretrained=False,
        num_classes=UNLEARNING_CLASSES,
        init_values=UNLEARNING_INIT_VALUES,
    )
    try:
        model.load_state_dict(state_dict, strict=True)
    except (RuntimeError, TypeError, ValueError) as exc:
        return [f"strict load into {UNLEARNING_ARCH} failed: {str(exc)[:500]}"]
    return []


def load_torch_file(path: Path) -> Any:
    import torch

    return torch.load(path, map_location="cpu", weights_only=True)


def _model_equality_report(path: Path) -> ValidationReport:
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return ValidationReport([f"{path.name} is not UTF-8 text"])
    errors = validate_model_equality(text)
    if errors:
        return ValidationReport(errors)
    labels = list(parse_labels(text).values())
    families = {family for family, _ in labels}
    note = f"{len(labels)} endpoints, {len(families)} family labels, {len(set(labels))} models"
    return ValidationReport([], [note])


def _watermark_report(obj: Any, expected_names: Collection[str] | None) -> ValidationReport:
    errors = validate_watermark(obj, expected_names=expected_names)
    if errors:
        return ValidationReport(errors)
    masks = _as_numpy(obj["masks"])
    notes = [f"{masks.shape[0]} masks, {masks.mean():.1%} of cells marked"]
    if expected_names is None:
        notes.append("image names not checked against the test set (test metadata not found)")
    return ValidationReport([], notes)


def _unlearning_report(obj: Any, strict_load: StrictLoad) -> ValidationReport:
    errors = validate_unlearning(obj)
    if errors:
        return ValidationReport(errors)
    note = f"{len(obj)} float32 tensors, all finite"
    try:
        load_errors = strict_load(obj)
    except ImportError as exc:
        return ValidationReport([], [note, f"strict load skipped ({exc}); install timm to run it"])
    if load_errors:
        return ValidationReport(load_errors, [note])
    return ValidationReport([], [note, f"strict load into {UNLEARNING_ARCH} passed"])


def validate_file(
    task: str,
    path: str | Path,
    *,
    load_pt: Loader | None = None,
    strict_load: StrictLoad | None = None,
    expected_names: Collection[str] | None = None,
) -> ValidationReport:
    info = get_task(task)
    path = Path(path)
    if not path.is_file():
        return ValidationReport([f"no such file: {path}"])
    if path.suffix not in info.suffixes:
        allowed = " or ".join(info.suffixes)
        return ValidationReport([f"{task} needs a {allowed} file, got {path.name}"])
    size = path.stat().st_size
    if info.max_bytes is not None and size > info.max_bytes:
        limit = info.max_bytes
        return ValidationReport([f"{path.name} is {size} bytes; the limit is {limit} bytes"])
    if task == "model_equality":
        return _model_equality_report(path)
    try:
        obj = (load_pt or load_torch_file)(path)
    except ImportError as exc:
        return ValidationReport([f"cannot read {path.name} ({exc}); run with the torch extra"])
    except Exception as exc:
        return ValidationReport(
            [f"could not load {path.name} with torch.load(weights_only=True): {exc}"]
        )
    if task == "watermark_localization":
        return _watermark_report(obj, expected_names)
    return _unlearning_report(obj, strict_load or unlearning_strict_load)
