"""Synthetic training images for watermark localization, built the way the dataset was built.

What the train split shows (see probe_process.py): half of the images were watermarked on the full
frame and a part restored to the clean original (holes); a quarter carry the watermark only on an
axis-aligned rectangle, embedded on that crop; a quarter carry only an object-shaped part of a
full-frame watermark.
"""

from __future__ import annotations

from collections.abc import Callable

import cv2
import numpy as np

SIDE = 512
GRID = 128
CELL = 4
KINDS = ("hole", "rect", "object")
KIND_WEIGHTS = (0.5, 0.25, 0.25)
MAX_ASPECT = {"pixelseal": 16.0, "stegastamp": 16.0, "trustmark": 2.0}

Embed = Callable[[np.ndarray, np.ndarray], np.ndarray]


def to_grid(mask: np.ndarray) -> np.ndarray:
    cells = mask.astype(np.float32).reshape(GRID, CELL, GRID, CELL).mean(axis=(1, 3))
    return (cells >= 0.5).astype(np.uint8)


def log_uniform(rng: np.random.Generator, low: float, high: float) -> float:
    return float(np.exp(rng.uniform(np.log(low), np.log(high))))


def blob_shape(rng: np.random.Generator, share: float) -> np.ndarray:
    coarse = int(rng.choice((3, 4, 6, 8, 12, 16)))
    field = cv2.resize(
        rng.standard_normal((coarse, coarse)).astype(np.float32),
        (SIDE, SIDE),
        interpolation=cv2.INTER_CUBIC,
    )
    return (field >= np.quantile(field, 1.0 - share)).astype(np.uint8)


def polygon_shape(rng: np.random.Generator, share: float) -> np.ndarray:
    radius = np.sqrt(share * SIDE * SIDE / np.pi)
    centre = rng.uniform(0.15 * SIDE, 0.85 * SIDE, size=2)
    corners = int(rng.integers(3, 9))
    angles = np.sort(rng.uniform(0, 2 * np.pi, corners))
    radii = radius * rng.uniform(0.5, 1.6, corners)
    points = centre + np.stack([radii * np.cos(angles), radii * np.sin(angles)], 1)
    shape = np.zeros((SIDE, SIDE), np.uint8)
    cv2.fillPoly(shape, [np.round(points).astype(np.int32)], 1)
    return shape


def band_shape(rng: np.random.Generator, share: float) -> np.ndarray:
    xs = np.arange(SIDE, dtype=np.float32)
    wave = rng.uniform(0, 40) * np.sin(xs / rng.uniform(30, 200) + rng.uniform(0, 6.28))
    edge = share * SIDE + wave
    shape = (np.arange(SIDE, dtype=np.float32)[:, None] < edge[None, :]).astype(np.uint8)
    return np.rot90(shape, int(rng.integers(0, 4))).copy()


def box_shape(rng: np.random.Generator, share: float) -> np.ndarray:
    y0, y1, x0, x1 = rect_frame(rng, 4.0, share)
    shape = np.zeros((SIDE, SIDE), np.uint8)
    shape[y0:y1, x0:x1] = 1
    return shape


def vary(rng: np.random.Generator, shape: np.ndarray) -> np.ndarray:
    if rng.random() < 0.5:
        shape = shape[:, ::-1]
    if rng.random() < 0.5:
        shape = shape[::-1]
    if rng.random() < 0.5:
        shape = shape.T
    dy, dx = (int(v) for v in rng.integers(-64, 65, size=2))
    moved = np.zeros_like(shape)
    src = shape[max(0, -dy) : SIDE - max(0, dy), max(0, -dx) : SIDE - max(0, dx)]
    moved[max(0, dy) : max(0, dy) + src.shape[0], max(0, dx) : max(0, dx) + src.shape[1]] = src
    return np.ascontiguousarray(moved)


def rect_frame(
    rng: np.random.Generator, max_aspect: float, share: float | None = None
) -> tuple[int, int, int, int]:
    while True:
        if share is None:
            width = log_uniform(rng, 9, SIDE)
            height = log_uniform(rng, 9, SIDE)
        else:
            aspect = log_uniform(rng, 1 / max_aspect, max_aspect)
            area = share * SIDE * SIDE
            width, height = np.sqrt(area * aspect), np.sqrt(area / aspect)
        width, height = int(round(width)), int(round(height))
        if not (4 <= width <= SIDE and 4 <= height <= SIDE):
            continue
        if max(width / height, height / width) > max_aspect:
            continue
        if share is None and width * height > 0.4 * SIDE * SIDE:
            continue
        y0 = int(rng.integers(0, SIDE - height + 1))
        x0 = int(rng.integers(0, SIDE - width + 1))
        return y0, y0 + height, x0, x0 + width


def draw_shape(rng: np.random.Generator, bank: list[np.ndarray], low: float, high: float):
    for _ in range(20):
        pick = rng.random()
        share = log_uniform(rng, low, high)
        if bank and pick < 0.5:
            shape = vary(rng, bank[int(rng.integers(len(bank)))])
        elif pick < 0.8:
            shape = blob_shape(rng, share)
        elif pick < 0.9:
            shape = polygon_shape(rng, share)
        elif pick < 0.95:
            shape = band_shape(rng, share)
        else:
            shape = box_shape(rng, share)
        if 0.001 <= shape.mean() <= 0.6:
            return shape
    return blob_shape(rng, 0.1)


def strengthen(cover: np.ndarray, marked: np.ndarray, strength: float) -> np.ndarray:
    if strength == 1.0:
        return marked
    change = marked.astype(np.float32) - cover.astype(np.float32)
    return np.clip(np.rint(cover + strength * change), 0, 255).astype(np.uint8)


def make_sample(
    rng: np.random.Generator,
    cover: np.ndarray,
    embed: Embed,
    nbits: int,
    method: str,
    bank: list[np.ndarray],
    strength: float = 1.0,
    bits: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, str]:
    kind = str(rng.choice(KINDS, p=KIND_WEIGHTS))
    if bits is None:
        bits = rng.integers(0, 2, nbits)
    if kind == "rect":
        y0, y1, x0, x1 = rect_frame(rng, MAX_ASPECT[method])
        mask = np.zeros((SIDE, SIDE), np.uint8)
        mask[y0:y1, x0:x1] = 1
        image = cover.copy()
        crop = np.ascontiguousarray(cover[y0:y1, x0:x1])
        image[y0:y1, x0:x1] = strengthen(crop, embed(crop, bits), strength)
        return image, mask, kind
    if kind == "hole":
        mask = 1 - draw_shape(rng, bank, 0.003, 0.5)
    else:
        mask = draw_shape(rng, bank, 0.004, 0.45)
    marked = strengthen(cover, embed(cover, bits), strength)
    image = np.where(mask[:, :, None].astype(bool), marked, cover)
    return image, mask, kind
