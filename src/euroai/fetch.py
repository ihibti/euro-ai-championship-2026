"""Where task data lives locally, and unpacking the watermark zips as the starter kit does."""

from __future__ import annotations

import zipfile
from pathlib import Path, PurePosixPath

DATA_ROOT = Path("data")
WATERMARK_SPLITS = ("train", "test")


def task_data_dir(task: str, root: Path = DATA_ROOT) -> Path:
    return root / task


def extraction_plan(directory: Path) -> list[tuple[Path, Path]]:
    release = directory / "release"
    return [
        (directory / f"{split}.zip", release)
        for split in WATERMARK_SPLITS
        if (directory / f"{split}.zip").is_file()
        and not (release / split / "metadata.json").is_file()
    ]


def extract_zip(zip_path: Path, destination: Path) -> int:
    with zipfile.ZipFile(zip_path) as archive:
        names = archive.namelist()
        unsafe = [
            name
            for name in names
            if PurePosixPath(name).is_absolute() or ".." in PurePosixPath(name).parts
        ]
        if unsafe:
            raise ValueError(f"{zip_path.name} has unsafe member paths: {', '.join(unsafe[:3])}")
        archive.extractall(destination)
    return len(names)
