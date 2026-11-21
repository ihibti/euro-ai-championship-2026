import zipfile

import pytest

from euroai.fetch import extract_zip, extraction_plan, task_data_dir


def make_zip(path, members):
    with zipfile.ZipFile(path, "w") as archive:
        for name, data in members.items():
            archive.writestr(name, data)


def test_task_data_dir():
    assert str(task_data_dir("unlearning")) == "data/unlearning"


def test_extraction_plan_then_extract(tmp_path):
    make_zip(tmp_path / "train.zip", {"train/metadata.json": "[]", "train/images/a.png": "x"})
    make_zip(tmp_path / "test.zip", {"test/metadata.json": "[]"})
    plan = extraction_plan(tmp_path)
    assert plan == [
        (tmp_path / "train.zip", tmp_path / "release"),
        (tmp_path / "test.zip", tmp_path / "release"),
    ]
    for zip_path, destination in plan:
        extract_zip(zip_path, destination)
    assert (tmp_path / "release" / "train" / "images" / "a.png").read_text() == "x"
    assert extraction_plan(tmp_path) == []


def test_extraction_plan_without_zips(tmp_path):
    assert extraction_plan(tmp_path) == []


def test_extract_rejects_unsafe_members(tmp_path):
    make_zip(tmp_path / "evil.zip", {"../outside.txt": "x"})
    with pytest.raises(ValueError, match="outside.txt"):
        extract_zip(tmp_path / "evil.zip", tmp_path / "release")
    assert not (tmp_path / "outside.txt").exists()
