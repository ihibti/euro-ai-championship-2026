import json
import subprocess
from datetime import datetime, timezone

from euroai.provenance import (
    build_manifest,
    git_state,
    make_run_dir,
    sha256_file,
    slugify,
    write_json,
)

FIXED = datetime(2026, 11, 21, 12, 0, 0, tzinfo=timezone.utc)


def test_slugify():
    assert slugify("My Run: v2!") == "my-run-v2"
    assert slugify("!!!") == "run"


def test_make_run_dir_is_unique(tmp_path):
    first = make_run_dir(tmp_path, "exp", FIXED)
    second = make_run_dir(tmp_path, "exp", FIXED)
    assert first.name == "20261121T120000Z_exp"
    assert second.name == "20261121T120000Z_exp-2"


def test_sha256_file(tmp_path):
    path = tmp_path / "x.bin"
    path.write_bytes(b"abc")
    assert sha256_file(path) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def test_git_state_outside_repo_and_with_fake_runner(tmp_path):
    assert git_state(tmp_path).commit is None

    def fake_run(cmd, **_):
        outputs = {"--abbrev-ref": "main", "--porcelain": " M file", "HEAD": "abc123"}
        key = next(k for k in outputs if k in cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout=outputs[key] + "\n", stderr="")

    state = git_state(tmp_path, run=fake_run)
    assert (state.commit, state.branch, state.dirty) == ("abc123", "main", True)


def test_manifest_roundtrip(tmp_path):
    manifest = build_manifest(
        config={"a": 1},
        seed=3,
        environment={"python_version": "3.12"},
        git=git_state(tmp_path),
        started_at=FIXED,
        extra={"task": "t"},
    )
    loaded = json.loads(write_json(tmp_path / "m.json", manifest).read_text())
    assert loaded["seed"] == 3 and loaded["task"] == "t" and loaded["git"]["commit"] is None
