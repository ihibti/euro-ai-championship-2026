import importlib.util
import subprocess
from pathlib import Path

import pytest

from euroai.leaks import MIN_SECRET_LENGTH, Leak, describe, find_leaks

ROOT = Path(__file__).resolve().parents[1]
FAKE_KEY = "k" * 12 + "0123456789abcdef"
FAKE_HF = "hf_" + "A1b2" * 9
KEY_HEADER = "-----BEGIN OPENSSH " + "PRIVATE KEY-----"


def test_clean_text_has_no_leaks():
    text = "HACKATHON_API_KEY=\nAPI_KEY = 'YOUR_API_KEY_HERE'\nsha256 " + "ab" * 32 + "\n"
    assert find_leaks("a.py", text, {"HACKATHON_API_KEY": FAKE_KEY}) == []


def test_known_value_is_found_with_its_line_and_name():
    text = f"first\nheaders = {{'X-API-Key': '{FAKE_KEY}'}}\n"
    assert find_leaks("run.py", text, {"HACKATHON_API_KEY": FAKE_KEY}) == [
        Leak("run.py", 2, "the value of HACKATHON_API_KEY")
    ]


def test_short_or_empty_values_are_ignored():
    short = "x" * (MIN_SECRET_LENGTH - 1)
    assert find_leaks("a.txt", f"{short}\n\n", {"A": short, "B": ""}) == []


def test_hugging_face_token_is_found_without_knowing_it():
    assert find_leaks("notes.md", f"token: {FAKE_HF}\n", {}) == [
        Leak("notes.md", 1, "a Hugging Face token")
    ]
    assert find_leaks("notes.md", "hf_hub_download and hf_short\n", {}) == []


def test_private_key_block_is_found():
    assert find_leaks("id", f"{KEY_HEADER}\nb3Blbn\n", {}) == [Leak("id", 1, "a private key")]


def test_description_never_contains_the_value():
    leaks = find_leaks("a.py", f"key = '{FAKE_KEY}' and {FAKE_HF}", {"HACKATHON_API_KEY": FAKE_KEY})
    text = " ".join(describe(leak) for leak in leaks) + repr(leaks)
    assert len(leaks) == 2
    assert FAKE_KEY not in text
    assert FAKE_HF not in text
    assert describe(leaks[0]) == "a.py:1 contains the value of HACKATHON_API_KEY"


def load_checker():
    spec = importlib.util.spec_from_file_location(
        "check_secrets", ROOT / "scripts" / "check_secrets.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def repo(tmp_path, monkeypatch):
    secrets = tmp_path / "secrets.env"
    secrets.write_text(f"HACKATHON_API_KEY={FAKE_KEY}\n")
    monkeypatch.setenv("EUROAI_SECRETS_FILE", str(secrets))
    monkeypatch.delenv("HACKATHON_API_KEY", raising=False)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    for name in ("GIT_DIR", "GIT_INDEX_FILE", "GIT_WORK_TREE"):
        monkeypatch.delenv(name, raising=False)
    root = tmp_path / "repo"
    root.mkdir()
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    return root


def test_checker_refuses_a_staged_secret_and_passes_once_removed(repo, capsys):
    checker = load_checker()
    leaky = repo / "debug.py"
    leaky.write_text(f"KEY = '{FAKE_KEY}'\n")
    (repo / "fine.py").write_text("print('ok')\n")
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
    assert checker.main(["--staged"], root=repo) == 1
    err = capsys.readouterr().err
    assert "debug.py:1 contains the value of HACKATHON_API_KEY" in err
    assert FAKE_KEY not in err
    leaky.write_text("KEY = None\n")
    assert checker.main(["--staged"], root=repo) == 1
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
    assert checker.main(["--staged"], root=repo) == 0


def test_checker_scans_untracked_files_but_not_ignored_ones(repo):
    checker = load_checker()
    (repo / ".gitignore").write_text("ignored.txt\n")
    (repo / "ignored.txt").write_text(FAKE_HF + "\n")
    assert checker.main([], root=repo) == 0
    (repo / "new.txt").write_text(FAKE_HF + "\n")
    assert checker.main([], root=repo) == 1
