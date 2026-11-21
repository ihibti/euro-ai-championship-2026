from pathlib import Path

from euroai.credentials import (
    DEFAULT_SECRETS_FILE,
    find_secret,
    missing_secret_message,
    parse_env_text,
    permission_warning,
    read_text_or_none,
    secrets_file,
)

SECRET = "not-a-real-key-123"


def test_parse_env_text():
    text = "\n".join(
        [
            "# comment",
            "",
            "HACKATHON_API_KEY=abc=def",
            "export HF_TOKEN = 'quoted value' ",
            'OTHER="double"',
            "not a pair",
            "EMPTY=",
        ]
    )
    assert parse_env_text(text) == {
        "HACKATHON_API_KEY": "abc=def",
        "HF_TOKEN": "quoted value",
        "OTHER": "double",
        "EMPTY": "",
    }


def test_secrets_file_default_and_override():
    assert secrets_file({}) == Path(DEFAULT_SECRETS_FILE).expanduser()
    assert secrets_file({"EUROAI_SECRETS_FILE": "/tmp/x.env"}) == Path("/tmp/x.env")


def test_environment_wins_over_file():
    lookup = find_secret("HACKATHON_API_KEY", {"HACKATHON_API_KEY": SECRET}, lambda _: "x=1")
    assert (lookup.value, lookup.source) == (SECRET, "environment")


def test_empty_environment_value_falls_back_to_file(tmp_path):
    path = tmp_path / "secrets.env"
    path.write_text(f"HACKATHON_API_KEY={SECRET}\n")
    env = {"HACKATHON_API_KEY": "", "EUROAI_SECRETS_FILE": str(path)}
    lookup = find_secret("HACKATHON_API_KEY", env, read_text_or_none)
    assert (lookup.value, lookup.source, lookup.path) == (SECRET, "file", path)


def test_missing_everywhere(tmp_path):
    env = {"EUROAI_SECRETS_FILE": str(tmp_path / "absent.env")}
    lookup = find_secret("HF_TOKEN", env, read_text_or_none)
    assert lookup.value is None and lookup.source == "missing"
    message = missing_secret_message(lookup)
    assert "HF_TOKEN" in message and str(tmp_path / "absent.env") in message


def test_value_never_in_repr_or_message():
    lookup = find_secret("HACKATHON_API_KEY", {"HACKATHON_API_KEY": SECRET}, lambda _: None)
    assert SECRET not in repr(lookup)
    assert SECRET not in missing_secret_message(lookup)


def test_permission_warning():
    path = Path("/home/x/.config/euroai/.env")
    assert permission_warning(path, 0o100600) is None
    warning = permission_warning(path, 0o100644)
    assert warning and "chmod 600" in warning and str(path) in warning


def test_parse_env_text_inline_comments():
    text = "\n".join(["A=abc # team key", 'B="x # y"', "C=abc#def", "D='q'  # note", "E="])
    assert parse_env_text(text) == {"A": "abc", "B": "x # y", "C": "abc#def", "D": "q", "E": ""}
