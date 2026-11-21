"""Refuse content that holds a secret: this machine's values, Hugging Face tokens, private keys.

uv run python scripts/check_secrets.py            every file git would add, tracked or new
uv run python scripts/check_secrets.py --staged   staged content only; the pre-commit hook

The values compared are HACKATHON_API_KEY and HF_TOKEN as our scripts would read them. Output
names the file and the line, never the value.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from euroai.credentials import API_KEY_VAR, HF_TOKEN_VAR, get_secret  # noqa: E402
from euroai.leaks import describe, find_leaks  # noqa: E402

MAX_BYTES = 20_000_000


def git(root: Path, *args: str) -> bytes:
    return subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True).stdout


def names(output: bytes) -> list[str]:
    return [name for name in output.decode("utf-8", "surrogateescape").split("\0") if name]


def staged_files(root: Path) -> list[tuple[str, bytes]]:
    listed = names(git(root, "diff", "--cached", "--name-only", "--diff-filter=ACMR", "-z"))
    return [(name, git(root, "show", f":{name}")) for name in listed]


def working_files(root: Path) -> list[tuple[str, bytes]]:
    found = []
    for name in names(git(root, "ls-files", "--cached", "--others", "--exclude-standard", "-z")):
        path = root / name
        if path.is_file() and not path.is_symlink() and path.stat().st_size <= MAX_BYTES:
            found.append((name, path.read_bytes()))
    return found


def main(argv: list[str] | None = None, root: Path = ROOT) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--staged", action="store_true", help="check staged content only")
    args = parser.parse_args(argv)
    lookups = [get_secret(name) for name in (API_KEY_VAR, HF_TOKEN_VAR)]
    secrets = {lookup.name: lookup.value for lookup in lookups if lookup.value}
    files = staged_files(root) if args.staged else working_files(root)
    leaks = [
        leak
        for name, data in files
        for leak in find_leaks(name, data.decode("utf-8", "replace"), secrets)
    ]
    for leak in leaks:
        print(f"refused: {describe(leak)}", file=sys.stderr)
    if leaks:
        print(
            "Remove the secret. If it was ever committed or pushed, tell the lead at once.",
            file=sys.stderr,
        )
        return 1
    print(f"no secrets in {len(files)} files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
