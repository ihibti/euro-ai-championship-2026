from pathlib import Path

import pytest

from euroai.templating import find_placeholders, render

ROOT = Path(__file__).resolve().parents[1]


def test_render_fills_and_drops_empty_lines():
    text = "a=<A>\n#SBATCH --reservation=<R>\nend\n"
    assert render(text, {"A": "1", "R": ""}) == "a=1\nend\n"


def test_render_refuses_unfilled_and_unknown():
    with pytest.raises(ValueError, match="B"):
        render("<A> <B>", {"A": "x"})
    with pytest.raises(KeyError):
        render("<A>", {"A": "x", "TYPO": "y"})


def test_lowercase_and_shell_syntax_are_not_placeholders():
    assert find_placeholders("cat <<'EOF'\n<lower> ${VAR} <A_1>") == {"A_1"}


def test_jureca_sbatch_template_has_documented_placeholders_only():
    text = (ROOT / "jsc" / "jureca_gpu.sbatch.template").read_text()
    assert find_placeholders(text) == {"BUDGET_ACCOUNT", "RESERVATION", "TIME"}
    rendered = render(text, {"BUDGET_ACCOUNT": "acct", "RESERVATION": "", "TIME": "00:30:00"})
    assert "--reservation" not in rendered
    assert "#SBATCH --account=acct" in rendered


def test_ssh_template_placeholders():
    text = (ROOT / "jsc" / "ssh_config.template").read_text()
    assert find_placeholders(text) == {"JUDOOR_USERNAME"}
