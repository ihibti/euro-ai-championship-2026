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


def test_jureca_sbatch_template_uses_event_values():
    text = (ROOT / "jsc" / "jureca_gpu.sbatch.template").read_text()
    assert find_placeholders(text) == {"JOB_NAME", "TASK_DIR", "TIME", "COMMAND"}
    task_dir = "/p/scratch/training2666/hibti1/Begg/unlearning"
    values = {
        "JOB_NAME": "unlearn",
        "TASK_DIR": task_dir,
        "TIME": "01:00:00",
        "COMMAND": "python unlearn.py",
    }
    lines = render(text, values).splitlines()
    for expected in [
        "#SBATCH --account=training2666",
        "#SBATCH --reservation=cispahack",
        "#SBATCH --partition=dc-gpu",
        "#SBATCH --nodes=1",
        "#SBATCH --ntasks-per-node=1",
        "#SBATCH --gres=gpu:4",
        "#SBATCH --cpus-per-task=32",
        "export CUDA_VISIBLE_DEVICES=0,1,2,3",
        f"#SBATCH --output={task_dir}/output/%j.out",
        f"#SBATCH --error={task_dir}/output/%j.err",
        'export SRUN_CPUS_PER_TASK="${SLURM_CPUS_PER_TASK}"',
        f'cd "{task_dir}"',
        "srun python unlearn.py",
    ]:
        assert expected in lines


def test_ssh_template_placeholders():
    text = (ROOT / "jsc" / "ssh_config.template").read_text()
    assert find_placeholders(text) == {"JUDOOR_USERNAME"}
