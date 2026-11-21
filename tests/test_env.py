from euroai.env import (
    detect_environment,
    loaded_modules,
    package_versions,
    slurm_info,
    virtualenv_path,
)


def test_slurm_info_filters_keys():
    env = {"SLURM_JOB_ID": "42", "SLURM_JOB_PARTITION": "p", "HOME": "/h"}
    assert slurm_info(env) == {"SLURM_JOB_ID": "42", "SLURM_JOB_PARTITION": "p"}


def test_loaded_modules_parses_lmod_variable():
    assert loaded_modules({"LOADEDMODULES": "A/1:B/2:"}) == ["A/1", "B/2"]
    assert loaded_modules({}) == []


def test_virtualenv_detection():
    assert virtualenv_path("/p", "/p", {}) is None
    assert virtualenv_path("/venv", "/base", {}) == "/venv"
    assert virtualenv_path("/p", "/p", {"VIRTUAL_ENV": "/v"}) == "/v"


def test_package_versions_uses_injected_lookup():
    versions = package_versions(("numpy", "sklearn"), version_of=lambda name: f"v-{name}")
    assert versions == {"numpy": "v-numpy", "sklearn": "v-scikit-learn"}


def test_detect_environment_without_probe():
    info = detect_environment({"SLURM_JOB_ID": "1"}, probe=False, version_of=lambda _: None)
    data = info.to_dict()
    assert data["in_slurm_job"] is True
    assert data["accelerators"] == {"probed": False}
    assert data["python_version"]
