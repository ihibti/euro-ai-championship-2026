import numpy as np
import pytest

from euroai.validators import (
    ME_ENDPOINTS,
    validate_file,
    validate_model_equality,
    validate_unlearning,
    validate_watermark,
)


def watermark_obj(count=3, grid=128):
    masks = np.zeros((count, grid, grid), dtype=np.uint8)
    masks[:, :10, :20] = 1
    return {"image_names": [f"img_{i:04d}" for i in range(count)], "masks": masks}


def me_lines(header=True):
    lines = ["endpoint,family,model"] if header else []
    for index, endpoint in enumerate(ME_ENDPOINTS):
        family = "A" if index < 16 else "B"
        lines.append(f"{endpoint},{family},{family}{index % 4}")
    return lines


def test_me_endpoints_are_the_33_scored_ones():
    assert ME_ENDPOINTS[0] == "ep-01" and ME_ENDPOINTS[-1] == "ep-33" and len(ME_ENDPOINTS) == 33


def test_watermark_valid():
    assert validate_watermark(watermark_obj(), expected_count=3) == []


def test_watermark_bool_masks_accepted():
    obj = watermark_obj()
    obj["masks"] = obj["masks"].astype(bool)
    assert validate_watermark(obj, expected_count=3) == []


def test_watermark_bad_shape():
    errors = validate_watermark(watermark_obj(grid=512), expected_count=3)
    assert any("128" in error for error in errors)


def test_watermark_value_255():
    obj = watermark_obj()
    obj["masks"][0, 0, 0] = 255
    assert any("0 or 1" in error for error in validate_watermark(obj, expected_count=3))


def test_watermark_duplicate_names():
    obj = watermark_obj()
    obj["image_names"][2] = obj["image_names"][0]
    assert any("duplicate" in error for error in validate_watermark(obj, expected_count=3))


def test_watermark_count_mismatch():
    obj = watermark_obj()
    obj["masks"] = obj["masks"][:2]
    assert any("names" in error for error in validate_watermark(obj, expected_count=3))
    assert any("1020" in error for error in validate_watermark(watermark_obj()))


def test_watermark_png_suffix_rejected():
    obj = watermark_obj()
    obj["image_names"][1] = "img_0001.png"
    assert any(".png" in error for error in validate_watermark(obj, expected_count=3))


def test_watermark_float_masks_rejected():
    obj = watermark_obj()
    obj["masks"] = obj["masks"].astype(np.float32)
    assert any("float32" in error for error in validate_watermark(obj, expected_count=3))


def test_watermark_expected_names():
    expected = ["img_0000", "img_0001", "other"]
    errors = validate_watermark(watermark_obj(), expected_count=3, expected_names=expected)
    assert any("img_0002" in error for error in errors)
    assert any("other" in error for error in errors)
    names = ["img_0000", "img_0001", "img_0002"]
    assert validate_watermark(watermark_obj(), expected_count=3, expected_names=names) == []


def test_watermark_keys():
    assert validate_watermark({"masks": np.zeros((1, 128, 128))}, expected_count=1)
    obj = watermark_obj()
    obj["extra"] = 1
    assert any("extra" in error for error in validate_watermark(obj, expected_count=3))
    assert validate_watermark([1, 2, 3], expected_count=3)


def test_me_valid_with_and_without_header():
    assert validate_model_equality("\n".join(me_lines()) + "\n") == []
    assert validate_model_equality("\n".join(me_lines(header=False))) == []


def test_me_ref_line_rejected():
    errors = validate_model_equality("\n".join([*me_lines(), "ref-01,P,S"]))
    assert any("ref-01" in error for error in errors)


def test_me_missing_endpoint():
    errors = validate_model_equality("\n".join(me_lines()[:-1]))
    assert any("missing" in error and "ep-33" in error for error in errors)


def test_me_duplicate_endpoint():
    errors = validate_model_equality("\n".join([*me_lines(), "ep-01,A,A0"]))
    assert any("ep-01" in error and "twice" in error for error in errors)


def test_me_model_label_across_families():
    lines = me_lines()
    lines[1] = "ep-01,A,SHARED"
    lines[33] = "ep-33,B,SHARED"
    errors = validate_model_equality("\n".join(lines))
    assert any("SHARED" in error for error in errors)


def test_me_quoted_comma_rejected():
    lines = me_lines()
    lines[1] = 'ep-01,"A,B",A1'
    assert any("3 fields" in error for error in validate_model_equality("\n".join(lines)))


def test_me_empty_label_rejected():
    lines = me_lines()
    lines[1] = "ep-01,,A1"
    assert any("empty" in error for error in validate_model_equality("\n".join(lines)))


def test_me_family_must_be_capital_letters():
    lines = me_lines()
    lines[1] = "ep-01,F01,F011"
    assert any("capital letters" in error for error in validate_model_equality("\n".join(lines)))


def test_me_model_must_be_family_letter_and_number():
    lines = me_lines()
    lines[1] = "ep-01,A,B1"
    assert any("family letter" in error for error in validate_model_equality("\n".join(lines)))
    lines[1] = "ep-01,A,A"
    assert any("family letter" in error for error in validate_model_equality("\n".join(lines)))


def test_unlearning_valid():
    assert validate_unlearning({"w": np.zeros((2, 2), dtype=np.float32)}) == []


@pytest.mark.parametrize("dtype", [np.float16, np.float64, np.int64])
def test_unlearning_other_dtypes_rejected(dtype):
    errors = validate_unlearning({"w": np.zeros(2, dtype=dtype)})
    assert any(np.dtype(dtype).name in error for error in errors)


def test_unlearning_not_a_mapping_or_empty():
    assert validate_unlearning([np.zeros(2, dtype=np.float32)])
    assert validate_unlearning({})
    assert validate_unlearning({"epoch": 3})


def test_unlearning_nan_and_inf_rejected():
    assert validate_unlearning({"w": np.array([1.0, np.nan], dtype=np.float32)})
    assert validate_unlearning({"w": np.array([np.inf], dtype=np.float32)})


def test_unlearning_checkpoint_wrapper_rejected():
    errors = validate_unlearning({"state_dict": {"w": np.zeros(2, dtype=np.float32)}})
    assert any("state_dict()" in error for error in errors)


def test_unlearning_many_errors_truncated():
    state = {f"w{i}": np.zeros(1, dtype=np.float64) for i in range(30)}
    errors = validate_unlearning(state)
    assert len(errors) == 11 and "20 more" in errors[-1]


def test_validate_file_oversize_does_not_load(tmp_path):
    path = tmp_path / "big.pt"
    with path.open("wb") as handle:
        handle.truncate(20_000_001)

    calls = []
    report = validate_file("watermark_localization", path, load_pt=calls.append)
    assert calls == []
    assert not report.ok and any("20000000" in error for error in report.errors)


def test_validate_file_wrong_suffix(tmp_path):
    path = tmp_path / "answer.txt"
    path.write_text("x")
    assert not validate_file("model_equality", path).ok


def test_validate_file_missing(tmp_path):
    assert not validate_file("model_equality", tmp_path / "absent.csv").ok


def test_validate_file_model_equality(tmp_path):
    path = tmp_path / "answer.csv"
    path.write_text("\n".join(me_lines()) + "\n")
    report = validate_file("model_equality", path)
    assert report.ok, report.errors
    assert any("33 endpoints" in note for note in report.notes)


def test_validate_file_watermark_uses_loader(tmp_path):
    path = tmp_path / "sub.pt"
    path.write_bytes(b"x")
    report = validate_file("watermark_localization", path, load_pt=lambda _: watermark_obj(1020))
    assert report.ok, report.errors
    assert any("1020 masks" in note for note in report.notes)
    other_names = [f"other_{i}" for i in range(1020)]
    wrong = validate_file(
        "watermark_localization",
        path,
        load_pt=lambda _: watermark_obj(1020),
        expected_names=other_names,
    )
    assert not wrong.ok


def test_validate_file_load_failure_is_an_error(tmp_path):
    path = tmp_path / "sub.pt"
    path.write_bytes(b"x")

    def broken(_):
        raise RuntimeError("bad pickle")

    def no_torch(_):
        raise ImportError("No module named 'torch'")

    assert not validate_file("unlearning", path, load_pt=broken).ok
    assert not validate_file("watermark_localization", path, load_pt=no_torch).ok


def test_validate_file_unlearning_strict_load_skipped_or_failed(tmp_path):
    path = tmp_path / "sub.pt"
    path.write_bytes(b"x")
    state = {"w": np.zeros(2, dtype=np.float32)}

    def unavailable(_):
        raise ImportError("timm")

    skipped = validate_file("unlearning", path, load_pt=lambda _: state, strict_load=unavailable)
    assert skipped.ok and any("skipped" in note for note in skipped.notes)
    failed = validate_file(
        "unlearning", path, load_pt=lambda _: state, strict_load=lambda _: ["missing keys"]
    )
    assert not failed.ok


@pytest.mark.torch
def test_unlearning_strict_load_real():
    timm = pytest.importorskip("timm")
    pytest.importorskip("torch")
    from euroai.validators import unlearning_strict_load

    model = timm.create_model(
        "vit_base_patch16_224", pretrained=False, num_classes=100, init_values=1e-4
    )
    state = model.state_dict()
    assert unlearning_strict_load(state) == []
    state.pop(next(iter(state)))
    assert unlearning_strict_load(state)


@pytest.mark.torch
def test_validate_file_real_torch_files(tmp_path):
    torch = pytest.importorskip("torch")
    names = [f"img_{i:04d}" for i in range(1020)]
    masks = torch.zeros((1020, 128, 128), dtype=torch.bool)
    masks[:, :8, :8] = True
    submission = tmp_path / "submission.pt"
    torch.save({"image_names": names, "masks": masks}, submission)
    report = validate_file("watermark_localization", submission, expected_names=names)
    assert report.ok, report.errors

    def no_strict(_):
        return []

    nan_state = tmp_path / "nan.pt"
    torch.save({"w": torch.zeros(3), "b": torch.tensor([1.0, float("nan")])}, nan_state)
    errors = validate_file("unlearning", nan_state, strict_load=no_strict).errors
    assert any("NaN" in error for error in errors)
    half_state = tmp_path / "half.pt"
    torch.save({"w": torch.zeros(3, dtype=torch.float16)}, half_state)
    errors = validate_file("unlearning", half_state, strict_load=no_strict).errors
    assert any("float16" in error for error in errors)
