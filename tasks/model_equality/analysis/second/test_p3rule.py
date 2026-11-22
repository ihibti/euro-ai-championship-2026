import json

import numpy as np
import p3rule as r


def test_run_length_counts_the_longest_run_of_river():
    assert r.run_length("River river, river. sea river") == 3
    assert r.is_long(" ".join(["river"] * 128))
    assert not r.is_long("river")


def test_estimated_tokens_counts_words_punctuation_and_space_runs():
    assert r.estimated_tokens(" ".join(["river"] * 256)) == 256
    assert r.estimated_tokens(" ".join(["River,"] * 128)) == 256
    assert r.estimated_tokens("   ".join(["river"] * 128)) == 255


def test_is_capped_needs_about_the_output_cap_not_the_word_count():
    assert r.estimated_tokens(" ".join(["river"] * 150)) == 150
    assert not r.is_capped(" ".join(["river"] * 150))
    assert not r.is_capped("Here you go:\n" + " ".join(["river"] * 150))


def test_feature_capped_changes_the_per_endpoint_counts(tmp_path, capsys):
    capped_run = " ".join(["river"] * 50 + ["stone"] * 220)
    assert r.is_capped(capped_run) and not r.is_long(capped_run)
    rows = [
        {
            "endpoint": endpoint,
            "id": "p3_repeat",
            "repeat": repeat,
            "prompt": "Repeat.",
            "response": response,
        }
        for endpoint, response in (("e1", capped_run), ("e2", capped_run), ("e3", "river"))
        for repeat in range(10)
    ]
    answers = tmp_path / "answers.jsonl"
    answers.write_text("".join(json.dumps(row) + "\n" for row in rows))

    def share_line(feature):
        r.main(
            [
                "--answers",
                str(answers),
                "--prefix",
                "e",
                "--probe",
                "p3_repeat",
                "--feature",
                feature,
                "--draws",
                "49",
            ]
        )
        return next(line for line in capsys.readouterr().out.splitlines() if "share" in line)

    assert "e1 0.00" in share_line("long")
    assert "e1 1.00" in share_line("capped")


def test_call_family_splits_off_the_high_rate_minority_as_singletons():
    names = [f"e{i}" for i in range(9)]
    long_counts = dict(zip(names, [0, 1, 2, 1, 0, 3, 8, 7, 9], strict=True))
    totals = dict.fromkeys(names, 10)
    call = r.call_family(names, long_counts, totals, np.random.default_rng(0), draws=199)
    assert call.p < 0.01
    assert call.lora == ["e6", "e7", "e8"]
    assert call.base == ["e0", "e1", "e2", "e3", "e4", "e5"]


def test_call_family_calls_nothing_on_a_homogeneous_family():
    names = [f"e{i}" for i in range(8)]
    long_counts = dict(zip(names, [1, 2, 1, 0, 2, 1, 1, 2], strict=True))
    totals = dict.fromkeys(names, 10)
    call = r.call_family(names, long_counts, totals, np.random.default_rng(0), draws=199)
    assert call.p >= 0.01
    assert call.lora == []
