from __future__ import annotations

import pytest

from urdu_document_ocr import EvaluationError
from urdu_document_ocr.evaluation import analyze_errors, evaluate_predictions


def test_error_analysis_uses_alignment_pairs_and_deterministic_order() -> None:
    report = evaluate_predictions(
        (("a", "ابب"), ("b", "کت"), ("c", "ز")),
        (("a", "اتب"), ("b", "کتم"), ("c", "")),
    )

    first = analyze_errors(report, worst_limit=3)
    second = analyze_errors(report, worst_limit=3)
    public = first.to_public_dict()

    assert first == second
    assert [sample.sample_id for sample in first.worst_samples] == ["c", "b", "a"]
    assert first.substitutions == 1
    assert first.deletions == 1
    assert first.insertions == 1
    assert first.substitution_pairs[0].reference_character == "ب"
    assert first.substitution_pairs[0].hypothesis_character == "ت"
    assert first.deleted_characters[0].character == "ز"
    assert first.inserted_characters[0].character == "م"
    assert public["substitution_pairs"][0]["reference_code_point"] == "U+0628"  # type: ignore[index]


def test_frequencies_sort_by_count_then_code_point() -> None:
    report = evaluate_predictions((("a", "ببا"),), (("a", "ب"),))
    analysis = analyze_errors(report)

    assert [(item.character, item.count) for item in analysis.reference_character_frequency] == [
        ("ب", 2),
        ("ا", 1),
    ]


def test_worst_sample_limit_and_validation() -> None:
    report = evaluate_predictions((("a", "x"),), (("a", "y"),))

    assert analyze_errors(report, worst_limit=0).worst_samples == ()
    with pytest.raises(EvaluationError, match="nonnegative"):
        analyze_errors(report, worst_limit=-1)
    with pytest.raises(TypeError):
        analyze_errors(object())  # type: ignore[arg-type]
