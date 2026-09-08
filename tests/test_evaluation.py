from __future__ import annotations

from urdu_document_ocr import EditCounts, EvaluationResult


def test_phase4_evaluation_result_constructor_remains_compatible() -> None:
    result = EvaluationResult(
        sample_count=2,
        exact_matches=1,
        character_edits=EditCounts(1, 2, 3, 12),
        word_edits=EditCounts(0, 1, 0, 4),
        data_fingerprint="data",
        config_fingerprint="config",
    )

    assert result.character_error_rate == 0.5
    assert result.word_error_rate == 0.25
    assert result.exact_match_rate == 0.5
    assert result.to_public_dict()["schema_version"] == 1


def test_zero_denominator_properties_are_explicit() -> None:
    result = EvaluationResult(
        sample_count=0,
        exact_matches=0,
        character_edits=EditCounts(0, 0, 0, 0),
        word_edits=EditCounts(0, 0, 0, 0),
        data_fingerprint="data",
        config_fingerprint="config",
    )

    assert result.character_error_rate is None
    assert result.word_error_rate is None
    assert result.exact_match_rate is None
