from __future__ import annotations

import math

import pytest

from urdu_document_ocr import EditCounts, EvaluationError, EvaluationResult
from urdu_document_ocr.evaluation import (
    METRIC_POLICY_FINGERPRINT,
    AlignmentStep,
    EditOperation,
    EvaluationReport,
    SampleEvaluation,
    align_sequences,
    ascii_space_tokens,
    character_error_rate,
    edit_counts,
    edit_distance,
    evaluate_predictions,
    evaluate_text,
    exact_match,
    word_error_rate,
)


@pytest.mark.parametrize(
    ("reference", "hypothesis", "expected"),
    [
        ("abc", "abc", (0, 0, 0)),
        ("abc", "adc", (1, 0, 0)),
        ("abc", "ac", (0, 1, 0)),
        ("abc", "abbc", (0, 0, 1)),
        ("kitten", "sitting", (2, 0, 1)),
        ("aaa", "aa", (0, 1, 0)),
        ("اردو", "ارزو", (1, 0, 0)),
        ("نیم\u200cخودکار", "نیمخودکار", (0, 1, 0)),
        ("۱۲۳،", "۱۲۳؟", (1, 0, 0)),
    ],
)
def test_hand_checked_character_alignments(reference, hypothesis, expected) -> None:
    alignment = align_sequences(reference, hypothesis)
    counts = edit_counts(alignment, len(reference))

    assert (counts.substitutions, counts.deletions, counts.insertions) == expected
    assert counts.total_errors == edit_distance(reference, hypothesis)


def test_alignment_tie_policy_prefers_substitution() -> None:
    first = align_sequences("ab", "ba")
    second = align_sequences("ab", "ba")

    assert first == second
    assert [step.operation for step in first] == [
        EditOperation.SUBSTITUTION,
        EditOperation.SUBSTITUTION,
    ]
    assert edit_distance("ab", "ba") == 2


def test_generic_sequence_and_literal_none_tokens() -> None:
    reference = (None, 1, 2)
    hypothesis = (None, 2, 3)
    alignment = align_sequences(reference, hypothesis)

    assert edit_distance(reference, hypothesis) == 2
    assert sum(step.operation is not EditOperation.MATCH for step in alignment) == 2


def test_standard_cer_can_exceed_one() -> None:
    assert character_error_rate("a", "bbbb") == 4.0


def test_wer_uses_only_ascii_space_and_keeps_punctuation() -> None:
    assert ascii_space_tokens("ایک  دو\tتین؟") == ("ایک", "دو\tتین؟")
    assert word_error_rate("ایک دو؟", "ایک دو۔") == 0.5


def test_exact_match_does_not_normalize_or_fold() -> None:
    assert exact_match("اُردو", "اُردو")
    assert not exact_match("اُردو", "اردو")
    assert not exact_match("ک", "ك")
    assert not exact_match("é", "e\u0301")


@pytest.mark.parametrize("hypothesis", ["", "abc"])
def test_zero_reference_rates_are_undefined(hypothesis: str) -> None:
    sample = evaluate_text("empty", "", hypothesis)

    assert sample.character_error_rate is None
    assert sample.word_error_rate is None
    assert sample.exact_match is (hypothesis == "")
    assert sample.character_edits.insertions == len(hypothesis)


def test_empty_hypothesis_has_deletions() -> None:
    sample = evaluate_text("deleted", "abc", "")

    assert sample.character_edits.deletions == 3
    assert sample.character_error_rate == 1.0
    assert sample.word_edits.deletions == 1
    assert sample.word_error_rate == 1.0


def test_corpus_uses_summed_denominators_not_mean_sample_rates() -> None:
    report = evaluate_predictions(
        (("short", "a"), ("long", "abcdefghij")),
        (("long", "abcdefghiX"), ("short", "")),
    )

    assert report.aggregate.character_edits.total_errors == 2
    assert report.aggregate.character_edits.reference_length == 11
    assert math.isclose(report.aggregate.character_error_rate or -1, 2 / 11)
    assert report.aggregate.exact_matches == 0
    assert report.aggregate.exact_match_rate == 0.0
    assert [sample.sample_id for sample in report.samples] == ["long", "short"]


def test_empty_corpus_rates_are_undefined() -> None:
    report = evaluate_predictions((), ())

    assert report.aggregate.character_error_rate is None
    assert report.aggregate.word_error_rate is None
    assert report.aggregate.exact_match_rate is None


@pytest.mark.parametrize(
    ("references", "predictions", "message"),
    [
        ((("a", "x"), ("a", "y")), (("a", "x"),), "duplicate"),
        ((("a", "x"),), (("a", "x"), ("a", "y")), "duplicate"),
        ((("a", "x"),), (), "differ"),
        ((), (("a", "x"),), "differ"),
    ],
)
def test_prediction_identity_mismatches_fail(references, predictions, message) -> None:
    with pytest.raises(EvaluationError, match=message):
        evaluate_predictions(references, predictions)


def test_invalid_alignment_count_contract_fails() -> None:
    with pytest.raises(EvaluationError, match="consume"):
        edit_counts(align_sequences("a", "a"), 2)
    with pytest.raises(TypeError):
        edit_counts((), True)
    with pytest.raises(ValueError):
        edit_counts((), -1)


def test_report_projection_has_raw_rates_and_safe_unicode() -> None:
    report = evaluate_predictions((("urdu", "اردو"),), (("urdu", "ارزو"),))
    public = report.to_public_dict()

    assert public["summary"]["character"]["error_rate"] == 0.25  # type: ignore[index]
    assert public["samples"][0]["reference"] == "اردو"  # type: ignore[index]


@pytest.mark.parametrize(
    ("function", "arguments"),
    [
        (ascii_space_tokens, (None,)),
        (character_error_rate, (None, "x")),
        (word_error_rate, ("x", None)),
        (exact_match, (1, "1")),
        (evaluate_text, ("", "x", "x")),
        (evaluate_text, ("x", None, "x")),
    ],
)
def test_metric_public_input_validation(function, arguments) -> None:
    with pytest.raises((TypeError, EvaluationError)):
        function(*arguments)


@pytest.mark.parametrize(
    "values",
    [
        None,
        [["id", "text"]],
        [("", "text")],
        [("id", 1)],
        [("id", "text", "extra")],
    ],
)
def test_identified_text_collection_validation(values) -> None:
    with pytest.raises(EvaluationError):
        evaluate_predictions(values, ())


def test_explicit_evaluation_fingerprints_must_be_nonempty_strings() -> None:
    with pytest.raises(EvaluationError, match="data_fingerprint"):
        evaluate_predictions((), (), data_fingerprint="")
    with pytest.raises(EvaluationError, match="config_fingerprint"):
        evaluate_predictions((), (), config_fingerprint="")


def test_metric_result_record_validation() -> None:
    counts = EditCounts(0, 0, 0, 1)
    valid_sample = evaluate_text("a", "x", "x")
    aggregate = EvaluationResult(1, 1, counts, counts, "data", "config")

    with pytest.raises(TypeError, match="operation"):
        AlignmentStep("match", "x", "x")  # type: ignore[arg-type]
    with pytest.raises(EvaluationError, match="sample_id"):
        SampleEvaluation("", "x", "x", counts, 0.0, counts, 0.0, True)
    with pytest.raises(EvaluationError, match="strings"):
        SampleEvaluation("a", 1, "x", counts, 0.0, counts, 0.0, True)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="EditCounts"):
        SampleEvaluation("a", "x", "x", object(), 0.0, counts, 0.0, True)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="float"):
        SampleEvaluation("a", "x", "x", counts, 0, counts, 0.0, True)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="boolean"):
        SampleEvaluation("a", "x", "x", counts, 0.0, counts, 0.0, 1)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="aggregate"):
        EvaluationReport(object(), ())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="SampleEvaluation"):
        EvaluationReport(aggregate, (object(),))  # type: ignore[arg-type]
    with pytest.raises(EvaluationError, match="sample count"):
        EvaluationReport(aggregate, ())
    with pytest.raises(EvaluationError, match="identifier"):
        EvaluationReport(aggregate, (valid_sample,), metric_policy_id="other")
    with pytest.raises(EvaluationError, match="fingerprint"):
        EvaluationReport(
            aggregate,
            (valid_sample,),
            metric_policy_fingerprint="0" * len(METRIC_POLICY_FINGERPRINT),
        )


def test_report_requires_sorted_unique_samples() -> None:
    first = evaluate_text("a", "x", "x")
    second = evaluate_text("b", "x", "x")
    counts = EditCounts(0, 0, 0, 2)
    aggregate = EvaluationResult(2, 2, counts, counts, "data", "config")

    with pytest.raises(EvaluationError, match="sorted"):
        EvaluationReport(aggregate, (second, first))
    with pytest.raises(EvaluationError, match="unique"):
        EvaluationReport(aggregate, (first, first))


def test_edit_counts_rejects_non_alignment_values() -> None:
    with pytest.raises(TypeError, match="AlignmentStep"):
        edit_counts((object(),), 0)  # type: ignore[arg-type]
