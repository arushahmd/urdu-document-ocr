"""Deterministic, model-independent OCR edit metrics."""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from hashlib import sha256
from typing import Generic, TypeVar

from urdu_document_ocr.errors import EvaluationError
from urdu_document_ocr.types import EditCounts, EvaluationResult

Token = TypeVar("Token")

METRIC_POLICY_ID = "ocr-edit-v1"
_METRIC_POLICY = {
    "alignment_tie_order": ["match", "substitution", "deletion", "insertion"],
    "cer_units": "unicode-code-points",
    "exact_match": "exact-code-point-equality",
    "identifier": METRIC_POLICY_ID,
    "normalization": "none",
    "wer_units": "nonempty-ascii-space-separated-substrings",
    "zero_reference_rate": None,
}
METRIC_POLICY_FINGERPRINT = sha256(
    json.dumps(_METRIC_POLICY, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode(
        "utf-8"
    )
).hexdigest()


class EditOperation(StrEnum):
    """One operation in a deterministic minimum-cost edit alignment."""

    MATCH = "match"
    SUBSTITUTION = "substitution"
    DELETION = "deletion"
    INSERTION = "insertion"


@dataclass(frozen=True, slots=True)
class AlignmentStep(Generic[Token]):
    """One aligned pair; operation distinguishes a gap from a literal ``None`` token."""

    operation: EditOperation
    reference: Token | None
    hypothesis: Token | None

    def __post_init__(self) -> None:
        if not isinstance(self.operation, EditOperation):
            raise TypeError("operation must be an EditOperation")


@dataclass(frozen=True, slots=True)
class SampleEvaluation:
    """Hand-inspectable character and word metrics for one identified sample."""

    sample_id: str
    reference: str
    hypothesis: str
    character_edits: EditCounts
    character_error_rate: float | None
    word_edits: EditCounts
    word_error_rate: float | None
    exact_match: bool

    def __post_init__(self) -> None:
        if not isinstance(self.sample_id, str) or not self.sample_id:
            raise EvaluationError("sample_id must be a nonempty string")
        if not isinstance(self.reference, str) or not isinstance(self.hypothesis, str):
            raise EvaluationError("reference and hypothesis must be strings")
        if not isinstance(self.character_edits, EditCounts) or not isinstance(
            self.word_edits, EditCounts
        ):
            raise TypeError("character_edits and word_edits must be EditCounts")
        for name, value in (
            ("character_error_rate", self.character_error_rate),
            ("word_error_rate", self.word_error_rate),
        ):
            if value is not None and (isinstance(value, bool) or not isinstance(value, float)):
                raise TypeError(f"{name} must be a float or None")
        if not isinstance(self.exact_match, bool):
            raise TypeError("exact_match must be a boolean")

    def to_public_dict(self) -> dict[str, object]:
        return {
            "sample_id": self.sample_id,
            "reference": self.reference,
            "hypothesis": self.hypothesis,
            "character_edits": self.character_edits.to_public_dict(),
            "character_error_rate": self.character_error_rate,
            "word_edits": self.word_edits.to_public_dict(),
            "word_error_rate": self.word_error_rate,
            "exact_match": self.exact_match,
        }


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    """Corpus totals plus deterministic per-sample records."""

    aggregate: EvaluationResult
    samples: tuple[SampleEvaluation, ...]
    metric_policy_id: str = METRIC_POLICY_ID
    metric_policy_fingerprint: str = METRIC_POLICY_FINGERPRINT

    def __post_init__(self) -> None:
        if not isinstance(self.aggregate, EvaluationResult):
            raise TypeError("aggregate must be an EvaluationResult")
        samples = tuple(self.samples)
        if any(not isinstance(sample, SampleEvaluation) for sample in samples):
            raise TypeError("samples must contain only SampleEvaluation values")
        if tuple(sorted(samples, key=lambda sample: sample.sample_id)) != samples:
            raise EvaluationError("samples must be sorted by sample_id")
        if len({sample.sample_id for sample in samples}) != len(samples):
            raise EvaluationError("sample IDs must be unique")
        if len(samples) != self.aggregate.sample_count:
            raise EvaluationError("sample count does not match aggregate")
        if self.metric_policy_id != METRIC_POLICY_ID:
            raise EvaluationError("metric policy identifier is unsupported")
        if self.metric_policy_fingerprint != METRIC_POLICY_FINGERPRINT:
            raise EvaluationError("metric policy fingerprint is invalid")
        object.__setattr__(self, "samples", samples)

    def to_public_dict(self) -> dict[str, object]:
        character = self.aggregate.character_edits.to_public_dict()
        character["error_rate"] = self.aggregate.character_error_rate
        word = self.aggregate.word_edits.to_public_dict()
        word["error_rate"] = self.aggregate.word_error_rate
        return {
            "schema_version": 1,
            "metric_policy": {
                "identifier": self.metric_policy_id,
                "fingerprint": self.metric_policy_fingerprint,
            },
            "identities": {
                "data_fingerprint": self.aggregate.data_fingerprint,
                "config_fingerprint": self.aggregate.config_fingerprint,
                "model_fingerprint": self.aggregate.model_fingerprint,
            },
            "summary": {
                "sample_count": self.aggregate.sample_count,
                "exact_match_count": self.aggregate.exact_matches,
                "exact_match_rate": self.aggregate.exact_match_rate,
                "character": character,
                "word": word,
            },
            "samples": [sample.to_public_dict() for sample in self.samples],
        }


def edit_distance(reference: Sequence[Token], hypothesis: Sequence[Token]) -> int:
    """Return unit-cost Levenshtein distance for arbitrary finite sequences."""

    previous = list(range(len(hypothesis) + 1))
    for reference_index, reference_token in enumerate(reference, start=1):
        current = [reference_index]
        for hypothesis_index, hypothesis_token in enumerate(hypothesis, start=1):
            substitution_cost = int(reference_token != hypothesis_token)
            current.append(
                min(
                    previous[hypothesis_index] + 1,
                    current[hypothesis_index - 1] + 1,
                    previous[hypothesis_index - 1] + substitution_cost,
                )
            )
        previous = current
    return previous[-1]


def align_sequences(
    reference: Sequence[Token],
    hypothesis: Sequence[Token],
) -> tuple[AlignmentStep[Token], ...]:
    """Return one optimal alignment using match/substitute/delete/insert tie order."""

    reference_values = tuple(reference)
    hypothesis_values = tuple(hypothesis)
    rows = len(reference_values) + 1
    columns = len(hypothesis_values) + 1
    costs = [[0] * columns for _ in range(rows)]
    for index in range(rows):
        costs[index][0] = index
    for index in range(columns):
        costs[0][index] = index
    for row in range(1, rows):
        for column in range(1, columns):
            costs[row][column] = min(
                costs[row - 1][column] + 1,
                costs[row][column - 1] + 1,
                costs[row - 1][column - 1]
                + int(reference_values[row - 1] != hypothesis_values[column - 1]),
            )

    row = len(reference_values)
    column = len(hypothesis_values)
    reverse_steps: list[AlignmentStep[Token]] = []
    while row or column:
        if (
            row
            and column
            and reference_values[row - 1] == hypothesis_values[column - 1]
            and costs[row][column] == costs[row - 1][column - 1]
        ):
            reverse_steps.append(
                AlignmentStep(
                    EditOperation.MATCH,
                    reference_values[row - 1],
                    hypothesis_values[column - 1],
                )
            )
            row -= 1
            column -= 1
        elif row and column and costs[row][column] == costs[row - 1][column - 1] + 1:
            reverse_steps.append(
                AlignmentStep(
                    EditOperation.SUBSTITUTION,
                    reference_values[row - 1],
                    hypothesis_values[column - 1],
                )
            )
            row -= 1
            column -= 1
        elif row and costs[row][column] == costs[row - 1][column] + 1:
            reverse_steps.append(
                AlignmentStep(EditOperation.DELETION, reference_values[row - 1], None)
            )
            row -= 1
        elif column and costs[row][column] == costs[row][column - 1] + 1:
            reverse_steps.append(
                AlignmentStep(EditOperation.INSERTION, None, hypothesis_values[column - 1])
            )
            column -= 1
        else:  # pragma: no cover - recurrence invariant guard
            raise EvaluationError("edit alignment backtrace became inconsistent")
    reverse_steps.reverse()
    return tuple(reverse_steps)


def edit_counts(alignment: Sequence[AlignmentStep[Token]], reference_length: int) -> EditCounts:
    """Count substitutions, deletions, and insertions from an explicit alignment."""

    if isinstance(reference_length, bool) or not isinstance(reference_length, int):
        raise TypeError("reference_length must be an integer")
    if reference_length < 0:
        raise ValueError("reference_length must be nonnegative")
    values = tuple(alignment)
    if any(not isinstance(step, AlignmentStep) for step in values):
        raise TypeError("alignment must contain only AlignmentStep values")
    consumed_reference = sum(step.operation is not EditOperation.INSERTION for step in values)
    if consumed_reference != reference_length:
        raise EvaluationError("alignment does not consume the declared reference length")
    return EditCounts(
        substitutions=sum(step.operation is EditOperation.SUBSTITUTION for step in values),
        deletions=sum(step.operation is EditOperation.DELETION for step in values),
        insertions=sum(step.operation is EditOperation.INSERTION for step in values),
        reference_length=reference_length,
    )


def _rate(counts: EditCounts) -> float | None:
    if counts.reference_length == 0:
        return None
    return counts.total_errors / counts.reference_length


def ascii_space_tokens(text: str) -> tuple[str, ...]:
    """Split into nonempty substrings using ASCII space as the only boundary."""

    if not isinstance(text, str):
        raise TypeError("text must be a string")
    return tuple(token for token in text.split(" ") if token)


def character_error_rate(reference: str, hypothesis: str) -> float | None:
    """Return standard code-point CER without normalization or capping."""

    if not isinstance(reference, str) or not isinstance(hypothesis, str):
        raise TypeError("reference and hypothesis must be strings")
    if not reference:
        return None
    return edit_distance(reference, hypothesis) / len(reference)


def word_error_rate(reference: str, hypothesis: str) -> float | None:
    """Return standard WER over nonempty ASCII-space-separated substrings."""

    reference_words = ascii_space_tokens(reference)
    hypothesis_words = ascii_space_tokens(hypothesis)
    if not reference_words:
        return None
    return edit_distance(reference_words, hypothesis_words) / len(reference_words)


def exact_match(reference: str, hypothesis: str) -> bool:
    """Compare complete strings by exact code-point equality."""

    if not isinstance(reference, str) or not isinstance(hypothesis, str):
        raise TypeError("reference and hypothesis must be strings")
    return reference == hypothesis


def evaluate_text(sample_id: str, reference: str, hypothesis: str) -> SampleEvaluation:
    """Calculate exact character/word metrics for one sample without normalization."""

    if not isinstance(sample_id, str) or not sample_id:
        raise EvaluationError("sample_id must be a nonempty string")
    if not isinstance(reference, str) or not isinstance(hypothesis, str):
        raise EvaluationError("reference and hypothesis must be strings")
    character_alignment = align_sequences(reference, hypothesis)
    character_counts = edit_counts(character_alignment, len(reference))
    reference_words = ascii_space_tokens(reference)
    hypothesis_words = ascii_space_tokens(hypothesis)
    word_alignment = align_sequences(reference_words, hypothesis_words)
    word_counts = edit_counts(word_alignment, len(reference_words))
    return SampleEvaluation(
        sample_id=sample_id,
        reference=reference,
        hypothesis=hypothesis,
        character_edits=character_counts,
        character_error_rate=_rate(character_counts),
        word_edits=word_counts,
        word_error_rate=_rate(word_counts),
        exact_match=exact_match(reference, hypothesis),
    )


def _identified_texts(
    values: Iterable[tuple[str, str]],
    *,
    role: str,
) -> dict[str, str]:
    result: dict[str, str] = {}
    try:
        pairs = tuple(values)
    except TypeError as error:
        raise EvaluationError(f"{role} must be an iterable of ID/text pairs") from error
    for pair in pairs:
        if not isinstance(pair, tuple) or len(pair) != 2:
            raise EvaluationError(f"{role} must contain two-item tuples")
        sample_id, text = pair
        if not isinstance(sample_id, str) or not sample_id:
            raise EvaluationError(f"{role} sample IDs must be nonempty strings")
        if not isinstance(text, str):
            raise EvaluationError(f"{role} text must be a string")
        if sample_id in result:
            raise EvaluationError(
                f"{role} contains duplicate sample IDs", context={"sample_id": sample_id}
            )
        result[sample_id] = text
    return result


def _reference_fingerprint(references: dict[str, str]) -> str:
    payload = [[sample_id, references[sample_id]] for sample_id in sorted(references)]
    return sha256(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def evaluate_predictions(
    references: Iterable[tuple[str, str]],
    predictions: Iterable[tuple[str, str]],
    *,
    data_fingerprint: str | None = None,
    config_fingerprint: str = METRIC_POLICY_FINGERPRINT,
    model_fingerprint: str | None = None,
) -> EvaluationReport:
    """Evaluate exact ID-matched corpora; missing, extra, and duplicate IDs fail."""

    reference_map = _identified_texts(references, role="references")
    prediction_map = _identified_texts(predictions, role="predictions")
    missing = sorted(reference_map.keys() - prediction_map.keys())
    extra = sorted(prediction_map.keys() - reference_map.keys())
    if missing or extra:
        context: dict[str, str | int] = {
            "missing_count": len(missing),
            "extra_count": len(extra),
        }
        if missing:
            context["first_missing_id"] = missing[0]
        if extra:
            context["first_extra_id"] = extra[0]
        raise EvaluationError("reference and prediction sample IDs differ", context=context)
    samples = tuple(
        evaluate_text(sample_id, reference_map[sample_id], prediction_map[sample_id])
        for sample_id in sorted(reference_map)
    )

    def totals(name: str) -> EditCounts:
        counts = [getattr(sample, name) for sample in samples]
        return EditCounts(
            substitutions=sum(count.substitutions for count in counts),
            deletions=sum(count.deletions for count in counts),
            insertions=sum(count.insertions for count in counts),
            reference_length=sum(count.reference_length for count in counts),
        )

    fingerprint = (
        _reference_fingerprint(reference_map) if data_fingerprint is None else data_fingerprint
    )
    if not isinstance(fingerprint, str) or not fingerprint:
        raise EvaluationError("data_fingerprint must be a nonempty string")
    if not isinstance(config_fingerprint, str) or not config_fingerprint:
        raise EvaluationError("config_fingerprint must be a nonempty string")
    aggregate = EvaluationResult(
        sample_count=len(samples),
        exact_matches=sum(sample.exact_match for sample in samples),
        character_edits=totals("character_edits"),
        word_edits=totals("word_edits"),
        data_fingerprint=fingerprint,
        config_fingerprint=config_fingerprint,
        model_fingerprint=model_fingerprint,
    )
    return EvaluationReport(aggregate=aggregate, samples=samples)


__all__ = [
    "METRIC_POLICY_FINGERPRINT",
    "METRIC_POLICY_ID",
    "AlignmentStep",
    "EditOperation",
    "EvaluationReport",
    "SampleEvaluation",
    "align_sequences",
    "ascii_space_tokens",
    "character_error_rate",
    "edit_counts",
    "edit_distance",
    "evaluate_predictions",
    "evaluate_text",
    "exact_match",
    "word_error_rate",
]
