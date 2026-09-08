"""Deterministic character-level OCR error analysis."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from urdu_document_ocr.errors import EvaluationError
from urdu_document_ocr.evaluation.metrics import (
    EditOperation,
    EvaluationReport,
    SampleEvaluation,
    align_sequences,
)


def _code_point(character: str) -> str:
    return f"U+{ord(character):04X}"


@dataclass(frozen=True, slots=True)
class CharacterFrequency:
    character: str
    count: int

    def __post_init__(self) -> None:
        if not isinstance(self.character, str) or len(self.character) != 1:
            raise TypeError("character must contain one Unicode code point")
        if isinstance(self.count, bool) or not isinstance(self.count, int) or self.count < 1:
            raise ValueError("count must be a positive integer")

    def to_public_dict(self) -> dict[str, str | int]:
        return {
            "character": self.character,
            "code_point": _code_point(self.character),
            "count": self.count,
        }


@dataclass(frozen=True, slots=True)
class SubstitutionConfusion:
    reference_character: str
    hypothesis_character: str
    count: int

    def __post_init__(self) -> None:
        if not isinstance(self.reference_character, str) or len(self.reference_character) != 1:
            raise TypeError("reference_character must contain one Unicode code point")
        if not isinstance(self.hypothesis_character, str) or len(self.hypothesis_character) != 1:
            raise TypeError("hypothesis_character must contain one Unicode code point")
        if isinstance(self.count, bool) or not isinstance(self.count, int) or self.count < 1:
            raise ValueError("count must be a positive integer")

    def to_public_dict(self) -> dict[str, str | int]:
        return {
            "reference_character": self.reference_character,
            "reference_code_point": _code_point(self.reference_character),
            "hypothesis_character": self.hypothesis_character,
            "hypothesis_code_point": _code_point(self.hypothesis_character),
            "count": self.count,
        }


@dataclass(frozen=True, slots=True)
class ErrorAnalysis:
    """Aggregate aligned errors and complete worst-sample evidence."""

    worst_samples: tuple[SampleEvaluation, ...]
    substitution_pairs: tuple[SubstitutionConfusion, ...]
    deleted_characters: tuple[CharacterFrequency, ...]
    inserted_characters: tuple[CharacterFrequency, ...]
    reference_character_frequency: tuple[CharacterFrequency, ...]
    hypothesis_character_frequency: tuple[CharacterFrequency, ...]
    substitutions: int
    deletions: int
    insertions: int

    def to_public_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "alignment_source": "ocr-edit-v1-deterministic-optimal-alignment",
            "totals": {
                "substitutions": self.substitutions,
                "deletions": self.deletions,
                "insertions": self.insertions,
            },
            "worst_samples": [sample.to_public_dict() for sample in self.worst_samples],
            "substitution_pairs": [pair.to_public_dict() for pair in self.substitution_pairs],
            "deleted_characters": [item.to_public_dict() for item in self.deleted_characters],
            "inserted_characters": [item.to_public_dict() for item in self.inserted_characters],
            "reference_character_frequency": [
                item.to_public_dict() for item in self.reference_character_frequency
            ],
            "hypothesis_character_frequency": [
                item.to_public_dict() for item in self.hypothesis_character_frequency
            ],
        }


def _frequencies(counter: Counter[str]) -> tuple[CharacterFrequency, ...]:
    return tuple(
        CharacterFrequency(character, count)
        for character, count in sorted(counter.items(), key=lambda item: (-item[1], ord(item[0])))
    )


def analyze_errors(report: EvaluationReport, *, worst_limit: int = 10) -> ErrorAnalysis:
    """Analyze only operations selected by the frozen optimal alignment.

    Defined CER values sort first by descending rate, then descending edit count,
    then sample ID. Undefined zero-reference rates follow them in sample-ID order.
    """

    if not isinstance(report, EvaluationReport):
        raise TypeError("report must be an EvaluationReport")
    if isinstance(worst_limit, bool) or not isinstance(worst_limit, int) or worst_limit < 0:
        raise EvaluationError("worst_limit must be a nonnegative integer")

    substitutions: Counter[tuple[str, str]] = Counter()
    deletions: Counter[str] = Counter()
    insertions: Counter[str] = Counter()
    reference_frequency: Counter[str] = Counter()
    hypothesis_frequency: Counter[str] = Counter()
    for sample in report.samples:
        reference_frequency.update(sample.reference)
        hypothesis_frequency.update(sample.hypothesis)
        for step in align_sequences(sample.reference, sample.hypothesis):
            if step.operation is EditOperation.SUBSTITUTION:
                assert isinstance(step.reference, str) and isinstance(step.hypothesis, str)
                substitutions[(step.reference, step.hypothesis)] += 1
            elif step.operation is EditOperation.DELETION:
                assert isinstance(step.reference, str)
                deletions[step.reference] += 1
            elif step.operation is EditOperation.INSERTION:
                assert isinstance(step.hypothesis, str)
                insertions[step.hypothesis] += 1

    ordered_samples = sorted(
        report.samples,
        key=lambda sample: (
            sample.character_error_rate is None,
            -(sample.character_error_rate or 0.0),
            -sample.character_edits.total_errors,
            sample.sample_id,
        ),
    )
    substitution_pairs = tuple(
        SubstitutionConfusion(reference, hypothesis, count)
        for (reference, hypothesis), count in sorted(
            substitutions.items(),
            key=lambda item: (-item[1], ord(item[0][0]), ord(item[0][1])),
        )
    )
    aggregate = report.aggregate.character_edits
    return ErrorAnalysis(
        worst_samples=tuple(ordered_samples[:worst_limit]),
        substitution_pairs=substitution_pairs,
        deleted_characters=_frequencies(deletions),
        inserted_characters=_frequencies(insertions),
        reference_character_frequency=_frequencies(reference_frequency),
        hypothesis_character_frequency=_frequencies(hypothesis_frequency),
        substitutions=aggregate.substitutions,
        deletions=aggregate.deletions,
        insertions=aggregate.insertions,
    )


__all__ = [
    "CharacterFrequency",
    "ErrorAnalysis",
    "SubstitutionConfusion",
    "analyze_errors",
]
