"""Non-mutating dataset, Unicode, containment, and image validation."""

from __future__ import annotations

import os
import unicodedata
import warnings
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum
from hashlib import sha256
from pathlib import Path
from statistics import median

import numpy as np
from numpy.typing import NDArray
from PIL import Image, UnidentifiedImageError

from urdu_document_ocr.data.manifest import dataset_fingerprint
from urdu_document_ocr.errors import DatasetValidationError
from urdu_document_ocr.types import DatasetSample, Vocabulary

NORMALIZATION_POLICY_VERSION = "nfc-v1"
_PERMITTED_FORMATS = frozenset({"JPEG", "PNG"})
_MAXIMUM_IMAGE_BYTES = 100 * 1024 * 1024
_MAXIMUM_IMAGE_PIXELS = 50_000_000
_UNUSUAL_MAXIMUM_WIDTH = 4_096
_UNUSUAL_ASPECT_RATIO = 50.0
_UNUSUAL_TEXT_LENGTH = 512
_MINIMUM_USUAL_DIMENSION = 4
_ZWNJ = "\u200c"


class _ImagePolicyError(ValueError):
    pass


class ValidationSeverity(StrEnum):
    ERROR = "error"
    WARNING = "warning"


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    code: str
    severity: ValidationSeverity
    message: str
    sample_id: str | None = None

    def __post_init__(self) -> None:
        if not self.code or not self.message:
            raise ValueError("validation issue code and message must be nonempty")
        if not isinstance(self.severity, ValidationSeverity):
            raise TypeError("severity must be ValidationSeverity")

    def to_public_dict(self) -> dict[str, str | None]:
        return {
            "code": self.code,
            "severity": self.severity.value,
            "message": self.message,
            "sample_id": self.sample_id,
        }


@dataclass(frozen=True, slots=True)
class NumericSummary:
    minimum: int | None
    median: float | None
    maximum: int | None

    def to_public_dict(self) -> dict[str, int | float | None]:
        return {"minimum": self.minimum, "median": self.median, "maximum": self.maximum}


@dataclass(frozen=True, slots=True)
class DatasetStatistics:
    sample_count: int
    document_count: int
    image_width: NumericSummary
    image_height: NumericSummary
    transcription_characters: NumericSummary
    transcription_words: NumericSummary
    samples_per_document: NumericSummary
    unique_character_count: int
    character_frequency: tuple[tuple[str, int], ...]
    empty_text_count: int
    image_error_count: int

    def to_public_dict(self) -> dict[str, object]:
        return {
            "sample_count": self.sample_count,
            "document_count": self.document_count,
            "image_width": self.image_width.to_public_dict(),
            "image_height": self.image_height.to_public_dict(),
            "transcription_characters": self.transcription_characters.to_public_dict(),
            "transcription_words": self.transcription_words.to_public_dict(),
            "samples_per_document": self.samples_per_document.to_public_dict(),
            "unique_character_count": self.unique_character_count,
            "character_frequency": [
                {"character": character, "count": count}
                for character, count in self.character_frequency
            ],
            "empty_text_count": self.empty_text_count,
            "image_error_count": self.image_error_count,
        }


@dataclass(frozen=True, slots=True)
class DatasetValidationReport:
    dataset_fingerprint: str
    issues: tuple[ValidationIssue, ...]
    statistics: DatasetStatistics
    schema_version: int = 1

    @property
    def error_count(self) -> int:
        return sum(issue.severity is ValidationSeverity.ERROR for issue in self.issues)

    @property
    def warning_count(self) -> int:
        return sum(issue.severity is ValidationSeverity.WARNING for issue in self.issues)

    @property
    def is_valid(self) -> bool:
        return self.error_count == 0

    def to_public_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "dataset_fingerprint": self.dataset_fingerprint,
            "is_valid": self.is_valid,
            "error_count": self.error_count,
            "warning_count": self.warning_count,
            "issues": [issue.to_public_dict() for issue in self.issues],
            "statistics": self.statistics.to_public_dict(),
        }


def _contains_forbidden_control(text: str) -> bool:
    return any(
        character != _ZWNJ and unicodedata.category(character) in {"Cc", "Cf", "Cs", "Zl", "Zp"}
        for character in text
    )


def normalize_transcription(text: str) -> str:
    """Explicitly produce the frozen NFC/single-space form without silent label folding."""

    if not isinstance(text, str):
        raise TypeError("text must be a string")
    if _contains_forbidden_control(text):
        raise ValueError("text contains a forbidden control character")
    normalized = unicodedata.normalize("NFC", text)
    normalized = " ".join(normalized.split())
    if not normalized:
        raise ValueError("text must not normalize to an empty value")
    return normalized


def _transcription_issue_codes(text: str) -> tuple[str, ...]:
    issues = []
    if not text or not text.strip():
        issues.append("empty_text")
        return tuple(issues)
    if _contains_forbidden_control(text):
        issues.append("invalid_text_control")
    if unicodedata.normalize("NFC", text) != text:
        issues.append("non_nfc_text")
    if text[:1].isspace() or text[-1:].isspace():
        issues.append("leading_or_trailing_whitespace")
    if "  " in text or any(character.isspace() and character != " " for character in text):
        issues.append("noncanonical_whitespace")
    return tuple(issues)


def _summary(values: Iterable[int]) -> NumericSummary:
    sequence = tuple(values)
    if not sequence:
        return NumericSummary(None, None, None)
    return NumericSummary(min(sequence), float(median(sequence)), max(sequence))


def _safe_root(dataset_root: str | os.PathLike[str]) -> Path:
    try:
        root = Path(dataset_root).resolve(strict=True)
    except (OSError, TypeError, ValueError) as error:
        raise DatasetValidationError("dataset root is unavailable") from error
    if not root.is_dir():
        raise DatasetValidationError("dataset root must be a directory")
    return root


def _resolved_sample_path(root: Path, sample: DatasetSample) -> Path | None:
    try:
        resolved = (root / sample.image_path).resolve(strict=False)
    except (OSError, RuntimeError):
        return None
    if not resolved.is_relative_to(root):
        return None
    return resolved


def _hash_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _inspect_image(path: Path) -> tuple[int, int, str]:
    if path.stat().st_size > _MAXIMUM_IMAGE_BYTES:
        raise _ImagePolicyError("image_too_large")
    with warnings.catch_warnings():
        warnings.simplefilter("error", Image.DecompressionBombWarning)
        with Image.open(path) as image:
            width, height = image.size
            image_format = image.format
            if image_format not in _PERMITTED_FORMATS:
                raise _ImagePolicyError("unsupported_image_format")
            if width < 1 or height < 1 or width * height > _MAXIMUM_IMAGE_PIXELS:
                raise _ImagePolicyError("invalid_dimensions")
            image.verify()
    return width, height, _hash_file(path)


def load_dataset_line_image(
    sample: DatasetSample,
    *,
    dataset_root: str | os.PathLike[str],
) -> NDArray[np.uint8]:
    """Safely resolve and decode one canonical PNG/JPEG sample as grayscale uint8."""

    if not isinstance(sample, DatasetSample):
        raise DatasetValidationError("sample must be a DatasetSample")
    root = _safe_root(dataset_root)
    image_path = _resolved_sample_path(root, sample)
    if image_path is None:
        raise DatasetValidationError(
            "image path resolves outside the dataset root",
            context={"sample_id": sample.sample_id, "image_path": sample.image_path},
        )
    if not image_path.exists() or not image_path.is_file():
        raise DatasetValidationError(
            "line image is missing or is not a regular file",
            context={"sample_id": sample.sample_id, "image_path": sample.image_path},
        )
    try:
        _inspect_image(image_path)
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(image_path) as image:
                grayscale = np.asarray(image.convert("L"), dtype=np.uint8).copy()
    except (
        OSError,
        SyntaxError,
        UnidentifiedImageError,
        Image.DecompressionBombError,
        Image.DecompressionBombWarning,
        _ImagePolicyError,
    ) as error:
        raise DatasetValidationError(
            "line image could not be decoded safely as PNG or JPEG",
            context={"sample_id": sample.sample_id, "image_path": sample.image_path},
        ) from error
    if grayscale.ndim != 2 or grayscale.size == 0:
        raise DatasetValidationError(
            "decoded line image has invalid dimensions",
            context={"sample_id": sample.sample_id, "image_path": sample.image_path},
        )
    return grayscale


def _issue(
    code: str,
    severity: ValidationSeverity,
    message: str,
    sample: DatasetSample | None = None,
) -> ValidationIssue:
    return ValidationIssue(code, severity, message, None if sample is None else sample.sample_id)


def _duplicate_issues(
    samples: tuple[DatasetSample, ...],
    image_digests: dict[tuple[str, str, str, str], str],
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    by_id: dict[str, list[DatasetSample]] = defaultdict(list)
    by_path: dict[str, list[DatasetSample]] = defaultdict(list)
    by_digest: dict[str, list[DatasetSample]] = defaultdict(list)
    for sample in samples:
        by_id[sample.sample_id].append(sample)
        by_path[sample.image_path].append(sample)
        digest = image_digests.get(_sample_identity_key(sample))
        if digest is not None:
            by_digest[digest].append(sample)

    for values in by_id.values():
        for sample in sorted(values, key=lambda item: _sample_identity_key(item))[1:]:
            issues.append(
                _issue(
                    "duplicate_sample_id",
                    ValidationSeverity.ERROR,
                    "sample_id must be unique",
                    sample,
                )
            )
    for values in by_path.values():
        for sample in sorted(values, key=lambda item: _sample_identity_key(item))[1:]:
            issues.append(
                _issue(
                    "duplicate_image_path",
                    ValidationSeverity.WARNING,
                    "image_path is referenced by more than one sample",
                    sample,
                )
            )
    for values in by_digest.values():
        ordered = sorted(values, key=lambda item: _sample_identity_key(item))
        for sample in ordered[1:]:
            issues.append(
                _issue(
                    "duplicate_image",
                    ValidationSeverity.WARNING,
                    "image bytes duplicate another sample",
                    sample,
                )
            )
        by_text: dict[str, list[DatasetSample]] = defaultdict(list)
        for sample in ordered:
            by_text[sample.text].append(sample)
        for same_text in by_text.values():
            for sample in same_text[1:]:
                issues.append(
                    _issue(
                        "duplicate_pair",
                        ValidationSeverity.WARNING,
                        "image and transcription duplicate another sample",
                        sample,
                    )
                )
        if len(by_text) > 1:
            for sample in ordered:
                issues.append(
                    _issue(
                        "conflicting_image_text",
                        ValidationSeverity.ERROR,
                        "identical image bytes have conflicting transcriptions",
                        sample,
                    )
                )
    return issues


def _sample_identity_key(sample: DatasetSample) -> tuple[str, str, str, str]:
    return sample.sample_id, sample.image_path, sample.document_id, sample.text


def _statistics(
    samples: tuple[DatasetSample, ...],
    widths: list[int],
    heights: list[int],
    issues: list[ValidationIssue],
) -> DatasetStatistics:
    character_counts = Counter(character for sample in samples for character in sample.text)
    document_counts = Counter(sample.document_id for sample in samples)
    image_error_samples = {
        issue.sample_id
        for issue in issues
        if issue.code
        in {
            "invalid_path",
            "missing_image",
            "not_regular_file",
            "unreadable_image",
            "unsupported_image_format",
            "image_too_large",
            "invalid_dimensions",
        }
    }
    return DatasetStatistics(
        sample_count=len(samples),
        document_count=len(document_counts),
        image_width=_summary(widths),
        image_height=_summary(heights),
        transcription_characters=_summary(len(sample.text) for sample in samples),
        transcription_words=_summary(len(sample.text.split()) for sample in samples),
        samples_per_document=_summary(document_counts.values()),
        unique_character_count=len(character_counts),
        character_frequency=tuple(sorted(character_counts.items(), key=lambda item: ord(item[0]))),
        empty_text_count=sum(not sample.text.strip() for sample in samples),
        image_error_count=len(image_error_samples),
    )


def validate_dataset(
    samples: Iterable[DatasetSample],
    *,
    dataset_root: str | os.PathLike[str] | None = None,
    vocabulary: Vocabulary | None = None,
    inspect_images: bool = True,
) -> DatasetValidationReport:
    """Return deterministic findings and statistics without changing samples or files."""

    values = tuple(samples)
    if any(not isinstance(sample, DatasetSample) for sample in values):
        raise DatasetValidationError("samples must contain only DatasetSample values")
    if vocabulary is not None and not isinstance(vocabulary, Vocabulary):
        raise DatasetValidationError("vocabulary must be Vocabulary or None")
    if not isinstance(inspect_images, bool):
        raise DatasetValidationError("inspect_images must be a boolean")
    root = None
    if inspect_images:
        if dataset_root is None:
            raise DatasetValidationError("dataset_root is required when inspect_images is true")
        root = _safe_root(dataset_root)

    issues: list[ValidationIssue] = []
    widths: list[int] = []
    heights: list[int] = []
    image_digests: dict[tuple[str, str, str, str], str] = {}
    vocabulary_characters = frozenset() if vocabulary is None else frozenset(vocabulary.characters)

    for sample in values:
        for code in _transcription_issue_codes(sample.text):
            messages = {
                "empty_text": "transcription must not be empty",
                "invalid_text_control": "transcription contains a forbidden control character",
                "non_nfc_text": "transcription is not Unicode NFC",
                "leading_or_trailing_whitespace": (
                    "transcription has leading or trailing whitespace"
                ),
                "noncanonical_whitespace": (
                    "transcription whitespace is not canonical single ASCII space"
                ),
            }
            issues.append(_issue(code, ValidationSeverity.ERROR, messages[code], sample))
        if len(sample.text) > _UNUSUAL_TEXT_LENGTH:
            issues.append(
                _issue(
                    "unusually_long_text",
                    ValidationSeverity.WARNING,
                    "transcription exceeds the review-length threshold",
                    sample,
                )
            )
        if vocabulary is not None:
            unknown = sorted(set(sample.text) - vocabulary_characters, key=ord)
            if unknown:
                codepoints = ",".join(f"U+{ord(character):04X}" for character in unknown)
                issues.append(
                    _issue(
                        "unsupported_character",
                        ValidationSeverity.WARNING,
                        f"transcription contains characters outside the vocabulary: {codepoints}",
                        sample,
                    )
                )

        if root is None:
            continue
        image_path = _resolved_sample_path(root, sample)
        if image_path is None:
            issues.append(
                _issue(
                    "invalid_path",
                    ValidationSeverity.ERROR,
                    "image path resolves outside the dataset root",
                    sample,
                )
            )
            continue
        if not image_path.exists():
            issues.append(
                _issue("missing_image", ValidationSeverity.ERROR, "image file is missing", sample)
            )
            continue
        if not image_path.is_file():
            issues.append(
                _issue(
                    "not_regular_file",
                    ValidationSeverity.ERROR,
                    "image path is not a regular file",
                    sample,
                )
            )
            continue
        try:
            width, height, digest = _inspect_image(image_path)
        except _ImagePolicyError as error:
            code = str(error)
            message = {
                "unsupported_image_format": "line image must decode as PNG or JPEG",
                "image_too_large": "line image exceeds the supported byte-size bound",
                "invalid_dimensions": "line image dimensions exceed the supported bounds",
            }[code]
            issues.append(_issue(code, ValidationSeverity.ERROR, message, sample))
            continue
        except (
            OSError,
            SyntaxError,
            UnidentifiedImageError,
            Image.DecompressionBombError,
            Image.DecompressionBombWarning,
        ):
            issues.append(
                _issue(
                    "unreadable_image",
                    ValidationSeverity.ERROR,
                    "line image could not be decoded safely",
                    sample,
                )
            )
            continue
        widths.append(width)
        heights.append(height)
        image_digests[_sample_identity_key(sample)] = digest
        aspect_ratio = max(width / height, height / width)
        if (
            min(width, height) < _MINIMUM_USUAL_DIMENSION
            or width > _UNUSUAL_MAXIMUM_WIDTH
            or aspect_ratio > _UNUSUAL_ASPECT_RATIO
        ):
            issues.append(
                _issue(
                    "unusual_dimensions",
                    ValidationSeverity.WARNING,
                    "line image dimensions deserve manual review",
                    sample,
                )
            )

    issues.extend(_duplicate_issues(values, image_digests))
    severity_order = {ValidationSeverity.ERROR: 0, ValidationSeverity.WARNING: 1}
    issues.sort(
        key=lambda item: (
            severity_order[item.severity],
            item.code,
            "" if item.sample_id is None else item.sample_id,
            item.message,
        )
    )
    return DatasetValidationReport(
        dataset_fingerprint(values),
        tuple(issues),
        _statistics(values, widths, heights, issues),
    )
