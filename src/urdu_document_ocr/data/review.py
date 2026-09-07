"""Immutable review-decision overlays for labeled line samples."""

from __future__ import annotations

import json
import os
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from enum import StrEnum
from hashlib import sha256
from pathlib import Path

from urdu_document_ocr.data.manifest import (
    _atomic_write_bytes,
    _canonical_json_bytes,
    _reject_duplicate_json_keys,
    _reject_nonstandard_json_constant,
    dataset_fingerprint,
)
from urdu_document_ocr.errors import ReviewOverlayError
from urdu_document_ocr.types import DatasetSample

_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_SAFE_LABEL_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")
_SAFE_TAG_PATTERN = re.compile(r"[a-z0-9][a-z0-9-]{0,31}")
_REQUIRED_FIELDS = frozenset(
    {"schema_version", "sample_id", "status", "tags", "source_dataset_fingerprint"}
)
_OPTIONAL_FIELDS = frozenset({"note", "reviewer"})


class ReviewStatus(StrEnum):
    VALID = "valid"
    INVALID = "invalid"
    NEEDS_REVIEW = "needs_review"


@dataclass(frozen=True, slots=True)
class ReviewDecision:
    sample_id: str
    status: ReviewStatus
    source_dataset_fingerprint: str
    tags: tuple[str, ...] = ()
    note: str | None = None
    reviewer: str | None = None
    schema_version: int = 1

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("schema_version must be 1")
        if not isinstance(self.sample_id, str) or not self.sample_id.strip():
            raise ValueError("sample_id must be a nonempty string")
        if not isinstance(self.status, ReviewStatus):
            raise TypeError("status must be ReviewStatus")
        if not isinstance(self.source_dataset_fingerprint, str):
            raise TypeError("source_dataset_fingerprint must be a string")
        fingerprint = self.source_dataset_fingerprint.lower()
        if _SHA256_PATTERN.fullmatch(fingerprint) is None:
            raise ValueError("source_dataset_fingerprint must be 64 lowercase hexadecimal digits")
        object.__setattr__(self, "source_dataset_fingerprint", fingerprint)
        if not isinstance(self.tags, tuple):
            raise TypeError("tags must be a tuple")
        tags = self.tags
        if any(
            not isinstance(tag, str) or _SAFE_TAG_PATTERN.fullmatch(tag) is None for tag in tags
        ):
            raise ValueError("tags must use lowercase letters, digits, and hyphens")
        if len(tags) != len(set(tags)):
            raise ValueError("tags must not contain duplicates")
        object.__setattr__(self, "tags", tuple(sorted(tags)))
        if self.note is not None and (
            not isinstance(self.note, str)
            or not self.note.strip()
            or len(self.note) > 240
            or any(character in self.note for character in "\r\n\x00")
        ):
            raise ValueError("note must be a nonempty single-line string of at most 240 characters")
        if self.reviewer is not None and (
            not isinstance(self.reviewer, str)
            or _SAFE_LABEL_PATTERN.fullmatch(self.reviewer) is None
        ):
            raise ValueError("reviewer must be a safe generic label")

    def to_public_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "schema_version": self.schema_version,
            "sample_id": self.sample_id,
            "status": self.status.value,
            "tags": list(self.tags),
            "source_dataset_fingerprint": self.source_dataset_fingerprint,
        }
        if self.note is not None:
            result["note"] = self.note
        if self.reviewer is not None:
            result["reviewer"] = self.reviewer
        return result


def _canonical_overlay_bytes(decisions: Iterable[ReviewDecision]) -> bytes:
    values = tuple(decisions)
    if any(not isinstance(decision, ReviewDecision) for decision in values):
        raise ReviewOverlayError("overlay must contain only ReviewDecision values")
    identifiers = [decision.sample_id for decision in values]
    if len(identifiers) != len(set(identifiers)):
        raise ReviewOverlayError("overlay contains duplicate decisions")
    ordered = sorted(values, key=lambda decision: decision.sample_id.encode("utf-8"))
    return b"".join(_canonical_json_bytes(decision.to_public_dict()) for decision in ordered)


def review_overlay_fingerprint(decisions: Iterable[ReviewDecision]) -> str:
    return sha256(_canonical_overlay_bytes(decisions)).hexdigest()


def _record_to_decision(record: Mapping[str, object], line_number: int) -> ReviewDecision:
    fields = set(record)
    if not _REQUIRED_FIELDS.issubset(fields) or not fields.issubset(
        _REQUIRED_FIELDS | _OPTIONAL_FIELDS
    ):
        raise ReviewOverlayError(
            "review decision fields do not match schema version 1",
            context={"line_number": line_number},
        )
    if type(record["schema_version"]) is not int or record["schema_version"] != 1:
        raise ReviewOverlayError(
            "review schema_version must be integer 1", context={"line_number": line_number}
        )
    if not isinstance(record["sample_id"], str):
        raise ReviewOverlayError(
            "review sample_id must be a string", context={"line_number": line_number}
        )
    if not isinstance(record["status"], str):
        raise ReviewOverlayError(
            "review status must be a string", context={"line_number": line_number}
        )
    if not isinstance(record["source_dataset_fingerprint"], str):
        raise ReviewOverlayError(
            "review source fingerprint must be a string", context={"line_number": line_number}
        )
    tags = record["tags"]
    if not isinstance(tags, list) or any(not isinstance(tag, str) for tag in tags):
        raise ReviewOverlayError(
            "review tags must be a string array", context={"line_number": line_number}
        )
    note = record.get("note")
    reviewer = record.get("reviewer")
    if note is not None and not isinstance(note, str):
        raise ReviewOverlayError(
            "review note must be a string", context={"line_number": line_number}
        )
    if reviewer is not None and not isinstance(reviewer, str):
        raise ReviewOverlayError("reviewer must be a string", context={"line_number": line_number})
    try:
        return ReviewDecision(
            sample_id=record["sample_id"],
            status=ReviewStatus(record["status"]),
            source_dataset_fingerprint=record["source_dataset_fingerprint"],
            tags=tuple(tags),
            note=note,
            reviewer=reviewer,
        )
    except (TypeError, ValueError) as error:
        raise ReviewOverlayError(
            "review decision violates schema constraints", context={"line_number": line_number}
        ) from error


def read_review_overlay(path: str | os.PathLike[str]) -> tuple[ReviewDecision, ...]:
    """Read strict review JSONL in file order and reject duplicate decisions."""

    try:
        source = Path(path)
        handle = source.open("r", encoding="utf-8", newline="")
    except (OSError, TypeError, ValueError) as error:
        raise ReviewOverlayError("review overlay could not be opened") from error
    decisions: list[ReviewDecision] = []
    identifiers: set[str] = set()
    try:
        with handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(
                        line,
                        object_pairs_hook=_reject_duplicate_json_keys,
                        parse_constant=_reject_nonstandard_json_constant,
                    )
                except ValueError as error:
                    raise ReviewOverlayError(
                        "review overlay contains malformed JSON",
                        context={"line_number": line_number},
                    ) from error
                if not isinstance(record, dict):
                    raise ReviewOverlayError(
                        "review line must contain one JSON object",
                        context={"line_number": line_number},
                    )
                decision = _record_to_decision(record, line_number)
                if decision.sample_id in identifiers:
                    raise ReviewOverlayError(
                        "review overlay contains duplicate decisions",
                        context={"line_number": line_number, "sample_id": decision.sample_id},
                    )
                identifiers.add(decision.sample_id)
                decisions.append(decision)
    except UnicodeError as error:
        raise ReviewOverlayError("review overlay must be valid UTF-8") from error
    except OSError as error:
        raise ReviewOverlayError("review overlay could not be read") from error
    return tuple(decisions)


def write_review_overlay(
    decisions: Iterable[ReviewDecision],
    path: str | os.PathLike[str],
    *,
    overwrite: bool = False,
) -> None:
    """Atomically write decisions sorted by sample ID."""

    payload = _canonical_overlay_bytes(decisions)
    _atomic_write_bytes(
        payload,
        path,
        overwrite=overwrite,
        error_type=ReviewOverlayError,
        artifact_name="review overlay",
    )


def apply_review_overlay(
    samples: Iterable[DatasetSample],
    decisions: Iterable[ReviewDecision],
    *,
    include_needs_review: bool = False,
) -> tuple[DatasetSample, ...]:
    """Return a derived sample tuple; source samples and image files remain untouched."""

    values = tuple(samples)
    overlay = tuple(decisions)
    if any(not isinstance(sample, DatasetSample) for sample in values):
        raise ReviewOverlayError("samples must contain only DatasetSample values")
    if not isinstance(include_needs_review, bool):
        raise TypeError("include_needs_review must be a boolean")
    _canonical_overlay_bytes(overlay)
    identifiers = {sample.sample_id for sample in values}
    if len(identifiers) != len(values):
        raise ReviewOverlayError("source samples contain duplicate sample_id values")
    unknown = sorted(
        decision.sample_id for decision in overlay if decision.sample_id not in identifiers
    )
    if unknown:
        raise ReviewOverlayError(
            "review overlay refers to an unknown sample_id", context={"sample_id": unknown[0]}
        )
    expected_fingerprint = dataset_fingerprint(values)
    if any(decision.source_dataset_fingerprint != expected_fingerprint for decision in overlay):
        raise ReviewOverlayError("review overlay source fingerprint is stale")

    by_id = {decision.sample_id: decision for decision in overlay}
    result = []
    for sample in values:
        decision = by_id.get(sample.sample_id)
        if decision is None:
            result.append(sample)
            continue
        if decision.status is ReviewStatus.INVALID:
            continue
        if decision.status is ReviewStatus.NEEDS_REVIEW and not include_needs_review:
            continue
        combined_tags = tuple(sorted(set(sample.tags) | set(decision.tags)))
        result.append(replace(sample, tags=combined_tags))
    return tuple(result)
