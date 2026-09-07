"""Dataset-derived CTC vocabulary construction and safe JSON persistence."""

from __future__ import annotations

import json
import os
from collections.abc import Iterable
from pathlib import Path

from urdu_document_ocr.data.manifest import (
    _atomic_write_bytes,
    _canonical_json_bytes,
    _reject_duplicate_json_keys,
    _reject_nonstandard_json_constant,
)
from urdu_document_ocr.data.validation import (
    NORMALIZATION_POLICY_VERSION,
    _transcription_issue_codes,
)
from urdu_document_ocr.errors import VocabularyError
from urdu_document_ocr.types import DatasetSample, Vocabulary

_VOCABULARY_FIELDS = frozenset(
    {
        "schema_version",
        "normalization_policy_version",
        "blank_index",
        "characters",
        "fingerprint",
    }
)


def build_vocabulary(samples: Iterable[DatasetSample]) -> Vocabulary:
    """Build a code-point-sorted vocabulary from canonical training transcriptions."""

    values = tuple(samples)
    if any(not isinstance(sample, DatasetSample) for sample in values):
        raise VocabularyError("samples must contain only DatasetSample values")
    invalid = next((sample for sample in values if _transcription_issue_codes(sample.text)), None)
    if invalid is not None:
        raise VocabularyError(
            "transcriptions must satisfy the normalization policy",
            context={"sample_id": invalid.sample_id},
        )
    characters = tuple(
        sorted({character for sample in values for character in sample.text}, key=ord)
    )
    if not characters:
        raise VocabularyError("at least one transcription character is required")
    return Vocabulary(characters, normalization_policy_version=NORMALIZATION_POLICY_VERSION)


def find_unseen_characters(
    samples: Iterable[DatasetSample], vocabulary: Vocabulary
) -> tuple[str, ...]:
    """Return sorted validation/test code points absent from a training vocabulary."""

    if not isinstance(vocabulary, Vocabulary):
        raise TypeError("vocabulary must be Vocabulary")
    characters = set(vocabulary.characters)
    unseen: set[str] = set()
    for sample in samples:
        if not isinstance(sample, DatasetSample):
            raise VocabularyError("samples must contain only DatasetSample values")
        unseen.update(set(sample.text) - characters)
    return tuple(sorted(unseen, key=ord))


def save_vocabulary(
    vocabulary: Vocabulary,
    path: str | os.PathLike[str],
    *,
    overwrite: bool = False,
) -> None:
    """Atomically persist canonical JSON with an integrity fingerprint."""

    if not isinstance(vocabulary, Vocabulary):
        raise TypeError("vocabulary must be Vocabulary")
    payload = _canonical_json_bytes(vocabulary.to_public_dict())
    _atomic_write_bytes(
        payload,
        path,
        overwrite=overwrite,
        error_type=VocabularyError,
        artifact_name="vocabulary",
    )


def load_vocabulary(path: str | os.PathLike[str]) -> Vocabulary:
    """Load strict vocabulary JSON and verify ordering, uniqueness, policy, and hash."""

    try:
        source = Path(path)
        raw = source.read_text(encoding="utf-8")
    except (OSError, TypeError, ValueError, UnicodeError) as error:
        raise VocabularyError("vocabulary could not be read as UTF-8 JSON") from error
    try:
        record = json.loads(
            raw,
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=_reject_nonstandard_json_constant,
        )
    except ValueError as error:
        raise VocabularyError("vocabulary contains malformed JSON") from error
    if not isinstance(record, dict) or set(record) != _VOCABULARY_FIELDS:
        raise VocabularyError("vocabulary fields do not match schema version 1")
    if type(record["schema_version"]) is not int or record["schema_version"] != 1:
        raise VocabularyError("vocabulary schema_version must be integer 1")
    if record["normalization_policy_version"] != NORMALIZATION_POLICY_VERSION:
        raise VocabularyError("vocabulary normalization policy is unsupported")
    if type(record["blank_index"]) is not int or record["blank_index"] != 0:
        raise VocabularyError("vocabulary blank_index must be integer 0")
    characters = record["characters"]
    if not isinstance(characters, list) or any(
        not isinstance(character, str) or len(character) != 1 for character in characters
    ):
        raise VocabularyError("vocabulary characters must be single code points")
    if len(characters) != len(set(characters)):
        raise VocabularyError("vocabulary characters must be unique")
    if characters != sorted(characters, key=ord):
        raise VocabularyError("vocabulary characters must use Unicode code-point order")
    if not isinstance(record["fingerprint"], str):
        raise VocabularyError("vocabulary fingerprint must be a string")
    try:
        vocabulary = Vocabulary(
            tuple(characters),
            schema_version=record["schema_version"],
            normalization_policy_version=record["normalization_policy_version"],
            blank_index=record["blank_index"],
        )
    except (TypeError, ValueError) as error:
        raise VocabularyError("vocabulary violates structural constraints") from error
    if record["fingerprint"] != vocabulary.fingerprint:
        raise VocabularyError("vocabulary fingerprint does not match its character payload")
    return vocabulary
