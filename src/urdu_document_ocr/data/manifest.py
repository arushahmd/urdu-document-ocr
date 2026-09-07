"""Strict UTF-8 JSONL manifests and logical dataset fingerprints."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Iterable, Mapping
from contextlib import suppress
from hashlib import sha256
from pathlib import Path
from typing import Any, TypeVar

from urdu_document_ocr.errors import ManifestError, UrduOCRError
from urdu_document_ocr.types import DatasetSample

_REQUIRED_FIELDS = frozenset({"schema_version", "sample_id", "image_path", "text", "document_id"})
_OPTIONAL_FIELDS = frozenset({"page_id", "line_index", "tags"})
_ALLOWED_FIELDS = _REQUIRED_FIELDS | _OPTIONAL_FIELDS
_ErrorT = TypeVar("_ErrorT", bound=UrduOCRError)


class _DuplicateJsonKey(ValueError):
    pass


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateJsonKey
        result[key] = value
    return result


def _reject_nonstandard_json_constant(_: str) -> None:
    raise ValueError


def _sample_record(sample: DatasetSample) -> dict[str, object]:
    if not isinstance(sample, DatasetSample):
        raise TypeError("samples must contain only DatasetSample values")
    record: dict[str, object] = {
        "schema_version": sample.schema_version,
        "sample_id": sample.sample_id,
        "image_path": sample.image_path,
        "text": sample.text,
        "document_id": sample.document_id,
    }
    if sample.page_id is not None:
        record["page_id"] = sample.page_id
    if sample.line_index is not None:
        record["line_index"] = sample.line_index
    if sample.tags:
        record["tags"] = list(sample.tags)
    return record


def _canonical_json_bytes(value: object, *, final_newline: bool = True) -> bytes:
    suffix = "\n" if final_newline else ""
    return (
        json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + suffix
    ).encode("utf-8")


def canonical_manifest_bytes(samples: Iterable[DatasetSample]) -> bytes:
    """Return canonical JSONL sorted by sample ID, independent of input order."""

    records = [(_sample_record(sample), sample.sample_id) for sample in samples]
    encoded = [(_canonical_json_bytes(record), sample_id) for record, sample_id in records]
    encoded.sort(key=lambda item: (item[1], item[0]))
    return b"".join(payload for payload, _ in encoded)


def dataset_fingerprint(samples: Iterable[DatasetSample]) -> str:
    """Hash canonical logical records; manifest ordering and filesystem state do not matter."""

    return sha256(canonical_manifest_bytes(samples)).hexdigest()


def _record_to_sample(record: Mapping[str, object], line_number: int) -> DatasetSample:
    fields = set(record)
    if not _REQUIRED_FIELDS.issubset(fields):
        raise ManifestError(
            "manifest record is missing required fields", context={"line_number": line_number}
        )
    if not fields.issubset(_ALLOWED_FIELDS):
        raise ManifestError(
            "manifest record contains unknown fields", context={"line_number": line_number}
        )
    if type(record["schema_version"]) is not int or record["schema_version"] != 1:
        raise ManifestError(
            "manifest schema_version must be integer 1", context={"line_number": line_number}
        )
    for name in ("sample_id", "image_path", "text", "document_id"):
        if not isinstance(record[name], str):
            raise ManifestError(
                "manifest record has an invalid field type",
                context={"line_number": line_number, "field": name},
            )
    page_id = record.get("page_id")
    if page_id is not None and not isinstance(page_id, str):
        raise ManifestError(
            "manifest record has an invalid field type",
            context={"line_number": line_number, "field": "page_id"},
        )
    line_index = record.get("line_index")
    if line_index is not None and (type(line_index) is not int or line_index < 0):
        raise ManifestError(
            "manifest record has an invalid field type",
            context={"line_number": line_number, "field": "line_index"},
        )
    tags_value = record.get("tags", [])
    if not isinstance(tags_value, list) or any(not isinstance(tag, str) for tag in tags_value):
        raise ManifestError(
            "manifest record has an invalid field type",
            context={"line_number": line_number, "field": "tags"},
        )
    try:
        return DatasetSample(
            schema_version=1,
            sample_id=record["sample_id"],
            image_path=record["image_path"],
            text=record["text"],
            document_id=record["document_id"],
            page_id=page_id,
            line_index=line_index,
            tags=tuple(tags_value),
        )
    except (TypeError, ValueError) as error:
        raise ManifestError(
            "manifest record violates schema constraints", context={"line_number": line_number}
        ) from error


def read_manifest(path: str | os.PathLike[str]) -> tuple[DatasetSample, ...]:
    """Read schema-v1 JSONL in file order; whitespace-only lines are ignored."""

    try:
        source = Path(path)
        handle = source.open("r", encoding="utf-8", newline="")
    except (OSError, TypeError, ValueError) as error:
        raise ManifestError("manifest could not be opened") from error

    samples: list[DatasetSample] = []
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
                except (json.JSONDecodeError, _DuplicateJsonKey, ValueError) as error:
                    raise ManifestError(
                        "manifest contains malformed JSON", context={"line_number": line_number}
                    ) from error
                if not isinstance(record, dict):
                    raise ManifestError(
                        "manifest line must contain one JSON object",
                        context={"line_number": line_number},
                    )
                sample = _record_to_sample(record, line_number)
                if sample.sample_id in identifiers:
                    raise ManifestError(
                        "manifest contains a duplicate sample_id",
                        context={"line_number": line_number, "sample_id": sample.sample_id},
                    )
                identifiers.add(sample.sample_id)
                samples.append(sample)
    except UnicodeError as error:
        raise ManifestError("manifest must be valid UTF-8") from error
    except OSError as error:
        raise ManifestError("manifest could not be read") from error
    return tuple(samples)


def _atomic_write_bytes(
    payload: bytes,
    path: str | os.PathLike[str],
    *,
    overwrite: bool,
    error_type: type[_ErrorT],
    artifact_name: str,
) -> None:
    try:
        target = Path(path)
    except (TypeError, ValueError) as error:
        raise error_type(f"invalid {artifact_name} destination") from error
    if not target.parent.is_dir():
        raise error_type(f"{artifact_name} parent directory does not exist")
    if target.exists() and not overwrite:
        raise error_type(f"{artifact_name} already exists; set overwrite=True explicitly")

    temporary: Path | None = None
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
        )
        temporary = Path(temporary_name)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        if target.exists() and not overwrite:
            raise error_type(f"{artifact_name} already exists; set overwrite=True explicitly")
        os.replace(temporary, target)
        temporary = None
    except UrduOCRError:
        raise
    except OSError as error:
        raise error_type(f"{artifact_name} could not be written") from error
    finally:
        if temporary is not None:
            with suppress(OSError):
                temporary.unlink(missing_ok=True)


def write_manifest(
    samples: Iterable[DatasetSample],
    path: str | os.PathLike[str],
    *,
    overwrite: bool = False,
) -> None:
    """Atomically write canonical JSONL; existing files require explicit overwrite."""

    try:
        values = tuple(samples)
        if any(not isinstance(sample, DatasetSample) for sample in values):
            raise TypeError("non-sample value")
        identifiers = [sample.sample_id for sample in values]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("duplicate sample_id")
        payload = canonical_manifest_bytes(values)
    except (TypeError, ValueError) as error:
        raise ManifestError("manifest samples are invalid") from error
    _atomic_write_bytes(
        payload,
        path,
        overwrite=overwrite,
        error_type=ManifestError,
        artifact_name="manifest",
    )
