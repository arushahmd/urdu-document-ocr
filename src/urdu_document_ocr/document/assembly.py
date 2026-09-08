"""Deterministic, privacy-safe OCR result assembly and serialization."""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from pathlib import Path

from urdu_document_ocr.data.manifest import _atomic_write_bytes
from urdu_document_ocr.errors import AssemblyError
from urdu_document_ocr.types import (
    DocumentOCRResult,
    LineOCRResult,
    PageOCRResult,
    PreprocessedPage,
)

LINE_SEPARATOR = "\n"
PAGE_SEPARATOR = "\n\n"


def assemble_page(
    page: PreprocessedPage,
    lines: Sequence[LineOCRResult],
) -> PageOCRResult:
    """Bind ordered line results to one source page and join successful text."""

    if not isinstance(page, PreprocessedPage):
        raise TypeError("page must be a PreprocessedPage")
    values = tuple(lines)
    if any(not isinstance(line, LineOCRResult) for line in values):
        raise AssemblyError("page lines must contain only LineOCRResult values")
    expected_order = list(range(len(values)))
    actual_order = [line.region.reading_order_index for line in values]
    if actual_order != expected_order:
        raise AssemblyError("page lines must retain contiguous reading-order indices")
    if any(line.region.page_index != page.page.page_index for line in values):
        raise AssemblyError("page line geometry does not belong to the source page")
    if page.is_blank and values:
        raise AssemblyError("a blank page cannot contain recognized line results")
    text = LINE_SEPARATOR.join(
        line.prediction.text for line in values if line.prediction is not None
    )
    return PageOCRResult(
        page_index=page.page.page_index,
        source_page_number=page.page.source_page_number,
        lines=values,
        assembled_text=text,
        is_blank=page.is_blank,
    )


def assemble_document(pages: Sequence[PageOCRResult]) -> DocumentOCRResult:
    """Preserve every ordered page and separate adjacent pages with two newlines."""

    values = tuple(pages)
    if any(not isinstance(page, PageOCRResult) for page in values):
        raise AssemblyError("document pages must contain only PageOCRResult values")
    indexes = [page.page_index for page in values]
    if indexes != list(range(len(values))):
        raise AssemblyError("document pages must retain contiguous zero-based ordering")
    return DocumentOCRResult(values, PAGE_SEPARATOR.join(page.assembled_text for page in values))


def document_to_text(result: DocumentOCRResult) -> str:
    """Return the exact text-only document projection."""

    if not isinstance(result, DocumentOCRResult):
        raise TypeError("result must be a DocumentOCRResult")
    return result.assembled_text


def document_to_dict(result: DocumentOCRResult) -> dict[str, object]:
    """Return a JSON-compatible projection with no pixels, tensors, or local paths."""

    if not isinstance(result, DocumentOCRResult):
        raise TypeError("result must be a DocumentOCRResult")
    return result.to_public_dict()


def document_to_json(result: DocumentOCRResult) -> str:
    """Return deterministic readable UTF-8 JSON with a final newline."""

    return (
        json.dumps(
            document_to_dict(result),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def _validate_output_target(path: str | os.PathLike[str], *, overwrite: bool) -> Path:
    if not isinstance(overwrite, bool):
        raise AssemblyError("overwrite must be a boolean")
    try:
        target = Path(path)
    except (TypeError, ValueError) as error:
        raise AssemblyError("result destination is invalid") from error
    if not target.name or not target.parent.is_dir() or target.parent.is_symlink():
        raise AssemblyError("result parent must be an existing regular directory")
    if target.is_symlink() or (target.exists() and not target.is_file()):
        raise AssemblyError("result destination must be a regular file path")
    return target


def _write_result(
    payload: bytes,
    path: str | os.PathLike[str],
    *,
    overwrite: bool,
    artifact_name: str,
) -> None:
    target = _validate_output_target(path, overwrite=overwrite)
    _atomic_write_bytes(
        payload,
        target,
        overwrite=overwrite,
        error_type=AssemblyError,
        artifact_name=artifact_name,
    )


def write_text_result(
    result: DocumentOCRResult,
    path: str | os.PathLike[str],
    *,
    overwrite: bool = False,
) -> None:
    """Atomically write the exact text projection as UTF-8."""

    _write_result(
        document_to_text(result).encode("utf-8"),
        path,
        overwrite=overwrite,
        artifact_name="text result",
    )


def write_json_result(
    result: DocumentOCRResult,
    path: str | os.PathLike[str],
    *,
    overwrite: bool = False,
) -> None:
    """Atomically write deterministic non-ASCII-escaped JSON as UTF-8."""

    _write_result(
        document_to_json(result).encode("utf-8"),
        path,
        overwrite=overwrite,
        artifact_name="JSON result",
    )


__all__ = [
    "LINE_SEPARATOR",
    "PAGE_SEPARATOR",
    "assemble_document",
    "assemble_page",
    "document_to_dict",
    "document_to_json",
    "document_to_text",
    "write_json_result",
    "write_text_result",
]
