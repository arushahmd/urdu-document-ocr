"""Typed, privacy-aware domain contracts for the OCR pipeline."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from enum import StrEnum
from hashlib import sha256
from pathlib import PurePosixPath, PureWindowsPath
from typing import Any

import numpy as np
from numpy.typing import NDArray

UInt8Array = NDArray[np.uint8]
BoolArray = NDArray[np.bool_]

_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


def _require_integer(name: str, value: object, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _require_nonempty(name: str, value: object) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if not value.strip():
        raise ValueError(f"{name} must not be empty")
    return value


def normalize_relative_dataset_path(value: str) -> str:
    """Return a portable POSIX relative path or reject a privacy-unsafe path."""

    _require_nonempty("image_path", value)
    if "\x00" in value:
        raise ValueError("image_path must not contain a NUL character")
    if value.startswith(("/", "\\")):
        raise ValueError("image_path must be relative")

    windows_path = PureWindowsPath(value)
    if windows_path.drive or windows_path.root:
        raise ValueError("image_path must not contain a drive, UNC root, or rooted path")

    portable_value = value.replace("\\", "/")
    if ":" in portable_value:
        raise ValueError("image_path must not contain a drive or URI scheme")

    path = PurePosixPath(portable_value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("image_path must not be absolute or traverse outside its dataset")
    normalized = path.as_posix()
    if normalized in {"", "."}:
        raise ValueError("image_path must identify a file")
    return normalized


@dataclass(frozen=True, slots=True)
class BoundingBox:
    """A nonempty half-open pixel rectangle: [x, right) by [y, bottom)."""

    x: int
    y: int
    width: int
    height: int

    def __post_init__(self) -> None:
        _require_integer("x", self.x, minimum=0)
        _require_integer("y", self.y, minimum=0)
        _require_integer("width", self.width, minimum=1)
        _require_integer("height", self.height, minimum=1)

    @property
    def x_min(self) -> int:
        return self.x

    @property
    def y_min(self) -> int:
        return self.y

    @property
    def x_max(self) -> int:
        return self.x + self.width

    @property
    def y_max(self) -> int:
        return self.y + self.height

    @property
    def right(self) -> int:
        return self.x_max

    @property
    def bottom(self) -> int:
        return self.y_max

    @property
    def area(self) -> int:
        return self.width * self.height

    def intersection(self, other: BoundingBox) -> BoundingBox | None:
        left = max(self.x_min, other.x_min)
        top = max(self.y_min, other.y_min)
        right = min(self.x_max, other.x_max)
        bottom = min(self.y_max, other.y_max)
        if right <= left or bottom <= top:
            return None
        return BoundingBox(left, top, right - left, bottom - top)

    def intersection_over_union(self, other: BoundingBox) -> float:
        overlap = self.intersection(other)
        if overlap is None:
            return 0.0
        return overlap.area / (self.area + other.area - overlap.area)

    def contains(self, x: int, y: int) -> bool:
        return self.x_min <= x < self.x_max and self.y_min <= y < self.y_max

    def clipped(self, image_width: int, image_height: int) -> BoundingBox | None:
        _require_integer("image_width", image_width, minimum=1)
        _require_integer("image_height", image_height, minimum=1)
        left = min(max(self.x_min, 0), image_width)
        top = min(max(self.y_min, 0), image_height)
        right = min(self.x_max, image_width)
        bottom = min(self.y_max, image_height)
        if right <= left or bottom <= top:
            return None
        return BoundingBox(left, top, right - left, bottom - top)

    def to_public_dict(self) -> dict[str, int]:
        return {"x": self.x, "y": self.y, "width": self.width, "height": self.height}


class SourceType(StrEnum):
    IMAGE = "image"
    PDF = "pdf"


@dataclass(frozen=True, slots=True)
class SourceMetadata:
    source_type: SourceType
    display_name: str | None = None
    input_sha256: str | None = None
    byte_size: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.source_type, SourceType):
            raise TypeError("source_type must be a SourceType")
        if self.display_name is not None:
            name = _require_nonempty("display_name", self.display_name).strip()
            if name != PureWindowsPath(name).name or name != PurePosixPath(name).name:
                raise ValueError("display_name must be a basename, not a path")
            object.__setattr__(self, "display_name", name)
        if self.input_sha256 is not None:
            digest = self.input_sha256.lower()
            if _SHA256_PATTERN.fullmatch(digest) is None:
                raise ValueError("input_sha256 must contain 64 hexadecimal characters")
            object.__setattr__(self, "input_sha256", digest)
        if self.byte_size is not None:
            _require_integer("byte_size", self.byte_size, minimum=0)

    def to_public_dict(self) -> dict[str, str | int | None]:
        return {
            "source_type": self.source_type.value,
            "display_name": self.display_name,
            "input_sha256": self.input_sha256,
            "byte_size": self.byte_size,
        }


def _validate_array(
    name: str,
    value: object,
    *,
    dtype: np.dtype[Any],
    shape_suffix: tuple[int, ...] | None = None,
) -> np.ndarray[Any, Any]:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{name} must be a NumPy array")
    if value.dtype != dtype:
        raise TypeError(f"{name} must have dtype {dtype}")
    if shape_suffix is None:
        if value.ndim != 2:
            raise ValueError(f"{name} must be a two-dimensional array")
    elif value.ndim != 2 + len(shape_suffix) or value.shape[2:] != shape_suffix:
        raise ValueError(
            f"{name} must have shape [height, width, {', '.join(map(str, shape_suffix))}]"
        )
    if value.shape[0] < 1 or value.shape[1] < 1:
        raise ValueError(f"{name} dimensions must be positive")
    return value


@dataclass(frozen=True, slots=True)
class PageImage:
    page_index: int
    source_page_number: int
    image: UInt8Array = field(repr=False, compare=False)
    source: SourceMetadata
    raster_dpi: int | None = None

    def __post_init__(self) -> None:
        _require_integer("page_index", self.page_index, minimum=0)
        _require_integer("source_page_number", self.source_page_number, minimum=1)
        if self.source_page_number != self.page_index + 1:
            raise ValueError("source_page_number must equal page_index + 1")
        _validate_array("image", self.image, dtype=np.dtype(np.uint8), shape_suffix=(3,))
        if not isinstance(self.source, SourceMetadata):
            raise TypeError("source must be SourceMetadata")
        if self.raster_dpi is not None:
            _require_integer("raster_dpi", self.raster_dpi, minimum=1)

    @property
    def height(self) -> int:
        return int(self.image.shape[0])

    @property
    def width(self) -> int:
        return int(self.image.shape[1])

    def to_public_dict(self) -> dict[str, object]:
        return {
            "page_index": self.page_index,
            "source_page_number": self.source_page_number,
            "image": {"height": self.height, "width": self.width, "channels": 3, "dtype": "uint8"},
            "source": self.source.to_public_dict(),
            "raster_dpi": self.raster_dpi,
        }


@dataclass(frozen=True, slots=True)
class PreprocessedPage:
    page: PageImage
    grayscale: UInt8Array = field(repr=False, compare=False)
    foreground_mask: BoolArray = field(repr=False, compare=False)
    is_blank: bool
    deskew_angle_degrees: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.page, PageImage):
            raise TypeError("page must be PageImage")
        _validate_array("grayscale", self.grayscale, dtype=np.dtype(np.uint8))
        _validate_array("foreground_mask", self.foreground_mask, dtype=np.dtype(np.bool_))
        expected_shape = self.page.image.shape[:2]
        if self.grayscale.shape != expected_shape or self.foreground_mask.shape != expected_shape:
            raise ValueError("preprocessed arrays must match the source page dimensions")
        if not isinstance(self.is_blank, bool):
            raise TypeError("is_blank must be a boolean")
        if isinstance(self.deskew_angle_degrees, bool) or not isinstance(
            self.deskew_angle_degrees, (int, float)
        ):
            raise TypeError("deskew_angle_degrees must be numeric")
        if not math.isfinite(float(self.deskew_angle_degrees)):
            raise ValueError("deskew_angle_degrees must be finite")

    def to_public_dict(self) -> dict[str, object]:
        return {
            "page": self.page.to_public_dict(),
            "grayscale": {"height": self.page.height, "width": self.page.width, "dtype": "uint8"},
            "foreground_mask": {
                "height": self.page.height,
                "width": self.page.width,
                "dtype": "bool",
            },
            "is_blank": self.is_blank,
            "deskew_angle_degrees": float(self.deskew_angle_degrees),
        }


class RegionKind(StrEnum):
    LINE = "line"
    SPANNING = "spanning"


@dataclass(frozen=True, slots=True)
class LineRegion:
    region_id: str
    page_index: int
    bounding_box: BoundingBox
    column_index: int | None
    reading_order_index: int
    region_kind: RegionKind = RegionKind.LINE

    def __post_init__(self) -> None:
        _require_nonempty("region_id", self.region_id)
        _require_integer("page_index", self.page_index, minimum=0)
        if not isinstance(self.bounding_box, BoundingBox):
            raise TypeError("bounding_box must be BoundingBox")
        if self.column_index is not None:
            _require_integer("column_index", self.column_index, minimum=0)
            if self.column_index not in {0, 1}:
                raise ValueError("column_index must be 0, 1, or None")
        _require_integer("reading_order_index", self.reading_order_index, minimum=0)
        if not isinstance(self.region_kind, RegionKind):
            raise TypeError("region_kind must be RegionKind")
        if self.region_kind is RegionKind.SPANNING and self.column_index is not None:
            raise ValueError("spanning regions cannot be assigned to a column")

    def to_public_dict(self) -> dict[str, object]:
        return {
            "region_id": self.region_id,
            "page_index": self.page_index,
            "bounding_box": self.bounding_box.to_public_dict(),
            "column_index": self.column_index,
            "reading_order_index": self.reading_order_index,
            "region_kind": self.region_kind.value,
        }


@dataclass(frozen=True, slots=True)
class OCRPrediction:
    text: str
    character_indices: tuple[int, ...] = ()
    sequence_length: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.text, str):
            raise TypeError("text must be a string")
        indices = tuple(self.character_indices)
        for index in indices:
            _require_integer("character index", index, minimum=1)
        object.__setattr__(self, "character_indices", indices)
        if self.sequence_length is not None:
            _require_integer("sequence_length", self.sequence_length, minimum=0)
            if self.sequence_length < len(indices):
                raise ValueError("sequence_length cannot be shorter than emitted indices")

    def to_public_dict(self) -> dict[str, object]:
        return {
            "text": self.text,
            "character_indices": list(self.character_indices),
            "sequence_length": self.sequence_length,
        }


@dataclass(frozen=True, slots=True)
class LineOCRResult:
    region: LineRegion
    prediction: OCRPrediction | None = None
    error_code: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.region, LineRegion):
            raise TypeError("region must be LineRegion")
        if (self.prediction is None) == (self.error_code is None):
            raise ValueError("exactly one of prediction or error_code must be supplied")
        if self.prediction is not None and not isinstance(self.prediction, OCRPrediction):
            raise TypeError("prediction must be OCRPrediction")
        if self.error_code is not None:
            _require_nonempty("error_code", self.error_code)

    def to_public_dict(self) -> dict[str, object]:
        return {
            "region": self.region.to_public_dict(),
            "prediction": None if self.prediction is None else self.prediction.to_public_dict(),
            "error_code": self.error_code,
        }


@dataclass(frozen=True, slots=True)
class PageOCRResult:
    page_index: int
    source_page_number: int
    lines: tuple[LineOCRResult, ...]
    assembled_text: str
    is_blank: bool = False

    def __post_init__(self) -> None:
        _require_integer("page_index", self.page_index, minimum=0)
        _require_integer("source_page_number", self.source_page_number, minimum=1)
        if self.source_page_number != self.page_index + 1:
            raise ValueError("source_page_number must equal page_index + 1")
        if not isinstance(self.assembled_text, str):
            raise TypeError("assembled_text must be a string")
        if not isinstance(self.is_blank, bool):
            raise TypeError("is_blank must be a boolean")
        lines = tuple(self.lines)
        if any(line.region.page_index != self.page_index for line in lines):
            raise ValueError("every line must belong to this page")
        order = [line.region.reading_order_index for line in lines]
        if order != sorted(order) or len(order) != len(set(order)):
            raise ValueError("lines must have unique ascending reading-order indices")
        object.__setattr__(self, "lines", lines)

    def to_public_dict(self) -> dict[str, object]:
        return {
            "page_index": self.page_index,
            "source_page_number": self.source_page_number,
            "lines": [line.to_public_dict() for line in self.lines],
            "assembled_text": self.assembled_text,
            "is_blank": self.is_blank,
        }


@dataclass(frozen=True, slots=True)
class DocumentOCRResult:
    pages: tuple[PageOCRResult, ...]
    assembled_text: str
    schema_version: int = 1

    def __post_init__(self) -> None:
        _require_integer("schema_version", self.schema_version, minimum=1)
        if not isinstance(self.assembled_text, str):
            raise TypeError("assembled_text must be a string")
        pages = tuple(self.pages)
        indexes = [page.page_index for page in pages]
        if indexes != list(range(len(pages))):
            raise ValueError("pages must be ordered with contiguous zero-based indices")
        object.__setattr__(self, "pages", pages)

    def to_public_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "pages": [page.to_public_dict() for page in self.pages],
            "assembled_text": self.assembled_text,
        }


@dataclass(frozen=True, slots=True)
class DatasetSample:
    schema_version: int
    sample_id: str
    image_path: str
    text: str
    document_id: str
    page_id: str | None = None
    line_index: int | None = None
    tags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("schema_version must be 1")
        object.__setattr__(self, "sample_id", _require_nonempty("sample_id", self.sample_id))
        object.__setattr__(self, "image_path", normalize_relative_dataset_path(self.image_path))
        if not isinstance(self.text, str):
            raise TypeError("text must be a string")
        if not self.text.strip():
            raise ValueError("text must not be empty")
        object.__setattr__(self, "document_id", _require_nonempty("document_id", self.document_id))
        if self.page_id is not None:
            object.__setattr__(self, "page_id", _require_nonempty("page_id", self.page_id))
        if self.line_index is not None:
            _require_integer("line_index", self.line_index, minimum=0)
        tags = tuple(self.tags)
        if any(not isinstance(tag, str) or not tag.strip() for tag in tags):
            raise ValueError("tags must contain nonempty strings")
        if len(tags) != len(set(tags)):
            raise ValueError("tags must not contain duplicates")
        object.__setattr__(self, "tags", tuple(sorted(tags)))

    def to_public_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "sample_id": self.sample_id,
            "image_path": self.image_path,
            "text": self.text,
            "document_id": self.document_id,
            "page_id": self.page_id,
            "line_index": self.line_index,
            "tags": list(self.tags),
        }


@dataclass(frozen=True, slots=True)
class Vocabulary:
    characters: tuple[str, ...]
    schema_version: int = 1
    normalization_policy_version: str = "nfc-v1"
    blank_index: int = 0

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("schema_version must be 1")
        _require_nonempty("normalization_policy_version", self.normalization_policy_version)
        if self.blank_index != 0:
            raise ValueError("CTC blank_index must be 0")
        characters = tuple(self.characters)
        if any(not isinstance(character, str) or len(character) != 1 for character in characters):
            raise ValueError("each vocabulary entry must be one Unicode code point")
        if len(characters) != len(set(characters)):
            raise ValueError("vocabulary characters must be unique")
        object.__setattr__(self, "characters", tuple(sorted(characters, key=ord)))

    @property
    def fingerprint(self) -> str:
        payload = {
            "blank_index": self.blank_index,
            "characters": list(self.characters),
            "normalization_policy_version": self.normalization_policy_version,
            "schema_version": self.schema_version,
        }
        encoded = json.dumps(
            payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True
        ).encode("utf-8")
        return sha256(encoded).hexdigest()

    def encode(self, text: str) -> tuple[int, ...]:
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        mapping = {character: index for index, character in enumerate(self.characters, start=1)}
        try:
            return tuple(mapping[character] for character in text)
        except KeyError as error:
            raise ValueError(
                f"character U+{ord(error.args[0]):04X} is not in the vocabulary"
            ) from None

    def decode(self, indices: tuple[int, ...] | list[int]) -> str:
        output: list[str] = []
        for index in indices:
            _require_integer("character index", index, minimum=1)
            if index > len(self.characters):
                raise ValueError(f"character index {index} is outside the vocabulary")
            output.append(self.characters[index - 1])
        return "".join(output)

    def to_public_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "normalization_policy_version": self.normalization_policy_version,
            "blank_index": self.blank_index,
            "characters": list(self.characters),
            "fingerprint": self.fingerprint,
        }


@dataclass(frozen=True, slots=True)
class EditCounts:
    substitutions: int
    deletions: int
    insertions: int
    reference_length: int

    def __post_init__(self) -> None:
        for name in ("substitutions", "deletions", "insertions", "reference_length"):
            _require_integer(name, getattr(self, name), minimum=0)

    @property
    def total_errors(self) -> int:
        return self.substitutions + self.deletions + self.insertions

    def to_public_dict(self) -> dict[str, int]:
        return {
            "substitutions": self.substitutions,
            "deletions": self.deletions,
            "insertions": self.insertions,
            "reference_length": self.reference_length,
        }


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    sample_count: int
    exact_matches: int
    character_edits: EditCounts
    word_edits: EditCounts
    data_fingerprint: str
    config_fingerprint: str
    model_fingerprint: str | None = None
    schema_version: int = 1

    def __post_init__(self) -> None:
        _require_integer("sample_count", self.sample_count, minimum=0)
        _require_integer("exact_matches", self.exact_matches, minimum=0)
        if self.exact_matches > self.sample_count:
            raise ValueError("exact_matches cannot exceed sample_count")
        if not isinstance(self.character_edits, EditCounts) or not isinstance(
            self.word_edits, EditCounts
        ):
            raise TypeError("character_edits and word_edits must be EditCounts")
        for name in ("data_fingerprint", "config_fingerprint"):
            _require_nonempty(name, getattr(self, name))
        if self.model_fingerprint is not None:
            _require_nonempty("model_fingerprint", self.model_fingerprint)
        if self.schema_version != 1:
            raise ValueError("schema_version must be 1")

    def to_public_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "sample_count": self.sample_count,
            "exact_matches": self.exact_matches,
            "character_edits": self.character_edits.to_public_dict(),
            "word_edits": self.word_edits.to_public_dict(),
            "data_fingerprint": self.data_fingerprint,
            "config_fingerprint": self.config_fingerprint,
            "model_fingerprint": self.model_fingerprint,
        }
