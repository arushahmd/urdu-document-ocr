"""Provenance-cleared deterministic Urdu line and page fixture generation."""

from __future__ import annotations

import io
import json
import math
import os
import unicodedata
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from functools import lru_cache
from hashlib import sha256
from importlib.metadata import version as distribution_version
from pathlib import Path
from textwrap import dedent
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, features
from PIL import __version__ as pillow_version

try:
    import freetype
    import uharfbuzz as hb
except ImportError:  # pragma: no cover - exercised by a subprocess capability test
    freetype = None  # type: ignore[assignment]
    hb = None  # type: ignore[assignment]

from urdu_document_ocr.config import DatasetSplitConfig, SyntheticDataConfig
from urdu_document_ocr.data.manifest import (
    _atomic_write_bytes,
    _canonical_json_bytes,
    canonical_manifest_bytes,
    dataset_fingerprint,
)
from urdu_document_ocr.data.splitting import split_dataset, write_split_manifests
from urdu_document_ocr.data.validation import (
    NORMALIZATION_POLICY_VERSION,
    _transcription_issue_codes,
    validate_dataset,
)
from urdu_document_ocr.data.vocabulary import (
    build_vocabulary,
    find_unseen_characters,
    save_vocabulary,
)
from urdu_document_ocr.errors import SyntheticDataError
from urdu_document_ocr.types import (
    BoundingBox,
    DatasetSample,
    RegionKind,
    normalize_relative_dataset_path,
)

GENERATOR_VERSION = "synthetic-v1"
BUNDLED_FONT_FAMILY = "Noto Nastaliq Urdu"
BUNDLED_FONT_RELEASE = "NotoNastaliqUrdu-v4.000"
BUNDLED_FONT_FILENAME = "NotoNastaliqUrdu[wght].ttf"
BUNDLED_FONT_SHA256 = "eff3a48f588f599f98e98350f1107e2e492edefbe864c8b61c73f2d605f1dce4"
REFERENCE_CONFIG = SyntheticDataConfig()

_FONT_DIRECTORY = Path(__file__).resolve().parents[1] / "assets" / "fonts" / "noto-nastaliq-urdu"
_FONT_PROVENANCE_FIELDS = frozenset(
    {
        "artifact_sha256",
        "artifact_size_bytes",
        "copyright",
        "family",
        "filename",
        "license_file",
        "license_identifier",
        "modification_status",
        "provenance_schema_version",
        "release_archive_sha256",
        "release_archive_size_bytes",
        "release_asset_url",
        "release_commit",
        "release_date",
        "release_inner_path",
        "release_tag",
        "release_url",
        "reserved_font_names",
        "redistribution_basis",
        "scope_note",
    }
)
_NUMERIC_BIDI_CLASSES = frozenset({"AN", "EN"})
_SHAPING_FEATURES = {"calt": True, "kern": True, "liga": True}
_FONT_WEIGHT = 400.0


@dataclass(frozen=True, slots=True)
class TextSource:
    source_id: str
    text: str
    origin: str = "project-authored"
    note: str = "Neutral Urdu fixture text authored for this repository."

    def __post_init__(self) -> None:
        if not isinstance(self.source_id, str) or not self.source_id.startswith("fixture-text-"):
            raise ValueError("source_id must use the fixture-text namespace")
        if not isinstance(self.text, str) or _transcription_issue_codes(self.text):
            raise ValueError("fixture text must satisfy the nfc-v1 transcription policy")
        if self.origin not in {"project-authored", "programmatically-composed"}:
            raise ValueError("unsupported fixture text origin")
        if not isinstance(self.note, str) or not self.note.strip():
            raise ValueError("fixture text note must be nonempty")

    @property
    def fingerprint(self) -> str:
        return sha256(self.text.encode("utf-8")).hexdigest()

    def to_public_dict(self) -> dict[str, object]:
        return {
            "fixture_id": self.source_id,
            "text": self.text,
            "origin": self.origin,
            "sha256": self.fingerprint,
            "normalization_policy": NORMALIZATION_POLICY_VERSION,
            "redistribution_status": "approved-public-fixture-content",
            "notes": self.note,
        }


PUBLIC_TEXT_SOURCES = (
    TextSource(
        "fixture-text-001",
        "جامع حروف: ک ك ی ي، اعداد ۰۱۲۳۴۵۶۷۸۹ اور 0123456789؛ سوال؟ اُردو نیم‌خودکار۔",
        note=(
            "Coverage line for distinct letters, both digit families, punctuation, "
            "a diacritic, and ZWNJ."
        ),
    ),
    TextSource("fixture-text-002", "اردو متن صاف اور روشن ہے۔"),
    TextSource("fixture-text-003", "یہ ایک سادہ آزمائشی سطر ہے۔"),
    TextSource("fixture-text-004", "لفظوں کے درمیان جگہ موجود ہے۔"),
    TextSource("fixture-text-005", "دائیں سے بائیں ترتیب درست ہے۔"),
    TextSource("fixture-text-006", "صفحہ ۱۲۳ پر نئی سطر ہے۔"),
    TextSource("fixture-text-007", "عدد ۴۵۶، نشان؟ اور وقفہ۔"),
    TextSource("fixture-text-008", "نیم‌خودکار عمل واضح ہے۔"),
    TextSource("fixture-text-009", "صاف متن۔"),
    TextSource(
        "fixture-text-010",
        "یہ لمبی آزمائشی سطر مختلف لفظوں اور نشانات کو ایک ساتھ دکھاتی ہے۔",
    ),
    TextSource("fixture-text-011", "یہ مصنوعی اور محفوظ نمونہ ہے۔"),
    TextSource("fixture-text-012", "چھوٹی سطر دائیں کنارے پر ہے۔"),
    TextSource("fixture-text-013", "سوال؟ جواب: متن تیار ہے۔"),
)


class PageLayoutFamily(StrEnum):
    ONE_COLUMN = "one_column"
    BALANCED_TWO_COLUMN = "balanced_two_column"
    UNEVEN_TWO_COLUMN = "uneven_two_column"
    SPANNING_TOP = "spanning_top"
    SPANNING_MIDDLE = "spanning_middle"
    SPANNING_FOOTER = "spanning_footer"
    ONE_COLUMN_WIDE = "one_column_wide"
    BLANK = "blank"
    NEAR_BLANK = "near_blank"
    MILD_SKEW = "mild_skew"


_PAGE_LAYOUT_ORDER = tuple(PageLayoutFamily)


@dataclass(frozen=True, slots=True)
class ShapingCapabilities:
    engine: str
    harfbuzz_version: str
    freetype_version: str
    pillow_version: str
    pillow_freetype_version: str | None
    pillow_raqm_available: bool
    font_identifier: str
    font_sha256: str
    rtl_strategy: str

    def to_public_dict(self) -> dict[str, object]:
        return {
            "engine": self.engine,
            "harfbuzz_version": self.harfbuzz_version,
            "freetype_version": self.freetype_version,
            "pillow_version": self.pillow_version,
            "pillow_freetype_version": self.pillow_freetype_version,
            "pillow_raqm_available": self.pillow_raqm_available,
            "font_identifier": self.font_identifier,
            "font_sha256": self.font_sha256,
            "rtl_strategy": self.rtl_strategy,
        }


@dataclass(frozen=True, slots=True)
class SyntheticLineRecord:
    sample_id: str
    text_source_id: str
    text_sha256: str
    font_identifier: str
    font_sha256: str
    master_seed: int
    derived_seed: int
    width: int
    height: int
    font_size: int
    padding_left: int
    padding_right: int
    padding_top: int
    padding_bottom: int
    background_intensity: int
    text_intensity: int
    blur_sigma: float
    noise_std: float
    skew_degrees: float
    image_path: str
    image_sha256: str
    generator_version: str = GENERATOR_VERSION

    def to_public_dict(self) -> dict[str, object]:
        return {
            "generator_version": self.generator_version,
            "sample_id": self.sample_id,
            "text_source_id": self.text_source_id,
            "text_sha256": self.text_sha256,
            "font": {
                "identifier": self.font_identifier,
                "sha256": self.font_sha256,
                "size": self.font_size,
            },
            "seed": {"master": self.master_seed, "derived": self.derived_seed},
            "image": {
                "path": self.image_path,
                "width": self.width,
                "height": self.height,
                "sha256": self.image_sha256,
            },
            "render": {
                "direction": "rtl",
                "language": "ur",
                "padding": {
                    "left": self.padding_left,
                    "right": self.padding_right,
                    "top": self.padding_top,
                    "bottom": self.padding_bottom,
                },
                "background_intensity": self.background_intensity,
                "text_intensity": self.text_intensity,
            },
            "degradations": {
                "blur_sigma": self.blur_sigma,
                "noise_std": self.noise_std,
                "skew_degrees": self.skew_degrees,
                "operation_order": [
                    "render",
                    "contrast_background",
                    "gaussian_blur",
                    "gaussian_noise",
                    "rotation",
                ],
            },
        }


@dataclass(frozen=True, slots=True)
class GeneratedLineSample:
    sample: DatasetSample
    record: SyntheticLineRecord
    image: np.ndarray[Any, np.dtype[np.uint8]] = field(repr=False, compare=False)
    png_bytes: bytes = field(repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.sample, DatasetSample):
            raise TypeError("sample must be DatasetSample")
        if not isinstance(self.record, SyntheticLineRecord):
            raise TypeError("record must be SyntheticLineRecord")
        if not isinstance(self.image, np.ndarray) or self.image.dtype != np.uint8:
            raise TypeError("image must be a uint8 NumPy array")
        if self.image.ndim != 2 or self.image.shape != (self.record.height, self.record.width):
            raise ValueError("line image dimensions must match its record")
        if sha256(self.png_bytes).hexdigest() != self.record.image_sha256:
            raise ValueError("PNG bytes must match the recorded image fingerprint")


@dataclass(frozen=True, slots=True)
class SyntheticPageLine:
    line_id: str
    text: str
    text_sha256: str
    composition_box: BoundingBox
    output_box: BoundingBox
    region_kind: RegionKind
    column_index: int | None
    reading_order_index: int

    def to_public_dict(self) -> dict[str, object]:
        return {
            "line_id": self.line_id,
            "text": self.text,
            "text_sha256": self.text_sha256,
            "composition_box": self.composition_box.to_public_dict(),
            "output_box": self.output_box.to_public_dict(),
            "region_kind": self.region_kind.value,
            "column_index": self.column_index,
            "reading_order_index": self.reading_order_index,
        }


@dataclass(frozen=True, slots=True)
class SyntheticPageRecord:
    page_id: str
    layout_family: PageLayoutFamily
    image_path: str
    image_sha256: str
    width: int
    height: int
    master_seed: int
    derived_seed: int
    background_intensity: int
    skew_degrees: float
    expected_blank: bool
    lines: tuple[SyntheticPageLine, ...]
    generator_version: str = GENERATOR_VERSION

    @property
    def expected_line_count(self) -> int:
        return len(self.lines)

    def to_public_dict(self) -> dict[str, object]:
        return {
            "generator_version": self.generator_version,
            "page_id": self.page_id,
            "layout_family": self.layout_family.value,
            "image_path": self.image_path,
            "image_sha256": self.image_sha256,
            "dimensions": {"width": self.width, "height": self.height},
            "seed": {"master": self.master_seed, "derived": self.derived_seed},
            "background_intensity": self.background_intensity,
            "skew_degrees": self.skew_degrees,
            "expected_blank": self.expected_blank,
            "expected_line_count": self.expected_line_count,
            "lines": [line.to_public_dict() for line in self.lines],
        }


@dataclass(frozen=True, slots=True)
class GeneratedPageFixture:
    record: SyntheticPageRecord
    image: np.ndarray[Any, np.dtype[np.uint8]] = field(repr=False, compare=False)
    png_bytes: bytes = field(repr=False)

    def __post_init__(self) -> None:
        if self.image.dtype != np.uint8 or self.image.shape != (
            self.record.height,
            self.record.width,
        ):
            raise ValueError("page image must match its record")
        if sha256(self.png_bytes).hexdigest() != self.record.image_sha256:
            raise ValueError("page PNG bytes must match its record")


@dataclass(frozen=True, slots=True)
class FixtureGenerationResult:
    output_role: str
    line_count: int
    page_count: int
    dataset_fingerprint: str
    split_fingerprint: str
    vocabulary_fingerprint: str
    text_source_fingerprint: str
    config_fingerprint: str
    artifact_manifest_sha256: str
    total_size_bytes: int

    def to_public_dict(self) -> dict[str, object]:
        return {
            "output_role": self.output_role,
            "line_count": self.line_count,
            "page_count": self.page_count,
            "dataset_fingerprint": self.dataset_fingerprint,
            "split_fingerprint": self.split_fingerprint,
            "vocabulary_fingerprint": self.vocabulary_fingerprint,
            "text_source_fingerprint": self.text_source_fingerprint,
            "config_fingerprint": self.config_fingerprint,
            "artifact_manifest_sha256": self.artifact_manifest_sha256,
            "total_size_bytes": self.total_size_bytes,
        }


def bundled_font_path() -> Path:
    """Return the reviewed local font asset; this function never downloads anything."""

    return _FONT_DIRECTORY / BUNDLED_FONT_FILENAME


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_font_provenance() -> dict[str, object]:
    """Read and validate the isolated asset-level provenance record."""

    try:
        payload = json.loads((_FONT_DIRECTORY / "provenance.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise SyntheticDataError("bundled font provenance could not be validated") from error
    if not isinstance(payload, dict) or set(payload) != _FONT_PROVENANCE_FIELDS:
        raise SyntheticDataError("bundled font provenance fields are invalid")
    if payload.get("license_identifier") != "OFL-1.1":
        raise SyntheticDataError("bundled font license identifier is invalid")
    if payload.get("modification_status") != "unmodified":
        raise SyntheticDataError("bundled font must remain unmodified")
    if not (_FONT_DIRECTORY / "FONT_LICENSE.txt").is_file():
        raise SyntheticDataError("bundled font license notice is missing")
    font = bundled_font_path()
    if not font.is_file() or _file_sha256(font) != payload.get("artifact_sha256"):
        raise SyntheticDataError("bundled font fingerprint does not match provenance")
    return payload


def _resolved_font(font_path: str | os.PathLike[str] | None) -> tuple[Path, str, str]:
    try:
        path = bundled_font_path() if font_path is None else Path(font_path)
        resolved = path.resolve(strict=True)
    except (OSError, TypeError, ValueError) as error:
        raise SyntheticDataError("font file is unavailable") from error
    if not resolved.is_file() or resolved.stat().st_size > 10 * 1024 * 1024:
        raise SyntheticDataError("font must be a regular file no larger than 10 MiB")
    digest = _file_sha256(resolved)
    if font_path is None:
        load_font_provenance()
        if digest != BUNDLED_FONT_SHA256:
            raise SyntheticDataError("bundled font fingerprint is invalid")
        identifier = f"{BUNDLED_FONT_FAMILY}:{BUNDLED_FONT_RELEASE}:wght-400"
    else:
        identifier = f"caller-font:{digest[:16]}"
    return resolved, identifier, digest


def _directional_runs(text: str) -> tuple[tuple[str, str], ...]:
    logical: list[tuple[str, list[str]]] = []
    for character in text:
        direction = (
            "ltr" if unicodedata.bidirectional(character) in _NUMERIC_BIDI_CLASSES else "rtl"
        )
        if logical and logical[-1][0] == direction:
            logical[-1][1].append(character)
        else:
            logical.append((direction, [character]))
    return tuple((direction, "".join(characters)) for direction, characters in reversed(logical))


def _new_harfbuzz_font(font_bytes: bytes, font_size: int) -> Any:
    assert hb is not None
    face = hb.Face(font_bytes)
    font = hb.Font(face)
    font.scale = (font_size * 64, font_size * 64)
    hb.ot_font_set_funcs(font)
    font.set_variations({"wght": _FONT_WEIGHT})
    return font


def _shape_run(text: str, direction: str, font: Any) -> tuple[tuple[Any, Any], ...]:
    assert hb is not None
    buffer = hb.Buffer()
    buffer.add_str(text)
    buffer.direction = direction
    buffer.script = "arab"
    buffer.language = "ur"
    hb.shape(font, buffer, _SHAPING_FEATURES)
    shaped = tuple(zip(buffer.glyph_infos, buffer.glyph_positions, strict=True))
    if any(info.codepoint == 0 for info, _ in shaped):
        raise SyntheticDataError("font lacks a glyph required by the transcription")
    return shaped


def _shaped_glyph_ids(text: str, font_bytes: bytes, font_size: int) -> tuple[int, ...]:
    font = _new_harfbuzz_font(font_bytes, font_size)
    return tuple(
        info.codepoint
        for direction, run in _directional_runs(text)
        for info, _ in _shape_run(run, direction, font)
    )


def assert_shaping_available(
    font_path: str | os.PathLike[str] | None = None,
) -> ShapingCapabilities:
    """Fail closed unless direct HarfBuzz shaping and FreeType rasterization work."""

    if hb is None or freetype is None:
        raise SyntheticDataError(
            "synthetic Urdu generation requires the uharfbuzz and freetype-py runtimes"
        )
    path, identifier, digest = _resolved_font(font_path)
    return _check_shaping_capabilities(str(path), identifier, digest)


@lru_cache(maxsize=8)
def _check_shaping_capabilities(
    path_string: str, identifier: str, digest: str
) -> ShapingCapabilities:
    """Cache a capability check by verified font content identity."""

    if hb is None or freetype is None:  # pragma: no cover - public guard normally catches this
        raise SyntheticDataError("complex-script shaping dependencies are unavailable")
    path = Path(path_string)
    try:
        font_bytes = path.read_bytes()
        joined = _shaped_glyph_ids("سلام", font_bytes, 48)
        isolated = tuple(
            glyph for character in "سلام" for glyph in _shaped_glyph_ids(character, font_bytes, 48)
        )
        alpha = _render_text_alpha("سلام", path, 48)
    except SyntheticDataError:
        raise
    except Exception as error:
        raise SyntheticDataError("complex-script shaping capability check failed") from error
    if joined == isolated or not joined or not alpha.any():
        raise SyntheticDataError("font did not produce contextual joined Urdu shaping")
    return ShapingCapabilities(
        engine="uharfbuzz-freetype-direct-v1",
        harfbuzz_version=hb.version_string(),
        freetype_version=".".join(str(item) for item in freetype.version()),
        pillow_version=pillow_version,
        pillow_freetype_version=features.version_module("freetype2"),
        pillow_raqm_available=features.check_feature("raqm"),
        font_identifier=identifier,
        font_sha256=digest,
        rtl_strategy="explicit-rtl-with-numeric-ltr-runs",
    )


def _render_text_alpha(text: str, font_path: Path, font_size: int) -> np.ndarray[Any, Any]:
    if hb is None or freetype is None:
        raise SyntheticDataError("complex-script shaping dependencies are unavailable")
    try:
        font_bytes = font_path.read_bytes()
        harfbuzz_font = _new_harfbuzz_font(font_bytes, font_size)
        face = freetype.Face(str(font_path))
        face.set_char_size(font_size * 64)
        if hasattr(face, "set_var_design_coords"):
            face.set_var_design_coords((_FONT_WEIGHT,))
        bitmaps: list[tuple[int, int, np.ndarray[Any, Any]]] = []
        pen_x = 0
        pen_y = 0
        for direction, run in _directional_runs(text):
            for info, position in _shape_run(run, direction, harfbuzz_font):
                face.load_glyph(
                    info.codepoint,
                    freetype.FT_LOAD_RENDER | freetype.FT_LOAD_TARGET_NORMAL,
                )
                slot = face.glyph
                bitmap = slot.bitmap
                x = round((pen_x + position.x_offset) / 64) + slot.bitmap_left
                y = -round((pen_y + position.y_offset) / 64) - slot.bitmap_top
                if bitmap.width and bitmap.rows:
                    width = int(bitmap.width)
                    pitch = abs(int(bitmap.pitch))
                    pixels = np.asarray(bitmap.buffer, dtype=np.uint8).reshape(
                        int(bitmap.rows), pitch
                    )[:, :width]
                    bitmaps.append((x, y, pixels.copy()))
                pen_x += int(position.x_advance)
                pen_y += int(position.y_advance)
    except SyntheticDataError:
        raise
    except Exception as error:
        raise SyntheticDataError("Urdu text shaping or glyph rasterization failed") from error
    if not bitmaps:
        raise SyntheticDataError("shaped text produced no visible glyphs")
    minimum_x = min(x for x, _, _ in bitmaps)
    minimum_y = min(y for _, y, _ in bitmaps)
    maximum_x = max(x + bitmap.shape[1] for x, _, bitmap in bitmaps)
    maximum_y = max(y + bitmap.shape[0] for _, y, bitmap in bitmaps)
    alpha = np.zeros((maximum_y - minimum_y, maximum_x - minimum_x), dtype=np.uint8)
    for x, y, bitmap in bitmaps:
        left = x - minimum_x
        top = y - minimum_y
        view = alpha[top : top + bitmap.shape[0], left : left + bitmap.shape[1]]
        np.maximum(view, bitmap, out=view)
    return alpha


def _png_bytes(image: np.ndarray[Any, Any]) -> bytes:
    output = io.BytesIO()
    Image.fromarray(image, mode="L").save(
        output,
        format="PNG",
        compress_level=9,
        optimize=False,
    )
    return output.getvalue()


def _derived_seed(master_seed: int, stable_id: str) -> int:
    payload = f"{master_seed}\0{stable_id}".encode()
    return int.from_bytes(sha256(payload).digest()[:8], "big", signed=False)


def _blend_alpha(
    alpha: np.ndarray[Any, Any], background: int, foreground: int
) -> np.ndarray[Any, Any]:
    values = background - alpha.astype(np.float64) * (background - foreground) / 255.0
    return np.rint(values).clip(0, 255).astype(np.uint8)


def _sample_optional_range(
    rng: np.random.Generator,
    probability: float,
    minimum: float,
    maximum: float,
) -> float:
    if float(rng.random()) >= probability:
        return 0.0
    if math.isclose(minimum, maximum):
        return float(minimum)
    return float(rng.uniform(minimum, maximum))


def _rotate_grayscale(image: np.ndarray[Any, Any], angle: float, fill: int) -> np.ndarray[Any, Any]:
    if math.isclose(angle, 0.0, abs_tol=1e-12):
        return image.copy()
    height, width = image.shape
    center = ((width - 1) / 2.0, (height - 1) / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=fill,
    )
    return np.ascontiguousarray(rotated, dtype=np.uint8)


def generate_line_sample(
    text: str,
    *,
    sample_id: str,
    document_id: str,
    config: SyntheticDataConfig = REFERENCE_CONFIG,
    font_path: str | os.PathLike[str] | None = None,
    image_path: str | None = None,
    text_source_id: str = "caller-supplied",
    page_id: str | None = None,
    line_index: int | None = None,
) -> GeneratedLineSample:
    """Render one naturally sized shaped line without writing to the filesystem."""

    if not isinstance(config, SyntheticDataConfig):
        raise TypeError("config must be SyntheticDataConfig")
    if not isinstance(text, str) or _transcription_issue_codes(text):
        raise SyntheticDataError("text must satisfy the nfc-v1 transcription policy")
    resolved_font, font_identifier, font_digest = _resolved_font(font_path)
    assert_shaping_available(font_path)
    relative_image_path = image_path or f"lines/{sample_id}.png"
    try:
        sample = DatasetSample(
            schema_version=1,
            sample_id=sample_id,
            image_path=relative_image_path,
            text=text,
            document_id=document_id,
            page_id=page_id,
            line_index=line_index,
            tags=("synthetic",),
        )
    except (TypeError, ValueError) as error:
        raise SyntheticDataError("synthetic line identity or path is invalid") from error

    derived_seed = _derived_seed(config.seed, sample_id)
    rng = np.random.default_rng(derived_seed)
    font_size = int(rng.integers(config.font_size_min, config.font_size_max + 1))
    padding_left = int(
        rng.integers(config.horizontal_padding_min, config.horizontal_padding_max + 1)
    )
    padding_right = int(
        rng.integers(config.horizontal_padding_min, config.horizontal_padding_max + 1)
    )
    padding_top = int(rng.integers(config.vertical_padding_min, config.vertical_padding_max + 1))
    padding_bottom = int(rng.integers(config.vertical_padding_min, config.vertical_padding_max + 1))
    background = int(
        rng.integers(config.background_intensity_min, config.background_intensity_max + 1)
    )
    foreground = int(rng.integers(config.text_intensity_min, config.text_intensity_max + 1))
    blur_sigma = _sample_optional_range(
        rng,
        float(config.blur_probability),
        float(config.blur_sigma_min),
        float(config.blur_sigma_max),
    )
    noise_std = _sample_optional_range(
        rng,
        float(config.noise_probability),
        float(config.noise_std_min),
        float(config.noise_std_max),
    )
    skew_magnitude = _sample_optional_range(
        rng,
        float(config.skew_probability),
        0.0,
        float(config.skew_max_abs_degrees),
    )
    skew_degrees = skew_magnitude * (-1.0 if int(rng.integers(0, 2)) == 0 else 1.0)

    alpha = _render_text_alpha(text, resolved_font, font_size)
    width = padding_left + alpha.shape[1] + padding_right
    height = padding_top + alpha.shape[0] + padding_bottom
    if width > config.max_line_width:
        raise SyntheticDataError("rendered line exceeds max_line_width")
    image = np.full((height, width), background, dtype=np.uint8)
    image[
        padding_top : padding_top + alpha.shape[0],
        padding_left : padding_left + alpha.shape[1],
    ] = _blend_alpha(alpha, background, foreground)
    if blur_sigma > 0.0:
        image = cv2.GaussianBlur(image, (0, 0), sigmaX=blur_sigma, sigmaY=blur_sigma)
    if noise_std > 0.0:
        noise = rng.normal(0.0, noise_std, size=image.shape)
        image = np.rint(image.astype(np.float64) + noise).clip(0, 255).astype(np.uint8)
    image = _rotate_grayscale(image, skew_degrees, background)
    encoded = _png_bytes(image)
    record = SyntheticLineRecord(
        sample_id=sample.sample_id,
        text_source_id=text_source_id,
        text_sha256=sha256(text.encode("utf-8")).hexdigest(),
        font_identifier=font_identifier,
        font_sha256=font_digest,
        master_seed=config.seed,
        derived_seed=derived_seed,
        width=width,
        height=height,
        font_size=font_size,
        padding_left=padding_left,
        padding_right=padding_right,
        padding_top=padding_top,
        padding_bottom=padding_bottom,
        background_intensity=background,
        text_intensity=foreground,
        blur_sigma=blur_sigma,
        noise_std=noise_std,
        skew_degrees=skew_degrees,
        image_path=sample.image_path,
        image_sha256=sha256(encoded).hexdigest(),
    )
    return GeneratedLineSample(sample, record, image, encoded)


@dataclass(frozen=True, slots=True)
class _PagePlacement:
    text: str
    left: int
    right: int
    y: int
    region_kind: RegionKind
    column_index: int | None
    reading_order_index: int
    centered: bool = False


def _fit_page_line(
    text: str,
    font_path: Path,
    maximum_width: int,
    preferred_size: int,
) -> np.ndarray[Any, Any]:
    font_size = preferred_size
    while font_size >= 24:
        alpha = _render_text_alpha(text, font_path, font_size)
        if alpha.shape[1] <= maximum_width:
            return alpha
        font_size -= 2
    raise SyntheticDataError("page line cannot fit inside its assigned region")


def _page_placements(
    family: PageLayoutFamily,
    config: SyntheticDataConfig,
    texts: Sequence[str],
) -> tuple[_PagePlacement, ...]:
    margin = config.page_margin
    content_right = config.page_width - margin
    content_width = config.page_width - 2 * margin
    column_width = (content_width - config.column_gutter) // 2
    left_left = margin
    left_right = margin + column_width
    right_left = content_right - column_width
    right_right = content_right
    source_index = (
        1
        if family
        in {
            PageLayoutFamily.UNEVEN_TWO_COLUMN,
            PageLayoutFamily.SPANNING_FOOTER,
            PageLayoutFamily.MILD_SKEW,
        }
        else 0
    )
    placements: list[_PagePlacement] = []

    def next_text(*, long: bool = False) -> str:
        nonlocal source_index
        if long:
            return texts[9 % len(texts)]
        value = texts[(source_index + 1) % len(texts)]
        source_index += 1
        return value

    def add(
        left: int,
        right: int,
        y: int,
        kind: RegionKind,
        column: int | None,
        order: int,
        *,
        long: bool = False,
        centered: bool = False,
    ) -> None:
        placements.append(
            _PagePlacement(next_text(long=long), left, right, y, kind, column, order, centered)
        )

    if family in {PageLayoutFamily.ONE_COLUMN, PageLayoutFamily.MILD_SKEW}:
        for order, y in enumerate((90, 290, 490, 690, 890)):
            add(margin + 80, content_right, y, RegionKind.LINE, None, order)
    elif family is PageLayoutFamily.ONE_COLUMN_WIDE:
        add(margin, content_right, 100, RegionKind.LINE, None, 0, long=True)
        for order, y in enumerate((350, 600, 850), start=1):
            add(margin + 100, content_right, y, RegionKind.LINE, None, order)
    elif family is PageLayoutFamily.BALANCED_TWO_COLUMN:
        for order, y in enumerate((90, 350, 610, 870)):
            add(right_left, right_right, y, RegionKind.LINE, 0, order)
        for order, y in enumerate((90, 350, 610, 870), start=4):
            add(left_left, left_right, y, RegionKind.LINE, 1, order)
    elif family is PageLayoutFamily.UNEVEN_TWO_COLUMN:
        for order, y in enumerate((90, 350, 610, 870)):
            add(right_left, right_right, y, RegionKind.LINE, 0, order)
        for order, y in enumerate((90, 350, 610), start=4):
            add(left_left, left_right, y, RegionKind.LINE, 1, order)
    elif family is PageLayoutFamily.SPANNING_TOP:
        add(margin, content_right, 55, RegionKind.SPANNING, None, 0, long=True, centered=True)
        for order, y in enumerate((330, 590, 850), start=1):
            add(right_left, right_right, y, RegionKind.LINE, 0, order)
        for order, y in enumerate((330, 590, 850), start=4):
            add(left_left, left_right, y, RegionKind.LINE, 1, order)
    elif family is PageLayoutFamily.SPANNING_MIDDLE:
        for order, y in enumerate((70, 260)):
            add(right_left, right_right, y, RegionKind.LINE, 0, order)
        for order, y in enumerate((70, 260), start=2):
            add(left_left, left_right, y, RegionKind.LINE, 1, order)
        add(margin, content_right, 485, RegionKind.SPANNING, None, 4, long=True, centered=True)
        for order, y in enumerate((720, 940), start=5):
            add(right_left, right_right, y, RegionKind.LINE, 0, order)
        for order, y in enumerate((720, 940), start=7):
            add(left_left, left_right, y, RegionKind.LINE, 1, order)
    elif family is PageLayoutFamily.SPANNING_FOOTER:
        for order, y in enumerate((90, 350, 610)):
            add(right_left, right_right, y, RegionKind.LINE, 0, order)
        for order, y in enumerate((90, 350, 610), start=3):
            add(left_left, left_right, y, RegionKind.LINE, 1, order)
        add(margin, content_right, 950, RegionKind.SPANNING, None, 6, long=True, centered=True)
    elif family not in {PageLayoutFamily.BLANK, PageLayoutFamily.NEAR_BLANK}:
        raise SyntheticDataError("unsupported page layout family")
    return tuple(placements)


def _paste_alpha(
    page: np.ndarray[Any, Any],
    alpha: np.ndarray[Any, Any],
    *,
    x: int,
    y: int,
    background: int,
    foreground: int,
) -> BoundingBox:
    height, width = alpha.shape
    if x < 0 or y < 0 or x + width > page.shape[1] or y + height > page.shape[0]:
        raise SyntheticDataError("page line placement exceeds the page boundary")
    rendered = _blend_alpha(alpha, background, foreground)
    view = page[y : y + height, x : x + width]
    np.minimum(view, rendered, out=view)
    return BoundingBox(x, y, width, height)


def _rotation_matrix(width: int, height: int, angle: float) -> np.ndarray[Any, Any]:
    center = ((width - 1) / 2.0, (height - 1) / 2.0)
    return cv2.getRotationMatrix2D(center, angle, 1.0)


def _transformed_box(
    box: BoundingBox,
    matrix: np.ndarray[Any, Any],
    width: int,
    height: int,
) -> BoundingBox:
    corners = np.array(
        [
            [box.x_min, box.y_min, 1.0],
            [box.x_max, box.y_min, 1.0],
            [box.x_min, box.y_max, 1.0],
            [box.x_max, box.y_max, 1.0],
        ]
    )
    transformed = corners @ matrix.T
    left = max(0, math.floor(float(transformed[:, 0].min())))
    top = max(0, math.floor(float(transformed[:, 1].min())))
    right = min(width, math.ceil(float(transformed[:, 0].max())))
    bottom = min(height, math.ceil(float(transformed[:, 1].max())))
    if right <= left or bottom <= top:
        raise SyntheticDataError("rotated page line has invalid geometry")
    return BoundingBox(left, top, right - left, bottom - top)


def generate_page_fixture(
    layout_family: PageLayoutFamily | str,
    *,
    page_id: str,
    config: SyntheticDataConfig = REFERENCE_CONFIG,
    font_path: str | os.PathLike[str] | None = None,
    texts: Sequence[str] | None = None,
    image_path: str | None = None,
) -> GeneratedPageFixture:
    """Compose one shaped Urdu page and explicit half-open ground truth in memory."""

    if not isinstance(config, SyntheticDataConfig):
        raise TypeError("config must be SyntheticDataConfig")
    try:
        family = PageLayoutFamily(layout_family)
        normalized_image_path = normalize_relative_dataset_path(
            image_path or f"pages/{page_id}.png"
        )
    except (TypeError, ValueError) as error:
        raise SyntheticDataError("page identity, layout, or image path is invalid") from error
    if not isinstance(page_id, str) or not page_id.startswith("syn-page-"):
        raise SyntheticDataError("page_id must use the syn-page namespace")
    page_texts = (
        tuple(source.text for source in PUBLIC_TEXT_SOURCES) if texts is None else tuple(texts)
    )
    if not page_texts or any(
        not isinstance(text, str) or _transcription_issue_codes(text) for text in page_texts
    ):
        raise SyntheticDataError("page texts must satisfy the nfc-v1 transcription policy")

    resolved_font, _, font_digest = _resolved_font(font_path)
    capabilities = assert_shaping_available(font_path)
    if capabilities.font_sha256 != font_digest:
        raise SyntheticDataError("shaping capability font identity changed")
    derived_seed = _derived_seed(config.seed, page_id)
    rng = np.random.default_rng(derived_seed)
    background = int(
        rng.integers(config.background_intensity_min, config.background_intensity_max + 1)
    )
    foreground = int(rng.integers(config.text_intensity_min, config.text_intensity_max + 1))
    page = np.full((config.page_height, config.page_width), background, dtype=np.uint8)
    page_lines: list[SyntheticPageLine] = []
    for placement in _page_placements(family, config, page_texts):
        available = placement.right - placement.left
        preferred = config.font_size_max if placement.region_kind is RegionKind.SPANNING else 44
        alpha = _fit_page_line(placement.text, resolved_font, available, preferred)
        x = (
            placement.left + (available - alpha.shape[1]) // 2
            if placement.centered
            else placement.right - alpha.shape[1]
        )
        box = _paste_alpha(
            page,
            alpha,
            x=x,
            y=placement.y,
            background=background,
            foreground=foreground,
        )
        page_lines.append(
            SyntheticPageLine(
                line_id=f"{page_id}-line-{placement.reading_order_index + 1:02d}",
                text=placement.text,
                text_sha256=sha256(placement.text.encode("utf-8")).hexdigest(),
                composition_box=box,
                output_box=box,
                region_kind=placement.region_kind,
                column_index=placement.column_index,
                reading_order_index=placement.reading_order_index,
            )
        )

    if family is PageLayoutFamily.NEAR_BLANK:
        for _ in range(8):
            x = int(rng.integers(config.page_margin, config.page_width - config.page_margin))
            y = int(rng.integers(config.page_margin, config.page_height - config.page_margin))
            page[y, x] = max(0, background - 6)

    angle = 0.0
    if family is PageLayoutFamily.MILD_SKEW:
        angle = min(2.0, float(config.skew_max_abs_degrees))
    if angle:
        matrix = _rotation_matrix(config.page_width, config.page_height, angle)
        page = _rotate_grayscale(page, angle, background)
        page_lines = [
            SyntheticPageLine(
                line.line_id,
                line.text,
                line.text_sha256,
                line.composition_box,
                _transformed_box(
                    line.composition_box, matrix, config.page_width, config.page_height
                ),
                line.region_kind,
                line.column_index,
                line.reading_order_index,
            )
            for line in page_lines
        ]
    page_lines.sort(key=lambda line: line.reading_order_index)
    encoded = _png_bytes(page)
    record = SyntheticPageRecord(
        page_id=page_id,
        layout_family=family,
        image_path=normalized_image_path,
        image_sha256=sha256(encoded).hexdigest(),
        width=config.page_width,
        height=config.page_height,
        master_seed=config.seed,
        derived_seed=derived_seed,
        background_intensity=background,
        skew_degrees=angle,
        expected_blank=family in {PageLayoutFamily.BLANK, PageLayoutFamily.NEAR_BLANK},
        lines=tuple(page_lines),
    )
    return GeneratedPageFixture(record, page, encoded)


def _text_source_report(sources: Sequence[TextSource]) -> tuple[dict[str, object], str]:
    source_records = [source.to_public_dict() for source in sources]
    fingerprint = sha256(_canonical_json_bytes(source_records, final_newline=False)).hexdigest()
    frequencies = Counter(character for source in sources for character in source.text)
    payload: dict[str, object] = {
        "schema_version": 1,
        "source_set_id": "project-authored-urdu-fixtures-v1",
        "source_set_fingerprint": fingerprint,
        "normalization_policy": NORMALIZATION_POLICY_VERSION,
        "phrase_count": len(sources),
        "all_nfc": all(
            unicodedata.normalize("NFC", source.text) == source.text for source in sources
        ),
        "unique_character_count": len(frequencies),
        "character_frequency": [
            {"character": character, "code_point": f"U+{ord(character):04X}", "count": count}
            for character, count in sorted(frequencies.items(), key=lambda item: ord(item[0]))
        ],
        "missing_from_fixture_vocabulary": [],
        "sources": source_records,
    }
    return payload, fingerprint


def _safe_output_root(output_directory: str | os.PathLike[str]) -> Path:
    try:
        root = Path(output_directory).resolve(strict=True)
    except (OSError, TypeError, ValueError) as error:
        raise SyntheticDataError("explicit output directory must already exist") from error
    if not root.is_dir() or root.parent == root:
        raise SyntheticDataError("output directory must be a non-root directory")
    return root


def _target(root: Path, relative_path: str) -> Path:
    try:
        normalized = normalize_relative_dataset_path(relative_path)
        target = (root / normalized).resolve(strict=False)
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise SyntheticDataError("generated output path is invalid") from error
    if not target.is_relative_to(root):
        raise SyntheticDataError("generated output path escapes the output directory")
    return target


def _preflight_output(root: Path, expected_files: set[str], overwrite: bool) -> None:
    if not isinstance(overwrite, bool):
        raise TypeError("overwrite must be a boolean")
    existing = {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()}
    unexpected = existing - expected_files
    if unexpected:
        raise SyntheticDataError("output directory contains files outside this generation plan")
    if not overwrite and existing:
        raise SyntheticDataError("output files exist; set overwrite=True explicitly")


def _write_bytes(root: Path, relative_path: str, payload: bytes, *, overwrite: bool) -> None:
    target = _target(root, relative_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_bytes(
        payload,
        target,
        overwrite=overwrite,
        error_type=SyntheticDataError,
        artifact_name="synthetic artifact",
    )


def _ensure_directory(root: Path, relative_path: str) -> Path:
    target = _target(root, relative_path)
    target.mkdir(parents=True, exist_ok=True)
    if not target.is_dir():
        raise SyntheticDataError("synthetic artifact directory is unavailable")
    return target


def generate_line_dataset(
    output_directory: str | os.PathLike[str],
    sample_count: int,
    *,
    config: SyntheticDataConfig = REFERENCE_CONFIG,
    texts: Sequence[str] | None = None,
    overwrite: bool = False,
    font_path: str | os.PathLike[str] | None = None,
) -> tuple[GeneratedLineSample, ...]:
    """Generate any positive requested line count inside an explicit existing directory."""

    if type(sample_count) is not int or not 1 <= sample_count <= 100_000:
        raise SyntheticDataError("sample_count must be an integer from 1 through 100000")
    corpus = tuple(source.text for source in PUBLIC_TEXT_SOURCES) if texts is None else tuple(texts)
    if not corpus or any(
        not isinstance(text, str) or _transcription_issue_codes(text) for text in corpus
    ):
        raise SyntheticDataError("line dataset texts must satisfy nfc-v1")
    root = _safe_output_root(output_directory)
    expected = {f"lines/syn-line-{index:06d}.png" for index in range(1, sample_count + 1)}
    expected.update({"manifest.jsonl", "generation-manifest.json"})
    _preflight_output(root, expected, overwrite)
    assert_shaping_available(font_path)
    generated: list[GeneratedLineSample] = []
    for index in range(1, sample_count + 1):
        source_index = (index - 1) % len(corpus)
        item = generate_line_sample(
            corpus[source_index],
            sample_id=f"syn-line-{index:06d}",
            document_id=f"synthetic-document-{((index - 1) // 4) + 1:06d}",
            config=config,
            font_path=font_path,
            text_source_id=f"dataset-text-{source_index + 1:06d}",
            line_index=(index - 1) % 4,
        )
        generated.append(item)
        _write_bytes(root, item.sample.image_path, item.png_bytes, overwrite=overwrite)
    samples = tuple(item.sample for item in generated)
    _write_bytes(root, "manifest.jsonl", canonical_manifest_bytes(samples), overwrite=overwrite)
    generation = {
        "schema_version": 1,
        "generator_version": GENERATOR_VERSION,
        "config": config.to_dict(),
        "config_fingerprint": config.fingerprint,
        "dataset_fingerprint": dataset_fingerprint(samples),
        "sample_count": sample_count,
        "line_records": [item.record.to_public_dict() for item in generated],
    }
    _write_bytes(
        root,
        "generation-manifest.json",
        _canonical_json_bytes(generation),
        overwrite=overwrite,
    )
    return tuple(generated)


def _overlay_png(page: GeneratedPageFixture) -> bytes:
    image = np.repeat(page.image[:, :, np.newaxis], 3, axis=2)
    canvas = Image.fromarray(image)
    draw = ImageDraw.Draw(canvas)
    colors = {None: (184, 70, 40), 0: (27, 105, 180), 1: (44, 135, 73)}
    for line in page.record.lines:
        box = line.output_box
        color = colors[line.column_index]
        draw.rectangle((box.x, box.y, box.x_max - 1, box.y_max - 1), outline=color, width=3)
        draw.text((box.x + 4, box.y + 3), str(line.reading_order_index + 1), fill=color)
    output = io.BytesIO()
    canvas.save(output, format="PNG", compress_level=9, optimize=False)
    return output.getvalue()


def _artifact_role(relative_path: str) -> str:
    if relative_path.startswith("lines/"):
        return "synthetic-line-image"
    if relative_path.startswith("pages/"):
        return "synthetic-page-image"
    if relative_path.startswith("visual/"):
        return "fixture-derived-layout-overlay"
    if relative_path.startswith("splits/"):
        return "grouped-split-artifact"
    return {
        "manifest.jsonl": "phase-7-line-manifest",
        "page-ground-truth.json": "page-ground-truth",
        "generation-manifest.json": "generation-record",
        "provenance.json": "fixture-provenance",
        "text-provenance.json": "authored-text-provenance",
        "synthetic-fixture-vocabulary.json": "train-only-synthetic-fixture-vocabulary",
        "README.md": "fixture-documentation",
    }.get(relative_path, "synthetic-fixture-artifact")


def generate_fixture_dataset(
    output_directory: str | os.PathLike[str],
    *,
    config: SyntheticDataConfig = REFERENCE_CONFIG,
    overwrite: bool = False,
) -> FixtureGenerationResult:
    """Regenerate the reviewed canonical public fixture package."""

    root = _safe_output_root(output_directory)
    line_paths = {f"lines/syn-line-{index:06d}.png" for index in range(1, 25)}
    page_paths = {
        f"pages/syn-page-{index:03d}-{family.value}.png"
        for index, family in enumerate(_PAGE_LAYOUT_ORDER, start=1)
    }
    expected = (
        line_paths
        | page_paths
        | {
            "manifest.jsonl",
            "page-ground-truth.json",
            "generation-manifest.json",
            "provenance.json",
            "text-provenance.json",
            "synthetic-fixture-vocabulary.json",
            "splits/train.jsonl",
            "splits/validation.jsonl",
            "splits/test.jsonl",
            "splits/split_metadata.json",
            "visual/synthetic-layout-overlay.png",
            "README.md",
            "artifact-manifest.json",
        }
    )
    _preflight_output(root, expected, overwrite)
    capabilities = assert_shaping_available()
    text_report, text_fingerprint = _text_source_report(PUBLIC_TEXT_SOURCES)

    document_ids = tuple(f"synthetic-document-{index:03d}" for index in range(1, 7))
    planning = tuple(
        DatasetSample(
            1,
            f"planning-{document_index:03d}-{line_index:02d}",
            f"lines/planning-{document_index:03d}-{line_index:02d}.png",
            PUBLIC_TEXT_SOURCES[0].text,
            document_id,
        )
        for document_index, document_id in enumerate(document_ids, start=1)
        for line_index in range(4)
    )
    split_config = DatasetSplitConfig(
        train_ratio=2 / 3,
        validation_ratio=1 / 6,
        test_ratio=1 / 6,
        seed=1337,
    )
    planned_split = split_dataset(planning, split_config)
    train_documents = {
        document_id
        for document_id, partition in planned_split.document_assignments
        if partition.value == "train"
    }
    remaining_sources = list(PUBLIC_TEXT_SOURCES[1:])
    line_plan: list[tuple[str, int, TextSource]] = []
    fallback_index = 1
    for document_id in document_ids:
        line_plan.append((document_id, 0, PUBLIC_TEXT_SOURCES[0]))
        for line_index in range(1, 4):
            if document_id in train_documents:
                source = remaining_sources.pop(0) if remaining_sources else PUBLIC_TEXT_SOURCES[1]
            else:
                source = PUBLIC_TEXT_SOURCES[fallback_index]
                fallback_index = 1 + fallback_index % 6
            line_plan.append((document_id, line_index, source))
    if remaining_sources:  # pragma: no cover - guards fixture scale edits
        raise SyntheticDataError("training fixture slots do not cover every authored source")

    lines: list[GeneratedLineSample] = []
    for index, (document_id, line_index, source) in enumerate(line_plan, start=1):
        line = generate_line_sample(
            source.text,
            sample_id=f"syn-line-{index:06d}",
            document_id=document_id,
            config=config,
            text_source_id=source.source_id,
            line_index=line_index,
        )
        lines.append(line)
        _write_bytes(root, line.sample.image_path, line.png_bytes, overwrite=overwrite)

    pages: list[GeneratedPageFixture] = []
    for index, family in enumerate(_PAGE_LAYOUT_ORDER, start=1):
        page_id = f"syn-page-{index:03d}-{family.value}"
        page = generate_page_fixture(family, page_id=page_id, config=config)
        pages.append(page)
        _write_bytes(root, page.record.image_path, page.png_bytes, overwrite=overwrite)

    samples = tuple(line.sample for line in lines)
    split = split_dataset(samples, split_config)
    vocabulary = build_vocabulary(split.train)
    unseen = find_unseen_characters(split.validation + split.test, vocabulary)
    if unseen:
        raise SyntheticDataError("canonical validation/test fixtures contain unseen characters")
    _write_bytes(root, "manifest.jsonl", canonical_manifest_bytes(samples), overwrite=overwrite)
    split_directory = _ensure_directory(root, "splits")
    write_split_manifests(split, split_directory, overwrite=overwrite)
    save_vocabulary(
        vocabulary,
        _target(root, "synthetic-fixture-vocabulary.json"),
        overwrite=overwrite,
    )

    report = validate_dataset(samples, dataset_root=root, vocabulary=vocabulary)
    if report.error_count or report.warning_count:
        raise SyntheticDataError("canonical fixture validation must have zero findings")
    _write_bytes(
        root,
        "page-ground-truth.json",
        _canonical_json_bytes(
            {
                "schema_version": 1,
                "generator_version": GENERATOR_VERSION,
                "coordinate_semantics": "half-open-[x,right)-[y,bottom)",
                "pages": [page.record.to_public_dict() for page in pages],
            }
        ),
        overwrite=overwrite,
    )
    _write_bytes(
        root,
        "text-provenance.json",
        _canonical_json_bytes(text_report),
        overwrite=overwrite,
    )
    provenance = {
        "schema_version": 1,
        "generator_version": GENERATOR_VERSION,
        "fixture_origin": "current-project-authored-and-programmatically-rendered",
        "historical_source_copied": False,
        "historical_text_copied": False,
        "historical_font_copied": False,
        "historical_image_copied": False,
        "font": load_font_provenance(),
        "text_source_fingerprint": text_fingerprint,
        "visual": {
            "path": "visual/synthetic-layout-overlay.png",
            "source_fixture": pages[3].record.image_path,
            "generation_process": "draw-half-open-ground-truth-boxes-and-reading-order-v1",
        },
    }
    _write_bytes(root, "provenance.json", _canonical_json_bytes(provenance), overwrite=overwrite)
    generation_manifest = {
        "schema_version": 1,
        "generator_version": GENERATOR_VERSION,
        "reference_environment": {
            "python": "CPython-3.11",
            "pillow": pillow_version,
            "numpy": distribution_version("numpy"),
            "opencv_python_headless": distribution_version("opencv-python-headless"),
            "uharfbuzz": distribution_version("uharfbuzz"),
            "freetype_py": distribution_version("freetype-py"),
        },
        "determinism": {
            "logical": "identical for the frozen source, config, seed, font, and generator",
            "bytes": "verified for the exact dependency versions in reference_environment",
            "sample_identity": "stable sequence IDs locked by this manifest",
        },
        "config": config.to_dict(),
        "config_fingerprint": config.fingerprint,
        "shaping": capabilities.to_public_dict(),
        "dataset_validation": report.to_public_dict(),
        "dataset_fingerprint": dataset_fingerprint(samples),
        "split": split.to_public_dict(),
        "vocabulary_fingerprint": vocabulary.fingerprint,
        "validation_test_unseen_characters": [],
        "line_records": [line.record.to_public_dict() for line in lines],
        "page_records": [page.record.to_public_dict() for page in pages],
    }
    _write_bytes(
        root,
        "generation-manifest.json",
        _canonical_json_bytes(generation_manifest),
        overwrite=overwrite,
    )
    overlay = _overlay_png(pages[3])
    _write_bytes(
        root,
        "visual/synthetic-layout-overlay.png",
        overlay,
        overwrite=overwrite,
    )
    fixture_readme = dedent("""\
    # Safe synthetic Urdu fixtures

    This directory contains 24 current-project synthetic line images and 10 synthetic pages.
    The Urdu phrases are project-authored neutral fixture content; no historical text, scan,
    annotation, or private data is present. `manifest.jsonl` uses the Phase 7 schema. The split
    is document-grouped, and `synthetic-fixture-vocabulary.json` is a train-only **synthetic
    fixture vocabulary**, not a historical or production vocabulary.

    Regenerate from the repository root:

    ```console
    python scripts/generate_sample_data.py --output data/sample --overwrite
    ```

    Existing unrelated files are rejected and no network access occurs.
    See `generation-manifest.json`, `text-provenance.json`, `provenance.json`, and
    `artifact-manifest.json` for frozen identities and hashes.
    """)
    _write_bytes(root, "README.md", fixture_readme.encode("utf-8"), overwrite=overwrite)

    artifact_entries = []
    for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        if not path.is_file() or path.name == "artifact-manifest.json":
            continue
        relative = path.relative_to(root).as_posix()
        payload = path.read_bytes()
        entry: dict[str, object] = {
            "path": relative,
            "size_bytes": len(payload),
            "sha256": sha256(payload).hexdigest(),
            "logical_role": _artifact_role(relative),
        }
        if relative == "visual/synthetic-layout-overlay.png":
            entry["source_fixture"] = pages[3].record.image_path
            entry["generation_process"] = "draw-half-open-ground-truth-boxes-and-reading-order-v1"
        artifact_entries.append(entry)
    artifact_payload = _canonical_json_bytes(
        {
            "schema_version": 1,
            "generator_version": GENERATOR_VERSION,
            "self_entry_policy": "artifact-manifest-excluded-to-avoid-recursive-hash",
            "artifacts": artifact_entries,
        }
    )
    _write_bytes(root, "artifact-manifest.json", artifact_payload, overwrite=overwrite)
    total_size = sum(path.stat().st_size for path in root.rglob("*") if path.is_file())
    return FixtureGenerationResult(
        output_role="canonical-public-fixture-set",
        line_count=len(lines),
        page_count=len(pages),
        dataset_fingerprint=dataset_fingerprint(samples),
        split_fingerprint=split.fingerprint,
        vocabulary_fingerprint=vocabulary.fingerprint,
        text_source_fingerprint=text_fingerprint,
        config_fingerprint=config.fingerprint,
        artifact_manifest_sha256=sha256(artifact_payload).hexdigest(),
        total_size_bytes=total_size,
    )
