"""Bounded PNG, JPEG, and PDF ingestion into owned RGB page arrays."""

from __future__ import annotations

import io
import math
import os
import warnings
from hashlib import sha256
from pathlib import Path
from typing import TypeAlias

import numpy as np
import pypdfium2 as pdfium
from PIL import Image, ImageOps, UnidentifiedImageError

from urdu_document_ocr.config import InputLimitsConfig
from urdu_document_ocr.errors import (
    DocumentLoadError,
    PdfPasswordError,
    PdfRenderError,
    ResourceLimitError,
    UnsupportedDocumentFormatError,
)
from urdu_document_ocr.types import PageImage, SourceMetadata, SourceType

DocumentSource: TypeAlias = str | os.PathLike[str] | bytes | bytearray | memoryview

_DEFAULT_LIMITS = InputLimitsConfig()
_PDF_HEADER = b"%PDF-"
_PNG_HEADER = b"\x89PNG\r\n\x1a\n"
_JPEG_HEADER = b"\xff\xd8\xff"
_SUPPORTED_IMAGE_FORMATS = frozenset({"JPEG", "PNG"})


def load_document(
    source: DocumentSource,
    *,
    limits: InputLimitsConfig = _DEFAULT_LIMITS,
    pdf_dpi: int | None = None,
    display_name: str | None = None,
) -> tuple[PageImage, ...]:
    """Load an image or PDF without retaining handles or exposing source paths.

    Format selection is based on a PDF header or Pillow's decoded format, never a
    filename extension. Every returned page owns its RGB ``uint8`` NumPy array.
    """

    if not isinstance(limits, InputLimitsConfig):
        raise TypeError("limits must be an InputLimitsConfig")

    data, default_name = _read_source(source, limits)
    safe_name = _validated_display_name(display_name if display_name is not None else default_name)
    digest = sha256(data).hexdigest()

    if _looks_like_pdf(data):
        dpi = _validated_pdf_dpi(pdf_dpi, limits)
        metadata = SourceMetadata(SourceType.PDF, safe_name, digest, len(data))
        return _load_pdf(data, metadata, limits, dpi)

    if pdf_dpi is not None:
        raise DocumentLoadError("pdf_dpi may only be supplied for PDF input")
    metadata = SourceMetadata(SourceType.IMAGE, safe_name, digest, len(data))
    return (_load_image(data, metadata, limits),)


def _read_source(source: DocumentSource, limits: InputLimitsConfig) -> tuple[bytes, str | None]:
    if isinstance(source, (bytes, bytearray, memoryview)):
        byte_count = source.nbytes if isinstance(source, memoryview) else len(source)
        _enforce_input_size(byte_count, limits)
        data = bytes(source)
        _enforce_input_size(len(data), limits)
        if not data:
            raise DocumentLoadError("document input is empty")
        return data, None

    if isinstance(source, (str, os.PathLike)):
        try:
            path = Path(source)
        except (TypeError, ValueError) as error:
            raise DocumentLoadError("document path is invalid") from error
        try:
            stat_result = path.stat()
        except OSError as error:
            raise DocumentLoadError("document path could not be opened") from error
        if not path.is_file():
            raise DocumentLoadError("document path does not identify a regular file")
        _enforce_input_size(stat_result.st_size, limits)
        try:
            data = path.read_bytes()
        except OSError as error:
            raise DocumentLoadError("document input could not be read") from error
        _enforce_input_size(len(data), limits)
        if not data:
            raise DocumentLoadError("document input is empty")
        return data, path.name

    raise TypeError("source must be a filesystem path or an in-memory byte buffer")


def _enforce_input_size(byte_count: int, limits: InputLimitsConfig) -> None:
    if byte_count > limits.max_input_bytes:
        raise ResourceLimitError(
            "document input exceeds the configured byte limit",
            context={
                "resource": "input_bytes",
                "actual": byte_count,
                "limit": limits.max_input_bytes,
            },
        )


def _validated_display_name(display_name: str | None) -> str | None:
    try:
        placeholder = SourceMetadata(SourceType.IMAGE, display_name)
    except (TypeError, ValueError) as error:
        raise DocumentLoadError("display_name must be a nonempty basename") from error
    return placeholder.display_name


def _looks_like_pdf(data: bytes) -> bool:
    prefix = data[:1024].lstrip(b"\x00\t\n\r\f ")
    return prefix.startswith(_PDF_HEADER)


def _has_supported_image_magic(data: bytes) -> bool:
    return data.startswith((_PNG_HEADER, _JPEG_HEADER))


def _check_pixel_limits(width: int, height: int, limits: InputLimitsConfig) -> int:
    if width < 1 or height < 1:
        raise DocumentLoadError("document page dimensions must be positive")
    pixels = width * height
    if pixels > limits.max_pixels_per_page:
        raise ResourceLimitError(
            "document page exceeds the configured pixel limit",
            context={
                "resource": "page_pixels",
                "actual": pixels,
                "limit": limits.max_pixels_per_page,
            },
        )
    if pixels > limits.max_document_pixels:
        raise ResourceLimitError(
            "document exceeds the configured cumulative pixel limit",
            context={
                "resource": "document_pixels",
                "actual": pixels,
                "limit": limits.max_document_pixels,
            },
        )
    return pixels


def _load_image(data: bytes, metadata: SourceMetadata, limits: InputLimitsConfig) -> PageImage:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(data)) as probe:
                image_format = probe.format
                if image_format not in _SUPPORTED_IMAGE_FORMATS:
                    raise UnsupportedDocumentFormatError(
                        "only PNG, JPEG, and PDF documents are supported",
                        context={"detected_format": image_format or "unknown"},
                    )
                _check_pixel_limits(probe.width, probe.height, limits)
                probe.verify()

            with Image.open(io.BytesIO(data)) as decoded:
                _check_pixel_limits(decoded.width, decoded.height, limits)
                oriented = ImageOps.exif_transpose(decoded)
                try:
                    rgb = _flatten_transparency_to_rgb(oriented)
                    try:
                        image_array = np.asarray(rgb, dtype=np.uint8).copy()
                    finally:
                        rgb.close()
                finally:
                    if oriented is not decoded:
                        oriented.close()
    except UnsupportedDocumentFormatError:
        raise
    except (Image.DecompressionBombError, Image.DecompressionBombWarning) as error:
        raise ResourceLimitError(
            "image exceeds safe decoder dimensions",
            context={"resource": "decoder_dimensions"},
        ) from error
    except UnidentifiedImageError as error:
        if _has_supported_image_magic(data):
            raise DocumentLoadError("image content could not be decoded") from error
        raise UnsupportedDocumentFormatError(
            "only PNG, JPEG, and PDF documents are supported",
            context={"detected_format": "unknown"},
        ) from error
    except (OSError, SyntaxError, ValueError) as error:
        raise DocumentLoadError("image content could not be decoded") from error

    _check_pixel_limits(int(image_array.shape[1]), int(image_array.shape[0]), limits)
    return PageImage(0, 1, image_array, metadata)


def _flatten_transparency_to_rgb(image: Image.Image) -> Image.Image:
    has_alpha = image.mode in {"LA", "RGBA"} or (image.mode == "P" and "transparency" in image.info)
    if not has_alpha:
        return image.convert("RGB")

    rgba = image.convert("RGBA")
    try:
        background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        try:
            return Image.alpha_composite(background, rgba).convert("RGB")
        finally:
            background.close()
    finally:
        rgba.close()


def _validated_pdf_dpi(pdf_dpi: int | None, limits: InputLimitsConfig) -> int:
    dpi = limits.default_pdf_dpi if pdf_dpi is None else pdf_dpi
    if isinstance(dpi, bool) or not isinstance(dpi, int):
        raise ResourceLimitError("PDF DPI must be an integer", context={"resource": "pdf_dpi"})
    if not limits.min_pdf_dpi <= dpi <= limits.max_pdf_dpi:
        raise ResourceLimitError(
            "PDF DPI is outside the configured range",
            context={
                "resource": "pdf_dpi",
                "actual": dpi,
                "minimum": limits.min_pdf_dpi,
                "maximum": limits.max_pdf_dpi,
            },
        )
    return dpi


def _points_to_pixels(page_points: float, dpi: int) -> int:
    if not math.isfinite(page_points) or page_points <= 0.0:
        raise PdfRenderError("PDF page dimensions are invalid")
    pixels = math.ceil(page_points * dpi / 72.0)
    if pixels < 1:
        raise PdfRenderError("PDF page dimensions are invalid")
    return pixels


def _open_pdf(data: bytes) -> pdfium.PdfDocument:
    try:
        return pdfium.PdfDocument(data)
    except pdfium.PdfiumError as error:
        if getattr(error, "err_code", None) == 4:
            raise PdfPasswordError(
                "password-protected PDFs are not supported", context={"stage": "open"}
            ) from error
        raise PdfRenderError(
            "PDF content could not be opened", context={"stage": "open"}
        ) from error
    except (OSError, ValueError) as error:
        raise PdfRenderError("PDF content could not be opened") from error


def _load_pdf(
    data: bytes,
    metadata: SourceMetadata,
    limits: InputLimitsConfig,
    dpi: int,
) -> tuple[PageImage, ...]:
    document = _open_pdf(data)
    try:
        try:
            page_count = len(document)
        except pdfium.PdfiumError as error:
            raise PdfRenderError("PDF page count could not be read") from error
        if page_count < 1:
            raise PdfRenderError("PDF must contain at least one page")
        if page_count > limits.max_pages:
            raise ResourceLimitError(
                "PDF exceeds the configured page-count limit",
                context={"resource": "pdf_pages", "actual": page_count, "limit": limits.max_pages},
            )

        dimensions: list[tuple[int, int]] = []
        cumulative_pixels = 0
        for page_index in range(page_count):
            try:
                width_points, height_points = document.get_page_size(page_index)
            except pdfium.PdfiumError as error:
                raise PdfRenderError("PDF page dimensions could not be read") from error
            width = _points_to_pixels(float(width_points), dpi)
            height = _points_to_pixels(float(height_points), dpi)
            page_pixels = width * height
            if page_pixels > limits.max_pixels_per_page:
                raise ResourceLimitError(
                    "PDF page exceeds the configured pixel limit",
                    context={
                        "resource": "page_pixels",
                        "source_page_number": page_index + 1,
                        "actual": page_pixels,
                        "limit": limits.max_pixels_per_page,
                    },
                )
            cumulative_pixels += page_pixels
            if cumulative_pixels > limits.max_document_pixels:
                raise ResourceLimitError(
                    "PDF exceeds the configured cumulative pixel limit",
                    context={
                        "resource": "document_pixels",
                        "source_page_number": page_index + 1,
                        "actual": cumulative_pixels,
                        "limit": limits.max_document_pixels,
                    },
                )
            dimensions.append((width, height))

        pages = [
            _render_pdf_page(document, index, dimensions[index], metadata, dpi)
            for index in range(page_count)
        ]
        return tuple(pages)
    finally:
        document.close()


def _render_pdf_page(
    document: pdfium.PdfDocument,
    page_index: int,
    expected_dimensions: tuple[int, int],
    metadata: SourceMetadata,
    dpi: int,
) -> PageImage:
    page = None
    bitmap = None
    rendered = None
    try:
        page = document.get_page(page_index)
        bitmap = page.render(scale=dpi / 72.0)
        rendered = bitmap.to_pil()
        if rendered.size != expected_dimensions:
            raise PdfRenderError(
                "PDF renderer dimensions did not match preflight geometry",
                context={"source_page_number": page_index + 1, "stage": "render"},
            )
        rgb = rendered.convert("RGB")
        try:
            image_array = np.asarray(rgb, dtype=np.uint8).copy()
        finally:
            rgb.close()
    except (ResourceLimitError, PdfRenderError):
        raise
    except pdfium.PdfiumError as error:
        raise PdfRenderError(
            "PDF page could not be rasterized",
            context={"source_page_number": page_index + 1, "stage": "render"},
        ) from error
    except (OSError, RuntimeError, ValueError) as error:
        raise PdfRenderError(
            "PDF page could not be rasterized",
            context={"source_page_number": page_index + 1, "stage": "render"},
        ) from error
    finally:
        if rendered is not None:
            rendered.close()
        if bitmap is not None:
            bitmap.close()
        if page is not None:
            page.close()

    return PageImage(page_index, page_index + 1, image_array, metadata, raster_dpi=dpi)
