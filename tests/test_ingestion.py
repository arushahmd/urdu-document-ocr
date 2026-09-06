from __future__ import annotations

import io
from dataclasses import replace
from pathlib import Path

import numpy as np
import pypdfium2 as pdfium
import pytest
from PIL import Image

from urdu_document_ocr import InputLimitsConfig, load_document
from urdu_document_ocr.document import ingestion
from urdu_document_ocr.document.ingestion import _points_to_pixels
from urdu_document_ocr.errors import (
    DocumentLoadError,
    PdfPasswordError,
    PdfRenderError,
    ResourceLimitError,
    UnsupportedDocumentFormatError,
)
from urdu_document_ocr.types import SourceType


def image_bytes(image: Image.Image, image_format: str, *, exif: Image.Exif | None = None) -> bytes:
    stream = io.BytesIO()
    save_options = {} if exif is None else {"exif": exif}
    image.save(stream, format=image_format, **save_options)
    return stream.getvalue()


def pdf_bytes(page_sizes: list[tuple[float, float]]) -> bytes:
    stream = io.BytesIO()
    document = pdfium.PdfDocument.new()
    try:
        for width, height in page_sizes:
            page = document.new_page(width, height)
            page.close()
        document.save(stream)
    finally:
        document.close()
    return stream.getvalue()


@pytest.mark.parametrize(
    ("mode", "color"),
    [("L", 100), ("RGB", (1, 2, 3)), ("RGBA", (1, 2, 3, 128)), ("P", 1)],
)
def test_png_modes_normalize_to_owned_rgb_uint8(mode: str, color: object) -> None:
    image = Image.new(mode, (7, 5), color=color)  # type: ignore[arg-type]
    if mode == "P":
        image.putpalette([255, 255, 255, 20, 30, 40] + [0] * 762)
    data = image_bytes(image, "PNG")

    page = load_document(data, display_name="page.png")[0]

    assert page.image.shape == (5, 7, 3)
    assert page.image.dtype == np.uint8
    assert page.image.flags.owndata
    assert page.source.source_type is SourceType.IMAGE
    assert page.source.input_sha256 is not None
    assert page.source.byte_size == len(data)
    assert page.raster_dpi is None


def test_transparent_pixels_are_flattened_onto_white() -> None:
    image = Image.new("RGBA", (2, 1), (25, 50, 75, 0))

    page = load_document(image_bytes(image, "PNG"))[0]

    assert page.image[0, 0].tolist() == [255, 255, 255]


def test_jpeg_and_exif_orientation_are_supported() -> None:
    image = Image.new("RGB", (8, 3), "white")
    exif = Image.Exif()
    exif[274] = 6

    page = load_document(image_bytes(image, "JPEG", exif=exif), display_name="scan.jpg")[0]

    assert page.image.shape == (8, 3, 3)
    assert page.to_public_dict()["source"]["display_name"] == "scan.jpg"  # type: ignore[index]


def test_path_extension_is_not_trusted_and_only_basename_is_public(tmp_path: Path) -> None:
    source = tmp_path / "private-folder" / "misleading.gif"
    source.parent.mkdir()
    source.write_bytes(image_bytes(Image.new("RGB", (3, 2), "white"), "PNG"))

    page = load_document(source)[0]
    serialized = str(page.to_public_dict())

    assert page.source.display_name == "misleading.gif"
    assert str(tmp_path) not in serialized


@pytest.mark.parametrize("image_format", ["GIF", "TIFF", "WEBP"])
def test_other_decodable_image_formats_are_rejected(image_format: str) -> None:
    data = image_bytes(Image.new("RGB", (4, 4), "white"), image_format)

    with pytest.raises(UnsupportedDocumentFormatError):
        load_document(data)


def test_corrupt_supported_image_and_unknown_content_are_typed() -> None:
    with pytest.raises(DocumentLoadError):
        load_document(b"\x89PNG\r\n\x1a\ncorrupt")
    with pytest.raises(UnsupportedDocumentFormatError):
        load_document(b"not a document")


def test_input_byte_limit_is_enforced_at_and_above_boundary() -> None:
    data = image_bytes(Image.new("RGB", (2, 2), "white"), "PNG")

    assert load_document(data, limits=replace(InputLimitsConfig(), max_input_bytes=len(data)))
    with pytest.raises(ResourceLimitError):
        load_document(data, limits=replace(InputLimitsConfig(), max_input_bytes=len(data) - 1))


def test_image_pixel_limit_and_pillow_global_protection_are_preserved() -> None:
    data = image_bytes(Image.new("RGB", (11, 10), "white"), "PNG")
    previous_limit = Image.MAX_IMAGE_PIXELS

    with pytest.raises(ResourceLimitError):
        load_document(
            data,
            limits=replace(InputLimitsConfig(), max_pixels_per_page=109, max_document_pixels=109),
        )

    assert previous_limit == Image.MAX_IMAGE_PIXELS


def test_invalid_sources_and_display_names_do_not_expose_paths(tmp_path: Path) -> None:
    missing = tmp_path / "sensitive" / "missing.png"

    with pytest.raises(DocumentLoadError) as captured:
        load_document(missing)
    assert str(missing) not in str(captured.value)
    with pytest.raises(DocumentLoadError):
        load_document(b"data", display_name=str(missing))
    with pytest.raises(DocumentLoadError):
        load_document(b"")


def test_pdf_rasterization_preserves_order_geometry_and_metadata() -> None:
    data = pdf_bytes([(72.0, 36.0), (100.1, 200.1)])

    pages = load_document(data, pdf_dpi=72, display_name="document.pdf")

    assert [(page.page_index, page.source_page_number) for page in pages] == [(0, 1), (1, 2)]
    assert pages[0].image.shape == (36, 72, 3)
    assert pages[1].image.shape == (201, 101, 3)
    assert all(page.image.dtype == np.uint8 and page.image.flags.owndata for page in pages)
    assert all(page.raster_dpi == 72 for page in pages)
    assert all(page.source.source_type is SourceType.PDF for page in pages)


@pytest.mark.parametrize(
    ("dpi", "expected_shape"),
    [(72, (5, 10, 3)), (200, (14, 28, 3)), (400, (28, 56, 3))],
)
def test_pdf_min_default_and_max_dpi(dpi: int, expected_shape: tuple[int, int, int]) -> None:
    supplied_dpi = None if dpi == 200 else dpi

    page = load_document(pdf_bytes([(10.0, 5.0)]), pdf_dpi=supplied_dpi)[0]

    assert page.image.shape == expected_shape
    assert page.raster_dpi == dpi


def test_pdf_point_conversion_uses_ceiling() -> None:
    assert _points_to_pixels(72.0, 200) == 200
    assert _points_to_pixels(100.1, 200) == 279
    assert _points_to_pixels(0.1, 72) == 1


def test_pdf_page_count_is_checked_before_render(monkeypatch: pytest.MonkeyPatch) -> None:
    data = pdf_bytes([(20.0, 20.0), (20.0, 20.0)])
    render_called = False

    def unexpected_render(*args: object, **kwargs: object) -> object:
        nonlocal render_called
        render_called = True
        raise AssertionError("render must not be called")

    monkeypatch.setattr(pdfium.PdfPage, "render", unexpected_render)
    with pytest.raises(ResourceLimitError):
        load_document(data, pdf_dpi=72, limits=replace(InputLimitsConfig(), max_pages=1))
    assert not render_called


def test_pdf_document_closes_when_preflight_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeDocument:
        closed = False

        def __len__(self) -> int:
            return 2

        def close(self) -> None:
            self.closed = True

    fake = FakeDocument()
    monkeypatch.setattr(ingestion, "_open_pdf", lambda data: fake)

    with pytest.raises(ResourceLimitError) as captured:
        load_document(b"%PDF-1.7\nplaceholder", limits=replace(InputLimitsConfig(), max_pages=1))

    assert fake.closed
    assert captured.value.context == {"resource": "pdf_pages", "actual": 2, "limit": 1}


def test_pdf_pixel_limits_are_preflighted_before_render(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = pdf_bytes([(100.0, 100.0)])
    render_called = False

    def unexpected_render(*args: object, **kwargs: object) -> object:
        nonlocal render_called
        render_called = True
        raise AssertionError("render must not be called")

    monkeypatch.setattr(pdfium.PdfPage, "render", unexpected_render)
    limits = replace(InputLimitsConfig(), max_pixels_per_page=9_999, max_document_pixels=10_000)
    with pytest.raises(ResourceLimitError):
        load_document(data, pdf_dpi=72, limits=limits)
    assert not render_called


def test_pdf_cumulative_limit_is_preflighted_before_render(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = pdf_bytes([(100.0, 100.0), (100.0, 100.0)])
    render_called = False

    def unexpected_render(*args: object, **kwargs: object) -> object:
        nonlocal render_called
        render_called = True
        raise AssertionError("render must not be called")

    monkeypatch.setattr(pdfium.PdfPage, "render", unexpected_render)
    limits = replace(InputLimitsConfig(), max_pixels_per_page=10_000, max_document_pixels=15_000)
    with pytest.raises(ResourceLimitError):
        load_document(data, pdf_dpi=72, limits=limits)
    assert not render_called


@pytest.mark.parametrize("dpi", [71, 401, 72.5, True])
def test_invalid_pdf_dpi_is_rejected(dpi: object) -> None:
    with pytest.raises(ResourceLimitError):
        load_document(pdf_bytes([(10.0, 10.0)]), pdf_dpi=dpi)  # type: ignore[arg-type]


def test_pdf_dpi_is_rejected_for_image_input() -> None:
    data = image_bytes(Image.new("RGB", (2, 2), "white"), "PNG")

    with pytest.raises(DocumentLoadError):
        load_document(data, pdf_dpi=200)


def test_corrupt_pdf_is_typed_and_noninteractive() -> None:
    with pytest.raises(PdfRenderError):
        load_document(b"%PDF-1.7\ncorrupt")


def test_zero_page_pdf_is_rejected() -> None:
    with pytest.raises(PdfRenderError):
        load_document(pdf_bytes([]))


def test_password_required_pdf_error_is_mapped_without_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def password_failure(*args: object, **kwargs: object) -> object:
        raise pdfium.PdfiumError("decoder detail", err_code=4)

    monkeypatch.setattr(pdfium, "PdfDocument", password_failure)
    with pytest.raises(PdfPasswordError) as captured:
        load_document(b"%PDF-1.7\nplaceholder")

    assert captured.value.to_public_dict() == {
        "code": "pdf_password_required",
        "message": "password-protected PDFs are not supported",
        "context": {"stage": "open"},
    }
