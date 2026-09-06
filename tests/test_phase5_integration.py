from __future__ import annotations

import io

import numpy as np
import pypdfium2 as pdfium
from PIL import Image, ImageDraw

import urdu_document_ocr
from urdu_document_ocr import load_document, preprocess_page


def test_public_surface_contains_phase5_capabilities_without_future_stages() -> None:
    exported = set(urdu_document_ocr.__all__)

    assert {"load_document", "preprocess_page", "PreprocessingConfig"} <= exported
    assert not {"segment_lines", "recognize", "train", "evaluate", "serve"} & exported


def test_jpeg_bytes_to_preprocessed_page() -> None:
    image = Image.new("RGB", (160, 90), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((20, 20, 130, 25), fill="black")
    draw.rectangle((35, 50, 140, 55), fill="black")
    stream = io.BytesIO()
    image.save(stream, format="JPEG", quality=95)

    pages = load_document(stream.getvalue(), display_name="integration.jpg")
    processed = preprocess_page(pages[0])

    assert len(pages) == 1
    assert not processed.is_blank
    assert processed.grayscale.shape == (90, 160)
    assert processed.foreground_mask.dtype == np.bool_


def test_multipage_pdf_bytes_to_ordered_preprocessed_pages() -> None:
    stream = io.BytesIO()
    document = pdfium.PdfDocument.new()
    try:
        for _ in range(2):
            page = document.new_page(72, 36)
            page.close()
        document.save(stream)
    finally:
        document.close()

    pages = load_document(stream.getvalue(), pdf_dpi=72)
    processed = tuple(preprocess_page(page) for page in pages)

    assert [page.page.page_index for page in processed] == [0, 1]
    assert [page.page.source_page_number for page in processed] == [1, 2]
    assert all(page.is_blank for page in processed)
