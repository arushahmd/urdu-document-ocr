from __future__ import annotations

import io

import numpy as np
import pytest
from fixtures.page_patterns import mask_with_boxes, preprocessed_from_mask, rgb_from_mask
from PIL import Image

from urdu_document_ocr import (
    InferenceConfig,
    InputLimitsConfig,
    OCRPrediction,
    pipeline,
    recognize_document,
    recognize_page,
)
from urdu_document_ocr.errors import RecognitionError, ResourceLimitError, SegmentationError
from urdu_document_ocr.types import BoundingBox, LineRegion


class OrderedRecognizer:
    fingerprint = "e" * 64

    def __init__(self, page_texts: tuple[tuple[str, ...], ...]) -> None:
        self.page_texts = list(page_texts)
        self.calls: list[tuple[tuple[np.ndarray, ...], int]] = []

    def recognize_batch(
        self, images: tuple[np.ndarray, ...], *, batch_size: int
    ) -> tuple[OCRPrediction, ...]:
        self.calls.append((images, batch_size))
        texts = self.page_texts.pop(0)
        return tuple(OCRPrediction(text) for text in texts)


def _three_line_page(*, page_index: int = 0):
    mask = mask_with_boxes(
        height=180,
        width=320,
        boxes=((60, 20, 160, 10), (70, 70, 150, 10), (80, 120, 140, 10)),
    )
    return preprocessed_from_mask(mask, page_index=page_index)


def _png_bytes(rgb: np.ndarray) -> bytes:
    stream = io.BytesIO()
    Image.fromarray(rgb).save(stream, format="PNG")
    return stream.getvalue()


def test_recognize_page_extracts_grayscale_crops_and_keeps_order() -> None:
    page = _three_line_page()
    recognizer = OrderedRecognizer((("پہلی سطر", "دوسری سطر", "تیسری سطر"),))

    result = recognize_page(page, recognizer, inference_config=InferenceConfig(batch_size=2))

    assert result.assembled_text == "پہلی سطر\nدوسری سطر\nتیسری سطر"
    assert [line.region.reading_order_index for line in result.lines] == [0, 1, 2]
    assert [line.prediction.text for line in result.lines if line.prediction] == [
        "پہلی سطر",
        "دوسری سطر",
        "تیسری سطر",
    ]
    crops, batch_size = recognizer.calls[0]
    assert batch_size == 2
    assert len(crops) == 3
    assert all(crop.ndim == 2 and crop.dtype == np.uint8 for crop in crops)


def test_blank_page_preserves_identity_without_calling_recognizer() -> None:
    blank = preprocessed_from_mask(
        np.zeros((120, 200), dtype=np.bool_), page_index=2, is_blank=True
    )
    recognizer = OrderedRecognizer(())

    result = recognize_page(blank, recognizer)

    assert (result.page_index, result.source_page_number) == (2, 3)
    assert result.is_blank and result.lines == () and result.assembled_text == ""
    assert recognizer.calls == []


def test_document_pipeline_loads_preprocesses_segments_and_assembles_image() -> None:
    page = _three_line_page()
    recognizer = OrderedRecognizer((("الف", "ب", "ج"),))
    source = _png_bytes(rgb_from_mask(page.foreground_mask))

    result = recognize_document(source, recognizer, display_name="safe-input.png")

    assert len(result.pages) == 1
    assert result.assembled_text == "الف\nب\nج"
    assert result.pages[0].source_page_number == 1


def test_pipeline_fails_closed_on_recognizer_shape_and_runtime_failures() -> None:
    page = _three_line_page()

    class WrongCountRecognizer:
        def recognize_batch(self, images, *, batch_size):
            return (OCRPrediction("صرف ایک"),)

    class BrokenRecognizer:
        def recognize_batch(self, images, *, batch_size):
            raise RuntimeError("private model detail")

    class ListRecognizer:
        def recognize_batch(self, images, *, batch_size):
            return [OCRPrediction("not immutable") for _image in images]

    with pytest.raises(RecognitionError, match="count"):
        recognize_page(page, WrongCountRecognizer())
    with pytest.raises(RecognitionError, match="line recognition failed") as captured:
        recognize_page(page, BrokenRecognizer())
    assert "private model detail" not in str(captured.value)
    with pytest.raises(RecognitionError, match="tuple"):
        recognize_page(page, ListRecognizer())
    with pytest.raises(TypeError, match="recognize_batch"):
        recognize_page(page, object())


def test_pipeline_propagates_structural_segmentation_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(_page, _config):
        raise SegmentationError("structural segmentation failure")

    monkeypatch.setattr(pipeline, "segment_page", fail)
    with pytest.raises(SegmentationError, match="structural"):
        recognize_page(_three_line_page(), OrderedRecognizer(()))


def test_pipeline_rejects_invalid_page_order_and_config_types(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    page = _three_line_page()
    wrong_order = (LineRegion("wrong-order", 0, BoundingBox(0, 0, 10, 5), None, 1),)
    monkeypatch.setattr(pipeline, "segment_page", lambda _page, _config: wrong_order)
    with pytest.raises(SegmentationError, match="reading-order"):
        recognize_page(page, OrderedRecognizer(()))
    with pytest.raises(TypeError, match="PreprocessedPage"):
        recognize_page(object(), OrderedRecognizer(()))  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="inference_config"):
        recognize_page(page, OrderedRecognizer(()), inference_config=object())  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "overrides",
    [
        {"input_limits": object()},
        {"preprocessing_config": object()},
        {"segmentation_config": object()},
        {"inference_config": object()},
    ],
)
def test_document_pipeline_rejects_wrong_configuration_types(overrides) -> None:
    with pytest.raises(TypeError):
        recognize_document(b"unused", OrderedRecognizer(()), **overrides)


def test_document_pipeline_reuses_ingestion_limits() -> None:
    page = _three_line_page()
    source = _png_bytes(rgb_from_mask(page.foreground_mask))
    limits = InputLimitsConfig(max_input_bytes=len(source) - 1)

    with pytest.raises(ResourceLimitError):
        recognize_document(source, OrderedRecognizer(()), input_limits=limits)
