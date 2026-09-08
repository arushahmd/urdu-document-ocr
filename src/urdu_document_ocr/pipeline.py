"""Thin orchestration from bounded documents to immutable OCR results."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from urdu_document_ocr.config import (
    InferenceConfig,
    InputLimitsConfig,
    PreprocessingConfig,
    SegmentationConfig,
)
from urdu_document_ocr.document import assemble_document, assemble_page, load_document
from urdu_document_ocr.errors import RecognitionError, SegmentationError, UrduOCRError
from urdu_document_ocr.types import (
    DocumentOCRResult,
    LineOCRResult,
    OCRPrediction,
    PageOCRResult,
    PreprocessedPage,
)
from urdu_document_ocr.vision import extract_line_crop, preprocess_page, segment_page

if TYPE_CHECKING:
    from urdu_document_ocr.document.ingestion import DocumentSource
    from urdu_document_ocr.recognition.base import OCRRecognizer

_LOGGER = logging.getLogger(__name__)


def recognize_page(
    page: PreprocessedPage,
    recognizer: OCRRecognizer,
    *,
    segmentation_config: SegmentationConfig | None = None,
    inference_config: InferenceConfig | None = None,
) -> PageOCRResult:
    """Segment and recognize one already-preprocessed page, failing closed."""

    if not isinstance(page, PreprocessedPage):
        raise TypeError("page must be a PreprocessedPage")
    active_inference = InferenceConfig() if inference_config is None else inference_config
    if not isinstance(active_inference, InferenceConfig):
        raise TypeError("inference_config must be an InferenceConfig or None")

    regions = segment_page(page, segmentation_config)
    expected_order = list(range(len(regions)))
    if [region.reading_order_index for region in regions] != expected_order:
        raise SegmentationError("segmentation did not return contiguous reading-order indices")
    if not regions:
        result = assemble_page(page, ())
        _LOGGER.info(
            "OCR page assembled",
            extra={"source_page_number": page.page.source_page_number, "line_count": 0},
        )
        return result

    crops = tuple(extract_line_crop(page, region, source="grayscale") for region in regions)
    method = getattr(recognizer, "recognize_batch", None)
    if not callable(method):
        raise TypeError("recognizer must implement OCRRecognizer.recognize_batch")
    try:
        predictions = method(crops, batch_size=active_inference.batch_size)
    except UrduOCRError:
        raise
    except Exception as error:
        raise RecognitionError("line recognition failed") from error
    if not isinstance(predictions, tuple) or any(
        not isinstance(prediction, OCRPrediction) for prediction in predictions
    ):
        raise RecognitionError("recognizer must return a tuple of OCRPrediction values")
    if len(predictions) != len(regions):
        raise RecognitionError("recognizer output count does not match segmented line count")

    line_results = tuple(
        LineOCRResult(region=region, prediction=prediction)
        for region, prediction in zip(regions, predictions, strict=True)
    )
    result = assemble_page(page, line_results)
    _LOGGER.info(
        "OCR page assembled",
        extra={
            "source_page_number": page.page.source_page_number,
            "line_count": len(line_results),
        },
    )
    return result


def recognize_document(
    source: DocumentSource,
    recognizer: OCRRecognizer,
    *,
    input_limits: InputLimitsConfig | None = None,
    pdf_dpi: int | None = None,
    display_name: str | None = None,
    preprocessing_config: PreprocessingConfig | None = None,
    segmentation_config: SegmentationConfig | None = None,
    inference_config: InferenceConfig | None = None,
) -> DocumentOCRResult:
    """Load, preprocess, recognize, and assemble a document page by page."""

    limits = InputLimitsConfig() if input_limits is None else input_limits
    preprocessing = PreprocessingConfig() if preprocessing_config is None else preprocessing_config
    segmentation = SegmentationConfig() if segmentation_config is None else segmentation_config
    inference = InferenceConfig() if inference_config is None else inference_config
    if not isinstance(limits, InputLimitsConfig):
        raise TypeError("input_limits must be an InputLimitsConfig or None")
    if not isinstance(preprocessing, PreprocessingConfig):
        raise TypeError("preprocessing_config must be a PreprocessingConfig or None")
    if not isinstance(segmentation, SegmentationConfig):
        raise TypeError("segmentation_config must be a SegmentationConfig or None")
    if not isinstance(inference, InferenceConfig):
        raise TypeError("inference_config must be an InferenceConfig or None")

    pages = load_document(
        source,
        limits=limits,
        pdf_dpi=pdf_dpi,
        display_name=display_name,
    )
    _LOGGER.info("OCR document loaded", extra={"page_count": len(pages)})
    page_results: list[PageOCRResult] = []
    for page in pages:
        preprocessed = preprocess_page(page, preprocessing)
        page_results.append(
            recognize_page(
                preprocessed,
                recognizer,
                segmentation_config=segmentation,
                inference_config=inference,
            )
        )
    return assemble_document(page_results)


__all__ = ["recognize_document", "recognize_page"]
