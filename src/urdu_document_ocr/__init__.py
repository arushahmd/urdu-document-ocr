"""Public package metadata for the Urdu document OCR reimplementation."""

import logging as _logging

from urdu_document_ocr.config import (
    InputLimitsConfig,
    MorphologyOperation,
    PreprocessingConfig,
    SegmentationConfig,
    ThresholdMethod,
)
from urdu_document_ocr.document import load_document
from urdu_document_ocr.errors import (
    DocumentLoadError,
    PdfPasswordError,
    PdfRenderError,
    PreprocessingError,
    ResourceLimitError,
    SegmentationError,
    UnsupportedDocumentFormatError,
    UrduOCRError,
)
from urdu_document_ocr.types import PageImage, PreprocessedPage, SourceMetadata, SourceType
from urdu_document_ocr.vision import extract_line_crop, preprocess_page, segment_page

__version__ = "0.1.0"

_logging.getLogger(__name__).addHandler(_logging.NullHandler())

__all__ = [
    "DocumentLoadError",
    "InputLimitsConfig",
    "MorphologyOperation",
    "PageImage",
    "PdfPasswordError",
    "PdfRenderError",
    "PreprocessedPage",
    "PreprocessingConfig",
    "PreprocessingError",
    "ResourceLimitError",
    "SegmentationConfig",
    "SegmentationError",
    "SourceMetadata",
    "SourceType",
    "ThresholdMethod",
    "UnsupportedDocumentFormatError",
    "UrduOCRError",
    "__version__",
    "extract_line_crop",
    "load_document",
    "preprocess_page",
    "segment_page",
]
