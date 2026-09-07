"""Public package metadata for the Urdu document OCR reimplementation."""

import logging as _logging

from urdu_document_ocr.config import (
    DatasetSplitConfig,
    InputLimitsConfig,
    MorphologyOperation,
    PreprocessingConfig,
    RecognizerConfig,
    SegmentationConfig,
    SyntheticDataConfig,
    ThresholdMethod,
)
from urdu_document_ocr.data import (
    DatasetPartition,
    DatasetSplitResult,
    DatasetStatistics,
    DatasetValidationReport,
    FixtureGenerationResult,
    GeneratedLineSample,
    GeneratedPageFixture,
    PageLayoutFamily,
    ReviewDecision,
    ReviewStatus,
    ShapingCapabilities,
    SplitCounts,
    SyntheticLineRecord,
    SyntheticPageLine,
    SyntheticPageRecord,
    TextSource,
    ValidationIssue,
    ValidationSeverity,
    apply_review_overlay,
    assert_shaping_available,
    build_vocabulary,
    bundled_font_path,
    dataset_fingerprint,
    find_unseen_characters,
    generate_fixture_dataset,
    generate_line_dataset,
    generate_line_sample,
    generate_page_fixture,
    load_font_provenance,
    load_vocabulary,
    normalize_transcription,
    read_manifest,
    read_review_overlay,
    review_overlay_fingerprint,
    save_vocabulary,
    split_dataset,
    validate_dataset,
    write_manifest,
    write_review_overlay,
    write_split_manifests,
)
from urdu_document_ocr.document import load_document
from urdu_document_ocr.errors import (
    CTCAlignmentError,
    DatasetSplitError,
    DatasetValidationError,
    DecodingError,
    DocumentLoadError,
    ManifestError,
    ModelInputError,
    PdfPasswordError,
    PdfRenderError,
    PreprocessingError,
    RecognitionError,
    ResourceLimitError,
    ReviewOverlayError,
    SegmentationError,
    SyntheticDataError,
    UnsupportedDocumentFormatError,
    UrduOCRError,
    VocabularyError,
)
from urdu_document_ocr.types import (
    DatasetSample,
    PageImage,
    PreprocessedPage,
    SourceMetadata,
    SourceType,
    Vocabulary,
)
from urdu_document_ocr.vision import extract_line_crop, preprocess_page, segment_page

__version__ = "0.1.0"

_logging.getLogger(__name__).addHandler(_logging.NullHandler())

_RECOGNITION_EXPORTS = frozenset(
    {
        "CRNNRecognizer",
        "CTCAlignmentReport",
        "OCRRecognizer",
        "RecognizerOutput",
        "compute_ctc_loss",
        "greedy_ctc_decode",
        "input_width_to_timesteps",
        "minimum_ctc_timesteps",
        "prepare_line_image",
        "validate_ctc_alignment",
    }
)


def __getattr__(name: str) -> object:
    """Load the optional recognition surface only when an ML object is requested."""

    if name not in _RECOGNITION_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    try:
        from urdu_document_ocr import recognition
    except ModuleNotFoundError as error:
        if error.name == "torch":
            raise ImportError(
                "PyTorch recognition support is optional; install urdu-document-ocr[ml]"
            ) from error
        raise
    value = getattr(recognition, name)
    globals()[name] = value
    return value


__all__ = [
    "CRNNRecognizer",
    "CTCAlignmentError",
    "CTCAlignmentReport",
    "DatasetPartition",
    "DatasetSample",
    "DatasetSplitConfig",
    "DatasetSplitError",
    "DatasetSplitResult",
    "DatasetStatistics",
    "DatasetValidationError",
    "DatasetValidationReport",
    "DecodingError",
    "DocumentLoadError",
    "FixtureGenerationResult",
    "GeneratedLineSample",
    "GeneratedPageFixture",
    "InputLimitsConfig",
    "ManifestError",
    "ModelInputError",
    "MorphologyOperation",
    "OCRRecognizer",
    "PageImage",
    "PageLayoutFamily",
    "PdfPasswordError",
    "PdfRenderError",
    "PreprocessedPage",
    "PreprocessingConfig",
    "PreprocessingError",
    "RecognitionError",
    "RecognizerConfig",
    "RecognizerOutput",
    "ResourceLimitError",
    "ReviewDecision",
    "ReviewOverlayError",
    "ReviewStatus",
    "SegmentationConfig",
    "SegmentationError",
    "ShapingCapabilities",
    "SourceMetadata",
    "SourceType",
    "SplitCounts",
    "SyntheticDataConfig",
    "SyntheticDataError",
    "SyntheticLineRecord",
    "SyntheticPageLine",
    "SyntheticPageRecord",
    "TextSource",
    "ThresholdMethod",
    "UnsupportedDocumentFormatError",
    "UrduOCRError",
    "ValidationIssue",
    "ValidationSeverity",
    "Vocabulary",
    "VocabularyError",
    "__version__",
    "apply_review_overlay",
    "assert_shaping_available",
    "build_vocabulary",
    "bundled_font_path",
    "compute_ctc_loss",
    "dataset_fingerprint",
    "extract_line_crop",
    "find_unseen_characters",
    "generate_fixture_dataset",
    "generate_line_dataset",
    "generate_line_sample",
    "generate_page_fixture",
    "greedy_ctc_decode",
    "input_width_to_timesteps",
    "load_document",
    "load_font_provenance",
    "load_vocabulary",
    "minimum_ctc_timesteps",
    "normalize_transcription",
    "prepare_line_image",
    "preprocess_page",
    "read_manifest",
    "read_review_overlay",
    "review_overlay_fingerprint",
    "save_vocabulary",
    "segment_page",
    "split_dataset",
    "validate_ctc_alignment",
    "validate_dataset",
    "write_manifest",
    "write_review_overlay",
    "write_split_manifests",
]
