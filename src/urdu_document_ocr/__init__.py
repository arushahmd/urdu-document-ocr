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
    TrainingConfig,
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
    CheckpointError,
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
    TrainingDataError,
    TrainingError,
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

_TRAINING_EXPORTS = frozenset(
    {
        "CHECKPOINT_SCHEMA_VERSION",
        "CheckpointMetadata",
        "OCRBatch",
        "OCRLineDataset",
        "OCRLineItem",
        "OCRTrainer",
        "TrainingEpochRecord",
        "TrainingPreflightReport",
        "TrainingResult",
        "collate_ocr_batch",
        "load_checkpoint",
        "save_checkpoint",
        "set_training_seed",
        "train_model",
        "validate_training_data",
    }
)


def __getattr__(name: str) -> object:
    """Load optional recognition or training objects only when requested."""

    if name not in _RECOGNITION_EXPORTS | _TRAINING_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    try:
        if name in _RECOGNITION_EXPORTS:
            from urdu_document_ocr import recognition as optional_module
        else:
            from urdu_document_ocr import training as optional_module
    except ModuleNotFoundError as error:
        if error.name in {"safetensors", "torch"}:
            raise ImportError(
                "PyTorch recognition and training support is optional; "
                "install urdu-document-ocr[ml]"
            ) from error
        raise
    value = getattr(optional_module, name)
    globals()[name] = value
    return value


__all__ = [
    "CHECKPOINT_SCHEMA_VERSION",
    "CRNNRecognizer",
    "CTCAlignmentError",
    "CTCAlignmentReport",
    "CheckpointError",
    "CheckpointMetadata",
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
    "OCRBatch",
    "OCRLineDataset",
    "OCRLineItem",
    "OCRRecognizer",
    "OCRTrainer",
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
    "TrainingConfig",
    "TrainingDataError",
    "TrainingEpochRecord",
    "TrainingError",
    "TrainingPreflightReport",
    "TrainingResult",
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
    "collate_ocr_batch",
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
    "load_checkpoint",
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
    "save_checkpoint",
    "save_vocabulary",
    "segment_page",
    "set_training_seed",
    "split_dataset",
    "train_model",
    "validate_ctc_alignment",
    "validate_dataset",
    "validate_training_data",
    "write_manifest",
    "write_review_overlay",
    "write_split_manifests",
]
