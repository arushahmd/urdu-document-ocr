"""Public OCR dataset engineering APIs."""

from urdu_document_ocr.data.manifest import (
    dataset_fingerprint,
    read_manifest,
    write_manifest,
)
from urdu_document_ocr.data.review import (
    ReviewDecision,
    ReviewStatus,
    apply_review_overlay,
    read_review_overlay,
    review_overlay_fingerprint,
    write_review_overlay,
)
from urdu_document_ocr.data.splitting import (
    DatasetPartition,
    DatasetSplitResult,
    SplitCounts,
    split_dataset,
    write_split_manifests,
)
from urdu_document_ocr.data.validation import (
    DatasetStatistics,
    DatasetValidationReport,
    NumericSummary,
    ValidationIssue,
    ValidationSeverity,
    normalize_transcription,
    validate_dataset,
)
from urdu_document_ocr.data.vocabulary import (
    build_vocabulary,
    find_unseen_characters,
    load_vocabulary,
    save_vocabulary,
)

__all__ = [
    "DatasetPartition",
    "DatasetSplitResult",
    "DatasetStatistics",
    "DatasetValidationReport",
    "NumericSummary",
    "ReviewDecision",
    "ReviewStatus",
    "SplitCounts",
    "ValidationIssue",
    "ValidationSeverity",
    "apply_review_overlay",
    "build_vocabulary",
    "dataset_fingerprint",
    "find_unseen_characters",
    "load_vocabulary",
    "normalize_transcription",
    "read_manifest",
    "read_review_overlay",
    "review_overlay_fingerprint",
    "save_vocabulary",
    "split_dataset",
    "validate_dataset",
    "write_manifest",
    "write_review_overlay",
    "write_split_manifests",
]
