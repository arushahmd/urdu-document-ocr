"""Typed, privacy-safe failures exposed by document and vision operations."""

from __future__ import annotations

from collections.abc import Mapping
from typing import ClassVar, TypeAlias

PublicContextValue: TypeAlias = str | int | float | bool | None


class UrduOCRError(Exception):
    """Base error with a stable public machine code and content-free message."""

    code: ClassVar[str] = "urdu_ocr_error"

    def __init__(
        self,
        message: str,
        *,
        context: Mapping[str, PublicContextValue] | None = None,
    ) -> None:
        super().__init__(message)
        self.context = {} if context is None else dict(context)

    def to_public_dict(self) -> dict[str, object]:
        return {"code": self.code, "message": str(self), "context": dict(self.context)}


class DocumentLoadError(UrduOCRError):
    """A supported document could not be opened or decoded."""

    code = "document_load_error"


class UnsupportedDocumentFormatError(DocumentLoadError):
    """Input is not one of the explicitly supported document formats."""

    code = "unsupported_document_format"


class ResourceLimitError(DocumentLoadError):
    """Input exceeds a configured byte, page, pixel, or DPI limit."""

    code = "resource_limit_exceeded"


class PdfRenderError(DocumentLoadError):
    """A PDF could not be opened, inspected, or rasterized safely."""

    code = "pdf_render_error"


class PdfPasswordError(PdfRenderError):
    """A password-protected PDF is unsupported by the V1 loader."""

    code = "pdf_password_required"


class PreprocessingError(UrduOCRError):
    """A valid page could not be preprocessed."""

    code = "preprocessing_error"


class SegmentationError(UrduOCRError):
    """A structurally valid preprocessed page could not be segmented."""

    code = "segmentation_error"


class ManifestError(UrduOCRError):
    """A JSONL manifest violates the public schema or safe I/O policy."""

    code = "manifest_error"


class DatasetValidationError(UrduOCRError):
    """Dataset validation could not safely begin."""

    code = "dataset_validation_error"


class DatasetSplitError(UrduOCRError):
    """A dataset cannot be assigned to deterministic document groups."""

    code = "dataset_split_error"


class VocabularyError(UrduOCRError):
    """A vocabulary artifact or source transcription is invalid."""

    code = "vocabulary_error"


class ReviewOverlayError(UrduOCRError):
    """A review overlay is invalid, conflicting, unknown, or stale."""

    code = "review_overlay_error"


class SyntheticDataError(UrduOCRError):
    """Synthetic rendering or fixture generation failed a safety contract."""

    code = "synthetic_data_error"


class RecognitionError(UrduOCRError):
    """Recognition input or computation violates the public model contract."""

    code = "recognition_error"


class ModelInputError(RecognitionError):
    """A tensor or line image violates the recognizer input contract."""

    code = "model_input_error"


class CTCAlignmentError(RecognitionError):
    """CTC labels, lengths, or alignment feasibility are invalid."""

    code = "ctc_alignment_error"


class DecodingError(RecognitionError):
    """Recognizer logits cannot be decoded with the supplied vocabulary."""

    code = "decoding_error"


class TrainingError(UrduOCRError):
    """A training operation cannot continue safely."""

    code = "training_error"


class TrainingDataError(TrainingError):
    """Training data violates a required image, split, or CTC invariant."""

    code = "training_data_error"


class CheckpointError(TrainingError):
    """A checkpoint cannot be saved or loaded without violating integrity rules."""

    code = "checkpoint_error"


class AssemblyError(UrduOCRError):
    """OCR results cannot be assembled or written without losing structure."""

    code = "assembly_error"
