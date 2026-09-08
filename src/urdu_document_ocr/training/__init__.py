"""Optional PyTorch training and safe-checkpoint API."""

from importlib.util import find_spec

if find_spec("torch") is None or find_spec("safetensors") is None:
    raise ImportError("OCR training support is optional; install urdu-document-ocr[ml]")

from urdu_document_ocr.training.checkpoint import (
    CHECKPOINT_SCHEMA_VERSION,
    CheckpointMetadata,
    load_checkpoint,
    save_checkpoint,
)
from urdu_document_ocr.training.dataset import (
    OCRBatch,
    OCRLineDataset,
    OCRLineItem,
    TrainingPreflightReport,
    collate_ocr_batch,
    validate_training_data,
)
from urdu_document_ocr.training.trainer import (
    OCRTrainer,
    TrainingEpochRecord,
    TrainingResult,
    set_training_seed,
    train_model,
)

__all__ = [
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
]
