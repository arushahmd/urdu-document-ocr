"""Checkpoint-backed line inference with ordered dynamic-width batches."""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from os import PathLike
from typing import TYPE_CHECKING

import numpy as np
import torch
from numpy.typing import NDArray

from urdu_document_ocr.config import RecognizerConfig
from urdu_document_ocr.errors import CheckpointError, RecognitionError
from urdu_document_ocr.recognition.base import OCRRecognizer
from urdu_document_ocr.recognition.decoding import greedy_ctc_decode
from urdu_document_ocr.recognition.model import (
    CRNNRecognizer,
    pad_prepared_line_images,
    prepare_line_image,
)
from urdu_document_ocr.types import OCRPrediction, Vocabulary

if TYPE_CHECKING:
    from urdu_document_ocr.training.checkpoint import CheckpointMetadata

_LOGGER = logging.getLogger(__name__)
_DEVICE_PATTERN = re.compile(r"(?:cpu|cuda(?::[0-9]+)?)")


def _validate_batch_size(batch_size: object) -> int:
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise RecognitionError("batch_size must be a positive integer")
    return batch_size


def _resolve_device(device: object) -> torch.device:
    if not isinstance(device, str) or _DEVICE_PATTERN.fullmatch(device) is None:
        raise RecognitionError("device must be cpu, cuda, or cuda:<index>")
    selected = torch.device(device)
    if selected.type == "cuda":
        if not torch.cuda.is_available():
            raise RecognitionError("the explicitly requested CUDA device is unavailable")
        index = torch.cuda.current_device() if selected.index is None else selected.index
        if index >= torch.cuda.device_count():
            raise RecognitionError("the explicitly requested CUDA device index is unavailable")
    return selected


@dataclass(slots=True)
class LoadedRecognizer:
    """One verified checkpoint kept in memory for repeated line and page inference."""

    model: CRNNRecognizer = field(repr=False)
    checkpoint_metadata: CheckpointMetadata
    device: str

    def __post_init__(self) -> None:
        if not isinstance(self.model, CRNNRecognizer):
            raise TypeError("model must be a CRNNRecognizer")
        from urdu_document_ocr.training.checkpoint import CheckpointMetadata

        if not isinstance(self.checkpoint_metadata, CheckpointMetadata):
            raise TypeError("checkpoint_metadata must be CheckpointMetadata")
        resolved = _resolve_device(self.device)
        if self.model.fingerprint != self.checkpoint_metadata.model_fingerprint:
            raise CheckpointError("loaded model fingerprint does not match checkpoint metadata")
        try:
            self.model.to(resolved)
        except (RuntimeError, ValueError) as error:
            raise RecognitionError(
                "recognizer could not be placed on the requested device"
            ) from error
        self.model.eval()
        self.device = str(resolved)

    @property
    def config(self) -> RecognizerConfig:
        return self.model.config

    @property
    def vocabulary(self) -> Vocabulary:
        return self.model.vocabulary

    @property
    def fingerprint(self) -> str:
        return self.model.fingerprint

    def recognize_batch(
        self,
        images: Sequence[NDArray[np.uint8]],
        *,
        batch_size: int,
    ) -> tuple[OCRPrediction, ...]:
        """Normalize, batch, run, and greedily decode lines without changing order."""

        active_batch_size = _validate_batch_size(batch_size)
        try:
            values = tuple(images)
        except TypeError as error:
            raise RecognitionError("images must be an iterable of grayscale line arrays") from error
        if not values:
            return ()

        predictions: list[OCRPrediction] = []
        for start in range(0, len(values), active_batch_size):
            current = values[start : start + active_batch_size]
            prepared = [prepare_line_image(image, self.config)[0] for image in current]
            batch, valid_widths = pad_prepared_line_images(
                prepared,
                max_width=self.config.max_width,
            )
            try:
                self.model.eval()
                with torch.inference_mode():
                    output = self.model(batch.to(self.device), valid_widths)
                    decoded = greedy_ctc_decode(
                        output.logits,
                        output.input_lengths,
                        self.vocabulary,
                    )
            except (CheckpointError, RecognitionError):
                raise
            except (RuntimeError, ValueError) as error:
                raise RecognitionError("checkpoint-backed line inference failed") from error
            if len(decoded) != len(current):
                raise RecognitionError("recognizer output count does not match the input batch")
            predictions.extend(decoded)
        return tuple(predictions)


def load_recognizer(
    checkpoint_directory: str | PathLike[str],
    *,
    device: str = "cpu",
) -> LoadedRecognizer:
    """Strictly verify a safetensors checkpoint and retain its model in memory."""

    from urdu_document_ocr.training.checkpoint import load_checkpoint_model

    selected = _resolve_device(device)
    model, metadata = load_checkpoint_model(checkpoint_directory)
    recognizer = LoadedRecognizer(model=model, checkpoint_metadata=metadata, device=str(selected))
    _LOGGER.info(
        "OCR checkpoint loaded",
        extra={"model_fingerprint": recognizer.fingerprint},
    )
    return recognizer


def recognize_lines(
    images: Sequence[NDArray[np.uint8]],
    recognizer: OCRRecognizer,
    *,
    batch_size: int = 16,
) -> tuple[OCRPrediction, ...]:
    """Recognize grayscale line crops in their supplied order."""

    active_batch_size = _validate_batch_size(batch_size)
    try:
        values = tuple(images)
    except TypeError as error:
        raise RecognitionError("images must be an iterable of grayscale line arrays") from error
    method = getattr(recognizer, "recognize_batch", None)
    if not callable(method):
        raise TypeError("recognizer must implement OCRRecognizer.recognize_batch")
    if not values:
        return ()
    predictions = method(values, batch_size=active_batch_size)
    if not isinstance(predictions, tuple) or any(
        not isinstance(item, OCRPrediction) for item in predictions
    ):
        raise RecognitionError("recognizer must return a tuple of OCRPrediction values")
    if len(predictions) != len(values):
        raise RecognitionError("recognizer output count does not match the input line count")
    return predictions


def recognize_line(
    image: NDArray[np.uint8],
    recognizer: OCRRecognizer,
) -> OCRPrediction:
    """Recognize one grayscale line through the reusable recognizer boundary."""

    return recognize_lines((image,), recognizer, batch_size=1)[0]


__all__ = ["LoadedRecognizer", "load_recognizer", "recognize_line", "recognize_lines"]
