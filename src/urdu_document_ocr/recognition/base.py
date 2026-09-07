"""Typed recognition boundary shared by the model, CTC, and future inference layer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch
from torch import Tensor

from urdu_document_ocr.config import RecognizerConfig
from urdu_document_ocr.errors import ModelInputError
from urdu_document_ocr.types import Vocabulary


@dataclass(frozen=True, slots=True)
class RecognizerOutput:
    """Raw batch-first CTC logits and their valid sequence lengths."""

    logits: Tensor
    input_lengths: Tensor

    def __post_init__(self) -> None:
        if not isinstance(self.logits, Tensor) or self.logits.ndim != 3:
            raise ModelInputError("logits must be a rank-3 PyTorch tensor")
        if not self.logits.is_floating_point():
            raise ModelInputError("logits must have a floating-point dtype")
        if not isinstance(self.input_lengths, Tensor) or self.input_lengths.ndim != 1:
            raise ModelInputError("input_lengths must be a rank-1 PyTorch tensor")
        if self.input_lengths.dtype != torch.int64 or self.input_lengths.device.type != "cpu":
            raise ModelInputError("input_lengths must be a CPU int64 tensor")
        if self.input_lengths.shape[0] != self.logits.shape[0]:
            raise ModelInputError("input_lengths must match the logits batch size")
        if torch.any(self.input_lengths <= 0) or torch.any(
            self.input_lengths > self.logits.shape[1]
        ):
            raise ModelInputError("input_lengths must be positive and within logits time")


class OCRRecognizer(Protocol):
    """Structural contract for a vocabulary-bound trainable logit recognizer."""

    config: RecognizerConfig
    vocabulary: Vocabulary

    @property
    def fingerprint(self) -> str:
        """Return the architecture/configuration/vocabulary identity."""

    def __call__(self, images: Tensor, valid_widths: Tensor) -> RecognizerOutput:
        """Produce raw CTC logits without applying softmax or decoding."""
