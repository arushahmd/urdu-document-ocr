"""Vocabulary-bound greedy CTC decoding without confidence fabrication."""

from __future__ import annotations

import torch
from torch import Tensor

from urdu_document_ocr.errors import DecodingError
from urdu_document_ocr.types import OCRPrediction, Vocabulary


def greedy_ctc_decode(
    logits: Tensor,
    output_lengths: Tensor,
    vocabulary: Vocabulary,
) -> tuple[OCRPrediction, ...]:
    """Collapse repeats, remove blank zero, then map indexes through ``vocabulary``."""

    if not isinstance(vocabulary, Vocabulary):
        raise TypeError("vocabulary must be a Vocabulary")
    if not isinstance(logits, Tensor) or logits.ndim != 3 or not logits.is_floating_point():
        raise DecodingError("logits must be a rank-3 floating-point PyTorch tensor")
    if not torch.isfinite(logits).all():
        raise DecodingError("logits must contain only finite values")
    expected_classes = len(vocabulary.characters) + 1
    if logits.shape[2] != expected_classes:
        raise DecodingError("classifier class count does not match the vocabulary")
    if not isinstance(output_lengths, Tensor) or output_lengths.ndim != 1:
        raise DecodingError("output_lengths must be a rank-1 PyTorch tensor")
    if output_lengths.dtype != torch.int64 or output_lengths.device.type != "cpu":
        raise DecodingError("output_lengths must be a CPU int64 tensor")
    if output_lengths.shape[0] != logits.shape[0]:
        raise DecodingError("output_lengths must match the logits batch size")
    if torch.any(output_lengths < 0) or torch.any(output_lengths > logits.shape[1]):
        raise DecodingError("output length is outside the logits time dimension")

    best_paths = logits.argmax(dim=2).detach().cpu()
    predictions: list[OCRPrediction] = []
    for batch_index, length in enumerate(output_lengths.tolist()):
        path = best_paths[batch_index, :length].tolist()
        collapsed = [
            value for index, value in enumerate(path) if index == 0 or value != path[index - 1]
        ]
        character_indices = tuple(value for value in collapsed if value != vocabulary.blank_index)
        try:
            text = vocabulary.decode(character_indices)
        except (TypeError, ValueError) as error:
            raise DecodingError("decoded character index is outside the vocabulary") from error
        predictions.append(
            OCRPrediction(
                text=text,
                character_indices=character_indices,
                sequence_length=length,
            )
        )
    return tuple(predictions)
