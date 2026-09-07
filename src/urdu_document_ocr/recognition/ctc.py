"""CTC loss and explicit alignment-feasibility validation."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise

import torch
from torch import Tensor, nn

from urdu_document_ocr.errors import CTCAlignmentError


@dataclass(frozen=True, slots=True)
class CTCAlignmentReport:
    """Minimum required timesteps and zero-based impossible sample indexes."""

    minimum_timesteps: tuple[int, ...]
    impossible_sample_indices: tuple[int, ...]

    @property
    def is_feasible(self) -> bool:
        return not self.impossible_sample_indices


def _cpu_int64_vector(name: str, value: Tensor) -> Tensor:
    if not isinstance(value, Tensor) or value.ndim != 1:
        raise CTCAlignmentError(f"{name} must be a rank-1 PyTorch tensor")
    if value.dtype != torch.int64 or value.device.type != "cpu":
        raise CTCAlignmentError(f"{name} must be a CPU int64 tensor")
    return value


def minimum_ctc_timesteps(target: Sequence[int] | Tensor) -> int:
    """Return target length plus one timestep per adjacent repeated label."""

    if isinstance(target, Tensor):
        values = _cpu_int64_vector("target", target).tolist()
    elif isinstance(target, Sequence) and not isinstance(target, (str, bytes, bytearray)):
        values = list(target)
    else:
        raise CTCAlignmentError("target must be a one-dimensional integer sequence")
    if not values:
        raise CTCAlignmentError("target must contain at least one character index")
    if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
        raise CTCAlignmentError("target must contain only integer character indexes")
    if any(value <= 0 for value in values):
        raise CTCAlignmentError("target character indexes must be positive and exclude blank")
    repeats = sum(left == right for left, right in pairwise(values))
    return len(values) + repeats


def validate_ctc_alignment(
    targets: Tensor,
    target_lengths: Tensor,
    input_lengths: Tensor,
    *,
    class_count: int,
    max_input_timesteps: int | None = None,
    blank_index: int = 0,
) -> CTCAlignmentReport:
    """Validate concatenated targets and identify samples impossible under CTC."""

    targets = _cpu_int64_vector("targets", targets)
    target_lengths = _cpu_int64_vector("target_lengths", target_lengths)
    input_lengths = _cpu_int64_vector("input_lengths", input_lengths)
    if isinstance(class_count, bool) or not isinstance(class_count, int) or class_count < 2:
        raise CTCAlignmentError("class_count must be an integer of at least 2")
    if blank_index != 0:
        raise CTCAlignmentError("CTC blank_index must be 0")
    if target_lengths.shape != input_lengths.shape or target_lengths.numel() < 1:
        raise CTCAlignmentError("target_lengths and input_lengths must match a nonempty batch")
    if torch.any(target_lengths <= 0):
        raise CTCAlignmentError("V1 target lengths must be positive")
    if torch.any(input_lengths <= 0):
        raise CTCAlignmentError("input lengths must be positive")
    if max_input_timesteps is not None:
        if (
            isinstance(max_input_timesteps, bool)
            or not isinstance(max_input_timesteps, int)
            or max_input_timesteps < 1
        ):
            raise CTCAlignmentError("max_input_timesteps must be a positive integer")
        if torch.any(input_lengths > max_input_timesteps):
            raise CTCAlignmentError("input length exceeds the logits time dimension")
    if int(target_lengths.sum().item()) != targets.numel():
        raise CTCAlignmentError("concatenated target count must equal sum(target_lengths)")
    if torch.any(targets <= blank_index):
        raise CTCAlignmentError("targets must exclude blank and contain positive indexes")
    if torch.any(targets >= class_count):
        raise CTCAlignmentError("target character index is outside the classifier classes")

    minimums: list[int] = []
    impossible: list[int] = []
    offset = 0
    for sample_index, (target_length, input_length) in enumerate(
        zip(target_lengths.tolist(), input_lengths.tolist(), strict=True)
    ):
        sample = targets[offset : offset + target_length]
        minimum = minimum_ctc_timesteps(sample)
        minimums.append(minimum)
        if input_length < minimum:
            impossible.append(sample_index)
        offset += target_length
    return CTCAlignmentReport(tuple(minimums), tuple(impossible))


def compute_ctc_loss(
    logits: Tensor,
    targets: Tensor,
    target_lengths: Tensor,
    input_lengths: Tensor,
    *,
    blank_index: int = 0,
) -> Tensor:
    """Compute mean CTC loss after rejecting malformed or impossible alignments."""

    if not isinstance(logits, Tensor) or logits.ndim != 3 or not logits.is_floating_point():
        raise CTCAlignmentError("logits must be a rank-3 floating-point PyTorch tensor")
    if logits.shape[0] < 1 or logits.shape[1] < 1 or logits.shape[2] < 2:
        raise CTCAlignmentError("logits dimensions must define a nonempty CTC batch")
    if not torch.isfinite(logits).all():
        raise CTCAlignmentError("logits must contain only finite values")
    report = validate_ctc_alignment(
        targets,
        target_lengths,
        input_lengths,
        class_count=logits.shape[2],
        max_input_timesteps=logits.shape[1],
        blank_index=blank_index,
    )
    if input_lengths.shape[0] != logits.shape[0]:
        raise CTCAlignmentError("length vectors must match the logits batch size")
    if not report.is_feasible:
        raise CTCAlignmentError(
            "CTC alignment is impossible for one or more samples",
            context={
                "impossible_count": len(report.impossible_sample_indices),
                "first_sample_index": report.impossible_sample_indices[0],
            },
        )
    criterion = nn.CTCLoss(blank=blank_index, reduction="mean", zero_infinity=True)
    log_probabilities = logits.log_softmax(dim=2).transpose(0, 1)
    return criterion(log_probabilities, targets, input_lengths, target_lengths)
