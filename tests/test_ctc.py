from __future__ import annotations

import pytest
import torch

from urdu_document_ocr import (
    CTCAlignmentError,
    compute_ctc_loss,
    minimum_ctc_timesteps,
    validate_ctc_alignment,
)


@pytest.mark.parametrize(
    ("target", "expected"),
    [
        ([1, 2, 3], 3),
        ([1, 1], 3),
        ([1, 1, 1], 5),
        ([1, 1, 2, 2, 2], 8),
        (torch.tensor([1, 2, 2], dtype=torch.int64), 4),
    ],
)
def test_minimum_ctc_timesteps_counts_adjacent_repeats(target: object, expected: int) -> None:
    assert minimum_ctc_timesteps(target) == expected  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "target",
    [[], [0], [1, -1], [1, True], [1, 2.0], "12", torch.tensor([1], dtype=torch.int32)],
)
def test_minimum_ctc_timesteps_rejects_invalid_targets(target: object) -> None:
    with pytest.raises(CTCAlignmentError):
        minimum_ctc_timesteps(target)  # type: ignore[arg-type]


def test_alignment_report_surfaces_exact_boundary_and_one_below() -> None:
    targets = torch.tensor([1, 2, 1, 1, 2, 2, 2], dtype=torch.int64)
    target_lengths = torch.tensor([2, 2, 3], dtype=torch.int64)
    input_lengths = torch.tensor([2, 2, 4], dtype=torch.int64)

    report = validate_ctc_alignment(
        targets,
        target_lengths,
        input_lengths,
        class_count=3,
        max_input_timesteps=4,
    )

    assert report.minimum_timesteps == (2, 3, 5)
    assert report.impossible_sample_indices == (1, 2)
    assert not report.is_feasible


def test_alignment_exact_repeated_minimum_is_feasible() -> None:
    report = validate_ctc_alignment(
        torch.tensor([1, 1], dtype=torch.int64),
        torch.tensor([2], dtype=torch.int64),
        torch.tensor([3], dtype=torch.int64),
        class_count=3,
        max_input_timesteps=3,
    )

    assert report.minimum_timesteps == (3,)
    assert report.is_feasible


@pytest.mark.parametrize(
    ("targets", "target_lengths", "input_lengths", "class_count", "maximum"),
    [
        (torch.tensor([0]), torch.tensor([1]), torch.tensor([1]), 2, 1),
        (torch.tensor([2]), torch.tensor([1]), torch.tensor([1]), 2, 1),
        (torch.tensor([1], dtype=torch.int32), torch.tensor([1]), torch.tensor([1]), 2, 1),
        (torch.tensor([1]), torch.tensor([0]), torch.tensor([1]), 2, 1),
        (torch.tensor([1]), torch.tensor([1]), torch.tensor([0]), 2, 1),
        (torch.tensor([1]), torch.tensor([2]), torch.tensor([2]), 2, 2),
        (torch.tensor([1]), torch.tensor([1, 1]), torch.tensor([1]), 2, 1),
        (torch.tensor([1]), torch.tensor([1]), torch.tensor([2]), 2, 1),
        (torch.tensor([1]), torch.tensor([1]), torch.tensor([1]), 1, 1),
    ],
)
def test_alignment_validation_rejects_malformed_contracts(
    targets: torch.Tensor,
    target_lengths: torch.Tensor,
    input_lengths: torch.Tensor,
    class_count: int,
    maximum: int,
) -> None:
    with pytest.raises(CTCAlignmentError):
        validate_ctc_alignment(
            targets,
            target_lengths,
            input_lengths,
            class_count=class_count,
            max_input_timesteps=maximum,
        )


def test_compute_ctc_loss_returns_scalar_with_finite_gradients() -> None:
    logits = torch.randn(2, 6, 4, requires_grad=True)
    targets = torch.tensor([1, 2, 3], dtype=torch.int64)
    target_lengths = torch.tensor([2, 1], dtype=torch.int64)
    input_lengths = torch.tensor([6, 5], dtype=torch.int64)

    loss = compute_ctc_loss(logits, targets, target_lengths, input_lengths)
    loss.backward()

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_compute_ctc_loss_rejects_impossible_alignment_before_loss() -> None:
    with pytest.raises(CTCAlignmentError, match="impossible") as caught:
        compute_ctc_loss(
            torch.randn(1, 2, 3),
            torch.tensor([1, 1], dtype=torch.int64),
            torch.tensor([2], dtype=torch.int64),
            torch.tensor([2], dtype=torch.int64),
        )

    assert caught.value.context == {"impossible_count": 1, "first_sample_index": 0}


@pytest.mark.parametrize(
    "logits",
    [
        torch.zeros(2, 3),
        torch.zeros(1, 3, 1),
        torch.zeros(1, 3, 3, dtype=torch.int64),
        torch.full((1, 3, 3), float("nan")),
    ],
)
def test_compute_ctc_loss_rejects_invalid_logits(logits: torch.Tensor) -> None:
    with pytest.raises(CTCAlignmentError):
        compute_ctc_loss(
            logits,
            torch.tensor([1]),
            torch.tensor([1]),
            torch.tensor([1]),
        )
