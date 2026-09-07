from __future__ import annotations

import pytest

from urdu_document_ocr.types import BoundingBox
from urdu_document_ocr.vision.geometry import (
    horizontal_gap,
    interval_gap,
    interval_overlap_length,
    padded_box,
    union_boxes,
    vertical_center_distance,
    vertical_overlap_ratio,
)


def test_interval_overlap_and_gap_use_half_open_coordinates() -> None:
    assert interval_overlap_length(0, 5, 3, 8) == 2
    assert interval_overlap_length(0, 5, 5, 8) == 0
    assert interval_gap(0, 5, 5, 8) == 0
    assert interval_gap(0, 5, 8, 10) == 3
    assert interval_gap(8, 10, 0, 5) == 3


@pytest.mark.parametrize("arguments", [(5, 4, 0, 1), (0, 1, 5, 4)])
def test_invalid_intervals_are_rejected(arguments: tuple[int, int, int, int]) -> None:
    with pytest.raises(ValueError):
        interval_overlap_length(*arguments)
    with pytest.raises(ValueError):
        interval_gap(*arguments)


def test_box_overlap_gap_and_center_distance() -> None:
    first = BoundingBox(10, 10, 20, 10)
    second = BoundingBox(35, 15, 10, 20)

    assert horizontal_gap(first, second) == 5
    assert vertical_overlap_ratio(first, second) == 0.5
    assert vertical_center_distance(first, second) == 10.0


def test_union_boxes_contains_every_input() -> None:
    combined = union_boxes((BoundingBox(10, 20, 5, 7), BoundingBox(3, 25, 20, 4)))

    assert combined == BoundingBox(3, 20, 20, 9)
    with pytest.raises(ValueError):
        union_boxes(())
    with pytest.raises(TypeError):
        union_boxes((BoundingBox(0, 0, 1, 1), "bad"))  # type: ignore[arg-type]


def test_padding_clips_to_page_and_retains_half_open_box() -> None:
    assert padded_box(BoundingBox(1, 2, 5, 4), 3, image_width=8, image_height=7) == BoundingBox(
        0, 0, 8, 7
    )
    assert padded_box(BoundingBox(3, 3, 2, 2), 0, image_width=10, image_height=10) == BoundingBox(
        3, 3, 2, 2
    )


@pytest.mark.parametrize(
    "arguments",
    [
        {"padding": -1, "image_width": 10, "image_height": 10},
        {"padding": 1.5, "image_width": 10, "image_height": 10},
        {"padding": 0, "image_width": 0, "image_height": 10},
    ],
)
def test_padding_rejects_invalid_bounds(arguments: dict[str, object]) -> None:
    with pytest.raises((TypeError, ValueError)):
        padded_box(BoundingBox(1, 1, 2, 2), **arguments)  # type: ignore[arg-type]
