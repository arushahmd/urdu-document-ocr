from __future__ import annotations

import pytest

from urdu_document_ocr.config import SegmentationConfig
from urdu_document_ocr.types import BoundingBox, RegionKind
from urdu_document_ocr.vision.layout import _infer_layout_and_order, _LineCandidate


def candidate(candidate_id: str, x: int, y: int, width: int, height: int = 8) -> _LineCandidate:
    return _LineCandidate(candidate_id, BoundingBox(x, y, width, height), width * height, 1)


def ids(assignments: tuple[object, ...]) -> list[str]:
    return [item.candidate.candidate_id for item in assignments]  # type: ignore[attr-defined]


def test_balanced_two_columns_use_explicit_urdu_rtl_order() -> None:
    lines = (
        candidate("left-1", 40, 20, 130),
        candidate("right-1", 230, 20, 130),
        candidate("left-2", 40, 55, 125),
        candidate("right-2", 235, 55, 125),
    )

    ordered = _infer_layout_and_order(lines, 8.0, SegmentationConfig())

    assert ids(ordered) == ["right-1", "right-2", "left-1", "left-2"]
    assert [item.column_index for item in ordered] == [0, 0, 1, 1]
    assert all(item.region_kind is RegionKind.LINE for item in ordered)


@pytest.mark.parametrize(
    ("right_y", "left_y"),
    [
        ((20, 50, 80, 110), (20, 50)),
        ((20, 50), (20, 50, 80, 110)),
        ((20, 50, 80), (22, 52)),
    ],
)
def test_uneven_column_counts_keep_each_column_top_to_bottom(
    right_y: tuple[int, ...], left_y: tuple[int, ...]
) -> None:
    lines = tuple(
        [candidate(f"right-{index}", 230, y, 120 - index * 3) for index, y in enumerate(right_y)]
        + [candidate(f"left-{index}", 45, y, 120 - index * 4) for index, y in enumerate(left_y)]
    )

    ordered = _infer_layout_and_order(lines, 8.0, SegmentationConfig())

    expected = [f"right-{index}" for index in range(len(right_y))] + [
        f"left-{index}" for index in range(len(left_y))
    ]
    assert ids(ordered) == expected


def test_smaller_valid_gutter_is_accepted() -> None:
    lines = (
        candidate("left-1", 40, 20, 150),
        candidate("right-1", 215, 20, 145),
        candidate("left-2", 45, 55, 145),
        candidate("right-2", 215, 55, 140),
    )

    ordered = _infer_layout_and_order(lines, 8.0, SegmentationConfig())

    assert [item.column_index for item in ordered] == [0, 0, 1, 1]


def test_small_whitespace_gap_is_not_two_column_evidence() -> None:
    lines = (
        candidate("left-1", 40, 20, 155),
        candidate("right-1", 205, 20, 155),
        candidate("left-2", 40, 55, 155),
        candidate("right-2", 205, 55, 155),
    )

    ordered = _infer_layout_and_order(lines, 8.0, SegmentationConfig())

    assert all(item.column_index is None for item in ordered)


def test_sparse_column_like_page_falls_back_to_one_column() -> None:
    lines = (candidate("left", 40, 30, 120), candidate("right", 240, 30, 120))

    ordered = _infer_layout_and_order(lines, 8.0, SegmentationConfig())

    assert ids(ordered) == ["right", "left"]
    assert all(item.column_index is None for item in ordered)


def test_wide_one_column_lines_do_not_create_spanning_or_columns() -> None:
    lines = (
        candidate("wide-1", 40, 20, 320),
        candidate("wide-2", 55, 60, 290),
        candidate("short", 140, 100, 120),
    )

    ordered = _infer_layout_and_order(lines, 8.0, SegmentationConfig())

    assert ids(ordered) == ["wide-1", "wide-2", "short"]
    assert all(item.column_index is None for item in ordered)
    assert all(item.region_kind is RegionKind.LINE for item in ordered)


def test_spanning_heading_columns_separator_and_footer_form_vertical_bands() -> None:
    lines = (
        candidate("heading", 20, 5, 360, 16),
        candidate("right-section-1", 235, 35, 125),
        candidate("left-section-1", 40, 35, 125),
        candidate("separator", 30, 70, 340, 10),
        candidate("right-section-2", 235, 100, 125),
        candidate("left-section-2", 40, 100, 125),
        candidate("footer", 35, 140, 330, 10),
    )

    ordered = _infer_layout_and_order(lines, 8.0, SegmentationConfig())

    assert ids(ordered) == [
        "heading",
        "right-section-1",
        "left-section-1",
        "separator",
        "right-section-2",
        "left-section-2",
        "footer",
    ]
    assert [item.region_kind for item in ordered] == [
        RegionKind.SPANNING,
        RegionKind.LINE,
        RegionKind.LINE,
        RegionKind.SPANNING,
        RegionKind.LINE,
        RegionKind.LINE,
        RegionKind.SPANNING,
    ]
    assert ordered[-1].candidate.candidate_id == "footer"


def test_y_tolerance_uses_stable_rightmost_tie_break() -> None:
    config = SegmentationConfig(minimum_column_lines=3)
    lines = (candidate("left", 50, 20, 80), candidate("right", 200, 21, 80))

    assert ids(_infer_layout_and_order(lines, 8.0, config)) == ["right", "left"]
