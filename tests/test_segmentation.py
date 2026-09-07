from __future__ import annotations

import numpy as np
import pytest
from fixtures.page_patterns import mask_with_boxes, preprocessed_from_mask

from urdu_document_ocr import extract_line_crop, segment_page
from urdu_document_ocr.config import SegmentationConfig
from urdu_document_ocr.types import BoundingBox, LineRegion, RegionKind
from urdu_document_ocr.vision.segmentation import _Component, _estimate_text_scale


def component(label: int, height: int, area: int) -> _Component:
    return _Component(label, BoundingBox(label * 10, 10, 8, height), area, 0.0, 0.0)


def test_text_scale_is_weighted_median_and_ignores_large_page_outlier() -> None:
    components = (
        component(1, 2, 4),
        component(2, 8, 64),
        component(3, 9, 72),
        component(4, 80, 8_000),
    )

    assert _estimate_text_scale(components, 100_000, SegmentationConfig()) == 9.0
    assert _estimate_text_scale((), 100_000, SegmentationConfig()) == 1.0


def test_single_fragmented_line_has_refined_padded_bounds() -> None:
    mask = mask_with_boxes(boxes=((20, 30, 30, 8), (58, 31, 25, 8), (90, 29, 30, 10)))

    regions = segment_page(preprocessed_from_mask(mask))

    assert len(regions) == 1
    assert regions[0].bounding_box == BoundingBox(18, 27, 104, 14)
    assert regions[0].column_index is None
    assert regions[0].region_kind is RegionKind.LINE
    assert regions[0].reading_order_index == 0


def test_multiple_unequal_short_and_centered_lines_order_top_to_bottom() -> None:
    boxes = (
        (200, 20, 120, 8),
        (230, 60, 12, 6),
        (110, 100, 180, 10),
        (170, 140, 70, 8),
    )

    regions = segment_page(preprocessed_from_mask(mask_with_boxes(boxes=boxes)))

    assert len(regions) == 4
    assert [region.bounding_box.y for region in regions] == sorted(
        region.bounding_box.y for region in regions
    )
    assert [region.reading_order_index for region in regions] == [0, 1, 2, 3]
    assert all(region.column_index is None for region in regions)


def test_disconnected_dots_above_and_below_remain_with_the_line() -> None:
    mask = mask_with_boxes(
        boxes=(
            (30, 30, 45, 8),
            (82, 30, 40, 8),
            (48, 25, 2, 2),
            (96, 40, 2, 2),
            (135, 5, 2, 2),
        )
    )

    (region,) = segment_page(preprocessed_from_mask(mask))

    assert region.bounding_box.contains(48, 25)
    assert region.bounding_box.contains(96, 41)
    assert not region.bounding_box.contains(135, 5)


def test_components_with_modest_vertical_overlap_form_one_line() -> None:
    mask = mask_with_boxes(boxes=((20, 30, 35, 10), (62, 37, 40, 10)))

    regions = segment_page(preprocessed_from_mask(mask))

    assert len(regions) == 1
    assert regions[0].bounding_box.y <= 30
    assert regions[0].bounding_box.bottom >= 47


def test_large_word_gap_merges_after_one_column_fallback() -> None:
    mask = mask_with_boxes(boxes=((30, 60, 40, 8), (115, 60, 45, 8)))

    regions = segment_page(preprocessed_from_mask(mask))

    assert len(regions) == 1
    assert regions[0].bounding_box.x <= 30
    assert regions[0].bounding_box.right >= 160


def test_small_scanner_specks_are_ignored_around_real_lines() -> None:
    mask = mask_with_boxes(boxes=((40, 60, 100, 8), (170, 90, 1, 1), (5, 5, 2, 2)))

    regions = segment_page(preprocessed_from_mask(mask))

    assert len(regions) == 1
    assert regions[0].bounding_box.right < 170
    assert regions[0].bounding_box.y > 5


def test_vertically_close_lines_do_not_merge() -> None:
    mask = mask_with_boxes(
        boxes=(
            (30, 40, 50, 8),
            (88, 41, 45, 8),
            (35, 51, 45, 8),
            (88, 52, 50, 8),
            (60, 37, 2, 2),
            (105, 61, 2, 2),
        )
    )

    regions = segment_page(preprocessed_from_mask(mask))

    assert len(regions) == 2
    assert regions[0].bounding_box.bottom <= regions[1].bounding_box.bottom


def test_empty_nonblank_mask_and_explicit_blank_page_return_no_regions() -> None:
    empty = mask_with_boxes()
    contradictory_blank = mask_with_boxes(boxes=((20, 20, 100, 8),))

    assert segment_page(preprocessed_from_mask(empty, is_blank=False)) == ()
    assert segment_page(preprocessed_from_mask(contradictory_blank, is_blank=True)) == ()


def test_segmentation_is_deterministic_and_does_not_mutate_page() -> None:
    mask = mask_with_boxes(boxes=((20, 20, 70, 8), (25, 60, 90, 8)))
    page = preprocessed_from_mask(mask)
    before_mask = page.foreground_mask.copy()
    before_grayscale = page.grayscale.copy()

    first = segment_page(page)
    second = segment_page(page)

    assert first == second
    assert np.array_equal(page.foreground_mask, before_mask)
    assert np.array_equal(page.grayscale, before_grayscale)


def test_segmentation_preserves_nonzero_page_index() -> None:
    page = preprocessed_from_mask(mask_with_boxes(boxes=((20, 20, 70, 8),)), page_index=3)

    (region,) = segment_page(page, SegmentationConfig(padding_scale=0.0))

    assert region.page_index == 3
    assert region.region_id == "page-3-line-0000"
    assert region.bounding_box == BoundingBox(20, 20, 70, 8)


def test_line_crop_returns_owned_grayscale_and_foreground_arrays() -> None:
    page = preprocessed_from_mask(mask_with_boxes(boxes=((20, 20, 60, 8),)))
    (region,) = segment_page(page)

    grayscale = extract_line_crop(page, region)
    foreground = extract_line_crop(page, region, source="foreground")

    assert grayscale.shape == (region.bounding_box.height, region.bounding_box.width)
    assert foreground.dtype == np.bool_
    assert grayscale.flags.owndata and foreground.flags.owndata
    foreground[:] = False
    assert page.foreground_mask.any()


def test_line_crop_validates_source_page_and_bounds() -> None:
    page = preprocessed_from_mask(mask_with_boxes(boxes=((20, 20, 60, 8),)))
    (region,) = segment_page(page)
    wrong_page = preprocessed_from_mask(mask_with_boxes(boxes=((20, 20, 60, 8),)), page_index=1)
    outside = LineRegion(
        "outside",
        0,
        BoundingBox(page.page.width - 1, 0, 2, 2),
        None,
        0,
    )

    with pytest.raises(ValueError, match="page_index"):
        extract_line_crop(wrong_page, region)
    with pytest.raises(ValueError, match="exceeds"):
        extract_line_crop(page, outside)
    with pytest.raises(ValueError, match="source"):
        extract_line_crop(page, region, source="rgb")  # type: ignore[arg-type]


def test_segment_page_validates_inputs() -> None:
    page = preprocessed_from_mask(mask_with_boxes(boxes=((20, 20, 60, 8),)))

    with pytest.raises(TypeError):
        segment_page(page.page)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        segment_page(page, config="default")  # type: ignore[arg-type]
