from __future__ import annotations

import numpy as np
from fixtures.page_patterns import mask_with_boxes, rgb_from_mask

from urdu_document_ocr import preprocess_page, segment_page
from urdu_document_ocr.types import PageImage, RegionKind, SourceMetadata, SourceType


def page_from_mask(mask: np.ndarray) -> PageImage:
    rgb = rgb_from_mask(mask)
    return PageImage(0, 1, rgb, SourceMetadata(SourceType.IMAGE, "phase6-generated.png"))


def test_generated_rgb_page_crosses_phase5_into_one_column_segmentation() -> None:
    mask = mask_with_boxes(
        boxes=(
            (60, 25, 55, 8),
            (123, 25, 70, 8),
            (75, 70, 45, 8),
            (128, 70, 80, 8),
            (90, 115, 90, 8),
        )
    )

    preprocessed = preprocess_page(page_from_mask(mask))
    regions = segment_page(preprocessed)

    assert not preprocessed.is_blank
    assert len(regions) == 3
    assert [region.reading_order_index for region in regions] == [0, 1, 2]
    assert all(region.column_index is None for region in regions)
    assert [region.bounding_box.y for region in regions] == sorted(
        region.bounding_box.y for region in regions
    )


def test_generated_rgb_page_crosses_phase5_into_spanning_two_column_rtl_layout() -> None:
    mask = mask_with_boxes(
        boxes=(
            (25, 5, 350, 16),
            (40, 45, 125, 8),
            (235, 45, 125, 8),
            (45, 85, 120, 8),
            (235, 85, 125, 8),
            (35, 140, 330, 10),
        )
    )

    preprocessed = preprocess_page(page_from_mask(mask))
    regions = segment_page(preprocessed)

    assert [region.region_id for region in regions] == [
        "page-0-line-0000",
        "page-0-line-0002",
        "page-0-line-0004",
        "page-0-line-0001",
        "page-0-line-0003",
        "page-0-line-0005",
    ]
    assert [region.column_index for region in regions] == [None, 0, 0, 1, 1, None]
    assert [region.region_kind for region in regions] == [
        RegionKind.SPANNING,
        RegionKind.LINE,
        RegionKind.LINE,
        RegionKind.LINE,
        RegionKind.LINE,
        RegionKind.SPANNING,
    ]
    assert regions[-1].region_id == "page-0-line-0005"
