from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from urdu_document_ocr.types import PageImage, PreprocessedPage, SourceMetadata, SourceType

BoxTuple = tuple[int, int, int, int]


def mask_with_boxes(
    *, height: int = 220, width: int = 400, boxes: Iterable[BoxTuple] = ()
) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.bool_)
    for x, y, box_width, box_height in boxes:
        mask[y : y + box_height, x : x + box_width] = True
    return mask


def preprocessed_from_mask(
    mask: np.ndarray, *, page_index: int = 0, is_blank: bool = False
) -> PreprocessedPage:
    foreground = mask.astype(np.bool_, copy=True)
    grayscale = np.where(foreground, 0, 255).astype(np.uint8)
    rgb = np.repeat(grayscale[:, :, np.newaxis], 3, axis=2)
    page = PageImage(
        page_index,
        page_index + 1,
        rgb,
        SourceMetadata(SourceType.IMAGE, "generated-layout.png"),
    )
    return PreprocessedPage(page, grayscale, foreground, is_blank)


def rgb_from_mask(mask: np.ndarray) -> np.ndarray:
    grayscale = np.where(mask, 0, 255).astype(np.uint8)
    return np.repeat(grayscale[:, :, np.newaxis], 3, axis=2)
