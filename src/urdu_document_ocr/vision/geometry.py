"""Pure half-open geometry used by line grouping and layout inference."""

from __future__ import annotations

from collections.abc import Iterable

from urdu_document_ocr.types import BoundingBox


def interval_overlap_length(
    first_start: int, first_end: int, second_start: int, second_end: int
) -> int:
    """Return the overlap length of two validated half-open intervals."""

    if first_end < first_start or second_end < second_start:
        raise ValueError("interval ends must not precede their starts")
    return max(0, min(first_end, second_end) - max(first_start, second_start))


def interval_gap(first_start: int, first_end: int, second_start: int, second_end: int) -> int:
    """Return zero for touching/overlapping intervals, otherwise their separation."""

    if first_end < first_start or second_end < second_start:
        raise ValueError("interval ends must not precede their starts")
    if first_end < second_start:
        return second_start - first_end
    if second_end < first_start:
        return first_start - second_end
    return 0


def vertical_overlap_ratio(first: BoundingBox, second: BoundingBox) -> float:
    """Vertical overlap divided by the shorter box height."""

    overlap = interval_overlap_length(first.y, first.bottom, second.y, second.bottom)
    return overlap / min(first.height, second.height)


def horizontal_gap(first: BoundingBox, second: BoundingBox) -> int:
    """Horizontal separation between boxes, or zero if their x intervals meet."""

    return interval_gap(first.x, first.right, second.x, second.right)


def vertical_center_distance(first: BoundingBox, second: BoundingBox) -> float:
    """Absolute distance between vertical centers."""

    return abs((first.y + first.height / 2.0) - (second.y + second.height / 2.0))


def union_boxes(boxes: Iterable[BoundingBox]) -> BoundingBox:
    """Return the smallest half-open box containing every supplied box."""

    values = tuple(boxes)
    if not values:
        raise ValueError("at least one bounding box is required")
    if any(not isinstance(box, BoundingBox) for box in values):
        raise TypeError("boxes must contain only BoundingBox values")
    left = min(box.x for box in values)
    top = min(box.y for box in values)
    right = max(box.right for box in values)
    bottom = max(box.bottom for box in values)
    return BoundingBox(left, top, right - left, bottom - top)


def padded_box(
    box: BoundingBox, padding: int, *, image_width: int, image_height: int
) -> BoundingBox:
    """Pad a box equally and clip it to positive image dimensions."""

    if not isinstance(box, BoundingBox):
        raise TypeError("box must be a BoundingBox")
    for name, value, minimum in (
        ("padding", padding, 0),
        ("image_width", image_width, 1),
        ("image_height", image_height, 1),
    ):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer")
        if value < minimum:
            raise ValueError(f"{name} must be at least {minimum}")
    left = max(0, box.x - padding)
    top = max(0, box.y - padding)
    right = min(image_width, box.right + padding)
    bottom = min(image_height, box.bottom + padding)
    if right <= left or bottom <= top:
        raise ValueError("box does not intersect the image")
    return BoundingBox(left, top, right - left, bottom - top)
