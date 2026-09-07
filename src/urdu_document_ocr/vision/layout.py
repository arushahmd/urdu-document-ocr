"""Conservative one/two-column inference and deterministic Urdu RTL ordering."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from urdu_document_ocr.config import SegmentationConfig
from urdu_document_ocr.types import BoundingBox, RegionKind
from urdu_document_ocr.vision.geometry import union_boxes, vertical_overlap_ratio

_GUTTER_SEARCH_FRACTION = 0.5
_PAIR_DISTANCE_SCALE = 1.5


@dataclass(frozen=True, slots=True)
class _LineCandidate:
    candidate_id: str
    bounding_box: BoundingBox
    foreground_area: int
    component_count: int


@dataclass(frozen=True, slots=True)
class _Gutter:
    start: int
    end: int

    @property
    def width(self) -> int:
        return self.end - self.start

    @property
    def center(self) -> float:
        return (self.start + self.end) / 2.0


@dataclass(frozen=True, slots=True)
class _LayoutAssignment:
    candidate: _LineCandidate
    column_index: int | None
    region_kind: RegionKind


def _content_box(candidates: tuple[_LineCandidate, ...]) -> BoundingBox:
    return union_boxes(candidate.bounding_box for candidate in candidates)


def _is_wide_centered(
    candidate: _LineCandidate,
    content: BoundingBox,
    config: SegmentationConfig,
) -> bool:
    box = candidate.bounding_box
    width_fraction = box.width / content.width
    center_distance = abs((box.x + box.width / 2.0) - (content.x + content.width / 2.0))
    return (
        width_fraction >= config.spanning_width_fraction
        and center_distance / content.width <= config.spanning_center_tolerance_fraction
    )


def _zero_runs(occupancy: np.ndarray, offset: int) -> tuple[tuple[int, int], ...]:
    zero = occupancy == 0
    if not zero.any():
        return ()
    changes = np.diff(np.pad(zero.astype(np.int8), (1, 1)))
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)
    return tuple(
        (int(start) + offset, int(end) + offset) for start, end in zip(starts, ends, strict=True)
    )


def _paired_line_count(
    left: tuple[_LineCandidate, ...],
    right: tuple[_LineCandidate, ...],
    text_scale: float,
) -> int:
    unmatched = set(range(len(right)))
    matched = 0
    for left_line in sorted(
        left,
        key=lambda item: (
            item.bounding_box.y + item.bounding_box.height / 2.0,
            item.candidate_id,
        ),
    ):
        left_center = left_line.bounding_box.y + left_line.bounding_box.height / 2.0
        choices = []
        for index in unmatched:
            right_box = right[index].bounding_box
            distance = abs(left_center - (right_box.y + right_box.height / 2.0))
            if distance <= max(1.0, _PAIR_DISTANCE_SCALE * text_scale):
                choices.append((distance, right[index].candidate_id, index))
        if choices:
            _, _, chosen = min(choices)
            unmatched.remove(chosen)
            matched += 1
    return matched


def _find_persistent_gutter(
    candidates: tuple[_LineCandidate, ...],
    text_scale: float,
    config: SegmentationConfig,
) -> _Gutter | None:
    """Find a central x interval crossed by no body candidate with bilateral evidence."""

    if len(candidates) < 2 * config.minimum_column_lines:
        return None
    content = _content_box(candidates)
    body = tuple(
        candidate for candidate in candidates if not _is_wide_centered(candidate, content, config)
    )
    if len(body) < 2 * config.minimum_column_lines:
        return None

    search_margin = math.floor(content.width * (1.0 - _GUTTER_SEARCH_FRACTION) / 2.0)
    search_start = content.x + search_margin
    search_end = content.right - search_margin
    if search_end <= search_start:
        return None

    occupancy = np.zeros(search_end - search_start, dtype=np.uint32)
    for candidate in body:
        left = max(search_start, candidate.bounding_box.x)
        right = min(search_end, candidate.bounding_box.right)
        if right > left:
            occupancy[left - search_start : right - search_start] += 1

    minimum_width = max(
        1,
        math.ceil(content.width * config.gutter_min_width_fraction),
        math.ceil(text_scale * config.gutter_min_width_scale),
    )
    viable: list[tuple[int, float, int, float, _Gutter]] = []
    content_center = content.x + content.width / 2.0
    for start, end in _zero_runs(occupancy, search_start):
        gutter = _Gutter(start, end)
        if gutter.width < minimum_width:
            continue
        left = tuple(item for item in body if item.bounding_box.right <= gutter.start)
        right = tuple(item for item in body if item.bounding_box.x >= gutter.end)
        if len(left) < config.minimum_column_lines or len(right) < config.minimum_column_lines:
            continue
        left_span = union_boxes(item.bounding_box for item in left)
        right_span = union_boxes(item.bounding_box for item in right)
        vertical_support = vertical_overlap_ratio(left_span, right_span)
        if vertical_support < config.minimum_column_vertical_overlap:
            continue
        paired = _paired_line_count(left, right, text_scale)
        if paired < config.minimum_paired_lines:
            continue
        center_distance = abs(gutter.center - content_center)
        viable.append((paired, vertical_support, gutter.width, -center_distance, gutter))

    if not viable:
        return None
    return max(viable, key=lambda item: item[:4])[4]


def _is_spanning(
    candidate: _LineCandidate,
    gutter: _Gutter,
    content: BoundingBox,
    config: SegmentationConfig,
) -> bool:
    box = candidate.bounding_box
    return (
        _is_wide_centered(candidate, content, config)
        and box.x < gutter.start
        and box.right > gutter.end
    )


def _vertical_order(
    assignments: tuple[_LayoutAssignment, ...], text_scale: float, config: SegmentationConfig
) -> tuple[_LayoutAssignment, ...]:
    if not assignments:
        return ()
    tolerance = max(0, round(text_scale * config.y_order_tolerance_scale))
    pending = sorted(
        assignments,
        key=lambda item: (
            item.candidate.bounding_box.y,
            -item.candidate.bounding_box.right,
            item.candidate.candidate_id,
        ),
    )
    ordered: list[_LayoutAssignment] = []
    index = 0
    while index < len(pending):
        band_top = pending[index].candidate.bounding_box.y
        end = index + 1
        while end < len(pending) and pending[end].candidate.bounding_box.y <= band_top + tolerance:
            end += 1
        band = pending[index:end]
        band.sort(
            key=lambda item: (
                -item.candidate.bounding_box.right,
                item.candidate.bounding_box.y,
                item.candidate.bounding_box.x,
                item.candidate.candidate_id,
            )
        )
        ordered.extend(band)
        index = end
    return tuple(ordered)


def _column_band_order(
    assignments: tuple[_LayoutAssignment, ...], text_scale: float, config: SegmentationConfig
) -> tuple[_LayoutAssignment, ...]:
    right = tuple(item for item in assignments if item.column_index == 0)
    left = tuple(item for item in assignments if item.column_index == 1)
    return _vertical_order(right, text_scale, config) + _vertical_order(left, text_scale, config)


def _infer_layout_and_order(
    candidates: tuple[_LineCandidate, ...],
    text_scale: float,
    config: SegmentationConfig,
) -> tuple[_LayoutAssignment, ...]:
    if not candidates:
        return ()
    gutter = _find_persistent_gutter(candidates, text_scale, config)
    if gutter is None:
        one_column = tuple(
            _LayoutAssignment(candidate, None, RegionKind.LINE) for candidate in candidates
        )
        return _vertical_order(one_column, text_scale, config)

    content = _content_box(candidates)
    assignments: list[_LayoutAssignment] = []
    for candidate in candidates:
        if _is_spanning(candidate, gutter, content, config):
            assignments.append(_LayoutAssignment(candidate, None, RegionKind.SPANNING))
        else:
            center = candidate.bounding_box.x + candidate.bounding_box.width / 2.0
            column = 0 if center > gutter.center else 1
            assignments.append(_LayoutAssignment(candidate, column, RegionKind.LINE))

    spanning = _vertical_order(
        tuple(item for item in assignments if item.region_kind is RegionKind.SPANNING),
        text_scale,
        config,
    )
    remaining = [item for item in assignments if item.region_kind is RegionKind.LINE]
    ordered: list[_LayoutAssignment] = []
    for spanning_line in spanning:
        split_y = (
            spanning_line.candidate.bounding_box.y
            + spanning_line.candidate.bounding_box.height / 2.0
        )
        before = tuple(
            item
            for item in remaining
            if item.candidate.bounding_box.y + item.candidate.bounding_box.height / 2.0 < split_y
        )
        ordered.extend(_column_band_order(before, text_scale, config))
        before_candidate_ids = {item.candidate.candidate_id for item in before}
        remaining = [
            item for item in remaining if item.candidate.candidate_id not in before_candidate_ids
        ]
        ordered.append(spanning_line)
    ordered.extend(_column_band_order(tuple(remaining), text_scale, config))
    return tuple(ordered)
