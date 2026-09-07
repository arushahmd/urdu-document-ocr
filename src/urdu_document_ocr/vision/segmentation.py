"""Scale-relative component grouping into ordered Urdu line geometry."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np

from urdu_document_ocr.config import SegmentationConfig
from urdu_document_ocr.errors import SegmentationError
from urdu_document_ocr.types import BoundingBox, LineRegion, PreprocessedPage
from urdu_document_ocr.vision.geometry import (
    horizontal_gap,
    interval_gap,
    padded_box,
    union_boxes,
    vertical_center_distance,
    vertical_overlap_ratio,
)
from urdu_document_ocr.vision.layout import (
    _content_box,
    _find_persistent_gutter,
    _infer_layout_and_order,
    _is_spanning,
    _LineCandidate,
)

_MINIMUM_COMPONENT_VERTICAL_OVERLAP = 0.25
_MINIMUM_FRAGMENT_VERTICAL_OVERLAP = 0.55
_FRAGMENT_CENTER_DISTANCE_SCALE = 0.35
_CORE_AREA_SCALE = 0.5


@dataclass(frozen=True, slots=True)
class _Component:
    label: int
    bounding_box: BoundingBox
    area: int
    centroid_x: float
    centroid_y: float


@dataclass(frozen=True, slots=True)
class _ComponentGroup:
    components: tuple[_Component, ...]

    @property
    def bounding_box(self) -> BoundingBox:
        return union_boxes(component.bounding_box for component in self.components)

    @property
    def foreground_area(self) -> int:
        return sum(component.area for component in self.components)


class _UnionFind:
    def __init__(self, size: int) -> None:
        self._parent = list(range(size))
        self._rank = [0] * size

    def find(self, item: int) -> int:
        while self._parent[item] != item:
            self._parent[item] = self._parent[self._parent[item]]
            item = self._parent[item]
        return item

    def union(self, first: int, second: int) -> None:
        first_root = self.find(first)
        second_root = self.find(second)
        if first_root == second_root:
            return
        if self._rank[first_root] < self._rank[second_root]:
            first_root, second_root = second_root, first_root
        self._parent[second_root] = first_root
        if self._rank[first_root] == self._rank[second_root]:
            self._rank[first_root] += 1


def segment_page(
    page: PreprocessedPage,
    config: SegmentationConfig | None = None,
) -> tuple[LineRegion, ...]:
    """Segment a preprocessed page and assign conservative Urdu reading order."""

    if not isinstance(page, PreprocessedPage):
        raise TypeError("page must be a PreprocessedPage")
    if config is None:
        config = SegmentationConfig()
    if not isinstance(config, SegmentationConfig):
        raise TypeError("config must be a SegmentationConfig or None")
    if page.is_blank or not page.foreground_mask.any():
        return ()

    try:
        groups, text_scale = _detect_line_groups(page.foreground_mask, config)
        if not groups:
            return ()
        candidates = _candidates_from_groups(groups, page.page.page_index)
        assignments = _infer_layout_and_order(candidates, text_scale, config)
        padding = max(0, round(text_scale * config.padding_scale))
        result = []
        for reading_index, assignment in enumerate(assignments):
            box = padded_box(
                assignment.candidate.bounding_box,
                padding,
                image_width=page.page.width,
                image_height=page.page.height,
            )
            result.append(
                LineRegion(
                    region_id=assignment.candidate.candidate_id,
                    page_index=page.page.page_index,
                    bounding_box=box,
                    column_index=assignment.column_index,
                    reading_order_index=reading_index,
                    region_kind=assignment.region_kind,
                )
            )
        return tuple(result)
    except SegmentationError:
        raise
    except (cv2.error, MemoryError) as error:
        raise SegmentationError(
            "page segmentation failed",
            context={"source_page_number": page.page.source_page_number, "stage": "segmentation"},
        ) from error


def extract_line_crop(
    page: PreprocessedPage,
    region: LineRegion,
    *,
    source: Literal["grayscale", "foreground"] = "grayscale",
) -> np.ndarray:
    """Return an owned crop for a matching region from grayscale or foreground data."""

    if not isinstance(page, PreprocessedPage):
        raise TypeError("page must be a PreprocessedPage")
    if not isinstance(region, LineRegion):
        raise TypeError("region must be a LineRegion")
    if region.page_index != page.page.page_index:
        raise ValueError("region page_index does not match the preprocessed page")
    if source not in {"grayscale", "foreground"}:
        raise ValueError("source must be grayscale or foreground")
    box = region.bounding_box
    if box.right > page.page.width or box.bottom > page.page.height:
        raise ValueError("region bounding box exceeds the page dimensions")
    array = page.grayscale if source == "grayscale" else page.foreground_mask
    return array[box.y : box.bottom, box.x : box.right].copy()


def _extract_components(
    foreground: np.ndarray, config: SegmentationConfig
) -> tuple[_Component, ...]:
    count, labels, statistics, centroids = cv2.connectedComponentsWithStats(
        foreground.astype(np.uint8), connectivity=8
    )
    del labels
    page_area = foreground.size
    minimum_area = max(
        config.component_min_area_pixels,
        math.ceil(page_area * config.component_min_area_fraction),
    )
    components = []
    for label in range(1, count):
        area = int(statistics[label, cv2.CC_STAT_AREA])
        if area < minimum_area:
            continue
        components.append(
            _Component(
                label=label,
                bounding_box=BoundingBox(
                    int(statistics[label, cv2.CC_STAT_LEFT]),
                    int(statistics[label, cv2.CC_STAT_TOP]),
                    int(statistics[label, cv2.CC_STAT_WIDTH]),
                    int(statistics[label, cv2.CC_STAT_HEIGHT]),
                ),
                area=area,
                centroid_x=float(centroids[label, 0]),
                centroid_y=float(centroids[label, 1]),
            )
        )
    return tuple(components)


def _estimate_text_scale(
    components: tuple[_Component, ...], page_area: int, config: SegmentationConfig
) -> float:
    if not components:
        return 1.0
    maximum_area = page_area * config.scale_max_component_area_fraction
    scale_components = tuple(item for item in components if item.area <= maximum_area)
    if not scale_components:
        scale_components = components
    ordered = sorted(scale_components, key=lambda item: (item.bounding_box.height, item.label))
    total_weight = sum(item.area for item in ordered)
    midpoint = total_weight / 2.0
    cumulative = 0
    for component in ordered:
        cumulative += component.area
        if cumulative >= midpoint:
            return float(component.bounding_box.height)
    return float(ordered[-1].bounding_box.height)


def _component_pair_is_compatible(
    first: _Component, second: _Component, text_scale: float, config: SegmentationConfig
) -> bool:
    if horizontal_gap(first.bounding_box, second.bounding_box) > math.ceil(
        text_scale * config.horizontal_grouping_gap_scale
    ):
        return False
    return (
        vertical_overlap_ratio(first.bounding_box, second.bounding_box)
        >= _MINIMUM_COMPONENT_VERTICAL_OVERLAP
        or vertical_center_distance(first.bounding_box, second.bounding_box)
        <= text_scale * config.vertical_grouping_tolerance_scale
    )


def _group_core_components(
    components: tuple[_Component, ...], text_scale: float, config: SegmentationConfig
) -> tuple[_ComponentGroup, ...]:
    if not components:
        return ()
    indexed = sorted(
        enumerate(components), key=lambda item: (item[1].bounding_box.x, item[1].label)
    )
    union_find = _UnionFind(len(components))
    active: list[tuple[int, _Component]] = []
    maximum_gap = math.ceil(text_scale * config.horizontal_grouping_gap_scale)
    for current_index, current in indexed:
        active = [
            item
            for item in active
            if item[1].bounding_box.right >= current.bounding_box.x - maximum_gap
        ]
        for previous_index, previous in active:
            if _component_pair_is_compatible(previous, current, text_scale, config):
                union_find.union(previous_index, current_index)
        active.append((current_index, current))

    grouped: dict[int, list[_Component]] = {}
    for index, component in enumerate(components):
        grouped.setdefault(union_find.find(index), []).append(component)
    groups = [
        _ComponentGroup(tuple(sorted(values, key=lambda item: item.label)))
        for values in grouped.values()
    ]
    return tuple(sorted(groups, key=_group_sort_key))


def _attach_marks(
    groups: tuple[_ComponentGroup, ...],
    marks: tuple[_Component, ...],
    text_scale: float,
    config: SegmentationConfig,
) -> tuple[_ComponentGroup, ...]:
    if not groups:
        return ()
    boxes = tuple(group.bounding_box for group in groups)
    attached: list[list[_Component]] = [list(group.components) for group in groups]
    maximum_distance = max(1, math.ceil(text_scale * config.mark_attachment_distance_scale))
    for mark in marks:
        choices = []
        for index, box in enumerate(boxes):
            vertical_gap = interval_gap(
                mark.bounding_box.y,
                mark.bounding_box.bottom,
                box.y,
                box.bottom,
            )
            x_gap = horizontal_gap(mark.bounding_box, box)
            if vertical_gap <= maximum_distance and x_gap <= maximum_distance:
                choices.append(
                    (
                        vertical_gap,
                        x_gap,
                        vertical_center_distance(mark.bounding_box, box),
                        box.y,
                        -box.right,
                        index,
                    )
                )
        if choices:
            chosen = min(choices)[-1]
            attached[chosen].append(mark)
    return tuple(
        _ComponentGroup(tuple(sorted(values, key=lambda item: item.label))) for values in attached
    )


def _fragment_pair_is_compatible(
    first: _ComponentGroup,
    second: _ComponentGroup,
    text_scale: float,
    maximum_gap: int,
) -> bool:
    first_box = first.bounding_box
    second_box = second.bounding_box
    if horizontal_gap(first_box, second_box) > maximum_gap:
        return False
    return (
        vertical_overlap_ratio(first_box, second_box) >= _MINIMUM_FRAGMENT_VERTICAL_OVERLAP
        or vertical_center_distance(first_box, second_box)
        <= text_scale * _FRAGMENT_CENTER_DISTANCE_SCALE
    )


def _merge_groups(
    groups: tuple[_ComponentGroup, ...], text_scale: float, config: SegmentationConfig
) -> tuple[_ComponentGroup, ...]:
    if len(groups) < 2:
        return groups
    maximum_gap = math.ceil(text_scale * config.line_merge_gap_scale)
    indexed = sorted(
        enumerate(groups), key=lambda item: (item[1].bounding_box.x, _group_sort_key(item[1]))
    )
    union_find = _UnionFind(len(groups))
    active: list[tuple[int, _ComponentGroup]] = []
    for current_index, current in indexed:
        active = [
            item
            for item in active
            if item[1].bounding_box.right >= current.bounding_box.x - maximum_gap
        ]
        for previous_index, previous in active:
            if _fragment_pair_is_compatible(previous, current, text_scale, maximum_gap):
                union_find.union(previous_index, current_index)
        active.append((current_index, current))

    merged: dict[int, list[_Component]] = {}
    for index, group in enumerate(groups):
        merged.setdefault(union_find.find(index), []).extend(group.components)
    result = [
        _ComponentGroup(tuple(sorted(values, key=lambda item: item.label)))
        for values in merged.values()
    ]
    return tuple(sorted(result, key=_group_sort_key))


def _filter_line_groups(
    groups: tuple[_ComponentGroup, ...], page_area: int, config: SegmentationConfig
) -> tuple[_ComponentGroup, ...]:
    minimum_area = max(
        2 * config.component_min_area_pixels + 1,
        math.ceil(page_area * config.line_min_area_fraction),
    )
    return tuple(group for group in groups if group.foreground_area >= minimum_area)


def _group_sort_key(group: _ComponentGroup) -> tuple[int, int, int, int, int]:
    box = group.bounding_box
    return box.y, box.x, box.width, box.height, min(item.label for item in group.components)


def _candidates_from_groups(
    groups: tuple[_ComponentGroup, ...], page_index: int
) -> tuple[_LineCandidate, ...]:
    ordered = sorted(groups, key=_group_sort_key)
    return tuple(
        _LineCandidate(
            candidate_id=f"page-{page_index}-line-{index:04d}",
            bounding_box=group.bounding_box,
            foreground_area=group.foreground_area,
            component_count=len(group.components),
        )
        for index, group in enumerate(ordered)
    )


def _partition_groups_at_gutter(
    groups: tuple[_ComponentGroup, ...],
    candidates: tuple[_LineCandidate, ...],
    text_scale: float,
    config: SegmentationConfig,
) -> tuple[_ComponentGroup, ...]:
    gutter = _find_persistent_gutter(candidates, text_scale, config)
    if gutter is None:
        return _merge_groups(groups, text_scale, config)
    content = _content_box(candidates)
    pools: dict[str, list[_ComponentGroup]] = {"spanning": [], "right": [], "left": []}
    for group, candidate in zip(groups, candidates, strict=True):
        if _is_spanning(candidate, gutter, content, config):
            pools["spanning"].append(group)
        else:
            center = group.bounding_box.x + group.bounding_box.width / 2.0
            side = "right" if center > gutter.center else "left"
            pools[side].append(group)
    merged = []
    for name in ("spanning", "right", "left"):
        merged.extend(_merge_groups(tuple(pools[name]), text_scale, config))
    return tuple(sorted(merged, key=_group_sort_key))


def _detect_line_groups(
    foreground: np.ndarray, config: SegmentationConfig
) -> tuple[tuple[_ComponentGroup, ...], float]:
    components = _extract_components(foreground, config)
    if not components:
        return (), 1.0
    text_scale = _estimate_text_scale(components, foreground.size, config)
    core_height = max(2, math.ceil(text_scale * config.core_min_height_scale))
    core_area = max(
        config.component_min_area_pixels,
        math.ceil(text_scale * text_scale * _CORE_AREA_SCALE),
    )
    core = tuple(
        item
        for item in components
        if item.bounding_box.height >= core_height or item.area >= core_area
    )
    core_labels = {item.label for item in core}
    marks = tuple(item for item in components if item.label not in core_labels)
    groups = _group_core_components(core, text_scale, config)
    groups = _attach_marks(groups, marks, text_scale, config)
    groups = _filter_line_groups(groups, foreground.size, config)
    if not groups:
        return (), text_scale

    initial_candidates = _candidates_from_groups(groups, page_index=0)
    groups = _partition_groups_at_gutter(groups, initial_candidates, text_scale, config)
    groups = _filter_line_groups(groups, foreground.size, config)
    candidates = _candidates_from_groups(groups, page_index=0)
    if _find_persistent_gutter(candidates, text_scale, config) is None:
        groups = _merge_groups(groups, text_scale, config)
        groups = _filter_line_groups(groups, foreground.size, config)
    return groups, text_scale
