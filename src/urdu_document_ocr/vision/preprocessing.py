"""Deterministic classical preprocessing without segmentation or text recognition."""

from __future__ import annotations

import math

import cv2
import numpy as np

from urdu_document_ocr.config import (
    MorphologyOperation,
    PreprocessingConfig,
    ThresholdMethod,
)
from urdu_document_ocr.errors import PreprocessingError
from urdu_document_ocr.types import PageImage, PreprocessedPage

_DEFAULT_CONFIG = PreprocessingConfig()
_POLARITY_DELTA = 1.0
_BORDER_FRACTION = 0.05


def preprocess_page(
    page: PageImage, config: PreprocessingConfig = _DEFAULT_CONFIG
) -> PreprocessedPage:
    """Convert one RGB page into grayscale and a True-is-foreground mask.

    The default path uses Otsu thresholding and makes no contrast, denoise,
    morphology, or deskew change. Optional deskew searches only small correction
    angles and applies one when both improvement and score separation are adequate.
    """

    if not isinstance(page, PageImage):
        raise TypeError("page must be a PageImage")
    if not isinstance(config, PreprocessingConfig):
        raise TypeError("config must be a PreprocessingConfig")

    try:
        grayscale = cv2.cvtColor(page.image, cv2.COLOR_RGB2GRAY)
        grayscale = np.ascontiguousarray(grayscale, dtype=np.uint8)
        enhanced = _enhance(grayscale, config)
        foreground = _foreground_mask(enhanced, config)
        foreground = _morphology(foreground, config)
        blank = _is_blank(enhanced, foreground, config)

        correction_angle = 0.0
        if config.deskew_enabled and not blank:
            correction_angle = _estimate_deskew(foreground, config)
            if correction_angle != 0.0:
                enhanced = _rotate_grayscale(enhanced, correction_angle)
                foreground = _foreground_mask(enhanced, config)
                foreground = _morphology(foreground, config)
                blank = _is_blank(enhanced, foreground, config)

        return PreprocessedPage(
            page=page,
            grayscale=enhanced.copy(),
            foreground_mask=foreground.astype(np.bool_, copy=True),
            is_blank=blank,
            deskew_angle_degrees=correction_angle,
        )
    except PreprocessingError:
        raise
    except (cv2.error, MemoryError, ValueError) as error:
        raise PreprocessingError(
            "page preprocessing failed",
            context={"source_page_number": page.source_page_number, "stage": "preprocessing"},
        ) from error


def _enhance(grayscale: np.ndarray, config: PreprocessingConfig) -> np.ndarray:
    result = grayscale.copy()
    if config.contrast_enabled:
        clahe = cv2.createCLAHE(
            clipLimit=float(config.clahe_clip_limit),
            tileGridSize=(config.clahe_grid_size, config.clahe_grid_size),
        )
        result = clahe.apply(result)
    if config.denoise_enabled:
        result = cv2.medianBlur(result, config.median_kernel_size)
    return np.ascontiguousarray(result, dtype=np.uint8)


def _foreground_mask(grayscale: np.ndarray, config: PreprocessingConfig) -> np.ndarray:
    background_is_dark = _background_is_dark(grayscale)
    if config.threshold_method is ThresholdMethod.OTSU:
        threshold, _ = cv2.threshold(grayscale, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        if background_is_dark:
            return grayscale > threshold
        return grayscale <= threshold

    threshold_map = _sauvola_threshold(
        grayscale,
        window_size=config.sauvola_window_size,
        k=float(config.sauvola_k),
        r=float(config.sauvola_r),
    )
    if background_is_dark:
        return grayscale.astype(np.float64) > threshold_map
    return grayscale.astype(np.float64) < threshold_map


def _border_pixels(grayscale: np.ndarray) -> np.ndarray:
    height, width = grayscale.shape
    thickness = max(1, math.ceil(min(height, width) * _BORDER_FRACTION))
    return np.concatenate(
        (
            grayscale[:thickness, :].ravel(),
            grayscale[-thickness:, :].ravel(),
            grayscale[:, :thickness].ravel(),
            grayscale[:, -thickness:].ravel(),
        )
    )


def _background_is_dark(grayscale: np.ndarray) -> bool:
    border_median = float(np.median(_border_pixels(grayscale)))
    darker_count = int(np.count_nonzero(grayscale < border_median - _POLARITY_DELTA))
    lighter_count = int(np.count_nonzero(grayscale > border_median + _POLARITY_DELTA))
    if lighter_count != darker_count:
        return lighter_count > darker_count
    return border_median < 127.5


def _sauvola_threshold(
    grayscale: np.ndarray, *, window_size: int, k: float, r: float
) -> np.ndarray:
    values = grayscale.astype(np.float64)
    kernel = (window_size, window_size)
    local_mean = cv2.boxFilter(
        values,
        ddepth=cv2.CV_64F,
        ksize=kernel,
        normalize=True,
        borderType=cv2.BORDER_REPLICATE,
    )
    local_square_mean = cv2.boxFilter(
        values * values,
        ddepth=cv2.CV_64F,
        ksize=kernel,
        normalize=True,
        borderType=cv2.BORDER_REPLICATE,
    )
    local_variance = np.maximum(local_square_mean - local_mean * local_mean, 0.0)
    local_standard_deviation = np.sqrt(local_variance)
    return local_mean * (1.0 + k * (local_standard_deviation / r - 1.0))


def _morphology(foreground: np.ndarray, config: PreprocessingConfig) -> np.ndarray:
    if config.morphology_operation is MorphologyOperation.NONE:
        return foreground.astype(np.bool_, copy=True)

    operation = (
        cv2.MORPH_OPEN
        if config.morphology_operation is MorphologyOperation.OPEN
        else cv2.MORPH_CLOSE
    )
    kernel = np.ones((config.morphology_kernel_size, config.morphology_kernel_size), dtype=np.uint8)
    cleaned = cv2.morphologyEx(foreground.astype(np.uint8), operation, kernel)
    return cleaned.astype(np.bool_)


def _meaningful_component_area(config: PreprocessingConfig, page_area: int) -> int:
    fractional_floor = math.ceil(config.meaningful_component_min_fraction * page_area)
    return max(config.meaningful_component_min_pixels, fractional_floor)


def _has_meaningful_component(foreground: np.ndarray, config: PreprocessingConfig) -> bool:
    if not foreground.any():
        return False
    _, _, statistics, _ = cv2.connectedComponentsWithStats(
        foreground.astype(np.uint8), connectivity=8
    )
    if len(statistics) <= 1:
        return False
    required_area = _meaningful_component_area(config, foreground.size)
    return bool(np.any(statistics[1:, cv2.CC_STAT_AREA] >= required_area))


def _is_blank(grayscale: np.ndarray, foreground: np.ndarray, config: PreprocessingConfig) -> bool:
    foreground_fraction = float(np.count_nonzero(foreground)) / foreground.size
    low_foreground = foreground_fraction < config.blank_max_foreground_fraction
    low_variation = float(np.std(grayscale, dtype=np.float64)) < config.blank_max_intensity_std
    lacks_component = not _has_meaningful_component(foreground, config)
    return bool(low_foreground and low_variation and lacks_component)


def _rotation_matrix(shape: tuple[int, int], angle: float) -> np.ndarray:
    height, width = shape
    center = ((width - 1) / 2.0, (height - 1) / 2.0)
    return cv2.getRotationMatrix2D(center, angle, 1.0)


def _rotate_mask(foreground: np.ndarray, angle: float) -> np.ndarray:
    height, width = foreground.shape
    rotated = cv2.warpAffine(
        foreground.astype(np.uint8),
        _rotation_matrix(foreground.shape, angle),
        (width, height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return rotated.astype(np.bool_)


def _projection_score(foreground: np.ndarray) -> float:
    ink_count = int(np.count_nonzero(foreground))
    if ink_count == 0:
        return 0.0
    row_profile = np.count_nonzero(foreground, axis=1).astype(np.float64)
    return float(np.dot(row_profile, row_profile) / (ink_count * ink_count))


def _deskew_candidates(maximum: float, step: float) -> tuple[float, ...]:
    step_count = math.floor(maximum / step)
    values = {round(index * step, 10) for index in range(-step_count, step_count + 1)}
    values.update({round(-maximum, 10), 0.0, round(maximum, 10)})
    return tuple(sorted(values))


def _estimate_deskew(foreground: np.ndarray, config: PreprocessingConfig) -> float:
    candidates = _deskew_candidates(
        float(config.deskew_max_angle_degrees), float(config.deskew_angle_step_degrees)
    )
    scores = np.array(
        [_projection_score(_rotate_mask(foreground, angle)) for angle in candidates],
        dtype=np.float64,
    )
    best_index = int(np.argmax(scores))
    best_angle = candidates[best_index]
    best_score = float(scores[best_index])
    baseline_score = float(scores[candidates.index(0.0)])

    if math.isclose(
        abs(best_angle),
        float(config.deskew_max_angle_degrees),
        abs_tol=float(config.deskew_angle_step_degrees) / 2.0,
    ):
        return 0.0
    if abs(best_angle) < float(config.deskew_angle_step_degrees) / 2.0:
        return 0.0
    if best_score <= 0.0 or baseline_score <= 0.0:
        return 0.0

    competitor_distance = max(1.0, 2.0 * float(config.deskew_angle_step_degrees))
    competing_scores = [
        float(score)
        for angle, score in zip(candidates, scores, strict=True)
        if abs(angle - best_angle) >= competitor_distance
    ]
    if not competing_scores:
        return 0.0
    competitor_score = max(competing_scores)
    improvement = max(0.0, (best_score - baseline_score) / baseline_score)
    separation = max(0.0, (best_score - competitor_score) / best_score)
    confidence = min(improvement, separation)
    if confidence < float(config.deskew_min_confidence):
        return 0.0
    return float(best_angle)


def _rotate_grayscale(grayscale: np.ndarray, angle: float) -> np.ndarray:
    height, width = grayscale.shape
    fill_value = round(float(np.median(_border_pixels(grayscale))))
    rotated = cv2.warpAffine(
        grayscale,
        _rotation_matrix(grayscale.shape, angle),
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=fill_value,
    )
    return np.ascontiguousarray(rotated, dtype=np.uint8)
