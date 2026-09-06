from __future__ import annotations

import cv2
import numpy as np
import pytest

from urdu_document_ocr import PreprocessingConfig, preprocess_page
from urdu_document_ocr.config import ConfigurationError, MorphologyOperation, ThresholdMethod
from urdu_document_ocr.types import PageImage, SourceMetadata, SourceType


def page_from_grayscale(grayscale: np.ndarray) -> PageImage:
    rgb = np.repeat(grayscale[:, :, np.newaxis], 3, axis=2).astype(np.uint8)
    return PageImage(0, 1, rgb, SourceMetadata(SourceType.IMAGE, "generated.png"))


def bars_page(*, background: int = 255, ink: int = 0) -> PageImage:
    grayscale = np.full((120, 180), background, dtype=np.uint8)
    for top, left, right in ((20, 25, 150), (50, 15, 160), (80, 35, 145)):
        grayscale[top : top + 5, left:right] = ink
    return page_from_grayscale(grayscale)


def test_preprocessing_defaults_and_mapping_are_strict() -> None:
    config = PreprocessingConfig.from_mapping(
        {"threshold_method": "sauvola", "morphology_operation": "open"}
    )

    assert config.threshold_method is ThresholdMethod.SAUVOLA
    assert config.morphology_operation is MorphologyOperation.OPEN
    assert PreprocessingConfig().threshold_method is ThresholdMethod.OTSU
    assert not PreprocessingConfig().deskew_enabled
    with pytest.raises(ConfigurationError, match="unknown preprocessing fields"):
        PreprocessingConfig.from_mapping({"future_parameter": 1})


@pytest.mark.parametrize(
    "overrides",
    [
        {"threshold_method": "global"},
        {"contrast_enabled": 1},
        {"clahe_clip_limit": 0},
        {"clahe_grid_size": 1},
        {"clahe_grid_size": 65},
        {"denoise_enabled": "yes"},
        {"median_kernel_size": 4},
        {"median_kernel_size": 7},
        {"morphology_operation": "dilate"},
        {"morphology_kernel_size": 2},
        {"sauvola_window_size": 2},
        {"sauvola_window_size": 10},
        {"sauvola_window_size": 257},
        {"sauvola_k": 0},
        {"sauvola_k": 1.1},
        {"sauvola_r": 256},
        {"blank_max_foreground_fraction": 0},
        {"blank_max_foreground_fraction": 1.1},
        {"blank_max_intensity_std": 0},
        {"meaningful_component_min_fraction": 0},
        {"meaningful_component_min_pixels": 0},
        {"deskew_enabled": 1},
        {"deskew_max_angle_degrees": 5.1},
        {"deskew_angle_step_degrees": 6},
        {"deskew_angle_step_degrees": 0.01},
        {"deskew_min_confidence": -0.1},
    ],
)
def test_preprocessing_config_rejects_invalid_values(overrides: dict[str, object]) -> None:
    with pytest.raises(ConfigurationError):
        PreprocessingConfig(**overrides)  # type: ignore[arg-type]


def test_grayscale_contract_is_deterministic_and_does_not_mutate_input() -> None:
    rgb = np.array([[[255, 0, 0], [0, 255, 0], [0, 0, 255]]], dtype=np.uint8)
    page = PageImage(0, 1, rgb, SourceMetadata(SourceType.IMAGE))
    before = page.image.copy()

    first = preprocess_page(page)
    second = preprocess_page(page)

    assert first.grayscale.tolist() == [[76, 150, 29]]
    assert first.grayscale.dtype == np.uint8
    assert first.grayscale.shape == (1, 3)
    assert np.array_equal(first.grayscale, second.grayscale)
    assert np.array_equal(first.foreground_mask, second.foreground_mask)
    assert np.array_equal(page.image, before)


@pytest.mark.parametrize(
    "grayscale",
    [
        np.full((100, 100), 255, dtype=np.uint8),
        np.full((100, 100), 248, dtype=np.uint8),
        np.zeros((100, 100), dtype=np.uint8),
    ],
)
def test_uniform_pages_are_blank_for_light_and_dark_polarity(grayscale: np.ndarray) -> None:
    processed = preprocess_page(page_from_grayscale(grayscale))

    assert processed.is_blank
    assert not processed.foreground_mask.any()
    assert not processed.deskew_applied


def test_one_tiny_speck_is_blank_under_all_three_signals() -> None:
    grayscale = np.full((100, 100), 255, dtype=np.uint8)
    grayscale[50, 50] = 0

    processed = preprocess_page(page_from_grayscale(grayscale))

    assert processed.is_blank
    assert processed.foreground_mask.sum() == 1


@pytest.mark.parametrize(("background", "ink"), [(255, 0), (0, 255), (210, 198)])
def test_text_like_bars_are_nonblank_and_true_means_ink(background: int, ink: int) -> None:
    processed = preprocess_page(bars_page(background=background, ink=ink))

    assert not processed.is_blank
    assert processed.foreground_mask[22, 50]
    assert not processed.foreground_mask[0, 0]


def test_noisy_page_with_meaningful_foreground_is_not_blank() -> None:
    generator = np.random.default_rng(20260906)
    grayscale = np.full((120, 160), 245, dtype=np.uint8)
    coordinates = generator.integers((0, 0), (120, 160), size=(400, 2))
    grayscale[coordinates[:, 0], coordinates[:, 1]] = 20
    grayscale[55:60, 25:130] = 30

    assert not preprocess_page(page_from_grayscale(grayscale)).is_blank


def test_sauvola_handles_both_polarities_deterministically() -> None:
    config = PreprocessingConfig(threshold_method=ThresholdMethod.SAUVOLA)

    dark_ink = preprocess_page(bars_page(), config)
    light_ink = preprocess_page(bars_page(background=0, ink=255), config)

    assert dark_ink.foreground_mask[22, 50]
    assert light_ink.foreground_mask[22, 50]
    assert not dark_ink.is_blank and not light_ink.is_blank


def test_optional_median_filter_removes_isolated_impulse() -> None:
    grayscale = np.full((25, 25), 255, dtype=np.uint8)
    grayscale[12, 12] = 0

    processed = preprocess_page(
        page_from_grayscale(grayscale), PreprocessingConfig(denoise_enabled=True)
    )

    assert processed.grayscale[12, 12] == 255
    assert not processed.foreground_mask.any()


def test_optional_clahe_changes_local_contrast_without_changing_contract() -> None:
    grayscale = np.tile(np.arange(100, 140, dtype=np.uint8), (40, 1))

    processed = preprocess_page(
        page_from_grayscale(grayscale), PreprocessingConfig(contrast_enabled=True)
    )

    assert processed.grayscale.dtype == np.uint8
    assert processed.grayscale.shape == grayscale.shape
    assert not np.array_equal(processed.grayscale, grayscale)


def test_small_opening_removes_speck_without_line_joining() -> None:
    grayscale = np.full((40, 60), 255, dtype=np.uint8)
    grayscale[10:15, 20:30] = 0
    grayscale[30, 50] = 0
    config = PreprocessingConfig(morphology_operation=MorphologyOperation.OPEN)

    processed = preprocess_page(page_from_grayscale(grayscale), config)

    assert processed.foreground_mask[12, 25]
    assert not processed.foreground_mask[30, 50]


def test_small_closing_repairs_only_tiny_local_gap() -> None:
    grayscale = np.full((40, 60), 255, dtype=np.uint8)
    grayscale[15:20, 10:40] = 0
    grayscale[15:20, 25] = 255
    config = PreprocessingConfig(morphology_operation=MorphologyOperation.CLOSE)

    processed = preprocess_page(page_from_grayscale(grayscale), config)

    assert processed.foreground_mask[17, 25]
    assert not processed.foreground_mask[17, 45]


def test_deskew_recovers_generated_small_rotation_with_tolerance() -> None:
    original = bars_page().image[:, :, 0]
    height, width = original.shape
    matrix = cv2.getRotationMatrix2D(((width - 1) / 2, (height - 1) / 2), 3.0, 1.0)
    skewed = cv2.warpAffine(original, matrix, (width, height), borderValue=255)
    config = PreprocessingConfig(deskew_enabled=True)

    processed = preprocess_page(page_from_grayscale(skewed), config)

    assert processed.deskew_applied
    assert processed.deskew_angle_degrees == pytest.approx(-3.0, abs=0.75)
    assert processed.grayscale.shape == original.shape


def test_deskew_is_a_noop_by_default() -> None:
    original = bars_page().image[:, :, 0]
    height, width = original.shape
    matrix = cv2.getRotationMatrix2D(((width - 1) / 2, (height - 1) / 2), 3.0, 1.0)
    skewed = cv2.warpAffine(original, matrix, (width, height), borderValue=255)

    processed = preprocess_page(page_from_grayscale(skewed))

    assert not processed.deskew_applied
    assert processed.deskew_angle_degrees == 0.0


def test_blank_and_low_confidence_pages_are_not_rotated() -> None:
    blank = preprocess_page(
        page_from_grayscale(np.full((80, 80), 255, dtype=np.uint8)),
        PreprocessingConfig(deskew_enabled=True),
    )
    symmetric = np.full((100, 100), 255, dtype=np.uint8)
    cv2.circle(symmetric, (50, 50), 20, 0, thickness=-1)
    low_confidence = preprocess_page(
        page_from_grayscale(symmetric),
        PreprocessingConfig(deskew_enabled=True, deskew_min_confidence=0.5),
    )

    assert not blank.deskew_applied
    assert not low_confidence.deskew_applied


def test_out_of_range_skew_is_not_forced_to_boundary() -> None:
    original = bars_page().image[:, :, 0]
    height, width = original.shape
    matrix = cv2.getRotationMatrix2D(((width - 1) / 2, (height - 1) / 2), 8.0, 1.0)
    skewed = cv2.warpAffine(original, matrix, (width, height), borderValue=255)

    processed = preprocess_page(
        page_from_grayscale(skewed), PreprocessingConfig(deskew_enabled=True)
    )

    assert not processed.deskew_applied
