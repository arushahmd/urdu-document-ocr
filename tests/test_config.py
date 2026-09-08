from __future__ import annotations

import pytest

from urdu_document_ocr.config import (
    ConfigurationError,
    InferenceConfig,
    InputLimitsConfig,
    RecognizerConfig,
    SegmentationConfig,
    TrainingConfig,
)


def test_input_limit_defaults_are_frozen_phase_3_values() -> None:
    config = InputLimitsConfig()

    assert config.max_input_bytes == 100 * 1024 * 1024
    assert config.max_pages == 100
    assert config.max_pixels_per_page == 50_000_000
    assert config.max_document_pixels == 500_000_000
    assert (config.min_pdf_dpi, config.default_pdf_dpi, config.max_pdf_dpi) == (72, 200, 400)


@pytest.mark.parametrize(
    "overrides",
    [
        {"max_input_bytes": 0},
        {"max_pages": 0},
        {"max_pixels_per_page": -1},
        {"max_document_pixels": 10, "max_pixels_per_page": 11},
        {"min_pdf_dpi": 300, "default_pdf_dpi": 200},
        {"default_pdf_dpi": 500, "max_pdf_dpi": 400},
        {"max_pages": True},
    ],
)
def test_input_limits_reject_invalid_values(overrides: dict[str, object]) -> None:
    with pytest.raises(ConfigurationError):
        InputLimitsConfig(**overrides)  # type: ignore[arg-type]


def test_recognizer_defaults_are_frozen_phase_3_values() -> None:
    assert RecognizerConfig() == RecognizerConfig(
        normalized_height=64, max_width=2048, blank_index=0
    )


def test_inference_config_has_one_strict_positive_batch_field() -> None:
    assert InferenceConfig() == InferenceConfig.from_mapping({"batch_size": 16})
    with pytest.raises(ConfigurationError, match="batch_size"):
        InferenceConfig(batch_size=0)
    with pytest.raises(ConfigurationError, match="unknown inference"):
        InferenceConfig.from_mapping({"device": "auto"})


def test_segmentation_defaults_are_scale_relative_and_conservative() -> None:
    config = SegmentationConfig()

    assert config.component_min_area_fraction == 0.000001
    assert config.horizontal_grouping_gap_scale == 4.0
    assert config.spanning_width_fraction == 0.65
    assert config.minimum_column_lines == 2
    assert config.fingerprint == SegmentationConfig.from_mapping(config.to_dict()).fingerprint


@pytest.mark.parametrize(
    "overrides",
    [
        {"component_min_area_fraction": 0},
        {"component_min_area_pixels": 0},
        {"line_min_area_fraction": 0.0000001},
        {"scale_max_component_area_fraction": 0.000001},
        {"core_min_height_scale": 1.1},
        {"horizontal_grouping_gap_scale": 0},
        {"horizontal_grouping_gap_scale": 21},
        {"vertical_grouping_tolerance_scale": 3.1},
        {"mark_attachment_distance_scale": 5.1},
        {"line_merge_gap_scale": 3},
        {"line_merge_gap_scale": 31},
        {"padding_scale": 1.1},
        {"spanning_width_fraction": 0.49},
        {"spanning_center_tolerance_fraction": 0.51},
        {"gutter_min_width_fraction": 0.51},
        {"gutter_min_width_scale": 21},
        {"minimum_column_lines": 1},
        {"minimum_paired_lines": 0},
        {"minimum_column_vertical_overlap": 0},
        {"y_order_tolerance_scale": 1.1},
    ],
)
def test_segmentation_config_rejects_invalid_values(overrides: dict[str, object]) -> None:
    with pytest.raises(ConfigurationError):
        SegmentationConfig(**overrides)  # type: ignore[arg-type]


def test_segmentation_mapping_rejects_unknown_fields() -> None:
    with pytest.raises(ConfigurationError, match="unknown segmentation fields"):
        SegmentationConfig.from_mapping({"unreviewed_layout_knob": True})


@pytest.mark.parametrize(
    "overrides",
    [
        {"normalized_height": 0},
        {"normalized_height": 63},
        {"normalized_height": 128, "max_width": 256},
        {"max_width": 0},
        {"max_width": 2049},
        {"normalized_height": 128, "max_width": 64},
        {"blank_index": 1},
    ],
)
def test_recognizer_rejects_invalid_dimensions_or_blank(overrides: dict[str, int]) -> None:
    with pytest.raises(ConfigurationError):
        RecognizerConfig(**overrides)


def test_strict_mapping_rejects_unknown_fields() -> None:
    with pytest.raises(ConfigurationError, match="unknown recognizer fields"):
        RecognizerConfig.from_mapping({"normalized_height": 64, "future_knob": True})


def test_config_fingerprint_is_canonical_and_stable() -> None:
    first = RecognizerConfig.from_mapping({"max_width": 1024, "normalized_height": 64})
    second = RecognizerConfig.from_mapping({"normalized_height": 64, "max_width": 1024})

    assert first.fingerprint == second.fingerprint
    assert len(first.fingerprint) == 64


def test_training_defaults_are_small_validated_contracts() -> None:
    config = TrainingConfig()

    assert config.seed == 1337
    assert config.batch_size == 16
    assert config.checkpoint_directory == "best"
    assert config.device == "cpu"
    assert config.learning_rate == 0.0005
    assert config.min_delta == 0.0
    assert config.max_image_width == 2048


@pytest.mark.parametrize(
    "overrides",
    [
        {"seed": -1},
        {"batch_size": 0},
        {"epochs": 0},
        {"learning_rate": 0.0},
        {"learning_rate": float("nan")},
        {"weight_decay": -0.1},
        {"gradient_clip": 0.0},
        {"device": "auto"},
        {"device": "gpu"},
        {"checkpoint_directory": "../outside"},
        {"checkpoint_directory": r"C:\private\checkpoints"},
        {"num_workers": -1},
        {"early_stopping_patience": 0},
        {"min_delta": -0.1},
        {"max_image_width": 0},
        {"max_image_width": 2049},
    ],
)
def test_training_config_rejects_invalid_values(overrides: dict[str, object]) -> None:
    with pytest.raises(ConfigurationError):
        TrainingConfig(**overrides)  # type: ignore[arg-type]


def test_training_config_normalizes_checkpoint_path() -> None:
    config = TrainingConfig(checkpoint_directory=r"runs\checkpoints")

    assert config.checkpoint_directory == "runs/checkpoints"
    assert config.to_dict()["checkpoint_directory"] == "runs/checkpoints"
