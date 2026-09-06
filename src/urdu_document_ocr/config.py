"""Strict configuration contracts for cross-phase OCR invariants."""

from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass, fields
from enum import StrEnum
from hashlib import sha256
from typing import Any, ClassVar, Self

from urdu_document_ocr.types import normalize_relative_dataset_path


class ConfigurationError(ValueError):
    """Raised when a configuration violates a public invariant."""


def _require_int(name: str, value: object, *, minimum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigurationError(f"{name} must be an integer")
    if value < minimum:
        raise ConfigurationError(f"{name} must be at least {minimum}")


def _require_number(name: str, value: object, *, minimum: float, inclusive: bool) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigurationError(f"{name} must be numeric")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ConfigurationError(f"{name} must be finite")
    invalid = numeric < minimum if inclusive else numeric <= minimum
    if invalid:
        relation = "at least" if inclusive else "greater than"
        raise ConfigurationError(f"{name} must be {relation} {minimum}")


def _require_boolean(name: str, value: object) -> None:
    if not isinstance(value, bool):
        raise ConfigurationError(f"{name} must be a boolean")


def _require_fraction(name: str, value: object, *, allow_zero: bool = False) -> None:
    _require_number(name, value, minimum=0.0, inclusive=allow_zero)
    if float(value) > 1.0:
        raise ConfigurationError(f"{name} must be at most 1.0")


class StrictConfig:
    """Mixin providing strict mapping construction and canonical fingerprints."""

    _config_name: ClassVar[str] = "configuration"

    @classmethod
    def from_mapping(cls, values: dict[str, Any]) -> Self:
        if not isinstance(values, dict):
            raise ConfigurationError(f"{cls._config_name} must be a mapping")
        allowed = {item.name for item in fields(cls)}
        unknown = sorted(set(values) - allowed)
        if unknown:
            raise ConfigurationError(f"unknown {cls._config_name} fields: {', '.join(unknown)}")
        try:
            return cls(**values)
        except TypeError as error:
            raise ConfigurationError(f"invalid {cls._config_name}: {error}") from error

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(
            self.to_dict(), ensure_ascii=False, separators=(",", ":"), sort_keys=True
        ).encode("utf-8")
        return sha256(payload).hexdigest()


@dataclass(frozen=True, slots=True)
class InputLimitsConfig(StrictConfig):
    """Resource and PDF rendering limits frozen by the Phase 3 architecture."""

    _config_name: ClassVar[str] = "input limits"

    max_input_bytes: int = 100 * 1024 * 1024
    max_pages: int = 100
    max_pixels_per_page: int = 50_000_000
    max_document_pixels: int = 500_000_000
    default_pdf_dpi: int = 200
    min_pdf_dpi: int = 72
    max_pdf_dpi: int = 400

    def __post_init__(self) -> None:
        _require_int("max_input_bytes", self.max_input_bytes, minimum=1)
        _require_int("max_pages", self.max_pages, minimum=1)
        _require_int("max_pixels_per_page", self.max_pixels_per_page, minimum=1)
        _require_int("max_document_pixels", self.max_document_pixels, minimum=1)
        _require_int("default_pdf_dpi", self.default_pdf_dpi, minimum=1)
        _require_int("min_pdf_dpi", self.min_pdf_dpi, minimum=1)
        _require_int("max_pdf_dpi", self.max_pdf_dpi, minimum=1)
        if self.max_document_pixels < self.max_pixels_per_page:
            raise ConfigurationError("max_document_pixels must be at least max_pixels_per_page")
        if not self.min_pdf_dpi <= self.default_pdf_dpi <= self.max_pdf_dpi:
            raise ConfigurationError(
                "PDF DPI values must satisfy min_pdf_dpi <= default_pdf_dpi <= max_pdf_dpi"
            )


class ThresholdMethod(StrEnum):
    """Supported foreground-threshold algorithms."""

    OTSU = "otsu"
    SAUVOLA = "sauvola"


class MorphologyOperation(StrEnum):
    """Small pre-segmentation mask cleanup operations."""

    NONE = "none"
    OPEN = "open"
    CLOSE = "close"


@dataclass(frozen=True, slots=True)
class PreprocessingConfig(StrictConfig):
    """Compact, validated configuration for the Phase 5 preprocessing pipeline."""

    _config_name: ClassVar[str] = "preprocessing"

    threshold_method: ThresholdMethod = ThresholdMethod.OTSU
    contrast_enabled: bool = False
    clahe_clip_limit: float = 2.0
    clahe_grid_size: int = 8
    denoise_enabled: bool = False
    median_kernel_size: int = 3
    morphology_operation: MorphologyOperation = MorphologyOperation.NONE
    morphology_kernel_size: int = 3
    sauvola_window_size: int = 31
    sauvola_k: float = 0.2
    sauvola_r: float = 128.0
    blank_max_foreground_fraction: float = 0.001
    blank_max_intensity_std: float = 3.0
    meaningful_component_min_fraction: float = 0.00001
    meaningful_component_min_pixels: int = 2
    deskew_enabled: bool = False
    deskew_max_angle_degrees: float = 5.0
    deskew_angle_step_degrees: float = 0.25
    deskew_min_confidence: float = 0.01

    def __post_init__(self) -> None:
        if isinstance(self.threshold_method, str) and not isinstance(
            self.threshold_method, ThresholdMethod
        ):
            try:
                object.__setattr__(self, "threshold_method", ThresholdMethod(self.threshold_method))
            except ValueError as error:
                raise ConfigurationError("threshold_method must be otsu or sauvola") from error
        if not isinstance(self.threshold_method, ThresholdMethod):
            raise ConfigurationError("threshold_method must be otsu or sauvola")

        if isinstance(self.morphology_operation, str) and not isinstance(
            self.morphology_operation, MorphologyOperation
        ):
            try:
                object.__setattr__(
                    self,
                    "morphology_operation",
                    MorphologyOperation(self.morphology_operation),
                )
            except ValueError as error:
                raise ConfigurationError(
                    "morphology_operation must be none, open, or close"
                ) from error
        if not isinstance(self.morphology_operation, MorphologyOperation):
            raise ConfigurationError("morphology_operation must be none, open, or close")

        _require_boolean("contrast_enabled", self.contrast_enabled)
        _require_number("clahe_clip_limit", self.clahe_clip_limit, minimum=0.0, inclusive=False)
        _require_int("clahe_grid_size", self.clahe_grid_size, minimum=2)
        if self.clahe_grid_size > 64:
            raise ConfigurationError("clahe_grid_size must be at most 64")
        _require_boolean("denoise_enabled", self.denoise_enabled)
        _require_int("median_kernel_size", self.median_kernel_size, minimum=3)
        if self.median_kernel_size % 2 == 0 or self.median_kernel_size > 5:
            raise ConfigurationError("median_kernel_size must be an odd integer no larger than 5")
        _require_int("morphology_kernel_size", self.morphology_kernel_size, minimum=3)
        if self.morphology_kernel_size % 2 == 0 or self.morphology_kernel_size > 5:
            raise ConfigurationError(
                "morphology_kernel_size must be an odd integer no larger than 5"
            )
        _require_int("sauvola_window_size", self.sauvola_window_size, minimum=3)
        if self.sauvola_window_size % 2 == 0 or self.sauvola_window_size > 255:
            raise ConfigurationError("sauvola_window_size must be odd and no larger than 255")
        _require_number("sauvola_k", self.sauvola_k, minimum=0.0, inclusive=False)
        if float(self.sauvola_k) > 1.0:
            raise ConfigurationError("sauvola_k must be at most 1.0")
        _require_number("sauvola_r", self.sauvola_r, minimum=0.0, inclusive=False)
        if float(self.sauvola_r) > 255.0:
            raise ConfigurationError("sauvola_r must be at most 255.0")
        _require_fraction("blank_max_foreground_fraction", self.blank_max_foreground_fraction)
        _require_number(
            "blank_max_intensity_std", self.blank_max_intensity_std, minimum=0.0, inclusive=False
        )
        _require_fraction(
            "meaningful_component_min_fraction", self.meaningful_component_min_fraction
        )
        _require_int(
            "meaningful_component_min_pixels", self.meaningful_component_min_pixels, minimum=1
        )
        _require_boolean("deskew_enabled", self.deskew_enabled)
        _require_number(
            "deskew_max_angle_degrees",
            self.deskew_max_angle_degrees,
            minimum=0.0,
            inclusive=False,
        )
        if float(self.deskew_max_angle_degrees) > 5.0:
            raise ConfigurationError("deskew_max_angle_degrees must be at most 5.0")
        _require_number(
            "deskew_angle_step_degrees",
            self.deskew_angle_step_degrees,
            minimum=0.05,
            inclusive=True,
        )
        if float(self.deskew_angle_step_degrees) > float(self.deskew_max_angle_degrees):
            raise ConfigurationError(
                "deskew_angle_step_degrees must not exceed deskew_max_angle_degrees"
            )
        _require_fraction("deskew_min_confidence", self.deskew_min_confidence, allow_zero=True)


@dataclass(frozen=True, slots=True)
class RecognizerConfig(StrictConfig):
    """Recognizer dimensions already fixed by the architecture."""

    _config_name: ClassVar[str] = "recognizer"

    normalized_height: int = 64
    max_width: int = 2048
    blank_index: int = 0

    def __post_init__(self) -> None:
        _require_int("normalized_height", self.normalized_height, minimum=1)
        _require_int("max_width", self.max_width, minimum=1)
        _require_int("blank_index", self.blank_index, minimum=0)
        if self.max_width < self.normalized_height:
            raise ConfigurationError("max_width must be at least normalized_height")
        if self.blank_index != 0:
            raise ConfigurationError("CTC blank_index must be 0")


_DEVICE_PATTERN = re.compile(r"(?:auto|cpu|mps|cuda(?::[0-9]+)?)")


@dataclass(frozen=True, slots=True)
class TrainingConfig(StrictConfig):
    """Small validated training configuration; training itself arrives in Phase 10."""

    _config_name: ClassVar[str] = "training"

    seed: int = 1337
    batch_size: int = 16
    epochs: int = 20
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    gradient_clip: float = 5.0
    device: str = "auto"
    checkpoint_directory: str = "checkpoints"
    num_workers: int = 0
    early_stopping_patience: int = 5
    max_image_width: int = 2048

    def __post_init__(self) -> None:
        _require_int("seed", self.seed, minimum=0)
        _require_int("batch_size", self.batch_size, minimum=1)
        _require_int("epochs", self.epochs, minimum=1)
        _require_number("learning_rate", self.learning_rate, minimum=0.0, inclusive=False)
        _require_number("weight_decay", self.weight_decay, minimum=0.0, inclusive=True)
        _require_number("gradient_clip", self.gradient_clip, minimum=0.0, inclusive=False)
        if not isinstance(self.device, str) or _DEVICE_PATTERN.fullmatch(self.device) is None:
            raise ConfigurationError("device must be auto, cpu, mps, cuda, or cuda:<index>")
        try:
            safe_checkpoint_path = normalize_relative_dataset_path(self.checkpoint_directory)
        except (TypeError, ValueError) as error:
            raise ConfigurationError(f"invalid checkpoint_directory: {error}") from error
        object.__setattr__(self, "checkpoint_directory", safe_checkpoint_path)
        _require_int("num_workers", self.num_workers, minimum=0)
        _require_int("early_stopping_patience", self.early_stopping_patience, minimum=1)
        _require_int("max_image_width", self.max_image_width, minimum=1)
