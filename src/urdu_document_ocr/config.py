"""Strict configuration contracts for cross-phase OCR invariants."""

from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass, fields
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
