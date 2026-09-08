"""Synthetic benchmark generation, vision scoring, and artifact integrity."""

from __future__ import annotations

import io
import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from hashlib import sha256
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np
from numpy.typing import NDArray
from PIL import Image

from urdu_document_ocr.config import (
    PreprocessingConfig,
    SegmentationConfig,
    SyntheticDataConfig,
)
from urdu_document_ocr.data import (
    PageLayoutFamily,
    dataset_fingerprint,
    generate_line_sample,
    generate_page_fixture,
    load_dataset_line_image,
    normalize_transcription,
)
from urdu_document_ocr.data.manifest import (
    _atomic_write_bytes,
    _reject_duplicate_json_keys,
    _reject_nonstandard_json_constant,
    canonical_manifest_bytes,
)
from urdu_document_ocr.errors import BenchmarkError
from urdu_document_ocr.evaluation.metrics import EvaluationReport, evaluate_predictions
from urdu_document_ocr.types import (
    BoundingBox,
    DatasetSample,
    LineRegion,
    PageImage,
    RegionKind,
    SourceMetadata,
    SourceType,
)
from urdu_document_ocr.vision import preprocess_page, segment_page

if TYPE_CHECKING:
    from urdu_document_ocr.recognition.base import OCRRecognizer

VISION_BENCHMARK_VERSION = "vision-synthetic-v1"
RECOGNITION_BENCHMARK_VERSION = "recognition-synthetic-v1"
BENCHMARK_SCHEMA_VERSION = 1


def canonical_json_bytes(value: object) -> bytes:
    """Serialize one deterministic JSON artifact with a final newline."""

    return (
        json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n"
    ).encode("utf-8")


def canonical_fingerprint(value: object) -> str:
    """Hash the canonical JSON representation of a logical value."""

    return sha256(canonical_json_bytes(value)).hexdigest()


def file_sha256(path: str | os.PathLike[str]) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def write_canonical_json(
    value: object,
    path: str | os.PathLike[str],
    *,
    overwrite: bool = False,
) -> None:
    """Atomically write canonical benchmark JSON."""

    _atomic_write_bytes(
        canonical_json_bytes(value),
        path,
        overwrite=overwrite,
        error_type=BenchmarkError,
        artifact_name="benchmark JSON",
    )


@dataclass(frozen=True, slots=True)
class ExpectedRegion:
    region_id: str
    bounding_box: BoundingBox
    region_kind: RegionKind
    column_index: int | None
    reading_order_index: int

    def __post_init__(self) -> None:
        if not isinstance(self.region_id, str) or not self.region_id:
            raise BenchmarkError("expected region ID must be nonempty")
        if not isinstance(self.bounding_box, BoundingBox):
            raise TypeError("bounding_box must be a BoundingBox")
        if not isinstance(self.region_kind, RegionKind):
            raise TypeError("region_kind must be a RegionKind")
        if self.column_index not in {None, 0, 1}:
            raise BenchmarkError("expected column index must be 0, 1, or None")
        if (
            isinstance(self.reading_order_index, bool)
            or not isinstance(self.reading_order_index, int)
            or self.reading_order_index < 0
        ):
            raise BenchmarkError("expected reading-order index must be nonnegative")

    def to_public_dict(self) -> dict[str, object]:
        return {
            "region_id": self.region_id,
            "bounding_box": self.bounding_box.to_public_dict(),
            "region_kind": self.region_kind.value,
            "column_index": self.column_index,
            "reading_order_index": self.reading_order_index,
        }


@dataclass(frozen=True, slots=True)
class VisionBenchmarkCase:
    page_id: str
    case_family: str
    layout_family: str
    expected_layout: str
    expected_blank: bool
    regions: tuple[ExpectedRegion, ...]
    image_sha256: str
    generator_seed: int
    image: NDArray[np.uint8] = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.page_id, str) or not self.page_id:
            raise BenchmarkError("vision page ID must be nonempty")
        if self.expected_layout not in {"blank", "one_column", "two_column"}:
            raise BenchmarkError("expected layout label is unsupported")
        if not isinstance(self.expected_blank, bool):
            raise TypeError("expected_blank must be a boolean")
        regions = tuple(self.regions)
        if [region.reading_order_index for region in regions] != list(range(len(regions))):
            raise BenchmarkError("expected regions must use contiguous reading order")
        if self.expected_blank != (len(regions) == 0):
            raise BenchmarkError("blank-state and expected regions disagree")
        if not isinstance(self.image, np.ndarray) or self.image.dtype != np.uint8:
            raise TypeError("vision case image must be a uint8 array")
        if self.image.ndim != 2 or min(self.image.shape) < 1:
            raise BenchmarkError("vision case image must be nonempty grayscale")
        if sha256(_png_bytes(self.image)).hexdigest() != self.image_sha256:
            raise BenchmarkError("vision image does not match its recorded hash")
        object.__setattr__(self, "regions", regions)


@dataclass(frozen=True, slots=True)
class RegionMatch:
    expected_index: int
    detected_index: int
    intersection_over_union: float

    def to_public_dict(self) -> dict[str, int | float]:
        return {
            "expected_index": self.expected_index,
            "detected_index": self.detected_index,
            "intersection_over_union": self.intersection_over_union,
        }


@dataclass(frozen=True, slots=True)
class VisionCaseResult:
    page_id: str
    case_family: str
    layout_family: str
    expected_layout: str
    detected_layout: str
    expected_blank: bool
    detected_blank: bool
    expected_regions: tuple[ExpectedRegion, ...]
    detected_regions: tuple[LineRegion, ...]
    matches: tuple[RegionMatch, ...]
    exact_line_count: bool
    exact_region_kind_count: int
    exact_column_count: int
    exact_reading_order: bool
    exact_layout: bool
    failures: tuple[str, ...]

    def to_public_dict(self) -> dict[str, object]:
        matched_iou = [match.intersection_over_union for match in self.matches]
        return {
            "page_id": self.page_id,
            "case_family": self.case_family,
            "layout_family": self.layout_family,
            "expected_layout": self.expected_layout,
            "detected_layout": self.detected_layout,
            "expected_blank": self.expected_blank,
            "detected_blank": self.detected_blank,
            "expected_line_count": len(self.expected_regions),
            "detected_line_count": len(self.detected_regions),
            "matched_region_count": len(self.matches),
            "mean_matched_iou": sum(matched_iou) / len(matched_iou) if matched_iou else None,
            "exact_line_count": self.exact_line_count,
            "exact_region_kind_count": self.exact_region_kind_count,
            "exact_column_count": self.exact_column_count,
            "exact_reading_order": self.exact_reading_order,
            "exact_layout": self.exact_layout,
            "expected_regions": [region.to_public_dict() for region in self.expected_regions],
            "detected_regions": [region.to_public_dict() for region in self.detected_regions],
            "matches": [match.to_public_dict() for match in self.matches],
            "failures": list(self.failures),
        }


@dataclass(frozen=True, slots=True)
class VisionBenchmarkResult:
    benchmark_version: str
    config_fingerprint: str
    iou_threshold: float
    cases: tuple[VisionCaseResult, ...]

    def __post_init__(self) -> None:
        if self.benchmark_version != VISION_BENCHMARK_VERSION:
            raise BenchmarkError("vision benchmark version is unsupported")
        values = tuple(self.cases)
        if tuple(sorted(values, key=lambda case: case.page_id)) != values:
            raise BenchmarkError("vision results must be sorted by page ID")
        object.__setattr__(self, "cases", values)

    def to_public_dict(self) -> dict[str, object]:
        expected = sum(len(case.expected_regions) for case in self.cases)
        detected = sum(len(case.detected_regions) for case in self.cases)
        matches = tuple(match for case in self.cases for match in case.matches)
        matched = len(matches)
        precision = matched / detected if detected else None
        recall = matched / expected if expected else None
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision is not None and recall is not None and precision + recall
            else None
        )
        failures = [
            {"page_id": case.page_id, "failures": list(case.failures)}
            for case in self.cases
            if case.failures
        ]
        return {
            "schema_version": BENCHMARK_SCHEMA_VERSION,
            "benchmark_version": self.benchmark_version,
            "evidence_class": "synthetic_vision",
            "label": "DETERMINISTIC SYNTHETIC DOCUMENT BENCHMARK",
            "config_fingerprint": self.config_fingerprint,
            "iou_threshold": self.iou_threshold,
            "summary": {
                "page_count": len(self.cases),
                "blank_page_correct_count": sum(
                    case.expected_blank == case.detected_blank for case in self.cases
                ),
                "exact_line_count_pages": sum(case.exact_line_count for case in self.cases),
                "expected_regions": expected,
                "detected_regions": detected,
                "matched_regions": matched,
                "region_precision": precision,
                "region_recall": recall,
                "region_f1": f1,
                "mean_matched_iou": (
                    sum(match.intersection_over_union for match in matches) / matched
                    if matched
                    else None
                ),
                "exact_region_kind_assignments": sum(
                    case.exact_region_kind_count for case in self.cases
                ),
                "exact_column_assignments": sum(case.exact_column_count for case in self.cases),
                "exact_reading_order_pages": sum(case.exact_reading_order for case in self.cases),
                "exact_layout_pages": sum(case.exact_layout for case in self.cases),
                "failure_page_count": len(failures),
            },
            "denominators": {
                "blank_page_correct_count": len(self.cases),
                "exact_line_count_pages": len(self.cases),
                "region_precision": detected,
                "region_recall": expected,
                "mean_matched_iou": matched,
                "exact_region_kind_assignments": matched,
                "exact_column_assignments": matched,
                "exact_reading_order_pages": len(self.cases),
                "exact_layout_pages": len(self.cases),
            },
            "cases": [case.to_public_dict() for case in self.cases],
            "failures": failures,
            "limitations": [
                "Deterministic synthetic pages only; not real-document accuracy.",
                "Supported one/two-column and spanning-region geometry only.",
            ],
        }


def match_regions(
    expected: Sequence[ExpectedRegion],
    detected: Sequence[LineRegion],
    *,
    iou_threshold: float,
) -> tuple[RegionMatch, ...]:
    """Greedily take highest-IoU nonconflicting pairs with stable index ties."""

    if (
        isinstance(iou_threshold, bool)
        or not isinstance(iou_threshold, (int, float))
        or not 0 < float(iou_threshold) <= 1
    ):
        raise BenchmarkError("IoU threshold must be within (0,1]")
    expected_values = tuple(expected)
    detected_values = tuple(detected)
    if any(not isinstance(region, ExpectedRegion) for region in expected_values):
        raise TypeError("expected values must be ExpectedRegion instances")
    if any(not isinstance(region, LineRegion) for region in detected_values):
        raise TypeError("detected values must be LineRegion instances")
    candidates = []
    for expected_index, expected_region in enumerate(expected_values):
        for detected_index, detected_region in enumerate(detected_values):
            iou = expected_region.bounding_box.intersection_over_union(detected_region.bounding_box)
            if iou >= float(iou_threshold):
                candidates.append((-iou, expected_index, detected_index))
    candidates.sort()
    used_expected: set[int] = set()
    used_detected: set[int] = set()
    matches: list[RegionMatch] = []
    for negative_iou, expected_index, detected_index in candidates:
        if expected_index in used_expected or detected_index in used_detected:
            continue
        used_expected.add(expected_index)
        used_detected.add(detected_index)
        matches.append(RegionMatch(expected_index, detected_index, -negative_iou))
    return tuple(sorted(matches, key=lambda match: match.expected_index))


def _layout_label(regions: Sequence[LineRegion], *, is_blank: bool) -> str:
    if is_blank and not regions:
        return "blank"
    columns = {region.column_index for region in regions if region.column_index is not None}
    return "two_column" if columns == {0, 1} else "one_column"


def run_vision_benchmark(
    cases: Sequence[VisionBenchmarkCase],
    *,
    preprocessing_config: PreprocessingConfig,
    segmentation_config: SegmentationConfig,
    iou_threshold: float,
    config_fingerprint: str,
) -> VisionBenchmarkResult:
    """Run preprocessing/segmentation against frozen synthetic ground truth."""

    values = tuple(cases)
    if not values or any(not isinstance(case, VisionBenchmarkCase) for case in values):
        raise BenchmarkError("vision benchmark cases must be nonempty and valid")
    if len({case.page_id for case in values}) != len(values):
        raise BenchmarkError("vision benchmark page IDs must be unique")
    if not isinstance(preprocessing_config, PreprocessingConfig) or not isinstance(
        segmentation_config, SegmentationConfig
    ):
        raise TypeError("vision preprocessing and segmentation configs are required")
    results: list[VisionCaseResult] = []
    for case in sorted(values, key=lambda item: item.page_id):
        rgb = np.repeat(case.image[:, :, np.newaxis], 3, axis=2)
        page = PageImage(
            page_index=0,
            source_page_number=1,
            image=rgb,
            source=SourceMetadata(SourceType.IMAGE, display_name=f"{case.page_id}.png"),
        )
        preprocessed = preprocess_page(page, preprocessing_config)
        detected = segment_page(preprocessed, segmentation_config)
        matches = match_regions(case.regions, detected, iou_threshold=iou_threshold)
        kind_count = sum(
            case.regions[match.expected_index].region_kind
            is detected[match.detected_index].region_kind
            for match in matches
        )
        column_count = sum(
            case.regions[match.expected_index].column_index
            == detected[match.detected_index].column_index
            for match in matches
        )
        expected_sequence = list(range(len(case.regions)))
        detected_sequence = [
            next(
                (
                    match.expected_index
                    for match in matches
                    if match.detected_index == detected_index
                ),
                -1,
            )
            for detected_index in range(len(detected))
        ]
        exact_order = detected_sequence == expected_sequence
        detected_layout = _layout_label(detected, is_blank=preprocessed.is_blank)
        failures: list[str] = []
        if preprocessed.is_blank != case.expected_blank:
            failures.append("blank_state")
        if len(detected) != len(case.regions):
            failures.append("line_count")
        if len(matches) != len(case.regions) or len(matches) != len(detected):
            failures.append("region_matching")
        if kind_count != len(matches):
            failures.append("region_kind")
        if column_count != len(matches):
            failures.append("column_assignment")
        if not exact_order:
            failures.append("reading_order")
        if detected_layout != case.expected_layout:
            failures.append("layout")
        results.append(
            VisionCaseResult(
                page_id=case.page_id,
                case_family=case.case_family,
                layout_family=case.layout_family,
                expected_layout=case.expected_layout,
                detected_layout=detected_layout,
                expected_blank=case.expected_blank,
                detected_blank=preprocessed.is_blank,
                expected_regions=case.regions,
                detected_regions=detected,
                matches=matches,
                exact_line_count=len(detected) == len(case.regions),
                exact_region_kind_count=kind_count,
                exact_column_count=column_count,
                exact_reading_order=exact_order,
                exact_layout=detected_layout == case.expected_layout,
                failures=tuple(failures),
            )
        )
    return VisionBenchmarkResult(
        benchmark_version=VISION_BENCHMARK_VERSION,
        config_fingerprint=config_fingerprint,
        iou_threshold=float(iou_threshold),
        cases=tuple(results),
    )


def _png_bytes(image: NDArray[np.uint8]) -> bytes:
    output = io.BytesIO()
    Image.fromarray(image).save(output, format="PNG", compress_level=9, optimize=False)
    return output.getvalue()


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise BenchmarkError(f"{name} must be a JSON object")
    return value


def _benchmark_texts(corpus: Mapping[str, Any]) -> tuple[str, ...]:
    values = corpus.get("vision_texts")
    if (
        not isinstance(values, list)
        or not values
        or any(not isinstance(text, str) for text in values)
    ):
        raise BenchmarkError("benchmark vision_texts must be a nonempty string list")
    normalized = tuple(values)
    if any(normalize_transcription(text) != text for text in normalized):
        raise BenchmarkError("benchmark vision text must satisfy nfc-v1")
    return normalized


def render_vision_cases(
    config: Mapping[str, Any],
    corpus: Mapping[str, Any],
    *,
    output_directory: str | os.PathLike[str] | None = None,
) -> tuple[VisionBenchmarkCase, ...]:
    """Regenerate frozen benchmark-only pages and optional temporary PNGs."""

    if config.get("benchmark_version") != VISION_BENCHMARK_VERSION:
        raise BenchmarkError("vision benchmark config version is unsupported")
    cases = config.get("cases")
    if not isinstance(cases, list) or not cases:
        raise BenchmarkError("vision config must contain cases")
    common = _mapping(config.get("generator_config"), "generator_config")
    texts = _benchmark_texts(corpus)
    output_root: Path | None = None
    if output_directory is not None:
        output_root = Path(output_directory)
        if not output_root.is_dir() or any(output_root.iterdir()):
            raise BenchmarkError("vision output directory must exist and be empty")
    results: list[VisionBenchmarkCase] = []
    for raw_case in cases:
        case = _mapping(raw_case, "vision case")
        overrides = _mapping(case.get("generator_overrides", {}), "generator_overrides")
        generator = SyntheticDataConfig.from_mapping({**common, **overrides})
        page_id = case.get("page_id")
        case_family = case.get("case_family")
        family = case.get("layout_family")
        expected_layout = case.get("expected_layout")
        if not all(isinstance(value, str) and value for value in (page_id, case_family, family)):
            raise BenchmarkError("vision case identifiers must be nonempty strings")
        generated = generate_page_fixture(
            PageLayoutFamily(family),
            page_id=page_id,
            config=generator,
            texts=texts,
            image_path=f"pages/{page_id}.png",
        )
        image = generated.image.copy()
        variant = case.get("image_variant", "base")
        if variant == "blur_noise":
            image = cv2.GaussianBlur(image, (3, 3), sigmaX=0.6, sigmaY=0.6)
            rng = np.random.default_rng(generator.seed)
            noise = rng.normal(0.0, 0.8, image.shape)
            image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        elif variant != "base":
            raise BenchmarkError("vision image variant is unsupported")
        payload = _png_bytes(image)
        record = replace(generated.record, image_sha256=sha256(payload).hexdigest())
        regions = tuple(
            ExpectedRegion(
                region_id=line.line_id,
                bounding_box=line.output_box,
                region_kind=line.region_kind,
                column_index=line.column_index,
                reading_order_index=line.reading_order_index,
            )
            for line in record.lines
        )
        result = VisionBenchmarkCase(
            page_id=record.page_id,
            case_family=case_family,
            layout_family=record.layout_family.value,
            expected_layout=expected_layout,
            expected_blank=record.expected_blank,
            regions=regions,
            image_sha256=record.image_sha256,
            generator_seed=generator.seed,
            image=image,
        )
        results.append(result)
        if output_root is not None:
            target = output_root / "pages" / f"{page_id}.png"
            target.parent.mkdir(parents=True, exist_ok=True)
            _atomic_write_bytes(
                payload,
                target,
                overwrite=False,
                error_type=BenchmarkError,
                artifact_name="vision benchmark image",
            )
    if len({case.page_id for case in results}) != len(results):
        raise BenchmarkError("vision config contains duplicate page IDs")
    return tuple(results)


def vision_logical_manifest(cases: Sequence[VisionBenchmarkCase]) -> dict[str, object]:
    return {
        "schema_version": BENCHMARK_SCHEMA_VERSION,
        "benchmark_version": VISION_BENCHMARK_VERSION,
        "coordinate_semantics": "half-open-[x,right)-[y,bottom)",
        "images_committed": False,
        "cases": [
            {
                "page_id": case.page_id,
                "case_family": case.case_family,
                "layout_family": case.layout_family,
                "expected_layout": case.expected_layout,
                "expected_blank": case.expected_blank,
                "expected_line_count": len(case.regions),
                "generator_seed": case.generator_seed,
                "image_sha256": case.image_sha256,
                "regions": [region.to_public_dict() for region in case.regions],
            }
            for case in sorted(cases, key=lambda item: item.page_id)
        ],
    }


@dataclass(frozen=True, slots=True)
class RecognitionBenchmarkData:
    samples: tuple[DatasetSample, ...]
    image_records: tuple[dict[str, object], ...]

    @property
    def fingerprint(self) -> str:
        return dataset_fingerprint(self.samples)


_URDU_DIGITS = str.maketrans("0123456789", "۰۱۲۳۴۵۶۷۸۹")


def _composed_text(template: str, number: int) -> str:
    ascii_number = f"{number:03d}"
    try:
        text = template.format(
            ascii_number=ascii_number,
            urdu_number=ascii_number.translate(_URDU_DIGITS),
        )
    except (KeyError, ValueError) as error:
        raise BenchmarkError("recognition corpus template is invalid") from error
    if normalize_transcription(text) != text:
        raise BenchmarkError("composed recognition text does not satisfy nfc-v1")
    return text


def render_recognition_dataset(
    config: Mapping[str, Any],
    corpus: Mapping[str, Any],
    *,
    output_directory: str | os.PathLike[str],
) -> RecognitionBenchmarkData:
    """Generate benchmark line images in an explicit empty temporary directory."""

    if config.get("benchmark_version") != RECOGNITION_BENCHMARK_VERSION:
        raise BenchmarkError("recognition benchmark config version is unsupported")
    document_count = config.get("document_count")
    lines_per_document = config.get("lines_per_document")
    if (
        isinstance(document_count, bool)
        or not isinstance(document_count, int)
        or document_count < 1
        or isinstance(lines_per_document, bool)
        or not isinstance(lines_per_document, int)
        or lines_per_document < 1
    ):
        raise BenchmarkError("recognition benchmark counts must be positive integers")
    templates = corpus.get("recognition_templates")
    if (
        not isinstance(templates, list)
        or len(templates) != lines_per_document
        or any(not isinstance(template, str) for template in templates)
    ):
        raise BenchmarkError("recognition templates must match lines_per_document")
    generator = SyntheticDataConfig.from_mapping(
        _mapping(config.get("generator_config"), "generator_config")
    )
    root = Path(output_directory)
    if not root.is_dir() or any(root.iterdir()):
        raise BenchmarkError("recognition output directory must exist and be empty")
    (root / "lines").mkdir()
    samples: list[DatasetSample] = []
    image_records: list[dict[str, object]] = []
    sample_number = 0
    for document_number in range(1, document_count + 1):
        document_id = f"benchmark-document-{document_number:04d}"
        for line_index, template in enumerate(templates):
            sample_number += 1
            sample_id = f"bench-line-{sample_number:06d}"
            text = _composed_text(template, sample_number)
            generated = generate_line_sample(
                text,
                sample_id=sample_id,
                document_id=document_id,
                config=generator,
                image_path=f"lines/{sample_id}.png",
                text_source_id=f"benchmark-template-{line_index + 1:02d}",
                line_index=line_index,
            )
            target = root / generated.sample.image_path
            _atomic_write_bytes(
                generated.png_bytes,
                target,
                overwrite=False,
                error_type=BenchmarkError,
                artifact_name="recognition benchmark image",
            )
            samples.append(generated.sample)
            image_records.append(
                {
                    "sample_id": sample_id,
                    "document_id": document_id,
                    "line_index": line_index,
                    "text_source_id": generated.record.text_source_id,
                    "text_sha256": generated.record.text_sha256,
                    "image_path": generated.record.image_path,
                    "image_sha256": generated.record.image_sha256,
                    "width": generated.record.width,
                    "height": generated.record.height,
                    "master_seed": generated.record.master_seed,
                    "derived_seed": generated.record.derived_seed,
                }
            )
    if len({sample.text for sample in samples}) != len(samples):
        raise BenchmarkError("recognition benchmark transcriptions must be globally unique")
    manifest = canonical_manifest_bytes(samples)
    _atomic_write_bytes(
        manifest,
        root / "manifest.jsonl",
        overwrite=False,
        error_type=BenchmarkError,
        artifact_name="recognition benchmark manifest",
    )
    return RecognitionBenchmarkData(tuple(samples), tuple(image_records))


def evaluate_checkpoint(
    samples: Sequence[DatasetSample],
    *,
    dataset_root: str | os.PathLike[str],
    recognizer: OCRRecognizer,
    batch_size: int = 16,
    data_fingerprint: str | None = None,
    config_fingerprint: str,
) -> EvaluationReport:
    """Run the existing line-inference stack and evaluate exact sample IDs."""

    from urdu_document_ocr.recognition.inference import recognize_lines

    values = tuple(samples)
    if not values or any(not isinstance(sample, DatasetSample) for sample in values):
        raise BenchmarkError("checkpoint evaluation samples must be nonempty")
    identifiers = [sample.sample_id for sample in values]
    if len(identifiers) != len(set(identifiers)):
        raise BenchmarkError("checkpoint evaluation sample IDs must be unique")
    vocabulary = getattr(recognizer, "vocabulary", None)
    if vocabulary is None:
        raise BenchmarkError("recognizer must expose its checkpoint vocabulary")
    try:
        for sample in values:
            vocabulary.encode(sample.text)
    except (TypeError, ValueError) as error:
        raise BenchmarkError("evaluation references contain unseen characters") from error
    images = tuple(load_dataset_line_image(sample, dataset_root=dataset_root) for sample in values)
    predictions = recognize_lines(images, recognizer, batch_size=batch_size)
    return evaluate_predictions(
        ((sample.sample_id, sample.text) for sample in values),
        (
            (sample.sample_id, prediction.text)
            for sample, prediction in zip(values, predictions, strict=True)
        ),
        data_fingerprint=data_fingerprint or dataset_fingerprint(values),
        config_fingerprint=config_fingerprint,
        model_fingerprint=getattr(recognizer, "fingerprint", None),
    )


def build_artifact_manifest(
    repository_root: str | os.PathLike[str],
    relative_paths: Sequence[str],
) -> dict[str, object]:
    """Build a repository-relative SHA-256 manifest without self-reference."""

    root = Path(repository_root).resolve(strict=True)
    entries: list[dict[str, object]] = []
    for value in sorted(relative_paths):
        path = (root / value).resolve(strict=True)
        if not path.is_relative_to(root) or not path.is_file():
            raise BenchmarkError("artifact path must name a repository file")
        relative = path.relative_to(root).as_posix()
        if relative == "benchmark/artifact_manifest.json":
            raise BenchmarkError("artifact manifest cannot include itself")
        entries.append(
            {
                "path": relative,
                "size_bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
        )
    return {
        "schema_version": BENCHMARK_SCHEMA_VERSION,
        "self_entry_policy": "artifact-manifest-excluded-to-avoid-recursive-hash",
        "artifacts": entries,
    }


def verify_artifact_manifest(
    repository_root: str | os.PathLike[str],
    manifest: Mapping[str, Any],
) -> tuple[int, int]:
    """Verify exact size/hash identity for every frozen benchmark artifact."""

    if manifest.get("schema_version") != BENCHMARK_SCHEMA_VERSION:
        raise BenchmarkError("artifact manifest schema version is unsupported")
    records = manifest.get("artifacts")
    if not isinstance(records, list) or not records:
        raise BenchmarkError("artifact manifest must contain artifacts")
    root = Path(repository_root).resolve(strict=True)
    matched = 0
    seen: set[str] = set()
    for raw in records:
        record = _mapping(raw, "artifact record")
        relative = record.get("path")
        size = record.get("size_bytes")
        digest = record.get("sha256")
        if (
            not isinstance(relative, str)
            or relative in seen
            or isinstance(size, bool)
            or not isinstance(size, int)
            or size < 0
            or not isinstance(digest, str)
            or len(digest) != 64
        ):
            raise BenchmarkError("artifact manifest record is malformed")
        seen.add(relative)
        try:
            path = (root / relative).resolve(strict=True)
        except OSError as error:
            raise BenchmarkError("frozen benchmark artifact is missing") from error
        if not path.is_relative_to(root) or not path.is_file():
            raise BenchmarkError("frozen benchmark artifact escapes repository")
        if path.stat().st_size != size or file_sha256(path) != digest:
            raise BenchmarkError(
                "frozen benchmark artifact failed integrity verification",
                context={"path": relative},
            )
        matched += 1
    return matched, len(records)


def read_benchmark_json(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Read one strict UTF-8 benchmark JSON object."""

    try:
        payload = json.loads(
            Path(path).read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=_reject_nonstandard_json_constant,
        )
    except (OSError, UnicodeError, ValueError) as error:
        raise BenchmarkError("benchmark JSON could not be read strictly") from error
    if not isinstance(payload, dict):
        raise BenchmarkError("benchmark JSON root must be an object")
    return payload


def verify_frozen_sources(
    repository_root: str | os.PathLike[str],
    freeze: Mapping[str, Any],
) -> tuple[int, int]:
    """Verify source artifacts recorded before final test evaluation."""

    records = freeze.get("source_artifacts")
    if not isinstance(records, list) or not records:
        raise BenchmarkError("freeze record must contain source artifacts")
    root = Path(repository_root).resolve(strict=True)
    matched = 0
    for raw in records:
        record = _mapping(raw, "freeze source artifact")
        relative = record.get("path")
        digest = record.get("sha256")
        if not isinstance(relative, str) or not isinstance(digest, str):
            raise BenchmarkError("freeze source artifact is malformed")
        try:
            path = (root / relative).resolve(strict=True)
        except OSError as error:
            raise BenchmarkError("frozen source artifact is missing") from error
        if not path.is_relative_to(root) or not path.is_file() or file_sha256(path) != digest:
            raise BenchmarkError(
                "frozen source artifact identity changed", context={"path": relative}
            )
        matched += 1
    return matched, len(records)


def verify_benchmark_results(
    repository_root: str | os.PathLike[str],
) -> dict[str, object]:
    """Cheaply recalculate committed metrics and rerun the synthetic vision layer."""

    from urdu_document_ocr.evaluation.errors import analyze_errors

    root = Path(repository_root).resolve(strict=True)
    benchmark_root = root / "benchmark"
    artifact_manifest = read_benchmark_json(benchmark_root / "artifact_manifest.json")
    artifact_matches = verify_artifact_manifest(root, artifact_manifest)
    freeze = read_benchmark_json(benchmark_root / "freeze.json")
    source_matches = verify_frozen_sources(root, freeze)
    fingerprints = _mapping(freeze.get("fingerprints"), "freeze fingerprints")

    recognition = read_benchmark_json(benchmark_root / "results" / "recognition_results.json")
    if recognition.get("benchmark_version") != RECOGNITION_BENCHMARK_VERSION:
        raise BenchmarkError("recognition benchmark version does not match")
    if recognition.get("freeze_sha256") != file_sha256(benchmark_root / "freeze.json"):
        raise BenchmarkError("recognition result freeze identity does not match")
    recognition_config = read_benchmark_json(benchmark_root / "recognition_config.json")
    evaluation_config = read_benchmark_json(benchmark_root / "evaluation_config.json")
    expected_evaluation_fingerprint = canonical_fingerprint(
        {"evaluation": evaluation_config, "recognition": recognition_config}
    )
    if recognition.get("frozen_config_fingerprint") != expected_evaluation_fingerprint:
        raise BenchmarkError("recognition evaluation config fingerprint does not match")
    if recognition.get("training_config_fingerprint") != fingerprints.get("training_config"):
        raise BenchmarkError("recognition training config fingerprint does not match")
    dataset_summary = _mapping(recognition.get("dataset"), "recognition dataset")
    dataset_fingerprints = _mapping(
        dataset_summary.get("fingerprints"), "recognition dataset fingerprints"
    )
    for name in ("dataset", "train", "validation", "test", "split", "vocabulary"):
        if dataset_fingerprints.get(name) != fingerprints.get(name):
            raise BenchmarkError("recognition dataset fingerprint does not match freeze")
    evaluation = _mapping(recognition.get("evaluation"), "recognition evaluation")
    metric_policy = _mapping(evaluation.get("metric_policy"), "evaluation metric policy")
    if metric_policy.get("identifier") != freeze.get("metric_policy_id") or metric_policy.get(
        "fingerprint"
    ) != freeze.get("metric_policy_fingerprint"):
        raise BenchmarkError("recognition metric policy does not match freeze")
    sample_records = evaluation.get("samples")
    identities = _mapping(evaluation.get("identities"), "evaluation identities")
    if not isinstance(sample_records, list):
        raise BenchmarkError("recognition evaluation samples are malformed")
    references: list[tuple[str, str]] = []
    predictions: list[tuple[str, str]] = []
    for raw in sample_records:
        sample = _mapping(raw, "evaluation sample")
        sample_id = sample.get("sample_id")
        reference = sample.get("reference")
        hypothesis = sample.get("hypothesis")
        if not all(isinstance(value, str) for value in (sample_id, reference, hypothesis)):
            raise BenchmarkError("evaluation sample text fields are malformed")
        references.append((sample_id, reference))
        predictions.append((sample_id, hypothesis))
    recalculated = evaluate_predictions(
        references,
        predictions,
        data_fingerprint=identities.get("data_fingerprint"),
        config_fingerprint=identities.get("config_fingerprint"),
        model_fingerprint=identities.get("model_fingerprint"),
    )
    if recalculated.to_public_dict() != evaluation:
        raise BenchmarkError("recognition result metrics do not recalculate exactly")
    committed_analysis = read_benchmark_json(benchmark_root / "results" / "error_analysis.json")
    if analyze_errors(recalculated, worst_limit=10).to_public_dict() != committed_analysis:
        raise BenchmarkError("recognition error analysis does not recalculate exactly")

    vision_config = read_benchmark_json(benchmark_root / "vision_config.json")
    corpus = read_benchmark_json(benchmark_root / "recognition_corpus.json")
    cases = render_vision_cases(vision_config, corpus)
    rerun = run_vision_benchmark(
        cases,
        preprocessing_config=PreprocessingConfig.from_mapping(
            _mapping(vision_config.get("preprocessing_config"), "preprocessing_config")
        ),
        segmentation_config=SegmentationConfig.from_mapping(
            _mapping(vision_config.get("segmentation_config"), "segmentation_config")
        ),
        iou_threshold=float(evaluation_config.get("vision_iou_threshold", 0)),
        config_fingerprint=canonical_fingerprint(vision_config),
    ).to_public_dict()
    committed_vision = read_benchmark_json(benchmark_root / "results" / "vision_results.json")
    if rerun != committed_vision:
        raise BenchmarkError("vision benchmark result does not reproduce exactly")
    if committed_vision.get(
        "benchmark_version"
    ) != VISION_BENCHMARK_VERSION or committed_vision.get("config_fingerprint") != fingerprints.get(
        "vision_config"
    ):
        raise BenchmarkError("vision result identity does not match freeze")

    summary = read_benchmark_json(benchmark_root / "results" / "summary.json")
    if summary.get("vision_summary") != committed_vision.get("summary"):
        raise BenchmarkError("benchmark summary does not match vision results")
    if summary.get("recognition_summary") != evaluation.get("summary"):
        raise BenchmarkError("benchmark summary does not match recognition results")
    expected_result_hashes = {
        "vision_results.json": file_sha256(benchmark_root / "results" / "vision_results.json"),
        "recognition_results.json": file_sha256(
            benchmark_root / "results" / "recognition_results.json"
        ),
        "error_analysis.json": file_sha256(benchmark_root / "results" / "error_analysis.json"),
    }
    if summary.get("result_sha256") != expected_result_hashes:
        raise BenchmarkError("benchmark summary result hashes do not match")
    return {
        "artifact_matches": artifact_matches[0],
        "artifact_total": artifact_matches[1],
        "source_matches": source_matches[0],
        "source_total": source_matches[1],
        "recognition_sample_count": len(sample_records),
        "vision_case_count": len(cases),
    }


__all__ = [
    "BENCHMARK_SCHEMA_VERSION",
    "RECOGNITION_BENCHMARK_VERSION",
    "VISION_BENCHMARK_VERSION",
    "ExpectedRegion",
    "RecognitionBenchmarkData",
    "RegionMatch",
    "VisionBenchmarkCase",
    "VisionBenchmarkResult",
    "build_artifact_manifest",
    "canonical_fingerprint",
    "canonical_json_bytes",
    "evaluate_checkpoint",
    "file_sha256",
    "match_regions",
    "read_benchmark_json",
    "render_recognition_dataset",
    "render_vision_cases",
    "run_vision_benchmark",
    "verify_artifact_manifest",
    "verify_benchmark_results",
    "verify_frozen_sources",
    "vision_logical_manifest",
    "write_canonical_json",
]
