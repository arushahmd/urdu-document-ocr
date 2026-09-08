from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

import pytest

from urdu_document_ocr import (
    BenchmarkError,
    PreprocessingConfig,
    SegmentationConfig,
    SyntheticDataConfig,
)
from urdu_document_ocr.data import (
    dataset_fingerprint,
    find_unseen_characters,
    load_vocabulary,
    read_manifest,
)
from urdu_document_ocr.evaluation import (
    ExpectedRegion,
    build_artifact_manifest,
    canonical_fingerprint,
    match_regions,
    render_recognition_dataset,
    render_vision_cases,
    run_vision_benchmark,
    verify_artifact_manifest,
    verify_benchmark_results,
    vision_logical_manifest,
)
from urdu_document_ocr.types import BoundingBox, LineRegion, RegionKind

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _expected(name: str, box: BoundingBox, order: int) -> ExpectedRegion:
    return ExpectedRegion(name, box, RegionKind.LINE, None, order)


def _detected(name: str, box: BoundingBox, order: int) -> LineRegion:
    return LineRegion(name, 0, box, None, order)


def test_greedy_iou_matching_is_one_to_one_and_stable() -> None:
    expected = (
        _expected("a", BoundingBox(0, 0, 10, 10), 0),
        _expected("b", BoundingBox(20, 0, 10, 10), 1),
    )
    detected = (
        _detected("x", BoundingBox(20, 0, 10, 10), 0),
        _detected("y", BoundingBox(0, 0, 10, 10), 1),
    )

    first = match_regions(expected, detected, iou_threshold=0.75)
    second = match_regions(expected, detected, iou_threshold=0.75)

    assert first == second
    assert [(match.expected_index, match.detected_index) for match in first] == [(0, 1), (1, 0)]
    assert all(match.intersection_over_union == 1.0 for match in first)


def test_iou_matching_uses_stable_indices_for_equal_candidates() -> None:
    expected = (_expected("a", BoundingBox(0, 0, 10, 10), 0),)
    detected = (
        _detected("x", BoundingBox(0, 0, 10, 10), 0),
        _detected("y", BoundingBox(0, 0, 10, 10), 1),
    )

    assert match_regions(expected, detected, iou_threshold=1.0)[0].detected_index == 0
    with pytest.raises(BenchmarkError, match="threshold"):
        match_regions(expected, detected, iou_threshold=0)


def _vision_config() -> dict[str, object]:
    return {
        "benchmark_version": "vision-synthetic-v1",
        "generator_config": SyntheticDataConfig(seed=801_001).to_dict(),
        "cases": [
            {
                "page_id": "syn-page-benchmark-blank",
                "case_family": "blank",
                "layout_family": "blank",
                "expected_layout": "blank",
                "image_variant": "base",
                "generator_overrides": {"seed": 801_001},
            },
            {
                "page_id": "syn-page-benchmark-one",
                "case_family": "one_column",
                "layout_family": "one_column",
                "expected_layout": "one_column",
                "image_variant": "base",
                "generator_overrides": {"seed": 801_002},
            },
        ],
    }


def _corpus() -> dict[str, object]:
    return {
        "vision_texts": [
            "محفوظ اردو متن واضح ہے۔",
            "نیم\u200cخودکار سطر درست ہے۔",
            "عدد ۱۲۳، سوال؟ جواب تیار ہے۔",
        ],
        "recognition_templates": [
            "محفوظ نمونہ {urdu_number}: اردو متن واضح ہے۔",
            "کوڈ {ascii_number}، نیم\u200cخودکار سطر درست ہے۔",
        ],
    }


def test_vision_generation_and_result_are_deterministic(tmp_path: Path) -> None:
    first = render_vision_cases(_vision_config(), _corpus())
    second = render_vision_cases(_vision_config(), _corpus())

    assert [case.image_sha256 for case in first] == [case.image_sha256 for case in second]
    assert first[0].expected_blank and first[0].regions == ()
    logical = vision_logical_manifest(first)
    assert logical["images_committed"] is False
    assert logical["cases"][0]["page_id"] == "syn-page-benchmark-blank"  # type: ignore[index]

    result = run_vision_benchmark(
        first,
        preprocessing_config=PreprocessingConfig(),
        segmentation_config=SegmentationConfig(),
        iou_threshold=0.75,
        config_fingerprint=canonical_fingerprint(_vision_config()),
    ).to_public_dict()
    assert result["summary"]["page_count"] == 2  # type: ignore[index]
    assert result["summary"]["blank_page_correct_count"] >= 1  # type: ignore[index]
    assert len(result["cases"]) == 2


def test_small_recognition_generation_is_unique_and_repeatable(tmp_path: Path) -> None:
    config = {
        "benchmark_version": "recognition-synthetic-v1",
        "document_count": 2,
        "lines_per_document": 2,
        "generator_config": SyntheticDataConfig(seed=802_001).to_dict(),
    }
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    first_root.mkdir()
    second_root.mkdir()

    first = render_recognition_dataset(config, _corpus(), output_directory=first_root)
    second = render_recognition_dataset(config, _corpus(), output_directory=second_root)

    assert len(first.samples) == 4
    assert len({sample.text for sample in first.samples}) == 4
    assert first.fingerprint == second.fingerprint
    assert first.image_records == second.image_records
    assert (first_root / "manifest.jsonl").read_bytes() == (
        second_root / "manifest.jsonl"
    ).read_bytes()


def test_artifact_manifest_verifies_and_detects_change(tmp_path: Path) -> None:
    target = tmp_path / "benchmark" / "config.json"
    target.parent.mkdir()
    target.write_text('{"safe":true}\n', encoding="utf-8")
    manifest = build_artifact_manifest(tmp_path, ("benchmark/config.json",))

    assert verify_artifact_manifest(tmp_path, manifest) == (1, 1)
    assert manifest["artifacts"][0]["sha256"] == sha256(target.read_bytes()).hexdigest()  # type: ignore[index]
    target.write_text(json.dumps({"safe": False}), encoding="utf-8")
    with pytest.raises(BenchmarkError, match="integrity"):
        verify_artifact_manifest(tmp_path, manifest)


def test_frozen_recognition_partitions_are_group_disjoint_and_train_vocabulary_covers() -> None:
    manifests = REPOSITORY_ROOT / "benchmark" / "manifests"
    train = read_manifest(manifests / "splits" / "train.jsonl")
    validation = read_manifest(manifests / "splits" / "validation.jsonl")
    test = read_manifest(manifests / "splits" / "test.jsonl")
    vocabulary = load_vocabulary(manifests / "vocabulary.json")
    summary = json.loads((manifests / "recognition_summary.json").read_text(encoding="utf-8"))

    assert (len(train), len(validation), len(test)) == (192, 32, 32)
    document_sets = tuple(
        {sample.document_id for sample in partition} for partition in (train, validation, test)
    )
    assert not document_sets[0] & document_sets[1]
    assert not document_sets[0] & document_sets[2]
    assert not document_sets[1] & document_sets[2]
    assert len({sample.text for sample in train + validation + test}) == 256
    assert find_unseen_characters(validation, vocabulary) == ()
    assert find_unseen_characters(test, vocabulary) == ()
    assert summary["fingerprints"]["train"] == dataset_fingerprint(train)
    assert summary["fingerprints"]["validation"] == dataset_fingerprint(validation)
    assert summary["fingerprints"]["test"] == dataset_fingerprint(test)
    assert all(value == 0 for value in summary["text_overlap"].values())


def test_benchmark_bulk_images_and_checkpoints_are_not_committed() -> None:
    benchmark = REPOSITORY_ROOT / "benchmark"

    assert not list(benchmark.rglob("*.png"))
    assert not [
        path
        for path in benchmark.rglob("*")
        if path.suffix.lower() in {".pt", ".pth", ".ckpt", ".safetensors"}
    ]


def test_committed_benchmark_results_recalculate_and_vision_reruns() -> None:
    verification = verify_benchmark_results(REPOSITORY_ROOT)

    assert verification["recognition_sample_count"] == 32
    assert verification["vision_case_count"] == 13
    assert verification["artifact_matches"] == verification["artifact_total"]
    assert verification["source_matches"] == verification["source_total"]
