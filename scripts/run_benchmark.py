"""Prepare, freeze, run, or verify the explicit synthetic Phase 12 benchmarks."""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import time
from importlib.metadata import version
from pathlib import Path

from urdu_document_ocr import (
    DatasetSplitConfig,
    OCRLineDataset,
    PreprocessingConfig,
    RecognizerConfig,
    SegmentationConfig,
    TrainingConfig,
    analyze_errors,
    assert_shaping_available,
    build_artifact_manifest,
    build_vocabulary,
    canonical_fingerprint,
    dataset_fingerprint,
    evaluate_checkpoint,
    find_unseen_characters,
    load_recognizer,
    load_vocabulary,
    read_benchmark_json,
    render_recognition_dataset,
    render_vision_cases,
    run_vision_benchmark,
    save_vocabulary,
    split_dataset,
    train_model,
    validate_training_data,
    verify_benchmark_results,
    verify_frozen_sources,
    vision_logical_manifest,
    write_canonical_json,
    write_manifest,
    write_split_manifests,
)
from urdu_document_ocr.evaluation import (
    METRIC_POLICY_FINGERPRINT,
    METRIC_POLICY_ID,
    RECOGNITION_BENCHMARK_VERSION,
    VISION_BENCHMARK_VERSION,
)
from urdu_document_ocr.evaluation.benchmark import file_sha256

_SOURCE_ARTIFACTS = (
    "benchmark/evaluation_config.json",
    "benchmark/manifests/recognition_generation.json",
    "benchmark/manifests/recognition_manifest.jsonl",
    "benchmark/manifests/recognition_summary.json",
    "benchmark/manifests/split_metadata.json",
    "benchmark/manifests/splits/test.jsonl",
    "benchmark/manifests/splits/train.jsonl",
    "benchmark/manifests/splits/validation.jsonl",
    "benchmark/manifests/vision_cases.json",
    "benchmark/manifests/vocabulary.json",
    "benchmark/provenance.json",
    "benchmark/recognition_config.json",
    "benchmark/recognition_corpus.json",
    "benchmark/vision_config.json",
    "scripts/run_benchmark.py",
    "src/urdu_document_ocr/evaluation/__init__.py",
    "src/urdu_document_ocr/evaluation/benchmark.py",
    "src/urdu_document_ocr/evaluation/errors.py",
    "src/urdu_document_ocr/evaluation/metrics.py",
    "src/urdu_document_ocr/errors.py",
    "src/urdu_document_ocr/types.py",
)

_REVIEWED_ARTIFACTS = (
    "benchmark/README.md",
    "benchmark/evaluation_config.json",
    "benchmark/freeze.json",
    "benchmark/manifests/recognition_generation.json",
    "benchmark/manifests/recognition_manifest.jsonl",
    "benchmark/manifests/recognition_summary.json",
    "benchmark/manifests/split_metadata.json",
    "benchmark/manifests/splits/test.jsonl",
    "benchmark/manifests/splits/train.jsonl",
    "benchmark/manifests/splits/validation.jsonl",
    "benchmark/manifests/vision_cases.json",
    "benchmark/manifests/vocabulary.json",
    "benchmark/provenance.json",
    "benchmark/recognition_config.json",
    "benchmark/recognition_corpus.json",
    "benchmark/results/error_analysis.json",
    "benchmark/results/recognition_results.json",
    "benchmark/results/summary.json",
    "benchmark/results/vision_results.json",
    "benchmark/vision_config.json",
    "scripts/run_benchmark.py",
)


def _empty_directory(path: Path, role: str) -> Path:
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise ValueError(f"{role} must already exist") from error
    if not resolved.is_dir() or resolved.parent == resolved or any(resolved.iterdir()):
        raise ValueError(f"{role} must be an empty non-root directory")
    return resolved


def _inputs(benchmark: Path) -> tuple[dict, dict, dict, dict]:
    return (
        read_benchmark_json(benchmark / "vision_config.json"),
        read_benchmark_json(benchmark / "recognition_config.json"),
        read_benchmark_json(benchmark / "evaluation_config.json"),
        read_benchmark_json(benchmark / "recognition_corpus.json"),
    )


def _partition_data(samples, recognition_config):
    split = split_dataset(
        samples,
        DatasetSplitConfig.from_mapping(recognition_config["split_config"]),
    )
    vocabulary = build_vocabulary(split.train)
    validation_unseen = find_unseen_characters(split.validation, vocabulary)
    test_unseen = find_unseen_characters(split.test, vocabulary)
    if validation_unseen or test_unseen:
        raise RuntimeError("benchmark validation/test text contains unseen characters")
    text_sets = {
        "train": {sample.text for sample in split.train},
        "validation": {sample.text for sample in split.validation},
        "test": {sample.text for sample in split.test},
    }
    text_overlap = {
        "train_validation": len(text_sets["train"] & text_sets["validation"]),
        "train_test": len(text_sets["train"] & text_sets["test"]),
        "validation_test": len(text_sets["validation"] & text_sets["test"]),
    }
    if any(text_overlap.values()):
        raise RuntimeError("benchmark transcriptions overlap across partitions")
    return split, vocabulary, validation_unseen, test_unseen, text_overlap


def _clean_work_subdirectories(work: Path) -> None:
    root = work.resolve(strict=True)
    for name in ("recognition", "training", "vision"):
        child = (root / name).resolve(strict=False)
        if not child.is_relative_to(root) or child.parent != root:
            raise RuntimeError("benchmark work cleanup target escaped its root")
        if child.exists():
            shutil.rmtree(child)


def prepare(repository: Path, benchmark: Path, work: Path) -> dict[str, object]:
    """Generate and commit only logical benchmark manifests, never bulk PNGs."""

    vision_config, recognition_config, evaluation_config, corpus = _inputs(benchmark)
    for value, filename in (
        (vision_config, "vision_config.json"),
        (recognition_config, "recognition_config.json"),
        (evaluation_config, "evaluation_config.json"),
        (corpus, "recognition_corpus.json"),
    ):
        write_canonical_json(value, benchmark / filename, overwrite=True)
    manifests = benchmark / "manifests"
    if manifests.exists() and any(manifests.rglob("*")):
        raise ValueError("benchmark manifests directory must not contain prior artifacts")
    manifests.mkdir(exist_ok=True)

    vision_work = work / "vision"
    recognition_work = work / "recognition"
    vision_work.mkdir()
    recognition_work.mkdir()
    vision_cases = render_vision_cases(
        vision_config,
        corpus,
        output_directory=vision_work,
    )
    vision_manifest = vision_logical_manifest(vision_cases)
    write_canonical_json(vision_manifest, manifests / "vision_cases.json")

    recognition = render_recognition_dataset(
        recognition_config,
        corpus,
        output_directory=recognition_work,
    )
    split, vocabulary, validation_unseen, test_unseen, text_overlap = _partition_data(
        recognition.samples, recognition_config
    )
    write_manifest(recognition.samples, manifests / "recognition_manifest.jsonl")
    split_directory = manifests / "splits"
    split_directory.mkdir()
    write_split_manifests(split, split_directory)
    (split_directory / "split_metadata.json").replace(manifests / "split_metadata.json")
    save_vocabulary(vocabulary, manifests / "vocabulary.json")
    generation = {
        "schema_version": 1,
        "benchmark_version": RECOGNITION_BENCHMARK_VERSION,
        "images_committed": False,
        "dataset_fingerprint": recognition.fingerprint,
        "images": list(recognition.image_records),
    }
    write_canonical_json(generation, manifests / "recognition_generation.json")
    summary = {
        "schema_version": 1,
        "benchmark_version": RECOGNITION_BENCHMARK_VERSION,
        "sample_counts": {
            "all": len(recognition.samples),
            "train": len(split.train),
            "validation": len(split.validation),
            "test": len(split.test),
        },
        "document_counts": {
            "all": len({sample.document_id for sample in recognition.samples}),
            "train": len({sample.document_id for sample in split.train}),
            "validation": len({sample.document_id for sample in split.validation}),
            "test": len({sample.document_id for sample in split.test}),
        },
        "fingerprints": {
            "dataset": recognition.fingerprint,
            "train": dataset_fingerprint(split.train),
            "validation": dataset_fingerprint(split.validation),
            "test": dataset_fingerprint(split.test),
            "split": split.fingerprint,
            "vocabulary": vocabulary.fingerprint,
        },
        "text_overlap": text_overlap,
        "validation_unseen_characters": list(validation_unseen),
        "test_unseen_characters": list(test_unseen),
        "normalization_policy": "nfc-v1",
    }
    write_canonical_json(summary, manifests / "recognition_summary.json")
    _clean_work_subdirectories(work)
    return {
        "vision_cases": len(vision_cases),
        "recognition_samples": len(recognition.samples),
        "train_samples": len(split.train),
        "validation_samples": len(split.validation),
        "test_samples": len(split.test),
        "test_predictions_observed": False,
    }


def freeze(repository: Path, benchmark: Path) -> dict[str, object]:
    """Write the immutable pre-result identity record after inputs are complete."""

    from urdu_document_ocr import CRNNRecognizer

    vision_config, recognition_config, evaluation_config, corpus = _inputs(benchmark)
    summary = read_benchmark_json(benchmark / "manifests" / "recognition_summary.json")
    vocabulary = load_vocabulary(benchmark / "manifests" / "vocabulary.json")
    recognizer_config = RecognizerConfig.from_mapping(recognition_config["recognizer_config"])
    training_config = TrainingConfig.from_mapping(recognition_config["training_config"])
    model = CRNNRecognizer(vocabulary, recognizer_config)
    capabilities = assert_shaping_available()
    sources = [
        {"path": path, "sha256": file_sha256(repository / path)} for path in _SOURCE_ARTIFACTS
    ]
    payload = {
        "schema_version": 1,
        "freeze_id": "urdu-ocr-synthetic-benchmarks-2026-09-08-v1",
        "frozen_on": "2026-09-08",
        "vision_benchmark_version": VISION_BENCHMARK_VERSION,
        "recognition_benchmark_version": RECOGNITION_BENCHMARK_VERSION,
        "metric_policy_id": METRIC_POLICY_ID,
        "metric_policy_fingerprint": METRIC_POLICY_FINGERPRINT,
        "font_sha256": capabilities.font_sha256,
        "generator_version": "synthetic-v1",
        "text_source_fingerprint": canonical_fingerprint(corpus),
        "seeds": {
            "vision_case_seeds": [
                case["generator_overrides"]["seed"] for case in vision_config["cases"]
            ],
            "recognition_generation": recognition_config["generator_config"]["seed"],
            "recognition_split": recognition_config["split_config"]["seed"],
            "recognition_training": recognition_config["training_config"]["seed"],
        },
        "fingerprints": {
            "vision_config": canonical_fingerprint(vision_config),
            "preprocessing_config": PreprocessingConfig.from_mapping(
                vision_config["preprocessing_config"]
            ).fingerprint,
            "segmentation_config": SegmentationConfig.from_mapping(
                vision_config["segmentation_config"]
            ).fingerprint,
            "recognition_config": canonical_fingerprint(recognition_config),
            "recognition_generator_config": canonical_fingerprint(
                recognition_config["generator_config"]
            ),
            "recognizer_config": recognizer_config.fingerprint,
            "model": model.fingerprint,
            "training_config": training_config.fingerprint,
            "evaluation_config": canonical_fingerprint(evaluation_config),
            **summary["fingerprints"],
        },
        "iou_threshold": evaluation_config["vision_iou_threshold"],
        "source_artifacts": sources,
        "freeze_sequence": [
            "source-text-and-config-finalized",
            "seeds-and-splits-finalized",
            "font-and-generator-identified",
            "vision-and-recognition-configs-finalized",
            "logical-manifests-generated-and-fingerprinted",
            "freeze-record-written",
            "final-vision-run-permitted",
            "final-recognition-training-and-test-permitted",
        ],
        "final_test_metrics_observed_before_freeze": False,
        "final_test_used_for_tuning": False,
        "post_test_config_changes": 0,
        "benchmark_invalidations": 0,
        "failures_removed": 0,
        "implementation_base_commit": "5f4973592732acdeb7bd347980fcba5582375098",
    }
    write_canonical_json(payload, benchmark / "freeze.json")
    return {
        "freeze_sha256": file_sha256(benchmark / "freeze.json"),
        "source_artifacts": len(sources),
        "final_test_metrics_observed": False,
    }


def preflight(benchmark: Path, work: Path) -> dict[str, object]:
    """Load every final train/validation image without observing test predictions."""

    _, recognition_config, _, corpus = _inputs(benchmark)
    recognition_work = work / "recognition"
    recognition_work.mkdir()
    try:
        recognition = render_recognition_dataset(
            recognition_config,
            corpus,
            output_directory=recognition_work,
        )
        split, vocabulary, validation_unseen, test_unseen, text_overlap = _partition_data(
            recognition.samples, recognition_config
        )
        recognizer_config = RecognizerConfig.from_mapping(recognition_config["recognizer_config"])
        report = validate_training_data(
            OCRLineDataset(
                split.train,
                vocabulary,
                dataset_root=recognition_work,
                recognizer_config=recognizer_config,
            ),
            OCRLineDataset(
                split.validation,
                vocabulary,
                dataset_root=recognition_work,
                recognizer_config=recognizer_config,
            ),
        )
        return {
            "train_samples_loaded": report.train_sample_count,
            "validation_samples_loaded": report.validation_sample_count,
            "validation_unseen_characters": len(validation_unseen),
            "test_unseen_characters": len(test_unseen),
            "cross_partition_text_overlaps": sum(text_overlap.values()),
            "test_predictions_observed": False,
        }
    finally:
        _clean_work_subdirectories(work)


def _verify_generated_inputs(benchmark: Path, vision_cases, recognition, split, vocabulary) -> None:
    expected_vision = (benchmark / "manifests" / "vision_cases.json").read_bytes()
    from urdu_document_ocr.data.manifest import canonical_manifest_bytes
    from urdu_document_ocr.evaluation.benchmark import canonical_json_bytes

    if canonical_json_bytes(vision_logical_manifest(vision_cases)) != expected_vision:
        raise RuntimeError("regenerated vision cases differ from the frozen manifest")
    expected_generation = read_benchmark_json(
        benchmark / "manifests" / "recognition_generation.json"
    )
    actual_generation = {
        "schema_version": 1,
        "benchmark_version": RECOGNITION_BENCHMARK_VERSION,
        "images_committed": False,
        "dataset_fingerprint": recognition.fingerprint,
        "images": list(recognition.image_records),
    }
    if actual_generation != expected_generation:
        raise RuntimeError("regenerated recognition images differ from frozen identities")
    if (
        canonical_manifest_bytes(recognition.samples)
        != (benchmark / "manifests" / "recognition_manifest.jsonl").read_bytes()
    ):
        raise RuntimeError("regenerated recognition manifest differs from freeze")
    for name, samples in (
        ("train", split.train),
        ("validation", split.validation),
        ("test", split.test),
    ):
        if (
            canonical_manifest_bytes(samples)
            != (benchmark / "manifests" / "splits" / f"{name}.jsonl").read_bytes()
        ):
            raise RuntimeError(f"regenerated {name} partition differs from freeze")
    if vocabulary != load_vocabulary(benchmark / "manifests" / "vocabulary.json"):
        raise RuntimeError("regenerated train-only vocabulary differs from freeze")


def _environment(capabilities) -> dict[str, object]:
    import cv2
    import numpy
    import torch
    from PIL import __version__ as pillow_version

    return {
        "python": f"CPython-{platform.python_version()}",
        "torch": torch.__version__,
        "numpy": numpy.__version__,
        "opencv_python_headless": cv2.__version__,
        "pillow": pillow_version,
        "uharfbuzz": version("uharfbuzz"),
        "freetype_py": version("freetype-py"),
        "harfbuzz": capabilities.harfbuzz_version,
        "freetype": capabilities.freetype_version,
        "device": "cpu",
        "font_sha256": capabilities.font_sha256,
    }


def run(repository: Path, benchmark: Path, work: Path, results: Path) -> dict[str, object]:
    """Run both final layers only after source/freeze verification succeeds."""

    from urdu_document_ocr import CRNNRecognizer

    freeze_record = read_benchmark_json(benchmark / "freeze.json")
    verify_frozen_sources(repository, freeze_record)
    vision_config, recognition_config, evaluation_config, corpus = _inputs(benchmark)
    vision_work = work / "vision"
    recognition_work = work / "recognition"
    vision_work.mkdir()
    recognition_work.mkdir()
    started = time.perf_counter()
    vision_cases = render_vision_cases(vision_config, corpus, output_directory=vision_work)
    recognition = render_recognition_dataset(
        recognition_config,
        corpus,
        output_directory=recognition_work,
    )
    split, vocabulary, _, _, _ = _partition_data(recognition.samples, recognition_config)
    _verify_generated_inputs(benchmark, vision_cases, recognition, split, vocabulary)

    vision_result = run_vision_benchmark(
        vision_cases,
        preprocessing_config=PreprocessingConfig.from_mapping(
            vision_config["preprocessing_config"]
        ),
        segmentation_config=SegmentationConfig.from_mapping(vision_config["segmentation_config"]),
        iou_threshold=evaluation_config["vision_iou_threshold"],
        config_fingerprint=canonical_fingerprint(vision_config),
    ).to_public_dict()
    vision_elapsed = time.perf_counter() - started

    training_config = TrainingConfig.from_mapping(recognition_config["training_config"])
    recognizer_config = RecognizerConfig.from_mapping(recognition_config["recognizer_config"])
    model = CRNNRecognizer(vocabulary, recognizer_config)
    training_started = time.perf_counter()
    training = train_model(
        model,
        split.train,
        split.validation,
        vocabulary,
        dataset_root=recognition_work,
        output_directory=work / "training",
        config=training_config,
    )
    recognizer = load_recognizer(work / "training" / training.checkpoint_directory)
    evaluation_fingerprint = canonical_fingerprint(
        {"evaluation": evaluation_config, "recognition": recognition_config}
    )
    report = evaluate_checkpoint(
        split.test,
        dataset_root=recognition_work,
        recognizer=recognizer,
        batch_size=recognition_config["inference_batch_size"],
        data_fingerprint=dataset_fingerprint(split.test),
        config_fingerprint=evaluation_fingerprint,
    )
    test_elapsed = time.perf_counter() - training_started
    error_analysis = analyze_errors(
        report, worst_limit=evaluation_config["worst_sample_limit"]
    ).to_public_dict()
    capabilities = assert_shaping_available()
    metadata = recognizer.checkpoint_metadata
    recognition_result = {
        "schema_version": 1,
        "benchmark_version": RECOGNITION_BENCHMARK_VERSION,
        "evidence_class": "synthetic_recognition",
        "label": "SYNTHETIC OCR RECOGNITION BENCHMARK",
        "frozen_config_fingerprint": evaluation_fingerprint,
        "freeze_sha256": file_sha256(benchmark / "freeze.json"),
        "dataset": read_benchmark_json(benchmark / "manifests" / "recognition_summary.json"),
        "model": {
            "architecture_id": metadata.architecture_id,
            "model_fingerprint": metadata.model_fingerprint,
            "model_weights_sha256": metadata.model_weights_sha256,
            "vocabulary_fingerprint": metadata.vocabulary_fingerprint,
            "checkpoint_format": "safetensors-plus-validated-json-v1",
            "checkpoint_committed": False,
        },
        "training": training.to_public_dict(),
        "training_config_fingerprint": training_config.fingerprint,
        "selection_criterion": "minimum validation CTC loss only",
        "decoder": "greedy-v1",
        "environment": _environment(capabilities),
        "evaluation": report.to_public_dict(),
        "prediction_fingerprint": canonical_fingerprint(
            [[sample.sample_id, sample.hypothesis] for sample in report.samples]
        ),
        "source_artifact_hashes": {
            "recognition_config": file_sha256(benchmark / "recognition_config.json"),
            "recognition_corpus": file_sha256(benchmark / "recognition_corpus.json"),
            "test_manifest": file_sha256(benchmark / "manifests" / "splits" / "test.jsonl"),
        },
        "limitations": [
            "Deterministic synthetically rendered Urdu/Nastaliq line images only.",
            "No historical, scanned-book, handwriting, arbitrary-document, or production claim.",
            "No canonical model checkpoint is published or retained.",
        ],
    }
    write_canonical_json(vision_result, results / "vision_results.json")
    write_canonical_json(recognition_result, results / "recognition_results.json")
    write_canonical_json(error_analysis, results / "error_analysis.json")
    summary = {
        "schema_version": 1,
        "vision_benchmark_version": VISION_BENCHMARK_VERSION,
        "recognition_benchmark_version": RECOGNITION_BENCHMARK_VERSION,
        "freeze_sha256": file_sha256(benchmark / "freeze.json"),
        "vision_summary": vision_result["summary"],
        "recognition_summary": report.to_public_dict()["summary"],
        "result_sha256": {
            "vision_results.json": file_sha256(results / "vision_results.json"),
            "recognition_results.json": file_sha256(results / "recognition_results.json"),
            "error_analysis.json": file_sha256(results / "error_analysis.json"),
        },
        "benchmark_invalidations": 0,
        "final_test_used_for_tuning": False,
        "post_test_config_changes": 0,
        "failures_removed": 0,
        "limitations": [
            "Both result layers are synthetic and do not establish real-world OCR accuracy."
        ],
    }
    write_canonical_json(summary, results / "summary.json")
    _clean_work_subdirectories(work)
    return {
        "vision_seconds": vision_elapsed,
        "recognition_training_and_test_seconds": test_elapsed,
        "best_epoch": training.best_epoch,
        "best_validation_loss": training.best_validation_loss,
        "test_sample_count": report.aggregate.sample_count,
        "character_error_rate": report.aggregate.character_error_rate,
        "word_error_rate": report.aggregate.word_error_rate,
        "exact_match_count": report.aggregate.exact_matches,
        "checkpoint_removed": not (work / "training").exists(),
    }


def create_artifact_manifest(repository: Path, benchmark: Path) -> dict[str, object]:
    """Create the final reviewed-file manifest after results and docs exist."""

    payload = build_artifact_manifest(repository, _REVIEWED_ARTIFACTS)
    write_canonical_json(payload, benchmark / "artifact_manifest.json")
    return {
        "artifact_manifest_sha256": file_sha256(benchmark / "artifact_manifest.json"),
        "artifact_count": len(payload["artifacts"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode", choices=("prepare", "preflight", "freeze", "run", "manifest", "verify")
    )
    parser.add_argument("--repository-directory", type=Path, default=Path.cwd())
    parser.add_argument("--benchmark-directory", type=Path, default=Path("benchmark"))
    parser.add_argument("--work-directory", type=Path)
    parser.add_argument("--results-directory", type=Path)
    args = parser.parse_args()

    repository = args.repository_directory.resolve(strict=True)
    benchmark = (
        args.benchmark_directory
        if args.benchmark_directory.is_absolute()
        else repository / args.benchmark_directory
    ).resolve(strict=True)
    if args.mode in {"prepare", "preflight", "run"}:
        if args.work_directory is None:
            parser.error("--work-directory is required for prepare, preflight, and run")
        work = _empty_directory(args.work_directory, "work directory")
    if args.mode == "prepare":
        result = prepare(repository, benchmark, work)
    elif args.mode == "preflight":
        result = preflight(benchmark, work)
    elif args.mode == "freeze":
        result = freeze(repository, benchmark)
    elif args.mode == "run":
        if args.results_directory is None:
            parser.error("--results-directory is required for run")
        results = _empty_directory(args.results_directory, "results directory")
        result = run(repository, benchmark, work, results)
    elif args.mode == "manifest":
        result = create_artifact_manifest(repository, benchmark)
    else:
        result = verify_benchmark_results(repository)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
