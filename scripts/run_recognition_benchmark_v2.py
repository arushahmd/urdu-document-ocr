"""Prepare, freeze, run once, and verify recognition-synthetic-v2.

The initial frozen Phase 12 benchmark remains owned by ``run_benchmark.py``.
This separate entry point is intentionally recognition-only so invoking the
existing script never launches the longer V2 training protocol.
"""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import time
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path

from urdu_document_ocr import (
    CRNNRecognizer,
    DatasetSplitConfig,
    OCRLineDataset,
    RecognizerConfig,
    SyntheticDataConfig,
    TrainingConfig,
    analyze_errors,
    assert_shaping_available,
    build_artifact_manifest,
    build_vocabulary,
    canonical_fingerprint,
    dataset_fingerprint,
    evaluate_checkpoint,
    evaluate_predictions,
    find_unseen_characters,
    generate_line_sample,
    load_recognizer,
    load_vocabulary,
    read_benchmark_json,
    save_vocabulary,
    split_dataset,
    train_model,
    validate_training_data,
    verify_artifact_manifest,
    verify_frozen_sources,
    write_canonical_json,
    write_manifest,
    write_split_manifests,
)
from urdu_document_ocr.data.manifest import canonical_manifest_bytes
from urdu_document_ocr.evaluation import METRIC_POLICY_FINGERPRINT, METRIC_POLICY_ID
from urdu_document_ocr.evaluation.benchmark import file_sha256
from urdu_document_ocr.training.trainer import set_training_seed

V2_VERSION = "recognition-synthetic-v2"
V2_LABEL = "SYNTHETIC OCR RECOGNITION BENCHMARK — V2"
V2_FREEZE_ID = "urdu-ocr-recognition-synthetic-v2-2026-09-09"
_URDU_DIGITS = str.maketrans("0123456789", "۰۱۲۳۴۵۶۷۸۹")


@dataclass(frozen=True, slots=True)
class V2Data:
    samples: tuple
    image_records: tuple[dict[str, object], ...]

    @property
    def fingerprint(self) -> str:
        return dataset_fingerprint(self.samples)


def _empty_directory(path: Path, role: str) -> Path:
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise ValueError(f"{role} must already exist") from error
    if not resolved.is_dir() or resolved.parent == resolved or any(resolved.iterdir()):
        raise ValueError(f"{role} must be an empty non-root directory")
    return resolved


def _clean_work(work: Path) -> None:
    root = work.resolve(strict=True)
    for name in ("recognition-v2", "training-v2"):
        target = (root / name).resolve(strict=False)
        if target.parent != root or not target.is_relative_to(root):
            raise RuntimeError("V2 cleanup target escaped the work directory")
        if target.exists():
            shutil.rmtree(target)


def _compose_text(template: str, number: int) -> str:
    ascii_number = f"{number:04d}"
    try:
        value = template.format(
            ascii_number=ascii_number,
            urdu_number=ascii_number.translate(_URDU_DIGITS),
        )
    except (KeyError, ValueError) as error:
        raise ValueError("V2 recognition corpus template is invalid") from error
    if not value or value != value.strip():
        raise ValueError("V2 recognition text must be nonempty without edge whitespace")
    return value


def render_v2_dataset(
    config: dict[str, object],
    corpus: dict[str, object],
    *,
    output_directory: Path,
) -> V2Data:
    """Render V2 identities with the reviewed synthetic-v1 shaping engine."""

    if config.get("benchmark_version") != V2_VERSION:
        raise ValueError("V2 benchmark version is unsupported")
    document_count = config.get("document_count")
    lines_per_document = config.get("lines_per_document")
    templates = corpus.get("recognition_templates")
    if (
        isinstance(document_count, bool)
        or not isinstance(document_count, int)
        or document_count < 1
        or isinstance(lines_per_document, bool)
        or not isinstance(lines_per_document, int)
        or lines_per_document < 1
        or not isinstance(templates, list)
        or len(templates) != lines_per_document
        or any(not isinstance(item, str) for item in templates)
    ):
        raise ValueError("V2 counts and corpus templates are inconsistent")
    raw_generator = config.get("generator_config")
    if not isinstance(raw_generator, dict):
        raise ValueError("V2 generator_config must be an object")
    generator = SyntheticDataConfig.from_mapping(raw_generator)
    root = output_directory.resolve(strict=True)
    if not root.is_dir() or any(root.iterdir()):
        raise ValueError("V2 output directory must be empty")
    lines = root / "lines"
    lines.mkdir()
    samples = []
    records: list[dict[str, object]] = []
    sample_number = 0
    for document_number in range(1, document_count + 1):
        document_id = f"v2-document-{document_number:04d}"
        for line_index, template in enumerate(templates):
            sample_number += 1
            sample_id = f"v2-line-{sample_number:06d}"
            text = _compose_text(template, sample_number)
            generated = generate_line_sample(
                text,
                sample_id=sample_id,
                document_id=document_id,
                config=generator,
                image_path=f"lines/{sample_id}.png",
                text_source_id=f"v2-template-{line_index + 1:02d}",
                line_index=line_index,
            )
            target = root / generated.sample.image_path
            target.write_bytes(generated.png_bytes)
            samples.append(generated.sample)
            records.append(
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
        raise ValueError("V2 transcriptions must be globally unique")
    (root / "manifest.jsonl").write_bytes(canonical_manifest_bytes(samples))
    return V2Data(tuple(samples), tuple(records))


def _inputs(benchmark: Path) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    config = read_benchmark_json(benchmark / "config.json")
    corpus = read_benchmark_json(benchmark / "corpus.json")
    provenance = read_benchmark_json(benchmark / "provenance.json")
    if config.get("benchmark_version") != V2_VERSION:
        raise ValueError("V2 config has the wrong benchmark version")
    if provenance.get("benchmark_version") != V2_VERSION:
        raise ValueError("V2 provenance has the wrong benchmark version")
    return config, corpus, provenance


def _partition(samples: tuple, config: dict[str, object]):
    raw_split = config.get("split_config")
    if not isinstance(raw_split, dict):
        raise ValueError("V2 split_config must be an object")
    split = split_dataset(samples, DatasetSplitConfig.from_mapping(raw_split))
    vocabulary = build_vocabulary(split.train)
    validation_unseen = find_unseen_characters(split.validation, vocabulary)
    test_unseen = find_unseen_characters(split.test, vocabulary)
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
    if validation_unseen or test_unseen or any(text_overlap.values()):
        raise RuntimeError("V2 split violates vocabulary or transcription isolation")
    return split, vocabulary, validation_unseen, test_unseen, text_overlap


def prepare(repository: Path, benchmark: Path, work: Path) -> dict[str, object]:
    """Create deterministic logical identities without model inference."""

    del repository
    config, corpus, _ = _inputs(benchmark)
    manifests = benchmark / "manifests"
    if manifests.exists():
        raise ValueError("V2 manifests already exist")
    manifests.mkdir()
    generated_root = work / "recognition-v2"
    generated_root.mkdir()
    try:
        data = render_v2_dataset(config, corpus, output_directory=generated_root)
        split, vocabulary, validation_unseen, test_unseen, text_overlap = _partition(
            data.samples, config
        )
        write_manifest(data.samples, manifests / "manifest.jsonl")
        split_directory = manifests / "splits"
        split_directory.mkdir()
        write_split_manifests(split, split_directory)
        (split_directory / "split_metadata.json").replace(manifests / "split_metadata.json")
        save_vocabulary(vocabulary, manifests / "vocabulary.json")
        write_canonical_json(
            {
                "schema_version": 1,
                "benchmark_version": V2_VERSION,
                "generator_version": "synthetic-v1",
                "images_committed": False,
                "dataset_fingerprint": data.fingerprint,
                "images": list(data.image_records),
            },
            manifests / "generation.json",
        )
        summary = {
            "schema_version": 1,
            "benchmark_version": V2_VERSION,
            "sample_counts": {
                "all": len(data.samples),
                "train": len(split.train),
                "validation": len(split.validation),
                "test": len(split.test),
            },
            "document_counts": {
                "all": len({sample.document_id for sample in data.samples}),
                "train": len({sample.document_id for sample in split.train}),
                "validation": len({sample.document_id for sample in split.validation}),
                "test": len({sample.document_id for sample in split.test}),
            },
            "fingerprints": {
                "dataset": data.fingerprint,
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
            "final_test_predictions_observed": False,
        }
        write_canonical_json(summary, manifests / "summary.json")
        return {
            "benchmark_version": V2_VERSION,
            "sample_counts": summary["sample_counts"],
            "document_counts": summary["document_counts"],
            "test_predictions_observed": False,
        }
    finally:
        _clean_work(work)


def _source_paths() -> tuple[str, ...]:
    return (
        "benchmark/recognition-synthetic-v2/config.json",
        "benchmark/recognition-synthetic-v2/corpus.json",
        "benchmark/recognition-synthetic-v2/provenance.json",
        "benchmark/recognition-synthetic-v2/manifests/generation.json",
        "benchmark/recognition-synthetic-v2/manifests/manifest.jsonl",
        "benchmark/recognition-synthetic-v2/manifests/split_metadata.json",
        "benchmark/recognition-synthetic-v2/manifests/splits/train.jsonl",
        "benchmark/recognition-synthetic-v2/manifests/splits/validation.jsonl",
        "benchmark/recognition-synthetic-v2/manifests/splits/test.jsonl",
        "benchmark/recognition-synthetic-v2/manifests/summary.json",
        "benchmark/recognition-synthetic-v2/manifests/vocabulary.json",
        "scripts/run_recognition_benchmark_v2.py",
        "src/urdu_document_ocr/data/manifest.py",
        "src/urdu_document_ocr/data/splitting.py",
        "src/urdu_document_ocr/data/synthetic.py",
        "src/urdu_document_ocr/data/vocabulary.py",
        "src/urdu_document_ocr/evaluation/benchmark.py",
        "src/urdu_document_ocr/evaluation/errors.py",
        "src/urdu_document_ocr/evaluation/metrics.py",
        "src/urdu_document_ocr/recognition/decoding.py",
        "src/urdu_document_ocr/recognition/inference.py",
        "src/urdu_document_ocr/recognition/model.py",
        "src/urdu_document_ocr/training/checkpoint.py",
        "src/urdu_document_ocr/training/dataset.py",
        "src/urdu_document_ocr/training/trainer.py",
        "src/urdu_document_ocr/types.py",
    )


def freeze(repository: Path, benchmark: Path) -> dict[str, object]:
    """Freeze all V2 identities before final-test inference is permitted."""

    config, corpus, provenance = _inputs(benchmark)
    if (benchmark / "freeze.json").exists() or (benchmark / "results").exists():
        raise ValueError("V2 freeze/results already exist")
    summary = read_benchmark_json(benchmark / "manifests" / "summary.json")
    vocabulary = load_vocabulary(benchmark / "manifests" / "vocabulary.json")
    raw_recognizer = config.get("recognizer_config")
    raw_training = config.get("training_config")
    if not isinstance(raw_recognizer, dict) or not isinstance(raw_training, dict):
        raise ValueError("V2 recognizer/training configs must be objects")
    recognizer_config = RecognizerConfig.from_mapping(raw_recognizer)
    training_config = TrainingConfig.from_mapping(raw_training)
    set_training_seed(training_config.seed)
    model = CRNNRecognizer(vocabulary, recognizer_config)
    capabilities = assert_shaping_available()
    payload = {
        "schema_version": 2,
        "freeze_id": V2_FREEZE_ID,
        "frozen_on": "2026-09-09",
        "recognition_benchmark_version": V2_VERSION,
        "evidence_class": "synthetic_recognition",
        "metric_policy_id": METRIC_POLICY_ID,
        "metric_policy_fingerprint": METRIC_POLICY_FINGERPRINT,
        "generator_version": "synthetic-v1",
        "font_sha256": capabilities.font_sha256,
        "text_source_fingerprint": canonical_fingerprint(corpus),
        "protocol_selection_fingerprint": canonical_fingerprint(
            provenance["development_protocol_selection"]
        ),
        "seeds": {
            "generation": config["generator_config"]["seed"],
            "split": config["split_config"]["seed"],
            "training": config["training_config"]["seed"],
        },
        "fingerprints": {
            "recognition_config": canonical_fingerprint(config),
            "generator_config": canonical_fingerprint(config["generator_config"]),
            "recognizer_config": recognizer_config.fingerprint,
            "model": model.fingerprint,
            "training_config": training_config.fingerprint,
            "metric_policy": METRIC_POLICY_FINGERPRINT,
            "corpus": canonical_fingerprint(corpus),
            **summary["fingerprints"],
        },
        "environment_contract": {
            "python": "CPython 3.11",
            "device": "cpu",
            "checkpoint_format": "safetensors-plus-validated-json-v1",
            "decoder": "greedy-v1",
            "selection_criterion": "minimum validation CTC loss only",
        },
        "source_artifacts": [
            {"path": path, "sha256": file_sha256(repository / path)} for path in _source_paths()
        ],
        "freeze_sequence": [
            "development-only-study-completed",
            "v2-protocol-selected-without-v2-test",
            "v2-corpus-and-config-finalized",
            "v2-logical-identities-and-grouped-split-finalized",
            "v2-train-only-vocabulary-finalized",
            "v2-model-training-metric-and-environment-finalized",
            "freeze-record-written",
            "one-final-test-run-permitted",
        ],
        "test_identity_overlap_with_development": 0,
        "final_test_metrics_observed_before_freeze": False,
        "final_test_used_for_tuning": False,
        "post_test_config_changes": 0,
        "final_test_run_limit": 1,
        "benchmark_invalidations": 0,
        "failures_removed": 0,
        "implementation_base_commit": "b94b4c1b687c4a79f76e151248e52a9df9ef3bd4",
    }
    write_canonical_json(payload, benchmark / "freeze.json")
    return {
        "freeze_sha256": file_sha256(benchmark / "freeze.json"),
        "source_artifacts": len(payload["source_artifacts"]),
        "final_test_metrics_observed": False,
    }


def _verify_generated(benchmark: Path, data: V2Data, split, vocabulary) -> None:
    if (
        canonical_manifest_bytes(data.samples)
        != (benchmark / "manifests/manifest.jsonl").read_bytes()
    ):
        raise RuntimeError("regenerated V2 manifest differs from freeze")
    generation = read_benchmark_json(benchmark / "manifests/generation.json")
    actual = {
        "schema_version": 1,
        "benchmark_version": V2_VERSION,
        "generator_version": "synthetic-v1",
        "images_committed": False,
        "dataset_fingerprint": data.fingerprint,
        "images": list(data.image_records),
    }
    if generation != actual:
        raise RuntimeError("regenerated V2 image identities differ from freeze")
    for name, values in (
        ("train", split.train),
        ("validation", split.validation),
        ("test", split.test),
    ):
        expected = benchmark / "manifests" / "splits" / f"{name}.jsonl"
        if canonical_manifest_bytes(values) != expected.read_bytes():
            raise RuntimeError(f"regenerated V2 {name} split differs from freeze")
    if vocabulary != load_vocabulary(benchmark / "manifests/vocabulary.json"):
        raise RuntimeError("regenerated V2 vocabulary differs from freeze")


def preflight(repository: Path, benchmark: Path, work: Path) -> dict[str, object]:
    """Verify frozen sources and load final train/validation images, never test inference."""

    freeze_record = read_benchmark_json(benchmark / "freeze.json")
    verify_frozen_sources(repository, freeze_record)
    config, corpus, _ = _inputs(benchmark)
    root = work / "recognition-v2"
    root.mkdir()
    try:
        data = render_v2_dataset(config, corpus, output_directory=root)
        split, vocabulary, validation_unseen, test_unseen, text_overlap = _partition(
            data.samples, config
        )
        _verify_generated(benchmark, data, split, vocabulary)
        recognizer_config = RecognizerConfig.from_mapping(config["recognizer_config"])
        report = validate_training_data(
            OCRLineDataset(
                split.train,
                vocabulary,
                dataset_root=root,
                recognizer_config=recognizer_config,
            ),
            OCRLineDataset(
                split.validation,
                vocabulary,
                dataset_root=root,
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
        _clean_work(work)


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


def run_once(repository: Path, benchmark: Path, work: Path) -> dict[str, object]:
    """Train from scratch and evaluate the frozen test partition exactly once."""

    results = benchmark / "results"
    if results.exists():
        raise ValueError("V2 results already exist; final test cannot be rerun")
    freeze_record = read_benchmark_json(benchmark / "freeze.json")
    verify_frozen_sources(repository, freeze_record)
    config, corpus, _ = _inputs(benchmark)
    root = work / "recognition-v2"
    training_root = work / "training-v2"
    root.mkdir()
    try:
        data = render_v2_dataset(config, corpus, output_directory=root)
        split, vocabulary, _, _, _ = _partition(data.samples, config)
        _verify_generated(benchmark, data, split, vocabulary)
        training_config = TrainingConfig.from_mapping(config["training_config"])
        recognizer_config = RecognizerConfig.from_mapping(config["recognizer_config"])

        # The V2 harness owns model initialization, so seed before construction.
        set_training_seed(training_config.seed)
        model = CRNNRecognizer(vocabulary, recognizer_config)
        training_started = time.perf_counter()
        training = train_model(
            model,
            split.train,
            split.validation,
            vocabulary,
            dataset_root=root,
            output_directory=training_root,
            config=training_config,
        )
        training_seconds = time.perf_counter() - training_started
        recognizer = load_recognizer(training_root / training.checkpoint_directory)
        evaluation_fingerprint = canonical_fingerprint(
            {
                "benchmark_version": V2_VERSION,
                "metric_policy": METRIC_POLICY_FINGERPRINT,
                "recognition_config": config,
            }
        )
        test_started = time.perf_counter()
        report = evaluate_checkpoint(
            split.test,
            dataset_root=root,
            recognizer=recognizer,
            batch_size=config["inference_batch_size"],
            data_fingerprint=dataset_fingerprint(split.test),
            config_fingerprint=evaluation_fingerprint,
        )
        test_seconds = time.perf_counter() - test_started
        capabilities = assert_shaping_available()
        metadata = recognizer.checkpoint_metadata
        predictions = {
            "schema_version": 1,
            "benchmark_version": V2_VERSION,
            "freeze_sha256": file_sha256(benchmark / "freeze.json"),
            "prediction_fingerprint": canonical_fingerprint(
                [[sample.sample_id, sample.hypothesis] for sample in report.samples]
            ),
            "predictions": [
                {"sample_id": sample.sample_id, "hypothesis": sample.hypothesis}
                for sample in report.samples
            ],
        }
        empty_count = sum(not sample.hypothesis for sample in report.samples)
        recognition_result = {
            "schema_version": 2,
            "benchmark_version": V2_VERSION,
            "evidence_class": "synthetic_recognition",
            "label": V2_LABEL,
            "freeze_sha256": file_sha256(benchmark / "freeze.json"),
            "frozen_config_fingerprint": evaluation_fingerprint,
            "dataset": read_benchmark_json(benchmark / "manifests/summary.json"),
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
            "runtime_seconds": {
                "training": training_seconds,
                "test_inference_and_metrics": test_seconds,
                "total_training_and_test": training_seconds + test_seconds,
            },
            "evaluation": report.to_public_dict(),
            "decoding_diagnostics": {
                "empty_prediction_count": empty_count,
                "empty_prediction_fraction": empty_count / len(report.samples),
                "mean_reference_codepoints": sum(len(sample.reference) for sample in report.samples)
                / len(report.samples),
                "mean_prediction_codepoints": sum(
                    len(sample.hypothesis) for sample in report.samples
                )
                / len(report.samples),
            },
            "prediction_fingerprint": predictions["prediction_fingerprint"],
            "source_artifact_hashes": {
                "config": file_sha256(benchmark / "config.json"),
                "corpus": file_sha256(benchmark / "corpus.json"),
                "test_manifest": file_sha256(benchmark / "manifests/splits/test.jsonl"),
            },
            "limitations": [
                "Deterministic synthetically rendered Urdu/Nastaliq line images only.",
                "No historical, scanned-book, handwriting, arbitrary-document, "
                "production, or real-world accuracy claim.",
                "No canonical model checkpoint is published or retained.",
            ],
        }
        results.mkdir()
        write_canonical_json(predictions, results / "predictions.json")
        write_canonical_json(recognition_result, results / "recognition_results.json")
        write_canonical_json(
            analyze_errors(report, worst_limit=config["worst_sample_limit"]).to_public_dict(),
            results / "error_analysis.json",
        )
        write_canonical_json(
            {
                "schema_version": 1,
                "benchmark_version": V2_VERSION,
                "freeze_sha256": file_sha256(benchmark / "freeze.json"),
                "final_test_runs": 1,
                "final_test_used_for_tuning": False,
                "post_test_config_changes": 0,
                "benchmark_invalidations": 0,
                "failures_removed": 0,
                "recognition_summary": report.to_public_dict()["summary"],
                "result_sha256": {
                    "predictions.json": file_sha256(results / "predictions.json"),
                    "recognition_results.json": file_sha256(results / "recognition_results.json"),
                    "error_analysis.json": file_sha256(results / "error_analysis.json"),
                },
            },
            results / "summary.json",
        )
        return {
            "final_test_runs": 1,
            "best_epoch": training.best_epoch,
            "best_validation_loss": training.best_validation_loss,
            "optimization_steps": training.optimization_steps,
            "training_seconds": training_seconds,
            "test_seconds": test_seconds,
            "test_sample_count": report.aggregate.sample_count,
            "character_error_rate": report.aggregate.character_error_rate,
            "word_error_rate": report.aggregate.word_error_rate,
            "exact_match_count": report.aggregate.exact_matches,
            "checkpoint_removed": True,
        }
    finally:
        _clean_work(work)


def _artifact_paths(benchmark: Path, repository: Path) -> tuple[str, ...]:
    candidates = [
        path
        for path in benchmark.rglob("*")
        if path.is_file() and path.name != "artifact_manifest.json"
    ]
    candidates.extend(
        [
            repository / "scripts/run_recognition_benchmark_v2.py",
            repository / "README.md",
            repository / "docs/evaluation.md",
            repository / "docs/training.md",
            repository / "docs/provenance/source-ledger.md",
            repository / "tests/test_phase12b_integration.py",
        ]
    )
    return tuple(sorted(path.relative_to(repository).as_posix() for path in candidates))


def create_manifest(repository: Path, benchmark: Path) -> dict[str, object]:
    if not (benchmark / "results/summary.json").is_file():
        raise ValueError("V2 results must exist before artifact manifest creation")
    payload = build_artifact_manifest(repository, _artifact_paths(benchmark, repository))
    write_canonical_json(payload, benchmark / "artifact_manifest.json")
    return {
        "artifact_manifest_sha256": file_sha256(benchmark / "artifact_manifest.json"),
        "artifact_count": len(payload["artifacts"]),
    }


def verify_v2(repository: Path, benchmark: Path) -> dict[str, object]:
    """Verify V2 hashes and recalculate metrics from committed predictions."""

    artifact = read_benchmark_json(benchmark / "artifact_manifest.json")
    artifact_matches = verify_artifact_manifest(repository, artifact)
    freeze_record = read_benchmark_json(benchmark / "freeze.json")
    source_matches = verify_frozen_sources(repository, freeze_record)
    result = read_benchmark_json(benchmark / "results/recognition_results.json")
    predictions = read_benchmark_json(benchmark / "results/predictions.json")
    summary = read_benchmark_json(benchmark / "results/summary.json")
    if (
        result.get("benchmark_version") != V2_VERSION
        or predictions.get("benchmark_version") != V2_VERSION
        or summary.get("benchmark_version") != V2_VERSION
    ):
        raise RuntimeError("V2 result version mismatch")
    freeze_hash = file_sha256(benchmark / "freeze.json")
    if any(item.get("freeze_sha256") != freeze_hash for item in (result, predictions, summary)):
        raise RuntimeError("V2 result freeze hash mismatch")
    evaluation = result.get("evaluation")
    if not isinstance(evaluation, dict) or not isinstance(evaluation.get("samples"), list):
        raise RuntimeError("V2 evaluation samples are malformed")
    references = []
    expected_predictions = []
    for item in evaluation["samples"]:
        references.append((item["sample_id"], item["reference"]))
        expected_predictions.append((item["sample_id"], item["hypothesis"]))
    committed_predictions = [
        (item["sample_id"], item["hypothesis"]) for item in predictions["predictions"]
    ]
    if committed_predictions != expected_predictions:
        raise RuntimeError("V2 standalone predictions differ from evaluation")
    identities = evaluation["identities"]
    recalculated = evaluate_predictions(
        references,
        committed_predictions,
        data_fingerprint=identities["data_fingerprint"],
        config_fingerprint=identities["config_fingerprint"],
        model_fingerprint=identities["model_fingerprint"],
    )
    if recalculated.to_public_dict() != evaluation:
        raise RuntimeError("V2 metrics do not recalculate exactly")
    config = read_benchmark_json(benchmark / "config.json")
    analysis = read_benchmark_json(benchmark / "results/error_analysis.json")
    if (
        analyze_errors(recalculated, worst_limit=config["worst_sample_limit"]).to_public_dict()
        != analysis
    ):
        raise RuntimeError("V2 error analysis does not recalculate exactly")
    expected_hashes = {
        "predictions.json": file_sha256(benchmark / "results/predictions.json"),
        "recognition_results.json": file_sha256(benchmark / "results/recognition_results.json"),
        "error_analysis.json": file_sha256(benchmark / "results/error_analysis.json"),
    }
    if summary.get("result_sha256") != expected_hashes:
        raise RuntimeError("V2 result hashes differ from summary")
    if summary.get("recognition_summary") != evaluation.get("summary"):
        raise RuntimeError("V2 summary differs from recalculated evaluation")
    return {
        "artifact_matches": artifact_matches[0],
        "artifact_total": artifact_matches[1],
        "source_matches": source_matches[0],
        "source_total": source_matches[1],
        "recognition_sample_count": len(evaluation["samples"]),
        "final_test_runs": summary["final_test_runs"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode", choices=("prepare", "freeze", "preflight", "run", "manifest", "verify")
    )
    parser.add_argument("--repository-directory", type=Path, default=Path.cwd())
    parser.add_argument(
        "--benchmark-directory",
        type=Path,
        default=Path("benchmark/recognition-synthetic-v2"),
    )
    parser.add_argument("--work-directory", type=Path)
    args = parser.parse_args()

    repository = args.repository_directory.resolve(strict=True)
    benchmark = (
        args.benchmark_directory
        if args.benchmark_directory.is_absolute()
        else repository / args.benchmark_directory
    ).resolve(strict=True)
    if args.mode in {"prepare", "preflight", "run"}:
        if args.work_directory is None:
            parser.error("--work-directory is required")
        work = _empty_directory(args.work_directory, "work directory")
    if args.mode == "prepare":
        result = prepare(repository, benchmark, work)
    elif args.mode == "freeze":
        result = freeze(repository, benchmark)
    elif args.mode == "preflight":
        result = preflight(repository, benchmark, work)
    elif args.mode == "run":
        result = run_once(repository, benchmark, work)
    elif args.mode == "manifest":
        result = create_manifest(repository, benchmark)
    else:
        result = verify_v2(repository, benchmark)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
