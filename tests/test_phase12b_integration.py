from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from urdu_document_ocr import find_unseen_characters, load_vocabulary, read_benchmark_json
from urdu_document_ocr.data.manifest import read_manifest
from urdu_document_ocr.evaluation.benchmark import file_sha256

ROOT = Path(__file__).resolve().parents[1]
V2_ROOT = ROOT / "benchmark" / "recognition-synthetic-v2"
V1_HASHES = {
    "benchmark/freeze.json": "2c4ad41e3bb4d57962bba409fb839182ae347a67c98feb95ae95aba55194a22c",
    "benchmark/results/recognition_results.json": (
        "e59c13611d258d2647507126db400ed2bc7b3e367daa3df6beb5cf80761d82a0"
    ),
    "benchmark/results/vision_results.json": (
        "2be9da4b531d37131fb22ce5bb8535b642449e22a7c3bc1752551a78afbb3fb9"
    ),
    "benchmark/results/error_analysis.json": (
        "bb1f471937e749f576ba2d00b6bb3721a00961fd6c2e4cad3c73e214a8df99bf"
    ),
    "benchmark/results/summary.json": (
        "fc259a4d2932b1c42d151d298db754394aa61c8a551e16d020fc95af4d6d915e"
    ),
    "benchmark/artifact_manifest.json": (
        "a84046b0fee523e601f405f0a3e5df845c07bcc691b36dfd1190ddfd38aa5a0d"
    ),
}


def _v2_module():
    path = ROOT / "scripts" / "run_recognition_benchmark_v2.py"
    spec = importlib.util.spec_from_file_location("phase12b_v2_runner", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(("relative", "expected"), V1_HASHES.items())
def test_phase12b_preserves_frozen_v1_bytes(relative: str, expected: str) -> None:
    assert file_sha256(ROOT / relative) == expected


def test_v2_version_handler_rejects_v1_config(tmp_path: Path) -> None:
    module = _v2_module()
    config = read_benchmark_json(V2_ROOT / "config.json")
    corpus = read_benchmark_json(V2_ROOT / "corpus.json")
    config["benchmark_version"] = "recognition-synthetic-v1"

    with pytest.raises(ValueError, match="version"):
        module.render_v2_dataset(config, corpus, output_directory=tmp_path)


def test_v2_freeze_precedes_one_final_test_run() -> None:
    freeze = read_benchmark_json(V2_ROOT / "freeze.json")
    summary = read_benchmark_json(V2_ROOT / "results" / "summary.json")

    assert freeze["recognition_benchmark_version"] == "recognition-synthetic-v2"
    assert freeze["final_test_metrics_observed_before_freeze"] is False
    assert freeze["final_test_used_for_tuning"] is False
    assert freeze["final_test_run_limit"] == 1
    assert freeze["test_identity_overlap_with_development"] == 0
    assert summary["final_test_runs"] == 1
    assert summary["post_test_config_changes"] == 0
    assert summary["benchmark_invalidations"] == 0
    assert summary["failures_removed"] == 0


def test_v2_group_split_and_train_only_vocabulary_are_isolated() -> None:
    split_root = V2_ROOT / "manifests" / "splits"
    train = read_manifest(split_root / "train.jsonl")
    validation = read_manifest(split_root / "validation.jsonl")
    test = read_manifest(split_root / "test.jsonl")
    vocabulary = load_vocabulary(V2_ROOT / "manifests" / "vocabulary.json")

    partitions = (train, validation, test)
    sample_sets = [{sample.sample_id for sample in values} for values in partitions]
    document_sets = [{sample.document_id for sample in values} for values in partitions]
    text_sets = [{sample.text for sample in values} for values in partitions]
    for left in range(len(partitions)):
        for right in range(left + 1, len(partitions)):
            assert sample_sets[left].isdisjoint(sample_sets[right])
            assert document_sets[left].isdisjoint(document_sets[right])
            assert text_sets[left].isdisjoint(text_sets[right])
    assert find_unseen_characters(validation, vocabulary) == ()
    assert find_unseen_characters(test, vocabulary) == ()


def test_v2_artifacts_and_metrics_recalculate_exactly() -> None:
    report = _v2_module().verify_v2(ROOT, V2_ROOT)

    assert report["artifact_matches"] == report["artifact_total"]
    assert report["source_matches"] == report["source_total"]
    assert report["final_test_runs"] == 1


def test_v2_commits_no_checkpoint() -> None:
    forbidden = {".pt", ".pth", ".pkl", ".pickle", ".safetensors"}
    assert not [path for path in V2_ROOT.rglob("*") if path.suffix.lower() in forbidden]
