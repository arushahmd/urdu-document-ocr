from __future__ import annotations

from pathlib import Path

from urdu_document_ocr import (
    CRNNRecognizer,
    SyntheticDataConfig,
    TrainingConfig,
    build_vocabulary,
    evaluate_checkpoint,
    load_recognizer,
    render_recognition_dataset,
)
from urdu_document_ocr.evaluation import analyze_errors, canonical_fingerprint
from urdu_document_ocr.training import save_checkpoint


def test_real_checkpoint_evaluation_returns_traceable_metrics(tmp_path: Path) -> None:
    config = {
        "benchmark_version": "recognition-synthetic-v1",
        "document_count": 1,
        "lines_per_document": 1,
        "generator_config": SyntheticDataConfig(
            seed=812_001,
            blur_probability=0.0,
            noise_probability=0.0,
            skew_probability=0.0,
        ).to_dict(),
    }
    corpus = {
        "vision_texts": ["محفوظ اردو متن۔"],
        "recognition_templates": ["محفوظ نمونہ {urdu_number}: اردو متن۔"],
    }
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    generated = render_recognition_dataset(config, corpus, output_directory=dataset_root)
    vocabulary = build_vocabulary(generated.samples)
    model = CRNNRecognizer(vocabulary)
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    metadata = save_checkpoint(
        checkpoint / "best",
        model,
        TrainingConfig(batch_size=1, epochs=1),
        epoch=1,
        validation_loss=1.0,
        train_dataset_fingerprint=generated.fingerprint,
        validation_dataset_fingerprint=generated.fingerprint,
    )
    recognizer = load_recognizer(checkpoint / "best")

    report = evaluate_checkpoint(
        generated.samples,
        dataset_root=dataset_root,
        recognizer=recognizer,
        batch_size=1,
        config_fingerprint=canonical_fingerprint(config),
    )
    analysis = analyze_errors(report)

    assert report.aggregate.sample_count == 1
    assert report.aggregate.model_fingerprint == metadata.model_fingerprint
    assert report.aggregate.character_error_rate is not None
    assert analysis.substitutions == report.aggregate.character_edits.substitutions
    assert analysis.deletions == report.aggregate.character_edits.deletions
    assert analysis.insertions == report.aggregate.character_edits.insertions
    assert not recognizer.model.training
