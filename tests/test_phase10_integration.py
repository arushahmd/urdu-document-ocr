from __future__ import annotations

import math
from pathlib import Path

import torch

from urdu_document_ocr import CRNNRecognizer, TrainingConfig
from urdu_document_ocr.data import load_vocabulary, read_manifest
from urdu_document_ocr.training import (
    OCRLineDataset,
    collate_ocr_batch,
    load_checkpoint,
    set_training_seed,
    train_model,
    validate_training_data,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_ROOT = REPOSITORY_ROOT / "data" / "sample"


def test_phase8_fixture_to_adamw_checkpoint_reload_and_forward(tmp_path) -> None:
    vocabulary = load_vocabulary(SAMPLE_ROOT / "synthetic-fixture-vocabulary.json")
    train_samples = read_manifest(SAMPLE_ROOT / "splits" / "train.jsonl")
    validation_samples = read_manifest(SAMPLE_ROOT / "splits" / "validation.jsonl")
    full_preflight = validate_training_data(
        OCRLineDataset(train_samples, vocabulary, dataset_root=SAMPLE_ROOT),
        OCRLineDataset(validation_samples, vocabulary, dataset_root=SAMPLE_ROOT),
    )
    assert full_preflight.train_sample_count == 16
    assert full_preflight.validation_sample_count == 4
    train_sample = min(train_samples, key=lambda sample: len(sample.text))
    validation_sample = min(validation_samples, key=lambda sample: len(sample.text))
    set_training_seed(29)
    model = CRNNRecognizer(vocabulary)
    before = model.classifier.weight.detach().clone()
    config = TrainingConfig(batch_size=1, epochs=1, seed=29)

    result = train_model(
        model,
        [train_sample],
        [validation_sample],
        vocabulary,
        dataset_root=SAMPLE_ROOT,
        output_directory=tmp_path / "run",
        config=config,
    )

    assert result.optimization_steps == 1
    assert math.isfinite(result.epochs[0].train_loss)
    assert math.isfinite(result.epochs[0].validation_loss)
    assert not torch.equal(before, model.classifier.weight)
    saved_weights = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
    reloaded = CRNNRecognizer(vocabulary)
    metadata = load_checkpoint(tmp_path / "run" / "best", reloaded, expected_training_config=config)
    assert metadata.epoch == 1
    assert all(
        torch.equal(saved_weights[name], tensor) for name, tensor in reloaded.state_dict().items()
    )

    dataset = OCRLineDataset([validation_sample], vocabulary, dataset_root=SAMPLE_ROOT)
    batch = collate_ocr_batch([dataset[0]], class_count=len(vocabulary.characters) + 1)
    with torch.inference_mode():
        output = reloaded(batch.images, batch.valid_widths)
    assert torch.isfinite(output.logits).all()
