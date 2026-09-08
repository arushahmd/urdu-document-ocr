from __future__ import annotations

import math
import random

import numpy as np
import pytest
import torch
from PIL import Image

from urdu_document_ocr import CRNNRecognizer, DatasetSample, TrainingConfig, Vocabulary
from urdu_document_ocr.errors import TrainingDataError, TrainingError
from urdu_document_ocr.training import (
    OCRBatch,
    OCRTrainer,
    TrainingPreflightReport,
    train_model,
)
from urdu_document_ocr.training import trainer as trainer_module


def _batch(*, width: int = 16, targets: tuple[int, ...] = (1,)) -> OCRBatch:
    return OCRBatch(
        images=torch.zeros((1, 1, 64, width), dtype=torch.float32),
        valid_widths=torch.tensor([width], dtype=torch.int64),
        targets=torch.tensor(targets, dtype=torch.int64),
        target_lengths=torch.tensor([len(targets)], dtype=torch.int64),
        sample_ids=("sample",),
    )


def _config(**overrides) -> TrainingConfig:
    return TrainingConfig(batch_size=1, epochs=1, **overrides)


def test_seed_is_repeatable_across_python_numpy_and_torch() -> None:
    first_generator = trainer_module.set_training_seed(41)
    first = (
        random.random(),
        np.random.random(),
        torch.rand(1),
        torch.rand(1, generator=first_generator),
    )
    second_generator = trainer_module.set_training_seed(41)
    second = (
        random.random(),
        np.random.random(),
        torch.rand(1),
        torch.rand(1, generator=second_generator),
    )

    assert first[0] == second[0]
    assert first[1] == second[1]
    assert torch.equal(first[2], second[2])
    assert torch.equal(first[3], second[3])


def test_one_train_step_updates_parameters_and_validation_does_not() -> None:
    trainer_module.set_training_seed(7)
    model = CRNNRecognizer(Vocabulary(("ا",)))
    trainer = OCRTrainer(model, _config())
    before = model.classifier.weight.detach().clone()

    loss = trainer.train_epoch([_batch()])
    after_train = model.classifier.weight.detach().clone()
    validation_loss = trainer.validate_epoch([_batch()])

    assert math.isfinite(loss) and math.isfinite(validation_loss)
    assert not torch.equal(before, after_train)
    assert torch.equal(after_train, model.classifier.weight)
    assert trainer.optimization_steps == 1


def test_gradient_clipping_is_called(monkeypatch) -> None:
    calls: list[float] = []

    def record_clip(_parameters, threshold):
        calls.append(threshold)
        return torch.tensor(0.0)

    monkeypatch.setattr(trainer_module, "clip_grad_norm_", record_clip)
    trainer = OCRTrainer(CRNNRecognizer(Vocabulary(("ا",))), _config(gradient_clip=1.25))
    trainer.train_epoch([_batch()])

    assert calls == [1.25]


def test_impossible_ctc_and_nonfinite_loss_fail_with_sample(monkeypatch) -> None:
    trainer = OCRTrainer(CRNNRecognizer(Vocabulary(("ا",))), _config())
    with pytest.raises(TrainingDataError, match="CTC alignment") as impossible:
        trainer.train_epoch([_batch(width=4, targets=(1, 1))])
    assert impossible.value.context["sample_id"] == "sample"

    monkeypatch.setattr(
        trainer_module,
        "compute_ctc_loss",
        lambda *_args, **_kwargs: torch.tensor(float("nan")),
    )
    with pytest.raises(TrainingError, match="nonfinite CTC loss"):
        trainer.train_epoch([_batch()])


def test_nonfinite_gradient_stops_before_optimizer_step() -> None:
    trainer = OCRTrainer(CRNNRecognizer(Vocabulary(("ا",))), _config())
    handle = trainer.model.classifier.weight.register_hook(lambda gradient: gradient * float("nan"))
    try:
        with pytest.raises(TrainingError, match="nonfinite gradient"):
            trainer.train_epoch([_batch()])
    finally:
        handle.remove()
    assert trainer.optimization_steps == 0


def test_sample_weighted_loss_aggregation() -> None:
    accumulator = trainer_module._LossAccumulator()
    accumulator.add(2.0, 3)
    accumulator.add(6.0, 1)

    assert accumulator.mean() == 3.0


def test_early_stopping_and_best_checkpoint_exact_epoch(tmp_path, monkeypatch) -> None:
    trainer = OCRTrainer(
        CRNNRecognizer(Vocabulary(("ا",))),
        TrainingConfig(batch_size=1, epochs=5, early_stopping_patience=2, min_delta=0.1),
    )
    losses = iter([4.0, 3.0, 4.0, 2.0, 4.0, 1.95, 4.0, 2.2])
    monkeypatch.setattr(trainer, "_run_epoch", lambda _batches, training: next(losses))
    saved: list[tuple[int, bool]] = []

    def record_save(_path, _model, _config, *, epoch, overwrite, **_kwargs):
        saved.append((epoch, overwrite))

    monkeypatch.setattr(trainer_module, "save_checkpoint", record_save)
    preflight = TrainingPreflightReport("a" * 64, "b" * 64, 1, 1)

    result = trainer.fit([], [], output_directory=tmp_path / "run", preflight=preflight)

    assert result.best_epoch == 2
    assert result.best_validation_loss == 2.0
    assert result.stopped_early is True
    assert len(result.epochs) == 4
    assert [record.improved for record in result.epochs] == [True, True, False, False]
    assert saved == [(1, False), (2, True)]


def test_train_shuffle_and_validation_order_are_seeded(tmp_path, monkeypatch) -> None:
    pixels = np.full((12, 24), 255, dtype=np.uint8)
    pixels[3:9, 4:20] = 0
    Image.fromarray(pixels).save(tmp_path / "line.png")
    vocabulary = Vocabulary(("ا",))
    train = tuple(
        DatasetSample(1, f"s{index}", "line.png", "ا", f"doc-{index}") for index in range(4)
    )
    validation = (DatasetSample(1, "v", "line.png", "ا", "validation-doc"),)

    def capture(self, train_loader, validation_loader, **_kwargs):
        return (
            tuple(item for batch in train_loader for item in batch.sample_ids),
            tuple(item for batch in validation_loader for item in batch.sample_ids),
        )

    monkeypatch.setattr(OCRTrainer, "fit", capture)

    first = train_model(
        CRNNRecognizer(vocabulary),
        train,
        validation,
        vocabulary,
        dataset_root=tmp_path,
        output_directory=tmp_path / "unused-1",
        config=TrainingConfig(batch_size=1, epochs=1, seed=73),
    )
    second = train_model(
        CRNNRecognizer(vocabulary),
        train,
        validation,
        vocabulary,
        dataset_root=tmp_path,
        output_directory=tmp_path / "unused-2",
        config=TrainingConfig(batch_size=1, epochs=1, seed=73),
    )

    assert first == second
    assert set(first[0]) == {"s0", "s1", "s2", "s3"}
    assert first[1] == ("v",)


def test_unavailable_cuda_and_nonempty_output_fail_closed(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(TrainingError, match="CUDA"):
        OCRTrainer(CRNNRecognizer(Vocabulary(("ا",))), _config(device="cuda"))

    trainer = OCRTrainer(CRNNRecognizer(Vocabulary(("ا",))), _config())
    output = tmp_path / "occupied"
    output.mkdir()
    (output / "unrelated.txt").write_text("keep", encoding="utf-8")
    with pytest.raises(TrainingError, match="must be empty"):
        trainer.fit(
            [],
            [],
            output_directory=output,
            preflight=TrainingPreflightReport("a" * 64, "b" * 64, 1, 1),
        )
    assert (output / "unrelated.txt").read_text(encoding="utf-8") == "keep"
