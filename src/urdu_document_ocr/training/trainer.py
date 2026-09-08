"""Plain AdamW OCR training with validation-loss checkpoint selection."""

from __future__ import annotations

import math
import os
import random
from collections.abc import Iterable
from contextlib import nullcontext
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.utils.data import DataLoader

from urdu_document_ocr.config import TrainingConfig
from urdu_document_ocr.errors import TrainingDataError, TrainingError
from urdu_document_ocr.recognition import compute_ctc_loss, validate_ctc_alignment
from urdu_document_ocr.recognition.model import CRNNRecognizer
from urdu_document_ocr.training.checkpoint import save_checkpoint
from urdu_document_ocr.training.dataset import (
    OCRBatch,
    OCRLineDataset,
    TrainingPreflightReport,
    collate_ocr_batch,
    validate_training_data,
)
from urdu_document_ocr.types import DatasetSample, Vocabulary


@dataclass(frozen=True, slots=True)
class TrainingEpochRecord:
    """Loss-only deterministic record for one completed epoch."""

    epoch: int
    train_loss: float
    validation_loss: float
    learning_rate: float
    improved: bool

    def to_public_dict(self) -> dict[str, int | float | bool]:
        return {
            "epoch": self.epoch,
            "train_loss": self.train_loss,
            "validation_loss": self.validation_loss,
            "learning_rate": self.learning_rate,
            "improved": self.improved,
        }


@dataclass(frozen=True, slots=True)
class TrainingResult:
    """Completed history and best-checkpoint identity without private paths."""

    epochs: tuple[TrainingEpochRecord, ...]
    best_epoch: int
    best_validation_loss: float
    stopped_early: bool
    optimization_steps: int
    checkpoint_directory: str
    train_dataset_fingerprint: str
    validation_dataset_fingerprint: str

    def to_public_dict(self) -> dict[str, object]:
        return {
            "epochs": [record.to_public_dict() for record in self.epochs],
            "best_epoch": self.best_epoch,
            "best_validation_loss": self.best_validation_loss,
            "stopped_early": self.stopped_early,
            "optimization_steps": self.optimization_steps,
            "checkpoint_directory": self.checkpoint_directory,
            "train_dataset_fingerprint": self.train_dataset_fingerprint,
            "validation_dataset_fingerprint": self.validation_dataset_fingerprint,
        }


def set_training_seed(seed: int, *, use_cuda: bool = False) -> torch.Generator:
    """Seed Python, NumPy, PyTorch, workers, and a returned DataLoader generator."""

    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise TrainingError("training seed must be a nonnegative integer")
    if not isinstance(use_cuda, bool):
        raise TrainingError("use_cuda must be a boolean")
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if use_cuda:
        if not torch.cuda.is_available():
            raise TrainingError("CUDA was requested but is unavailable")
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return generator


def _seed_worker(_worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


@dataclass(slots=True)
class _LossAccumulator:
    weighted_sum: float = 0.0
    sample_count: int = 0

    def add(self, loss: float, batch_size: int) -> None:
        if not math.isfinite(loss) or batch_size < 1:
            raise TrainingError("loss statistics must be finite and sample-weighted")
        self.weighted_sum += loss * batch_size
        self.sample_count += batch_size

    def mean(self) -> float:
        if self.sample_count == 0:
            raise TrainingError("an epoch must contain at least one sample")
        return self.weighted_sum / self.sample_count


def _is_improvement(candidate: float, best: float, min_delta: float) -> bool:
    return candidate < best - min_delta


class OCRTrainer:
    """Fixed-rate AdamW trainer with explicit device and best-only persistence."""

    def __init__(self, model: CRNNRecognizer, config: TrainingConfig) -> None:
        if not isinstance(model, CRNNRecognizer):
            raise TypeError("model must be a CRNNRecognizer")
        if not isinstance(config, TrainingConfig):
            raise TypeError("config must be a TrainingConfig")
        if config.max_image_width != model.config.max_width:
            raise TrainingError("training and recognizer maximum image widths must match")
        if config.device.startswith("cuda") and not torch.cuda.is_available():
            raise TrainingError("configured CUDA device is unavailable")
        try:
            self.device = torch.device(config.device)
            if self.device.type == "cuda":
                torch.empty(0, device=self.device)
        except (RuntimeError, ValueError) as error:
            raise TrainingError("configured training device could not be initialized") from error
        self.model = model.to(self.device)
        self.config = config
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=config.weight_decay,
            amsgrad=False,
            foreach=False,
            fused=False,
        )
        self.optimization_steps = 0

    def _validate_alignment(self, batch: OCRBatch, output: object) -> None:
        logits = getattr(output, "logits", None)
        input_lengths = getattr(output, "input_lengths", None)
        if not isinstance(logits, Tensor) or not isinstance(input_lengths, Tensor):
            raise TrainingError("recognizer output does not satisfy the training contract")
        report = validate_ctc_alignment(
            batch.targets,
            batch.target_lengths,
            input_lengths,
            class_count=logits.shape[2],
            max_input_timesteps=logits.shape[1],
            blank_index=self.model.config.blank_index,
        )
        if not report.is_feasible:
            index = report.impossible_sample_indices[0]
            raise TrainingDataError(
                "sample cannot satisfy CTC alignment",
                context={"sample_id": batch.sample_ids[index]},
            )

    def _run_epoch(self, batches: Iterable[OCRBatch], *, training: bool) -> float:
        previous_mode = self.model.training
        self.model.train(training)
        statistics = _LossAccumulator()
        context = nullcontext() if training else torch.inference_mode()
        try:
            with context:
                for batch in batches:
                    if not isinstance(batch, OCRBatch):
                        raise TrainingError("DataLoader must yield OCRBatch values")
                    images = batch.images.to(self.device)
                    if training:
                        self.optimizer.zero_grad(set_to_none=True)
                    output = self.model(images, batch.valid_widths)
                    self._validate_alignment(batch, output)
                    loss = compute_ctc_loss(
                        output.logits,
                        batch.targets,
                        batch.target_lengths,
                        output.input_lengths,
                        blank_index=self.model.config.blank_index,
                    )
                    if not torch.isfinite(loss):
                        raise TrainingError(
                            "nonfinite CTC loss stopped training",
                            context={"sample_id": batch.sample_ids[0]},
                        )
                    if training:
                        loss.backward()
                        if any(
                            parameter.grad is not None and not torch.isfinite(parameter.grad).all()
                            for parameter in self.model.parameters()
                        ):
                            raise TrainingError(
                                "nonfinite gradient stopped training",
                                context={"sample_id": batch.sample_ids[0]},
                            )
                        clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
                        self.optimizer.step()
                        self.optimization_steps += 1
                    statistics.add(float(loss.detach().cpu().item()), batch.batch_size)
        finally:
            if not training:
                self.model.train(previous_mode)
        return statistics.mean()

    def train_epoch(self, batches: Iterable[OCRBatch]) -> float:
        """Run one optimizing epoch and return sample-weighted mean CTC loss."""

        return self._run_epoch(batches, training=True)

    def validate_epoch(self, batches: Iterable[OCRBatch]) -> float:
        """Run one mutation-free validation epoch and return sample-weighted loss."""

        return self._run_epoch(batches, training=False)

    def fit(
        self,
        train_loader: DataLoader[OCRBatch],
        validation_loader: DataLoader[OCRBatch],
        *,
        output_directory: str | os.PathLike[str],
        preflight: TrainingPreflightReport,
    ) -> TrainingResult:
        """Train, select the lowest validation loss, and save only ``best/``."""

        if not isinstance(preflight, TrainingPreflightReport):
            raise TypeError("preflight must be a TrainingPreflightReport")
        output = Path(output_directory)
        if output.is_symlink():
            raise TrainingError("training output directory cannot be a symbolic link")
        if output.exists():
            if not output.is_dir():
                raise TrainingError("training output must be a directory")
            try:
                if any(output.iterdir()):
                    raise TrainingError("training output directory must be empty")
            except OSError as error:
                raise TrainingError("training output directory could not be inspected") from error
        else:
            try:
                output.mkdir(parents=False)
            except OSError as error:
                raise TrainingError("training output directory could not be created") from error

        checkpoint = output / self.config.checkpoint_directory
        try:
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            raise TrainingError("checkpoint parent could not be created") from error
        history: list[TrainingEpochRecord] = []
        best_loss = math.inf
        best_epoch = 0
        without_improvement = 0
        stopped_early = False
        checkpoint_exists = False
        for epoch in range(1, self.config.epochs + 1):
            train_loss = self.train_epoch(train_loader)
            validation_loss = self.validate_epoch(validation_loader)
            improved = _is_improvement(validation_loss, best_loss, self.config.min_delta)
            if improved:
                best_loss = validation_loss
                best_epoch = epoch
                without_improvement = 0
                save_checkpoint(
                    checkpoint,
                    self.model,
                    self.config,
                    epoch=epoch,
                    validation_loss=validation_loss,
                    train_dataset_fingerprint=preflight.train_dataset_fingerprint,
                    validation_dataset_fingerprint=preflight.validation_dataset_fingerprint,
                    overwrite=checkpoint_exists,
                )
                checkpoint_exists = True
            else:
                without_improvement += 1
            history.append(
                TrainingEpochRecord(
                    epoch=epoch,
                    train_loss=train_loss,
                    validation_loss=validation_loss,
                    learning_rate=self.config.learning_rate,
                    improved=improved,
                )
            )
            if without_improvement >= self.config.early_stopping_patience:
                stopped_early = True
                break
        if best_epoch == 0:
            raise TrainingError("training completed without a finite best validation loss")
        return TrainingResult(
            epochs=tuple(history),
            best_epoch=best_epoch,
            best_validation_loss=best_loss,
            stopped_early=stopped_early,
            optimization_steps=self.optimization_steps,
            checkpoint_directory=self.config.checkpoint_directory,
            train_dataset_fingerprint=preflight.train_dataset_fingerprint,
            validation_dataset_fingerprint=preflight.validation_dataset_fingerprint,
        )


def train_model(
    model: CRNNRecognizer,
    train_samples: Iterable[DatasetSample],
    validation_samples: Iterable[DatasetSample],
    vocabulary: Vocabulary,
    *,
    dataset_root: str | os.PathLike[str],
    output_directory: str | os.PathLike[str],
    config: TrainingConfig | None = None,
) -> TrainingResult:
    """Run the complete validated Dataset/DataLoader/AdamW/checkpoint flow."""

    active_config = TrainingConfig() if config is None else config
    if not isinstance(active_config, TrainingConfig):
        raise TypeError("config must be a TrainingConfig")
    if not isinstance(model, CRNNRecognizer) or model.vocabulary != vocabulary:
        raise TrainingError("model and training vocabulary must match exactly")
    train_dataset = OCRLineDataset(
        tuple(train_samples),
        vocabulary,
        dataset_root=dataset_root,
        recognizer_config=model.config,
    )
    validation_dataset = OCRLineDataset(
        tuple(validation_samples),
        vocabulary,
        dataset_root=dataset_root,
        recognizer_config=model.config,
    )
    preflight = validate_training_data(train_dataset, validation_dataset)
    generator = set_training_seed(
        active_config.seed,
        use_cuda=active_config.device.startswith("cuda"),
    )
    validation_generator = torch.Generator(device="cpu")
    validation_generator.manual_seed(active_config.seed + 1)
    collator = partial(
        collate_ocr_batch,
        class_count=len(vocabulary.characters) + 1,
        max_width=active_config.max_image_width,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=active_config.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=active_config.num_workers,
        collate_fn=collator,
        generator=generator,
        worker_init_fn=_seed_worker,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=active_config.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=active_config.num_workers,
        collate_fn=collator,
        generator=validation_generator,
        worker_init_fn=_seed_worker,
    )
    trainer = OCRTrainer(model, active_config)
    return trainer.fit(
        train_loader,
        validation_loader,
        output_directory=output_directory,
        preflight=preflight,
    )
