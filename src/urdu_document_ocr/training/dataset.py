"""Validated OCR line datasets, dynamic collation, and training preflight."""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import Dataset

from urdu_document_ocr.config import RecognizerConfig
from urdu_document_ocr.data import (
    dataset_fingerprint,
    find_unseen_characters,
    load_dataset_line_image,
    validate_dataset,
)
from urdu_document_ocr.errors import DatasetValidationError, ModelInputError, TrainingDataError
from urdu_document_ocr.recognition import (
    input_width_to_timesteps,
    prepare_line_image,
    validate_ctc_alignment,
)
from urdu_document_ocr.recognition.model import pad_prepared_line_images
from urdu_document_ocr.types import DatasetSample, Vocabulary


@dataclass(frozen=True, slots=True)
class OCRLineItem:
    """One prepared line and its structural identity."""

    image: Tensor
    valid_width: int
    target: Tensor
    sample_id: str


@dataclass(frozen=True, slots=True)
class OCRBatch:
    """A white-padded variable-width batch with concatenated CTC targets."""

    images: Tensor
    valid_widths: Tensor
    targets: Tensor
    target_lengths: Tensor
    sample_ids: tuple[str, ...]

    @property
    def batch_size(self) -> int:
        return len(self.sample_ids)


@dataclass(frozen=True, slots=True)
class TrainingPreflightReport:
    """Deterministic identities and counts proven before optimizer or output creation."""

    train_dataset_fingerprint: str
    validation_dataset_fingerprint: str
    train_sample_count: int
    validation_sample_count: int


class OCRLineDataset(Dataset[OCRLineItem]):
    """Load canonical manifest samples beneath one explicit dataset root."""

    def __init__(
        self,
        samples: Sequence[DatasetSample],
        vocabulary: Vocabulary,
        *,
        dataset_root: str | os.PathLike[str],
        recognizer_config: RecognizerConfig | None = None,
    ) -> None:
        values = tuple(samples)
        if any(not isinstance(sample, DatasetSample) for sample in values):
            raise TrainingDataError("samples must contain only DatasetSample values")
        if not isinstance(vocabulary, Vocabulary):
            raise TypeError("vocabulary must be a Vocabulary")
        self.vocabulary = vocabulary
        self.recognizer_config = (
            RecognizerConfig() if recognizer_config is None else recognizer_config
        )
        if not isinstance(self.recognizer_config, RecognizerConfig):
            raise TypeError("recognizer_config must be a RecognizerConfig")
        try:
            root = Path(dataset_root).resolve(strict=True)
        except (OSError, TypeError, ValueError) as error:
            raise TrainingDataError("dataset root could not be resolved") from error
        if not root.is_dir():
            raise TrainingDataError("dataset root must be a directory")

        report = validate_dataset(
            values,
            dataset_root=None,
            vocabulary=vocabulary,
            inspect_images=False,
        )
        errors = tuple(issue for issue in report.issues if issue.severity.value == "error")
        if errors:
            first = errors[0]
            raise TrainingDataError(
                "training samples do not satisfy the canonical dataset contract",
                context={"sample_id": first.sample_id, "issue": first.code},
            )
        unseen = find_unseen_characters(values, vocabulary)
        if unseen:
            raise TrainingDataError(
                "training samples contain characters outside the vocabulary",
                context={"first_codepoint": f"U+{ord(unseen[0]):04X}", "count": len(unseen)},
            )
        self.samples = values
        self._dataset_root = root

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> OCRLineItem:
        sample = self.samples[index]
        try:
            pixels = load_dataset_line_image(sample, dataset_root=self._dataset_root)
            image, valid_width = prepare_line_image(pixels, self.recognizer_config)
            target_values = self.vocabulary.encode(sample.text)
        except (DatasetValidationError, ModelInputError, TypeError, ValueError) as error:
            raise TrainingDataError(
                "training sample could not be prepared",
                context={"sample_id": sample.sample_id, "image_path": sample.image_path},
            ) from error
        return OCRLineItem(
            image=image,
            valid_width=valid_width,
            target=torch.tensor(target_values, dtype=torch.int64),
            sample_id=sample.sample_id,
        )


def collate_ocr_batch(
    items: Sequence[OCRLineItem],
    *,
    class_count: int,
    max_width: int = 2048,
) -> OCRBatch:
    """Right-pad lines with white ``+1`` to the next width-stride boundary."""

    values = tuple(items)
    if not values:
        raise TrainingDataError("cannot collate an empty OCR batch")
    if isinstance(class_count, bool) or not isinstance(class_count, int) or class_count < 2:
        raise TrainingDataError("class_count must be an integer of at least 2")
    if isinstance(max_width, bool) or not isinstance(max_width, int) or max_width < 4:
        raise TrainingDataError("max_width must be an integer of at least 4")
    sample_ids: list[str] = []
    targets: list[Tensor] = []
    for item in values:
        if not isinstance(item, OCRLineItem):
            raise TrainingDataError("batch values must be OCRLineItem instances")
        if not isinstance(item.sample_id, str) or not item.sample_id:
            raise TrainingDataError("batch sample IDs must be nonempty strings")
        if item.sample_id in sample_ids:
            raise TrainingDataError("batch sample IDs must be unique")
        if (
            not isinstance(item.image, Tensor)
            or item.image.dtype != torch.float32
            or item.image.ndim != 3
            or tuple(item.image.shape[:2]) != (1, 64)
        ):
            raise TrainingDataError("each image must be float32 with shape [1,64,width]")
        if not torch.isfinite(item.image).all() or item.image.min() < -1 or item.image.max() > 1:
            raise TrainingDataError("batch images must contain finite values within [-1,1]")
        if (
            isinstance(item.valid_width, bool)
            or not isinstance(item.valid_width, int)
            or item.valid_width != item.image.shape[2]
            or not 4 <= item.valid_width <= max_width
        ):
            raise TrainingDataError("valid width must equal image width within model limits")
        if (
            not isinstance(item.target, Tensor)
            or item.target.dtype != torch.int64
            or item.target.device.type != "cpu"
            or item.target.ndim != 1
            or item.target.numel() < 1
        ):
            raise TrainingDataError("each target must be a nonempty CPU int64 vector")
        if torch.any(item.target <= 0) or torch.any(item.target >= class_count):
            raise TrainingDataError("target indexes must be valid nonblank classifier indexes")
        sample_ids.append(item.sample_id)
        targets.append(item.target)

    try:
        images, valid_widths = pad_prepared_line_images(
            [item.image for item in values], max_width=max_width
        )
    except ModelInputError as error:
        raise TrainingDataError("line images could not be padded for the recognizer") from error
    return OCRBatch(
        images=images,
        valid_widths=valid_widths,
        targets=torch.cat(targets),
        target_lengths=torch.tensor([target.numel() for target in targets], dtype=torch.int64),
        sample_ids=tuple(sample_ids),
    )


def validate_training_data(
    train_dataset: OCRLineDataset,
    validation_dataset: OCRLineDataset,
) -> TrainingPreflightReport:
    """Load every image once and prove split, vocabulary, width, and CTC invariants."""

    if not isinstance(train_dataset, OCRLineDataset) or not isinstance(
        validation_dataset, OCRLineDataset
    ):
        raise TypeError("train_dataset and validation_dataset must be OCRLineDataset instances")
    if not train_dataset.samples or not validation_dataset.samples:
        raise TrainingDataError("training and validation partitions must both be nonempty")
    if train_dataset.vocabulary != validation_dataset.vocabulary:
        raise TrainingDataError("training and validation datasets must share one vocabulary")

    train_ids = {sample.sample_id for sample in train_dataset.samples}
    validation_ids = {sample.sample_id for sample in validation_dataset.samples}
    overlap = sorted(train_ids & validation_ids)
    if overlap:
        raise TrainingDataError(
            "training and validation sample IDs overlap",
            context={"sample_id": overlap[0], "count": len(overlap)},
        )
    train_documents = {sample.document_id for sample in train_dataset.samples}
    validation_documents = {sample.document_id for sample in validation_dataset.samples}
    document_overlap = sorted(train_documents & validation_documents)
    if document_overlap:
        raise TrainingDataError(
            "training and validation document groups overlap",
            context={"document_id": document_overlap[0], "count": len(document_overlap)},
        )

    class_count = len(train_dataset.vocabulary.characters) + 1
    image_labels: dict[str, tuple[str, str]] = {}
    for partition_name, dataset in (
        ("train", train_dataset),
        ("validation", validation_dataset),
    ):
        for sample, item in zip(dataset.samples, dataset, strict=True):
            image_identity = sha256(item.image.numpy().tobytes()).hexdigest()
            previous = image_labels.get(image_identity)
            if previous is not None and previous[0] != sample.text:
                raise TrainingDataError(
                    "identical prepared image pixels have conflicting transcriptions",
                    context={"sample_id": item.sample_id, "other_sample_id": previous[1]},
                )
            image_labels[image_identity] = (sample.text, item.sample_id)
            input_lengths = torch.tensor(
                [input_width_to_timesteps(item.valid_width)], dtype=torch.int64
            )
            report = validate_ctc_alignment(
                item.target,
                torch.tensor([item.target.numel()], dtype=torch.int64),
                input_lengths,
                class_count=class_count,
                blank_index=dataset.recognizer_config.blank_index,
            )
            if not report.is_feasible:
                raise TrainingDataError(
                    "sample cannot satisfy CTC alignment",
                    context={"partition": partition_name, "sample_id": item.sample_id},
                )
    return TrainingPreflightReport(
        train_dataset_fingerprint=dataset_fingerprint(train_dataset.samples),
        validation_dataset_fingerprint=dataset_fingerprint(validation_dataset.samples),
        train_sample_count=len(train_dataset),
        validation_sample_count=len(validation_dataset),
    )
