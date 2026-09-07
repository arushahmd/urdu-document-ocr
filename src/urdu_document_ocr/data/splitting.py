"""Deterministic document-grouped train/validation/test splitting."""

from __future__ import annotations

import os
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum
from hashlib import sha256
from pathlib import Path

from urdu_document_ocr.config import DatasetSplitConfig
from urdu_document_ocr.data.manifest import (
    _atomic_write_bytes,
    _canonical_json_bytes,
    canonical_manifest_bytes,
    dataset_fingerprint,
)
from urdu_document_ocr.data.validation import (
    NORMALIZATION_POLICY_VERSION,
    _transcription_issue_codes,
)
from urdu_document_ocr.errors import DatasetSplitError
from urdu_document_ocr.types import DatasetSample

SPLIT_ALGORITHM = "group-greedy-v1"


class DatasetPartition(StrEnum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


_PARTITION_ORDER = (
    DatasetPartition.TRAIN,
    DatasetPartition.VALIDATION,
    DatasetPartition.TEST,
)


@dataclass(frozen=True, slots=True)
class SplitCounts:
    train: int
    validation: int
    test: int

    def __post_init__(self) -> None:
        if any(
            type(value) is not int or value < 0
            for value in (self.train, self.validation, self.test)
        ):
            raise ValueError("split counts must be nonnegative integers")

    @property
    def total(self) -> int:
        return self.train + self.validation + self.test

    def to_public_dict(self) -> dict[str, int]:
        return {"train": self.train, "validation": self.validation, "test": self.test}


@dataclass(frozen=True, slots=True)
class DatasetSplitResult:
    train: tuple[DatasetSample, ...]
    validation: tuple[DatasetSample, ...]
    test: tuple[DatasetSample, ...]
    document_assignments: tuple[tuple[str, DatasetPartition], ...]
    config: DatasetSplitConfig
    dataset_fingerprint: str
    algorithm: str = SPLIT_ALGORITHM

    def __post_init__(self) -> None:
        if not isinstance(self.config, DatasetSplitConfig):
            raise TypeError("config must be DatasetSplitConfig")
        if self.algorithm != SPLIT_ALGORITHM:
            raise ValueError("algorithm must be group-greedy-v1")
        partitions = (self.train, self.validation, self.test)
        if any(not isinstance(values, tuple) for values in partitions) or any(
            not isinstance(sample, DatasetSample) for values in partitions for sample in values
        ):
            raise TypeError("split partitions must be tuples of DatasetSample values")
        all_samples = self.train + self.validation + self.test
        identifiers = [sample.sample_id for sample in all_samples]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("split partitions must contain every sample at most once")
        assignments = dict(self.document_assignments)
        if len(assignments) != len(self.document_assignments):
            raise ValueError("document assignments must be unique")
        expected: dict[str, DatasetPartition] = {}
        for partition, values in zip(_PARTITION_ORDER, partitions, strict=True):
            for sample in values:
                previous = expected.setdefault(sample.document_id, partition)
                if previous is not partition:
                    raise ValueError("a document cannot cross split partitions")
        if assignments != expected:
            raise ValueError("document assignments must match partition contents")
        if dataset_fingerprint(all_samples) != self.dataset_fingerprint:
            raise ValueError("dataset fingerprint must match split samples")

    @property
    def achieved_sample_counts(self) -> SplitCounts:
        return SplitCounts(len(self.train), len(self.validation), len(self.test))

    @property
    def achieved_document_counts(self) -> SplitCounts:
        counts = {partition: 0 for partition in _PARTITION_ORDER}
        for _, partition in self.document_assignments:
            counts[partition] += 1
        return SplitCounts(
            counts[DatasetPartition.TRAIN],
            counts[DatasetPartition.VALIDATION],
            counts[DatasetPartition.TEST],
        )

    @property
    def achieved_ratios(self) -> dict[str, float]:
        counts = self.achieved_sample_counts
        if counts.total == 0:
            return {partition.value: 0.0 for partition in _PARTITION_ORDER}
        return {
            "train": counts.train / counts.total,
            "validation": counts.validation / counts.total,
            "test": counts.test / counts.total,
        }

    @property
    def fingerprint(self) -> str:
        payload = {
            "algorithm": self.algorithm,
            "dataset_fingerprint": self.dataset_fingerprint,
            "document_assignments": [
                {"document_id": document_id, "partition": partition.value}
                for document_id, partition in self.document_assignments
            ],
            "ratios": {
                "test": self.config.test_ratio,
                "train": self.config.train_ratio,
                "validation": self.config.validation_ratio,
            },
            "seed": self.config.seed,
        }
        return sha256(_canonical_json_bytes(payload, final_newline=False)).hexdigest()

    def to_public_dict(self) -> dict[str, object]:
        return {
            "algorithm": self.algorithm,
            "seed": self.config.seed,
            "requested_ratios": {
                "train": self.config.train_ratio,
                "validation": self.config.validation_ratio,
                "test": self.config.test_ratio,
            },
            "achieved_ratios": self.achieved_ratios,
            "achieved_sample_counts": self.achieved_sample_counts.to_public_dict(),
            "achieved_document_counts": self.achieved_document_counts.to_public_dict(),
            "document_assignments": [
                {"document_id": document_id, "partition": partition.value}
                for document_id, partition in self.document_assignments
            ],
            "dataset_fingerprint": self.dataset_fingerprint,
            "split_fingerprint": self.fingerprint,
        }


def _tie_digest(document_id: str, seed: int) -> bytes:
    return sha256(f"{seed}\0{document_id}".encode()).digest()


def _require_split_samples(samples: Iterable[DatasetSample]) -> tuple[DatasetSample, ...]:
    values = tuple(samples)
    if any(not isinstance(sample, DatasetSample) for sample in values):
        raise DatasetSplitError("samples must contain only DatasetSample values")
    identifiers = [sample.sample_id for sample in values]
    if len(identifiers) != len(set(identifiers)):
        raise DatasetSplitError("sample_id values must be unique before splitting")
    if any(_transcription_issue_codes(sample.text) for sample in values):
        raise DatasetSplitError("transcriptions must satisfy the normalization policy")
    return values


def split_dataset(
    samples: Iterable[DatasetSample],
    config: DatasetSplitConfig | None = None,
) -> DatasetSplitResult:
    """Assign complete document groups using the frozen group-greedy-v1 algorithm."""

    values = _require_split_samples(samples)
    if config is None:
        config = DatasetSplitConfig()
    if not isinstance(config, DatasetSplitConfig):
        raise TypeError("config must be DatasetSplitConfig or None")

    grouped: dict[str, list[DatasetSample]] = defaultdict(list)
    for sample in values:
        grouped[sample.document_id].append(sample)
    ordered_groups = sorted(
        grouped.items(),
        key=lambda item: (-len(item[1]), _tie_digest(item[0], config.seed), item[0].encode()),
    )

    ratios = {
        DatasetPartition.TRAIN: config.train_ratio,
        DatasetPartition.VALIDATION: config.validation_ratio,
        DatasetPartition.TEST: config.test_ratio,
    }
    eligible = tuple(partition for partition in _PARTITION_ORDER if ratios[partition] > 0.0)
    targets = {partition: len(values) * ratios[partition] for partition in _PARTITION_ORDER}
    counts = {partition: 0 for partition in _PARTITION_ORDER}
    assignments: dict[str, DatasetPartition] = {}
    partitions: dict[DatasetPartition, list[DatasetSample]] = {
        partition: [] for partition in _PARTITION_ORDER
    }

    for document_id, group in ordered_groups:
        size = len(group)
        choices = []
        for fixed_order, partition in enumerate(eligible):
            deviation = sum(
                abs(
                    counts[candidate] + (size if candidate is partition else 0) - targets[candidate]
                )
                for candidate in _PARTITION_ORDER
            )
            choices.append((deviation, fixed_order, partition))
        _, _, chosen = min(choices)
        counts[chosen] += size
        assignments[document_id] = chosen
        partitions[chosen].extend(group)

    for partition in _PARTITION_ORDER:
        partitions[partition].sort(key=lambda sample: sample.sample_id)
    ordered_assignments = tuple(
        sorted(assignments.items(), key=lambda item: item[0].encode("utf-8"))
    )
    return DatasetSplitResult(
        train=tuple(partitions[DatasetPartition.TRAIN]),
        validation=tuple(partitions[DatasetPartition.VALIDATION]),
        test=tuple(partitions[DatasetPartition.TEST]),
        document_assignments=ordered_assignments,
        config=config,
        dataset_fingerprint=dataset_fingerprint(values),
    )


def _split_metadata(result: DatasetSplitResult, output_hashes: dict[str, str]) -> bytes:
    payload = result.to_public_dict()
    payload.update(
        {
            "schema_version": 1,
            "normalization_policy_version": NORMALIZATION_POLICY_VERSION,
            "output_sha256": output_hashes,
        }
    )
    return _canonical_json_bytes(payload)


def write_split_manifests(
    result: DatasetSplitResult,
    target_directory: str | os.PathLike[str],
    *,
    overwrite: bool = False,
) -> None:
    """Write three canonical manifests and metadata into an existing directory."""

    if not isinstance(result, DatasetSplitResult):
        raise TypeError("result must be DatasetSplitResult")
    try:
        target = Path(target_directory)
    except (TypeError, ValueError) as error:
        raise DatasetSplitError("invalid split output directory") from error
    if not target.is_dir():
        raise DatasetSplitError("split output directory must already exist")

    artifacts = {
        "train.jsonl": canonical_manifest_bytes(result.train),
        "validation.jsonl": canonical_manifest_bytes(result.validation),
        "test.jsonl": canonical_manifest_bytes(result.test),
    }
    output_hashes = {
        name: sha256(payload).hexdigest() for name, payload in sorted(artifacts.items())
    }
    artifacts["split_metadata.json"] = _split_metadata(result, output_hashes)
    if not overwrite and any((target / name).exists() for name in artifacts):
        raise DatasetSplitError("split artifacts already exist; set overwrite=True explicitly")
    for name, payload in artifacts.items():
        _atomic_write_bytes(
            payload,
            target / name,
            overwrite=overwrite,
            error_type=DatasetSplitError,
            artifact_name="split artifact",
        )
