"""Safetensors checkpoints with strict deterministic JSON metadata."""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from uuid import uuid4

import torch
from safetensors import __version__ as safetensors_version
from safetensors.torch import load_file, save_file

from urdu_document_ocr.config import RecognizerConfig, TrainingConfig
from urdu_document_ocr.data import load_vocabulary, save_vocabulary
from urdu_document_ocr.data.manifest import (
    _canonical_json_bytes,
    _reject_duplicate_json_keys,
    _reject_nonstandard_json_constant,
)
from urdu_document_ocr.errors import CheckpointError
from urdu_document_ocr.recognition.model import MODEL_ARCHITECTURE_ID, CRNNRecognizer

CHECKPOINT_SCHEMA_VERSION = 1
_EXPECTED_FILES = frozenset({"metadata.json", "model.safetensors", "vocabulary.json"})
_SAFE_IDENTIFIER = re.compile(r"[A-Za-z0-9._+-]{1,64}")
_METADATA_FIELDS = frozenset(
    {
        "schema_version",
        "package_version",
        "architecture_id",
        "model_fingerprint",
        "recognizer_config",
        "recognizer_config_fingerprint",
        "vocabulary_fingerprint",
        "vocabulary_file_sha256",
        "class_count",
        "blank_index",
        "epoch",
        "validation_loss",
        "training_config",
        "training_config_fingerprint",
        "train_dataset_fingerprint",
        "validation_dataset_fingerprint",
        "model_weights_sha256",
        "dependency_versions",
        "metadata_fingerprint",
    }
)


def _package_version() -> str:
    try:
        return version("urdu-document-ocr")
    except PackageNotFoundError:
        return "0.1.0"


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _metadata_fingerprint(record: dict[str, object]) -> str:
    identity = dict(record)
    identity.pop("metadata_fingerprint", None)
    return sha256(_canonical_json_bytes(identity)).hexdigest()


def _fsync(path: Path) -> None:
    with path.open("r+b") as handle:
        os.fsync(handle.fileno())


def _write_bytes(path: Path, payload: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


@dataclass(frozen=True, slots=True)
class CheckpointMetadata:
    """Validated public identity of one model-only checkpoint."""

    schema_version: int
    package_version: str
    architecture_id: str
    model_fingerprint: str
    recognizer_config: dict[str, object]
    recognizer_config_fingerprint: str
    vocabulary_fingerprint: str
    vocabulary_file_sha256: str
    class_count: int
    blank_index: int
    epoch: int
    validation_loss: float
    training_config: dict[str, object]
    training_config_fingerprint: str
    train_dataset_fingerprint: str
    validation_dataset_fingerprint: str
    model_weights_sha256: str
    dependency_versions: dict[str, str]
    metadata_fingerprint: str

    def to_public_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "package_version": self.package_version,
            "architecture_id": self.architecture_id,
            "model_fingerprint": self.model_fingerprint,
            "recognizer_config": dict(self.recognizer_config),
            "recognizer_config_fingerprint": self.recognizer_config_fingerprint,
            "vocabulary_fingerprint": self.vocabulary_fingerprint,
            "vocabulary_file_sha256": self.vocabulary_file_sha256,
            "class_count": self.class_count,
            "blank_index": self.blank_index,
            "epoch": self.epoch,
            "validation_loss": self.validation_loss,
            "training_config": dict(self.training_config),
            "training_config_fingerprint": self.training_config_fingerprint,
            "train_dataset_fingerprint": self.train_dataset_fingerprint,
            "validation_dataset_fingerprint": self.validation_dataset_fingerprint,
            "model_weights_sha256": self.model_weights_sha256,
            "dependency_versions": dict(self.dependency_versions),
            "metadata_fingerprint": self.metadata_fingerprint,
        }


def _metadata_from_record(record: object) -> CheckpointMetadata:
    if not isinstance(record, dict) or set(record) != _METADATA_FIELDS:
        raise CheckpointError("checkpoint metadata fields do not match schema version 1")
    integer_fields = ("schema_version", "class_count", "blank_index", "epoch")
    if any(type(record[name]) is not int for name in integer_fields):
        raise CheckpointError("checkpoint integer metadata fields are malformed")
    if record["schema_version"] != CHECKPOINT_SCHEMA_VERSION:
        raise CheckpointError("checkpoint schema version is unsupported")
    if record["class_count"] < 2 or record["blank_index"] != 0 or record["epoch"] < 1:
        raise CheckpointError("checkpoint dimensions, blank index, or epoch are invalid")
    if not isinstance(record["validation_loss"], (int, float)) or isinstance(
        record["validation_loss"], bool
    ):
        raise CheckpointError("checkpoint validation loss must be numeric")
    if not math.isfinite(float(record["validation_loss"])):
        raise CheckpointError("checkpoint validation loss must be finite")
    string_fields = (
        "package_version",
        "architecture_id",
        "model_fingerprint",
        "recognizer_config_fingerprint",
        "vocabulary_fingerprint",
        "vocabulary_file_sha256",
        "training_config_fingerprint",
        "train_dataset_fingerprint",
        "validation_dataset_fingerprint",
        "model_weights_sha256",
        "metadata_fingerprint",
    )
    if any(not isinstance(record[name], str) or not record[name] for name in string_fields):
        raise CheckpointError("checkpoint string metadata fields are malformed")
    if any(
        _SAFE_IDENTIFIER.fullmatch(record[name]) is None
        for name in ("package_version", "architecture_id")
    ):
        raise CheckpointError("checkpoint package or architecture identifier is unsafe")
    hash_fields = (
        *(name for name in string_fields if name.endswith("fingerprint")),
        "vocabulary_file_sha256",
        "model_weights_sha256",
    )
    if any(not _is_sha256(record[name]) for name in hash_fields):
        raise CheckpointError("checkpoint hashes must be lowercase SHA-256 values")
    if not isinstance(record["recognizer_config"], dict) or not isinstance(
        record["training_config"], dict
    ):
        raise CheckpointError("checkpoint configuration metadata must be mappings")
    dependencies = record["dependency_versions"]
    if (
        not isinstance(dependencies, dict)
        or set(dependencies) != {"safetensors", "torch"}
        or any(
            not isinstance(value, str) or _SAFE_IDENTIFIER.fullmatch(value) is None
            for value in dependencies.values()
        )
    ):
        raise CheckpointError("checkpoint dependency version metadata is malformed")
    if record["metadata_fingerprint"] != _metadata_fingerprint(record):
        raise CheckpointError("checkpoint metadata fingerprint does not match its payload")
    try:
        return CheckpointMetadata(**record)  # type: ignore[arg-type]
    except TypeError as error:
        raise CheckpointError("checkpoint metadata could not be constructed") from error


def _read_metadata(path: Path) -> CheckpointMetadata:
    try:
        raw = path.read_text(encoding="utf-8")
        record = json.loads(
            raw,
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=_reject_nonstandard_json_constant,
        )
    except (OSError, UnicodeError, ValueError) as error:
        raise CheckpointError("checkpoint metadata is not strict UTF-8 JSON") from error
    return _metadata_from_record(record)


def _validated_checkpoint_directory(path: str | os.PathLike[str]) -> Path:
    try:
        checkpoint = Path(path)
    except (TypeError, ValueError) as error:
        raise CheckpointError("checkpoint directory is invalid") from error
    if checkpoint.is_symlink() or not checkpoint.is_dir():
        raise CheckpointError("checkpoint directory must be a regular directory")
    try:
        entries = {entry.name for entry in checkpoint.iterdir()}
    except OSError as error:
        raise CheckpointError("checkpoint directory could not be inspected") from error
    if entries != _EXPECTED_FILES or any((checkpoint / name).is_symlink() for name in entries):
        raise CheckpointError("checkpoint directory contents do not match schema version 1")
    if any(not (checkpoint / name).is_file() for name in entries):
        raise CheckpointError("checkpoint entries must be regular files")
    return checkpoint


def save_checkpoint(
    checkpoint_directory: str | os.PathLike[str],
    model: CRNNRecognizer,
    training_config: TrainingConfig,
    *,
    epoch: int,
    validation_loss: float,
    train_dataset_fingerprint: str,
    validation_dataset_fingerprint: str,
    overwrite: bool = False,
) -> CheckpointMetadata:
    """Atomically save model tensors, vocabulary, and deterministic identity metadata."""

    if not isinstance(model, CRNNRecognizer):
        raise TypeError("model must be a CRNNRecognizer")
    if not isinstance(training_config, TrainingConfig):
        raise TypeError("training_config must be a TrainingConfig")
    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 1:
        raise CheckpointError("checkpoint epoch must be a positive integer")
    if (
        isinstance(validation_loss, bool)
        or not isinstance(validation_loss, (int, float))
        or not math.isfinite(float(validation_loss))
    ):
        raise CheckpointError("checkpoint validation loss must be finite")
    for name, value in (
        ("train_dataset_fingerprint", train_dataset_fingerprint),
        ("validation_dataset_fingerprint", validation_dataset_fingerprint),
    ):
        if not _is_sha256(value):
            raise CheckpointError(f"{name} must be a lowercase SHA-256 value")
    if not isinstance(overwrite, bool):
        raise CheckpointError("overwrite must be a boolean")

    target = Path(checkpoint_directory)
    if target.is_symlink() or target.parent.is_symlink():
        raise CheckpointError("checkpoint path cannot use symbolic links")
    parent = target.parent.resolve(strict=False)
    if not parent.is_dir() or parent.is_symlink() or not target.name:
        raise CheckpointError("checkpoint parent must be an existing regular directory")
    if target.exists():
        if not overwrite:
            raise CheckpointError("checkpoint already exists; set overwrite=True to replace it")
        _validated_checkpoint_directory(target)

    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}-", dir=parent))
    backup: Path | None = None
    try:
        weights_path = temporary / "model.safetensors"
        tensors = {
            name: tensor.detach().cpu().contiguous() for name, tensor in model.state_dict().items()
        }
        save_file(tensors, str(weights_path))
        _fsync(weights_path)
        vocabulary_path = temporary / "vocabulary.json"
        save_vocabulary(model.vocabulary, vocabulary_path)
        _fsync(vocabulary_path)
        record: dict[str, object] = {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "package_version": _package_version(),
            "architecture_id": MODEL_ARCHITECTURE_ID,
            "model_fingerprint": model.fingerprint,
            "recognizer_config": model.config.to_dict(),
            "recognizer_config_fingerprint": model.config.fingerprint,
            "vocabulary_fingerprint": model.vocabulary.fingerprint,
            "vocabulary_file_sha256": _sha256_file(vocabulary_path),
            "class_count": len(model.vocabulary.characters) + 1,
            "blank_index": model.config.blank_index,
            "epoch": epoch,
            "validation_loss": float(validation_loss),
            "training_config": training_config.to_dict(),
            "training_config_fingerprint": training_config.fingerprint,
            "train_dataset_fingerprint": train_dataset_fingerprint,
            "validation_dataset_fingerprint": validation_dataset_fingerprint,
            "model_weights_sha256": _sha256_file(weights_path),
            "dependency_versions": {
                "safetensors": safetensors_version,
                "torch": torch.__version__,
            },
        }
        record["metadata_fingerprint"] = _metadata_fingerprint(record)
        metadata = _metadata_from_record(record)
        _write_bytes(temporary / "metadata.json", _canonical_json_bytes(record))

        if target.exists():
            backup = parent / f".{target.name}.backup-{uuid4().hex}"
            os.replace(target, backup)
        try:
            os.replace(temporary, target)
        except OSError:
            if backup is not None:
                os.replace(backup, target)
                backup = None
            raise
        if backup is not None:
            shutil.rmtree(backup, ignore_errors=True)
            backup = None
        return metadata
    except CheckpointError:
        raise
    except Exception as error:
        raise CheckpointError("checkpoint could not be saved atomically") from error
    finally:
        if temporary.exists():
            shutil.rmtree(temporary, ignore_errors=True)
        if backup is not None and backup.exists() and not target.exists():
            os.replace(backup, target)


def load_checkpoint(
    checkpoint_directory: str | os.PathLike[str],
    model: CRNNRecognizer,
    *,
    expected_training_config: TrainingConfig | None = None,
) -> CheckpointMetadata:
    """Verify every checkpoint identity and tensor before mutating ``model``."""

    if not isinstance(model, CRNNRecognizer):
        raise TypeError("model must be a CRNNRecognizer")
    if expected_training_config is not None and not isinstance(
        expected_training_config, TrainingConfig
    ):
        raise TypeError("expected_training_config must be TrainingConfig or None")
    checkpoint = _validated_checkpoint_directory(checkpoint_directory)
    metadata = _read_metadata(checkpoint / "metadata.json")
    try:
        recognizer_config = RecognizerConfig.from_mapping(metadata.recognizer_config)
        training_config = TrainingConfig.from_mapping(metadata.training_config)
        vocabulary = load_vocabulary(checkpoint / "vocabulary.json")
    except Exception as error:
        raise CheckpointError("checkpoint configuration or vocabulary is invalid") from error
    if recognizer_config.fingerprint != metadata.recognizer_config_fingerprint:
        raise CheckpointError("recognizer configuration fingerprint mismatch")
    if training_config.fingerprint != metadata.training_config_fingerprint:
        raise CheckpointError("training configuration fingerprint mismatch")
    if expected_training_config is not None and training_config != expected_training_config:
        raise CheckpointError("checkpoint training configuration does not match the expected one")
    if metadata.architecture_id != MODEL_ARCHITECTURE_ID:
        raise CheckpointError("checkpoint architecture identifier is incompatible")
    if recognizer_config != model.config:
        raise CheckpointError("checkpoint recognizer configuration does not match the model")
    if vocabulary != model.vocabulary or vocabulary.fingerprint != metadata.vocabulary_fingerprint:
        raise CheckpointError("checkpoint vocabulary does not match the model")
    if metadata.model_fingerprint != model.fingerprint:
        raise CheckpointError("checkpoint model fingerprint does not match the model")
    if metadata.class_count != len(model.vocabulary.characters) + 1:
        raise CheckpointError("checkpoint class count does not match the model")
    if metadata.blank_index != model.config.blank_index:
        raise CheckpointError("checkpoint blank index does not match the model")
    if _sha256_file(checkpoint / "vocabulary.json") != metadata.vocabulary_file_sha256:
        raise CheckpointError("checkpoint vocabulary file hash mismatch")
    weights_path = checkpoint / "model.safetensors"
    if _sha256_file(weights_path) != metadata.model_weights_sha256:
        raise CheckpointError("checkpoint model weight hash mismatch")
    try:
        tensors = load_file(str(weights_path), device="cpu")
    except Exception as error:
        raise CheckpointError("checkpoint safetensors file could not be decoded") from error
    expected = model.state_dict()
    if set(tensors) != set(expected):
        raise CheckpointError("checkpoint tensor names do not exactly match the model")
    for name, expected_tensor in expected.items():
        candidate = tensors[name]
        if candidate.shape != expected_tensor.shape or candidate.dtype != expected_tensor.dtype:
            raise CheckpointError("checkpoint tensor shape or dtype does not match the model")
        if not torch.isfinite(candidate).all():
            raise CheckpointError("checkpoint tensors must contain only finite values")
    try:
        model.load_state_dict(tensors, strict=True)
    except RuntimeError as error:
        raise CheckpointError("checkpoint tensors could not be loaded strictly") from error
    return metadata


def load_checkpoint_model(
    checkpoint_directory: str | os.PathLike[str],
) -> tuple[CRNNRecognizer, CheckpointMetadata]:
    """Construct and strictly load the model described by a safe checkpoint."""

    checkpoint = _validated_checkpoint_directory(checkpoint_directory)
    metadata = _read_metadata(checkpoint / "metadata.json")
    try:
        config = RecognizerConfig.from_mapping(metadata.recognizer_config)
        vocabulary = load_vocabulary(checkpoint / "vocabulary.json")
        model = CRNNRecognizer(vocabulary, config)
    except CheckpointError:
        raise
    except Exception as error:
        raise CheckpointError("checkpoint model could not be constructed safely") from error
    verified_metadata = load_checkpoint(checkpoint, model)
    if verified_metadata != metadata:
        raise CheckpointError("checkpoint metadata changed during model loading")
    return model, verified_metadata
