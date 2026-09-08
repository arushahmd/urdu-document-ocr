from __future__ import annotations

import json
import shutil
from hashlib import sha256

import pytest
import torch
from safetensors.torch import load_file, save_file

from urdu_document_ocr import CRNNRecognizer, TrainingConfig, Vocabulary
from urdu_document_ocr.errors import CheckpointError
from urdu_document_ocr.training import checkpoint as checkpoint_module
from urdu_document_ocr.training import load_checkpoint, save_checkpoint


@pytest.fixture(scope="module")
def saved_checkpoint(tmp_path_factory):
    root = tmp_path_factory.mktemp("checkpoint")
    vocabulary = Vocabulary(("ا", "ب"))
    config = TrainingConfig(batch_size=1, epochs=1)
    torch.manual_seed(17)
    model = CRNNRecognizer(vocabulary)
    metadata = save_checkpoint(
        root / "best",
        model,
        config,
        epoch=1,
        validation_loss=2.5,
        train_dataset_fingerprint="a" * 64,
        validation_dataset_fingerprint="b" * 64,
    )
    return root / "best", vocabulary, config, metadata


def _copy_checkpoint(saved_checkpoint, tmp_path):
    source = saved_checkpoint[0]
    target = tmp_path / "best"
    shutil.copytree(source, target)
    return target


def _rewrite_metadata(path, **updates) -> None:
    metadata_path = path / "metadata.json"
    record = json.loads(metadata_path.read_text(encoding="utf-8"))
    record.update(updates)
    record["metadata_fingerprint"] = checkpoint_module._metadata_fingerprint(record)
    metadata_path.write_bytes(checkpoint_module._canonical_json_bytes(record))


def test_checkpoint_structure_hashes_and_roundtrip_exact_weights(saved_checkpoint) -> None:
    path, vocabulary, config, metadata = saved_checkpoint

    assert {item.name for item in path.iterdir()} == {
        "metadata.json",
        "model.safetensors",
        "vocabulary.json",
    }
    assert sha256((path / "model.safetensors").read_bytes()).hexdigest() == (
        metadata.model_weights_sha256
    )
    model = CRNNRecognizer(vocabulary)
    loaded = load_checkpoint(path, model, expected_training_config=config)
    stored = load_file(str(path / "model.safetensors"))

    assert loaded == metadata
    assert all(torch.equal(tensor, stored[name]) for name, tensor in model.state_dict().items())
    assert str(path.parent) not in (path / "metadata.json").read_text(encoding="utf-8")


def test_checkpoint_does_not_silently_overwrite(saved_checkpoint) -> None:
    path, vocabulary, config, _metadata = saved_checkpoint
    with pytest.raises(CheckpointError, match="overwrite=True"):
        save_checkpoint(
            path,
            CRNNRecognizer(vocabulary),
            config,
            epoch=2,
            validation_loss=2.0,
            train_dataset_fingerprint="a" * 64,
            validation_dataset_fingerprint="b" * 64,
        )


def test_checkpoint_explicit_overwrite_replaces_valid_checkpoint(
    saved_checkpoint, tmp_path
) -> None:
    source, vocabulary, config, _metadata = saved_checkpoint
    path = tmp_path / "best"
    shutil.copytree(source, path)
    model = CRNNRecognizer(vocabulary)
    with torch.no_grad():
        model.classifier.bias.add_(1.0)

    metadata = save_checkpoint(
        path,
        model,
        config,
        epoch=2,
        validation_loss=2.0,
        train_dataset_fingerprint="a" * 64,
        validation_dataset_fingerprint="b" * 64,
        overwrite=True,
    )
    loaded = CRNNRecognizer(vocabulary)
    load_checkpoint(path, loaded)

    assert metadata.epoch == 2
    assert torch.equal(model.classifier.bias, loaded.classifier.bias)


def test_failed_overwrite_preserves_existing_checkpoint(saved_checkpoint, monkeypatch) -> None:
    path, vocabulary, config, metadata = saved_checkpoint
    monkeypatch.setattr(
        checkpoint_module,
        "save_file",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("simulated")),
    )
    with pytest.raises(CheckpointError, match="atomically"):
        save_checkpoint(
            path,
            CRNNRecognizer(vocabulary),
            config,
            epoch=2,
            validation_loss=2.0,
            train_dataset_fingerprint="a" * 64,
            validation_dataset_fingerprint="b" * 64,
            overwrite=True,
        )
    assert sha256((path / "model.safetensors").read_bytes()).hexdigest() == (
        metadata.model_weights_sha256
    )


def test_failed_atomic_install_restores_existing_checkpoint(
    saved_checkpoint, tmp_path, monkeypatch
) -> None:
    source, vocabulary, config, metadata = saved_checkpoint
    path = tmp_path / "best"
    shutil.copytree(source, path)
    real_replace = checkpoint_module.os.replace
    directory_calls = 0

    def fail_second_directory_replace(source_path, target_path):
        nonlocal directory_calls
        if checkpoint_module.Path(source_path).is_dir():
            directory_calls += 1
            if directory_calls == 2:
                raise OSError("simulated atomic install failure")
        return real_replace(source_path, target_path)

    monkeypatch.setattr(checkpoint_module.os, "replace", fail_second_directory_replace)
    with pytest.raises(CheckpointError, match="atomically"):
        save_checkpoint(
            path,
            CRNNRecognizer(vocabulary),
            config,
            epoch=2,
            validation_loss=2.0,
            train_dataset_fingerprint="a" * 64,
            validation_dataset_fingerprint="b" * 64,
            overwrite=True,
        )

    assert path.is_dir()
    assert sha256((path / "model.safetensors").read_bytes()).hexdigest() == (
        metadata.model_weights_sha256
    )


def test_load_rejects_vocabulary_and_training_config_mismatch(saved_checkpoint) -> None:
    path, _vocabulary, _config, _metadata = saved_checkpoint
    with pytest.raises(CheckpointError, match="vocabulary"):
        load_checkpoint(path, CRNNRecognizer(Vocabulary(("ا",))))
    with pytest.raises(CheckpointError, match="training configuration"):
        load_checkpoint(
            path,
            CRNNRecognizer(Vocabulary(("ا", "ب"))),
            expected_training_config=TrainingConfig(batch_size=2),
        )


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"architecture_id": "different"}, "architecture"),
        ({"blank_index": 1}, "blank index"),
        ({"model_fingerprint": "0" * 64}, "model fingerprint"),
        ({"vocabulary_fingerprint": "0" * 64}, "vocabulary"),
    ],
)
def test_load_rejects_metadata_mismatches(saved_checkpoint, tmp_path, updates, message) -> None:
    path = _copy_checkpoint(saved_checkpoint, tmp_path)
    _rewrite_metadata(path, **updates)
    with pytest.raises(CheckpointError, match=message):
        load_checkpoint(path, CRNNRecognizer(Vocabulary(("ا", "ب"))))


def test_load_rejects_corrupted_weights_and_json(saved_checkpoint, tmp_path) -> None:
    path = _copy_checkpoint(saved_checkpoint, tmp_path)
    with (path / "model.safetensors").open("ab") as handle:
        handle.write(b"tampered")
    with pytest.raises(CheckpointError, match="weight hash"):
        load_checkpoint(path, CRNNRecognizer(Vocabulary(("ا", "ب"))))

    path = _copy_checkpoint(saved_checkpoint, tmp_path / "json")
    (path / "metadata.json").write_text('{"x":1,"x":2}', encoding="utf-8")
    with pytest.raises(CheckpointError, match="strict UTF-8 JSON"):
        load_checkpoint(path, CRNNRecognizer(Vocabulary(("ا", "ب"))))


@pytest.mark.parametrize("unexpected", [False, True])
def test_load_rejects_missing_and_unexpected_tensors(
    saved_checkpoint, tmp_path, unexpected
) -> None:
    path = _copy_checkpoint(saved_checkpoint, tmp_path)
    weights_path = path / "model.safetensors"
    tensors = load_file(str(weights_path))
    if unexpected:
        tensors["unexpected"] = torch.zeros(1)
    else:
        tensors.pop(next(iter(tensors)))
    save_file(tensors, str(weights_path))
    _rewrite_metadata(
        path,
        model_weights_sha256=sha256(weights_path.read_bytes()).hexdigest(),
    )
    with pytest.raises(CheckpointError, match="tensor names"):
        load_checkpoint(path, CRNNRecognizer(Vocabulary(("ا", "ب"))))


def test_load_rejects_wrong_tensor_shape(saved_checkpoint, tmp_path) -> None:
    path = _copy_checkpoint(saved_checkpoint, tmp_path)
    weights_path = path / "model.safetensors"
    tensors = load_file(str(weights_path))
    name = "classifier.bias"
    tensors[name] = tensors[name][:-1]
    save_file(tensors, str(weights_path))
    _rewrite_metadata(
        path,
        model_weights_sha256=sha256(weights_path.read_bytes()).hexdigest(),
    )
    with pytest.raises(CheckpointError, match="shape or dtype"):
        load_checkpoint(path, CRNNRecognizer(Vocabulary(("ا", "ب"))))
