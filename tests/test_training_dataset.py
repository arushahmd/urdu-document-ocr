from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from PIL import Image

from urdu_document_ocr import DatasetSample, RecognizerConfig, Vocabulary
from urdu_document_ocr.errors import TrainingDataError
from urdu_document_ocr.training import OCRLineDataset, validate_training_data


def _sample(name: str = "one", *, text: str = "اب", document: str = "doc") -> DatasetSample:
    return DatasetSample(1, name, f"{name}.png", text, document)


def _write_line(root, name: str = "one", *, width: int = 24, height: int = 12) -> None:
    pixels = np.full((height, width), 255, dtype=np.uint8)
    pixels[3:9, 5:19] = 0
    Image.fromarray(pixels).save(root / f"{name}.png")


def test_dataset_loads_prepares_encodes_deterministically_without_mutation(tmp_path) -> None:
    _write_line(tmp_path)
    sample = _sample()
    samples = [sample]
    dataset = OCRLineDataset(samples, Vocabulary(("ا", "ب")), dataset_root=tmp_path)

    first = dataset[0]
    second = dataset[0]

    assert first.sample_id == "one"
    assert first.image.shape == (1, 64, 128)
    assert first.valid_width == 128
    assert first.target.tolist() == [1, 2]
    assert first.image.equal(second.image)
    assert samples == [sample]


def test_dataset_rejects_unknown_character_and_bad_inputs(tmp_path) -> None:
    _write_line(tmp_path)
    with pytest.raises(TrainingDataError, match="outside the vocabulary"):
        OCRLineDataset([_sample(text="اج")], Vocabulary(("ا", "ب")), dataset_root=tmp_path)
    with pytest.raises(TrainingDataError, match="dataset root"):
        OCRLineDataset([_sample()], Vocabulary(("ا", "ب")), dataset_root=tmp_path / "missing")
    with pytest.raises(TrainingDataError, match="DatasetSample"):
        OCRLineDataset([object()], Vocabulary(("ا",)), dataset_root=tmp_path)  # type: ignore[list-item]


def test_dataset_rejects_invalid_and_over_width_images(tmp_path) -> None:
    (tmp_path / "bad.png").write_bytes(b"not an image")
    bad = OCRLineDataset([_sample("bad")], Vocabulary(("ا", "ب")), dataset_root=tmp_path)
    with pytest.raises(TrainingDataError, match="could not be prepared"):
        bad[0]

    _write_line(tmp_path, "wide", width=500, height=10)
    wide = OCRLineDataset(
        [_sample("wide")],
        Vocabulary(("ا", "ب")),
        dataset_root=tmp_path,
        recognizer_config=RecognizerConfig(max_width=256),
    )
    with pytest.raises(TrainingDataError, match="could not be prepared"):
        wide[0]


def test_dataset_loader_defends_root_containment(tmp_path) -> None:
    outside = tmp_path.parent / "outside-line.png"
    _write_line(tmp_path.parent, "outside-line")
    sample = _sample()
    object.__setattr__(sample, "image_path", "../outside-line.png")
    dataset = OCRLineDataset([sample], Vocabulary(("ا", "ب")), dataset_root=tmp_path)

    with pytest.raises(TrainingDataError, match="could not be prepared") as caught:
        dataset[0]

    assert caught.value.context["image_path"] == "../outside-line.png"
    assert outside.exists()


def test_preflight_proves_splits_and_ctc_feasibility(tmp_path) -> None:
    _write_line(tmp_path, "train")
    _write_line(tmp_path, "val")
    vocabulary = Vocabulary(("ا", "ب"))
    train = OCRLineDataset(
        [_sample("train", document="train-doc")], vocabulary, dataset_root=tmp_path
    )
    validation = OCRLineDataset(
        [_sample("val", document="val-doc")], vocabulary, dataset_root=tmp_path
    )

    report = validate_training_data(train, validation)

    assert report.train_sample_count == report.validation_sample_count == 1
    assert len(report.train_dataset_fingerprint) == 64


def test_preflight_rejects_empty_overlap_leakage_conflict_and_impossible_ctc(tmp_path) -> None:
    for name in ("one", "two", "three"):
        _write_line(tmp_path, name)
    vocabulary = Vocabulary(("ا", "ب"))
    empty = OCRLineDataset([], vocabulary, dataset_root=tmp_path)
    one = OCRLineDataset([_sample()], vocabulary, dataset_root=tmp_path)
    with pytest.raises(TrainingDataError, match="nonempty"):
        validate_training_data(empty, one)

    same_id = OCRLineDataset(
        [replace(_sample(), document_id="other")], vocabulary, dataset_root=tmp_path
    )
    with pytest.raises(TrainingDataError, match="sample IDs overlap"):
        validate_training_data(one, same_id)

    same_document = OCRLineDataset(
        [_sample("two", document="doc")], vocabulary, dataset_root=tmp_path
    )
    with pytest.raises(TrainingDataError, match="document groups overlap"):
        validate_training_data(one, same_document)

    conflicting = OCRLineDataset(
        [_sample("three", text="با", document="other")], vocabulary, dataset_root=tmp_path
    )
    with pytest.raises(TrainingDataError, match="conflicting transcriptions"):
        validate_training_data(one, conflicting)

    _write_line(tmp_path, "tiny", width=4, height=64)
    impossible = OCRLineDataset(
        [_sample("tiny", text="اا", document="other")], vocabulary, dataset_root=tmp_path
    )
    with pytest.raises(TrainingDataError, match="CTC alignment"):
        validate_training_data(one, impossible)
