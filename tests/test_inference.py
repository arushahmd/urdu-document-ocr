from __future__ import annotations

import shutil
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from urdu_document_ocr import (
    CRNNRecognizer,
    LoadedRecognizer,
    OCRPrediction,
    TrainingConfig,
    Vocabulary,
    load_recognizer,
    recognize_line,
    recognize_lines,
)
from urdu_document_ocr.data import save_vocabulary
from urdu_document_ocr.errors import CheckpointError, ModelInputError, RecognitionError
from urdu_document_ocr.recognition import inference as inference_module
from urdu_document_ocr.recognition.base import RecognizerOutput
from urdu_document_ocr.training import save_checkpoint


@pytest.fixture(scope="module")
def inference_checkpoint(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("phase11-checkpoint")
    torch.manual_seed(111)
    save_checkpoint(
        root / "best",
        CRNNRecognizer(Vocabulary(("ا", "ب"))),
        TrainingConfig(batch_size=1, epochs=1),
        epoch=1,
        validation_loss=1.0,
        train_dataset_fingerprint="a" * 64,
        validation_dataset_fingerprint="b" * 64,
    )
    return root / "best"


@pytest.fixture
def loaded(inference_checkpoint: Path) -> LoadedRecognizer:
    return load_recognizer(inference_checkpoint)


class FakeRecognizer:
    config = CRNNRecognizer(Vocabulary(("ا",))).config
    vocabulary = Vocabulary(("ا",))
    fingerprint = "f" * 64

    def __init__(self, texts: tuple[str, ...]) -> None:
        self.texts = texts
        self.calls: list[tuple[int, int]] = []

    def recognize_batch(
        self, images: tuple[np.ndarray, ...], *, batch_size: int
    ) -> tuple[OCRPrediction, ...]:
        self.calls.append((len(images), batch_size))
        return tuple(OCRPrediction(text) for text in self.texts[: len(images)])


def test_public_single_and_empty_batch_boundaries() -> None:
    recognizer = FakeRecognizer(("اردو",))
    image = np.full((12, 30), 255, dtype=np.uint8)

    assert recognize_line(image, recognizer).text == "اردو"
    assert recognizer.calls == [(1, 1)]
    assert recognize_lines((), recognizer, batch_size=5) == ()
    assert recognizer.calls == [(1, 1)]


def test_public_batch_rejects_broken_recognizer_and_iterable() -> None:
    image = np.full((12, 30), 255, dtype=np.uint8)

    class BrokenIterable:
        def __iter__(self):
            raise TypeError("broken")

    class ListRecognizer:
        def recognize_batch(self, images, *, batch_size):
            return [OCRPrediction("اردو")]

    with pytest.raises(RecognitionError, match="iterable"):
        recognize_lines(BrokenIterable(), FakeRecognizer(()))  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="recognize_batch"):
        recognize_lines((image,), object())  # type: ignore[arg-type]
    with pytest.raises(RecognitionError, match="tuple"):
        recognize_lines((image,), ListRecognizer())
    with pytest.raises(RecognitionError, match="count"):
        recognize_lines((image,), FakeRecognizer(()))


@pytest.mark.parametrize("batch_size", [0, -1, True, 1.5])
def test_public_batch_rejects_invalid_sizes(batch_size: object) -> None:
    with pytest.raises(RecognitionError, match="batch_size"):
        recognize_lines((), FakeRecognizer(()), batch_size=batch_size)  # type: ignore[arg-type]


def test_loaded_recognizer_batches_variable_widths_without_reordering(
    loaded: LoadedRecognizer, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[torch.Tensor, list[int], bool, bool]] = []
    emitted = {8: (1,), 12: (2,), 16: (1, 0, 2), 20: (2, 0, 1), 24: ()}

    def controlled_forward(
        model: CRNNRecognizer, images: torch.Tensor, valid_widths: torch.Tensor
    ) -> RecognizerOutput:
        widths = valid_widths.tolist()
        calls.append((images.detach().cpu(), widths, model.training, torch.is_grad_enabled()))
        lengths = torch.tensor([width // 4 for width in widths], dtype=torch.int64)
        logits = torch.full(
            (len(widths), max(lengths).item(), 3),
            -10.0,
            device=images.device,
        )
        logits[:, :, 0] = 10.0
        for batch_index, width in enumerate(widths):
            for time_index, character_index in enumerate(emitted[width]):
                logits[batch_index, time_index, :] = -10.0
                logits[batch_index, time_index, character_index] = 10.0
        return RecognizerOutput(logits, lengths)

    monkeypatch.setattr(CRNNRecognizer, "forward", controlled_forward)
    images = tuple(np.zeros((64, width), dtype=np.uint8) for width in (8, 12, 16, 20, 24))

    predictions = loaded.recognize_batch(images, batch_size=2)

    assert [prediction.text for prediction in predictions] == ["ا", "ب", "اب", "با", ""]
    assert [widths for _batch, widths, _training, _grad in calls] == [
        [8, 12],
        [16, 20],
        [24],
    ]
    assert [tuple(batch.shape) for batch, *_rest in calls] == [
        (2, 1, 64, 12),
        (2, 1, 64, 20),
        (1, 1, 64, 24),
    ]
    assert torch.all(calls[0][0][0, :, :, 8:] == 1.0)
    assert all(not training and not grad_enabled for _, _, training, grad_enabled in calls)


@pytest.mark.parametrize("batch_size", [1, 10])
def test_loaded_recognizer_handles_unit_and_oversized_batches(
    loaded: LoadedRecognizer, monkeypatch: pytest.MonkeyPatch, batch_size: int
) -> None:
    call_sizes: list[int] = []

    def blank_forward(
        _model: CRNNRecognizer, images: torch.Tensor, valid_widths: torch.Tensor
    ) -> RecognizerOutput:
        call_sizes.append(images.shape[0])
        lengths = torch.tensor([width // 4 for width in valid_widths.tolist()], dtype=torch.int64)
        logits = torch.zeros((images.shape[0], max(lengths).item(), 3), device=images.device)
        return RecognizerOutput(logits, lengths)

    monkeypatch.setattr(CRNNRecognizer, "forward", blank_forward)
    images = tuple(np.full((64, width), 255, dtype=np.uint8) for width in (8, 12, 16))

    assert len(loaded.recognize_batch(images, batch_size=batch_size)) == 3
    assert call_sizes == ([1, 1, 1] if batch_size == 1 else [3])


def test_loaded_recognizer_rejects_over_width_and_invalid_device(
    loaded: LoadedRecognizer, inference_checkpoint: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(ModelInputError, match="exceeds"):
        loaded.recognize_batch(
            (np.full((64, loaded.config.max_width + 1), 255, dtype=np.uint8),),
            batch_size=1,
        )
    with pytest.raises(RecognitionError, match="device"):
        load_recognizer(inference_checkpoint, device="auto")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RecognitionError, match="unavailable"):
        load_recognizer(inference_checkpoint, device="cuda")

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    with pytest.raises(RecognitionError, match="index"):
        load_recognizer(inference_checkpoint, device="cuda:1")


def test_loaded_recognizer_validates_construction_and_runtime_failures(
    loaded: LoadedRecognizer, monkeypatch: pytest.MonkeyPatch
) -> None:
    image = np.full((64, 8), 255, dtype=np.uint8)

    with pytest.raises(TypeError, match="CRNNRecognizer"):
        LoadedRecognizer(object(), loaded.checkpoint_metadata, "cpu")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="CheckpointMetadata"):
        LoadedRecognizer(loaded.model, object(), "cpu")  # type: ignore[arg-type]
    with pytest.raises(CheckpointError, match="fingerprint"):
        LoadedRecognizer(
            loaded.model,
            replace(loaded.checkpoint_metadata, model_fingerprint="0" * 64),
            "cpu",
        )

    def runtime_failure(_model, _images, _widths):
        raise RuntimeError("device detail")

    monkeypatch.setattr(CRNNRecognizer, "forward", runtime_failure)
    with pytest.raises(RecognitionError, match="inference failed") as captured:
        loaded.recognize_batch((image,), batch_size=1)
    assert "device detail" not in str(captured.value)


def test_loaded_recognizer_validates_empty_broken_and_decoder_count(
    loaded: LoadedRecognizer, monkeypatch: pytest.MonkeyPatch
) -> None:
    class BrokenIterable:
        def __iter__(self):
            raise TypeError("broken")

    assert loaded.recognize_batch((), batch_size=3) == ()
    with pytest.raises(RecognitionError, match="iterable"):
        loaded.recognize_batch(BrokenIterable(), batch_size=1)  # type: ignore[arg-type]

    def blank_forward(_model, images, widths):
        lengths = torch.tensor([width // 4 for width in widths.tolist()], dtype=torch.int64)
        return RecognizerOutput(
            torch.zeros((images.shape[0], int(lengths.max().item()), 3)), lengths
        )

    monkeypatch.setattr(CRNNRecognizer, "forward", blank_forward)
    monkeypatch.setattr(inference_module, "greedy_ctc_decode", lambda *_args: ())
    with pytest.raises(RecognitionError, match="output count"):
        loaded.recognize_batch((np.full((64, 8), 255, dtype=np.uint8),), batch_size=1)


def test_checkpoint_is_loaded_once_then_reused(
    inference_checkpoint: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from urdu_document_ocr.training import checkpoint as checkpoint_module

    original = checkpoint_module.load_checkpoint_model
    loads = 0

    def counted(path: str | Path):
        nonlocal loads
        loads += 1
        return original(path)

    monkeypatch.setattr(checkpoint_module, "load_checkpoint_model", counted)
    recognizer = load_recognizer(inference_checkpoint)
    monkeypatch.setattr(
        CRNNRecognizer,
        "forward",
        lambda _model, images, widths: RecognizerOutput(
            torch.zeros((images.shape[0], max(widths).item() // 4, 3)),
            torch.tensor([width // 4 for width in widths.tolist()], dtype=torch.int64),
        ),
    )

    recognize_line(np.full((64, 8), 255, dtype=np.uint8), recognizer)
    recognize_line(np.full((64, 12), 255, dtype=np.uint8), recognizer)

    assert loads == 1


def test_load_recognizer_rejects_corrupt_and_mismatched_checkpoint(
    inference_checkpoint: Path, tmp_path: Path
) -> None:
    corrupt = tmp_path / "corrupt"
    shutil.copytree(inference_checkpoint, corrupt)
    with (corrupt / "model.safetensors").open("ab") as handle:
        handle.write(b"tampered")
    with pytest.raises(CheckpointError, match="weight hash"):
        load_recognizer(corrupt)

    mismatch = tmp_path / "mismatch"
    shutil.copytree(inference_checkpoint, mismatch)
    save_vocabulary(Vocabulary(("ا",)), mismatch / "vocabulary.json", overwrite=True)
    with pytest.raises(CheckpointError, match="vocabulary"):
        load_recognizer(mismatch)
