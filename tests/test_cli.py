from __future__ import annotations

import importlib.util
import io
import json
import sys
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from urdu_document_ocr import (
    DatasetSample,
    OCRPrediction,
    Vocabulary,
    save_vocabulary,
    write_manifest,
)
from urdu_document_ocr.cli import build_parser, main

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_ROOT = REPOSITORY_ROOT / "data" / "sample"
SAMPLE_LINE = SAMPLE_ROOT / "lines" / "syn-line-000001.png"
HAS_ML = importlib.util.find_spec("torch") is not None
HAS_API = (
    importlib.util.find_spec("fastapi") is not None
    and importlib.util.find_spec("uvicorn") is not None
)


class FakeRecognizer:
    fingerprint = "f" * 64

    def __init__(self, text: str = "اردو متن") -> None:
        self.text = text
        self.vocabulary = Vocabulary(tuple(sorted(set(text), key=ord)))
        self.calls = 0

    def recognize_batch(self, images, *, batch_size):
        self.calls += 1
        return tuple(OCRPrediction(self.text) for _ in images)


def _patch_loaded_recognizer(monkeypatch: pytest.MonkeyPatch, recognizer: FakeRecognizer) -> None:
    import urdu_document_ocr.recognition as recognition

    monkeypatch.setattr(recognition, "load_recognizer", lambda *_args, **_kwargs: recognizer)


def _pdf_from_image(path: Path) -> bytes:
    output = io.BytesIO()
    with Image.open(path) as source:
        source.convert("RGB").save(output, format="PDF", resolution=72.0)
    return output.getvalue()


def _write_line(path: Path, *, offset: int) -> None:
    image = Image.new("L", (128, 64), color=255)
    draw = ImageDraw.Draw(image)
    draw.rectangle((20 + offset, 20, 100, 43), fill=0)
    image.save(path)


def _dataset(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    root = tmp_path / "dataset"
    root.mkdir()
    train_image = root / "train.png"
    validation_image = root / "validation.png"
    _write_line(train_image, offset=0)
    _write_line(validation_image, offset=3)
    train_manifest = tmp_path / "train.jsonl"
    validation_manifest = tmp_path / "validation.jsonl"
    write_manifest(
        (DatasetSample(1, "train-1", "train.png", "ا", "train-document"),),
        train_manifest,
    )
    write_manifest(
        (
            DatasetSample(
                1,
                "validation-1",
                "validation.png",
                "ا",
                "validation-document",
            ),
        ),
        validation_manifest,
    )
    vocabulary_path = tmp_path / "vocabulary.json"
    save_vocabulary(Vocabulary(("ا",)), vocabulary_path)
    return root, train_manifest, validation_manifest, vocabulary_path


def test_parser_help_version_unknown_command_and_command_inventory(capsys) -> None:
    with pytest.raises(SystemExit, match="0"):
        main(["--help"])
    help_output = capsys.readouterr()
    assert "recognize" in help_output.out
    assert "generate-synthetic" in help_output.out
    assert help_output.err == ""

    with pytest.raises(SystemExit, match="0"):
        main(["--version"])
    assert capsys.readouterr().out == "urdu-ocr 0.1.0\n"

    with pytest.raises(SystemExit, match="2"):
        main(["not-a-command"])
    assert "invalid choice" in capsys.readouterr().err
    assert set(build_parser()._subparsers._group_actions[0].choices) == {
        "recognize",
        "segment",
        "train",
        "evaluate",
        "validate-data",
        "split-data",
        "generate-synthetic",
        "serve",
    }


@pytest.mark.skipif(not HAS_ML, reason="ML extra is not installed")
def test_recognize_text_json_file_pdf_and_overwrite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    recognizer = FakeRecognizer("سلام")
    _patch_loaded_recognizer(monkeypatch, recognizer)

    assert main(["recognize", str(SAMPLE_LINE), "--checkpoint", "unused"]) == 0
    text_output = capsys.readouterr()
    assert "سلام" in text_output.out and text_output.err == ""

    assert (
        main(
            [
                "recognize",
                str(SAMPLE_LINE),
                "--checkpoint",
                "unused",
                "--format",
                "json",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["assembled_text"] == "سلام"

    output = tmp_path / "recognized.txt"
    assert (
        main(
            [
                "recognize",
                str(SAMPLE_LINE),
                "--checkpoint",
                "unused",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    assert output.read_text(encoding="utf-8") == "سلام"
    assert (
        main(
            [
                "recognize",
                str(SAMPLE_LINE),
                "--checkpoint",
                "unused",
                "--output",
                str(output),
            ]
        )
        == 2
    )
    assert "overwrite=True" in capsys.readouterr().err

    pdf = tmp_path / "line.pdf"
    pdf.write_bytes(_pdf_from_image(SAMPLE_LINE))
    assert main(["recognize", str(pdf), "--checkpoint", "unused", "--pdf-dpi", "72"]) == 0
    assert "سلام" in capsys.readouterr().out


@pytest.mark.skipif(not HAS_ML, reason="ML extra is not installed")
def test_recognize_expected_failures_are_concise(tmp_path: Path, capsys) -> None:
    unsupported = tmp_path / "not-an-image.bin"
    unsupported.write_bytes(b"not an image")
    assert main(["recognize", str(unsupported), "--checkpoint", "missing"]) == 2
    error = capsys.readouterr()
    assert error.out == ""
    assert error.err.startswith("error:")
    assert "Traceback" not in error.err and str(tmp_path) not in error.err

    assert (
        main(
            [
                "recognize",
                str(SAMPLE_LINE),
                "--checkpoint",
                "missing",
                "--device",
                "not-a-device",
            ]
        )
        == 2
    )
    assert "device" in capsys.readouterr().err


@pytest.mark.skipif(not HAS_ML, reason="ML extra is not installed")
def test_recognize_cli_runs_a_real_temporary_checkpoint(tmp_path: Path, capsys) -> None:
    from urdu_document_ocr import CRNNRecognizer, TrainingConfig, load_vocabulary
    from urdu_document_ocr.training import save_checkpoint

    vocabulary = load_vocabulary(SAMPLE_ROOT / "synthetic-fixture-vocabulary.json")
    checkpoint = tmp_path / "best"
    save_checkpoint(
        checkpoint,
        CRNNRecognizer(vocabulary),
        TrainingConfig(batch_size=1, epochs=1),
        epoch=1,
        validation_loss=1.0,
        train_dataset_fingerprint="a" * 64,
        validation_dataset_fingerprint="b" * 64,
    )
    assert (
        main(
            [
                "recognize",
                str(SAMPLE_LINE),
                "--checkpoint",
                str(checkpoint),
                "--format",
                "json",
                "--batch-size",
                "1",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["schema_version"] == 1
    assert len(payload["pages"]) == 1
    assert str(tmp_path) not in json.dumps(payload)


def test_segment_image_blank_and_json_file(tmp_path: Path, capsys) -> None:
    assert main(["segment", str(SAMPLE_LINE)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["page_count"] == 1
    assert payload["pages"][0]["regions"]
    assert [region["reading_order_index"] for region in payload["pages"][0]["regions"]] == list(
        range(len(payload["pages"][0]["regions"]))
    )
    assert "image" not in json.dumps(payload)

    blank = tmp_path / "blank.png"
    Image.new("RGB", (200, 100), color="white").save(blank)
    output = tmp_path / "segments.json"
    assert main(["segment", str(blank), "--output", str(output)]) == 0
    blank_payload = json.loads(output.read_text(encoding="utf-8"))
    assert blank_payload["pages"][0]["is_blank"] is True
    assert blank_payload["pages"][0]["regions"] == []


def test_validate_split_and_generation_commands(tmp_path: Path, capsys) -> None:
    root, train_manifest, validation_manifest, _vocabulary = _dataset(tmp_path)
    combined = tmp_path / "combined.jsonl"
    samples = (
        DatasetSample(1, "a", "train.png", "ا", "document-a"),
        DatasetSample(1, "b", "validation.png", "ا", "document-b"),
    )
    write_manifest(samples, combined)

    assert main(["validate-data", "--manifest", str(combined), "--dataset-root", str(root)]) == 0
    validation = json.loads(capsys.readouterr().out)
    assert validation["is_valid"] is True and validation["statistics"]["sample_count"] == 2

    missing_manifest = tmp_path / "missing.jsonl"
    write_manifest(
        (DatasetSample(1, "missing", "missing.png", "ا", "document-missing"),),
        missing_manifest,
    )
    assert (
        main(["validate-data", "--manifest", str(missing_manifest), "--dataset-root", str(root)])
        == 2
    )
    assert json.loads(capsys.readouterr().out)["error_count"] == 1

    malformed = tmp_path / "malformed.jsonl"
    malformed.write_text("{not-json}\n", encoding="utf-8")
    assert main(["validate-data", "--manifest", str(malformed), "--dataset-root", str(root)]) == 2
    malformed_error = capsys.readouterr()
    assert malformed_error.out == ""
    assert "malformed JSON" in malformed_error.err and "Traceback" not in malformed_error.err

    split_dir = tmp_path / "splits"
    split_dir.mkdir()
    split_args = [
        "split-data",
        "--manifest",
        str(combined),
        "--output-dir",
        str(split_dir),
        "--train-ratio",
        "0.5",
        "--validation-ratio",
        "0.5",
        "--test-ratio",
        "0",
    ]
    assert main(split_args) == 0
    split = json.loads(capsys.readouterr().out)
    assert split["achieved_sample_counts"]["train"] == 1
    assert (split_dir / "split_metadata.json").is_file()
    assert main(split_args) == 2
    assert "overwrite=True" in capsys.readouterr().err

    first = tmp_path / "synthetic-a"
    second = tmp_path / "synthetic-b"
    first.mkdir()
    second.mkdir()
    common = ["--count", "2", "--seed", "41"]
    assert main(["generate-synthetic", "--output-dir", str(first), *common]) == 0
    first_result = json.loads(capsys.readouterr().out)
    assert main(["generate-synthetic", "--output-dir", str(second), *common]) == 0
    second_result = json.loads(capsys.readouterr().out)
    assert first_result == second_result
    assert first_result["sample_count"] == 2
    assert (first / "manifest.jsonl").read_bytes() == (second / "manifest.jsonl").read_bytes()
    assert main(["generate-synthetic", "--output-dir", str(first), *common]) == 2
    assert "overwrite=True" in capsys.readouterr().err
    assert main(["generate-synthetic", "--output-dir", str(first), *common, "--overwrite"]) == 0
    assert json.loads(capsys.readouterr().out) == first_result

    assert train_manifest.is_file() and validation_manifest.is_file()


@pytest.mark.skipif(not HAS_ML, reason="ML extra is not installed")
def test_evaluate_reuses_metrics_and_error_analysis(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    root = tmp_path / "dataset"
    root.mkdir()
    image = root / "line.png"
    image.write_bytes(SAMPLE_LINE.read_bytes())
    manifest = tmp_path / "test.jsonl"
    write_manifest((DatasetSample(1, "line", "line.png", "اردو متن", "document"),), manifest)
    _patch_loaded_recognizer(monkeypatch, FakeRecognizer("اردو متن"))

    assert (
        main(
            [
                "evaluate",
                "--manifest",
                str(manifest),
                "--dataset-root",
                str(root),
                "--checkpoint",
                "unused",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    summary = payload["evaluation"]["summary"]
    assert summary["sample_count"] == 1
    assert summary["character"]["error_rate"] == 0.0
    assert summary["word"]["error_rate"] == 0.0
    assert payload["error_analysis"]["totals"] == {
        "deletions": 0,
        "insertions": 0,
        "substitutions": 0,
    }


@pytest.mark.skipif(not HAS_ML, reason="ML extra is not installed")
def test_train_cli_runs_one_optimization_and_writes_checkpoint(tmp_path: Path, capsys) -> None:
    root, train_manifest, validation_manifest, vocabulary = _dataset(tmp_path)
    output = tmp_path / "training-output"
    assert (
        main(
            [
                "train",
                "--train-manifest",
                str(train_manifest),
                "--validation-manifest",
                str(validation_manifest),
                "--dataset-root",
                str(root),
                "--vocabulary",
                str(vocabulary),
                "--output-dir",
                str(output),
                "--epochs",
                "1",
                "--batch-size",
                "1",
                "--patience",
                "1",
                "--max-image-width",
                "256",
            ]
        )
        == 0
    )
    summary = json.loads(capsys.readouterr().out)
    assert summary["best_epoch"] == 1
    assert summary["optimization_steps"] == 1
    assert set(path.name for path in (output / "best").iterdir()) == {
        "metadata.json",
        "model.safetensors",
        "vocabulary.json",
    }
    assert str(tmp_path) not in json.dumps(summary)


@pytest.mark.skipif(not HAS_API, reason="API extra is not installed")
def test_serve_uses_safe_defaults_without_opening_a_port(
    monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    import uvicorn

    import urdu_document_ocr.api as api

    sentinel = object()
    created: dict[str, object] = {}

    def fake_create_app(**kwargs):
        created.update(kwargs)
        return sentinel

    calls: list[tuple[object, str, int]] = []
    monkeypatch.setattr(api, "create_app", fake_create_app)
    monkeypatch.setattr(uvicorn, "run", lambda app, *, host, port: calls.append((app, host, port)))
    assert main(["serve", "--checkpoint", "checkpoint"]) == 0
    assert calls == [(sentinel, "127.0.0.1", 8000)]
    assert created == {"checkpoint_directory": "checkpoint", "device": "cpu", "batch_size": 16}
    assert capsys.readouterr().out == ""


def test_serve_missing_optional_dependency_has_install_guidance(
    monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    monkeypatch.setitem(sys.modules, "uvicorn", None)
    assert main(["serve", "--checkpoint", "checkpoint"]) == 2
    result = capsys.readouterr()
    assert result.out == ""
    assert "urdu-document-ocr[ml,api]" in result.err


def test_unexpected_cli_failure_is_exit_one_without_traceback(
    monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    import urdu_document_ocr.cli as cli

    monkeypatch.setattr(
        sys,
        "argv",
        ["urdu-ocr", "segment", str(SAMPLE_LINE)],
    )
    monkeypatch.setattr(
        cli, "_run_segment", lambda _arguments: (_ for _ in ()).throw(RuntimeError())
    )
    assert main() == 1
    error = capsys.readouterr().err
    assert error == "error: operation failed unexpectedly\n"
    assert "Traceback" not in error
