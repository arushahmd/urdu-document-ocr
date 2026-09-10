from __future__ import annotations

import asyncio
import importlib.util
import io
import json
from pathlib import Path

import httpx
import pytest
from PIL import Image

from urdu_document_ocr import (
    InputLimitsConfig,
    OCRPrediction,
    Vocabulary,
    load_vocabulary,
)
from urdu_document_ocr.api import create_app

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_ROOT = REPOSITORY_ROOT / "data" / "sample"
SAMPLE_LINE = SAMPLE_ROOT / "lines" / "syn-line-000001.png"
DATASET_FINGERPRINT = "73be137ce8e62843354fa4699c54f61f62ff1364d554562c4a5018bd58c4fc6d"
HAS_ML = importlib.util.find_spec("torch") is not None


class FakeRecognizer:
    fingerprint = "d" * 64

    def __init__(self, page_texts: tuple[tuple[str, ...], ...] | None = None) -> None:
        self.vocabulary = Vocabulary(tuple(" ابپتجدرسمںو"))
        self.page_texts = list(page_texts or ())
        self.call_count = 0

    def recognize_batch(self, images, *, batch_size):
        if self.page_texts:
            texts = self.page_texts[self.call_count]
            assert len(images) == len(texts)
        else:
            texts = tuple("سلام" for _ in images)
        self.call_count += 1
        return tuple(OCRPrediction(text) for text in texts)


def _request(app, method: str, path: str, **kwargs) -> httpx.Response:
    async def send() -> httpx.Response:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.request(method, path, **kwargs)

    return asyncio.run(send())


def _pdf_bytes(paths: tuple[Path, ...]) -> bytes:
    output = io.BytesIO()
    images = [Image.open(path).convert("RGB") for path in paths]
    try:
        images[0].save(
            output,
            format="PDF",
            save_all=True,
            append_images=images[1:],
            resolution=72.0,
        )
    finally:
        for image in images:
            image.close()
    return output.getvalue()


def _post(app, payload: bytes, *, mime: str = "image/png", query: str = "") -> httpx.Response:
    return _request(
        app,
        "POST",
        f"/ocr{query}",
        files={"file": ("untrusted-name.bin", payload, mime)},
    )


def _assert_forbidden_fields_absent(payload: object) -> None:
    forbidden = {
        "image",
        "images",
        "mask",
        "masks",
        "logits",
        "weights",
        "checkpoint_path",
        "device",
        "confidence",
    }
    if isinstance(payload, dict):
        assert forbidden.isdisjoint(payload)
        for value in payload.values():
            _assert_forbidden_fields_absent(value)
    elif isinstance(payload, list):
        for value in payload:
            _assert_forbidden_fields_absent(value)


def test_health_is_structural_and_does_not_run_ocr() -> None:
    recognizer = FakeRecognizer()
    response = _request(create_app(recognizer=recognizer), "GET", "/health")
    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "model_fingerprint": "d" * 64,
        "vocabulary_fingerprint": recognizer.vocabulary.fingerprint,
    }
    assert recognizer.call_count == 0


@pytest.mark.parametrize("image_format,mime", [("PNG", "application/pdf"), ("JPEG", "text/plain")])
def test_ocr_accepts_actual_png_and_jpeg_while_distrusting_mime(
    image_format: str, mime: str
) -> None:
    output = io.BytesIO()
    with Image.open(SAMPLE_LINE) as image:
        image.convert("RGB").save(output, format=image_format)
    response = _post(create_app(recognizer=FakeRecognizer()), output.getvalue(), mime=mime)
    assert response.status_code == 200
    payload = response.json()
    assert payload["schema_version"] == 1
    assert payload["assembled_text"] == "سلام"
    _assert_forbidden_fields_absent(payload)


def test_pdf_exact_urdu_rtl_spanning_and_blank_page_order() -> None:
    page_ids = (
        "syn-page-003-uneven_two_column",
        "syn-page-008-blank",
        "syn-page-006-spanning_footer",
    )
    ground_truth = json.loads((SAMPLE_ROOT / "page-ground-truth.json").read_text(encoding="utf-8"))[
        "pages"
    ]
    records = tuple(
        next(item for item in ground_truth if item["page_id"] == page_id) for page_id in page_ids
    )
    paths = tuple(SAMPLE_ROOT / record["image_path"] for record in records)
    page_texts = tuple(
        tuple(line["text"] for line in record["lines"]) for record in (records[0], records[2])
    )
    recognizer = FakeRecognizer(page_texts)

    response = _post(
        create_app(recognizer=recognizer),
        _pdf_bytes(paths),
        mime="image/jpeg",
        query="?pdf_dpi=72",
    )
    assert response.status_code == 200
    payload = response.json()
    assert [page["page_index"] for page in payload["pages"]] == [0, 1, 2]
    assert payload["pages"][1]["is_blank"] is True
    assert payload["pages"][1]["lines"] == []
    assert [line["region"]["column_index"] for line in payload["pages"][0]["lines"]] == [
        0,
        0,
        0,
        0,
        1,
        1,
        1,
    ]
    assert payload["pages"][2]["lines"][-1]["region"]["region_kind"] == "spanning"
    assert page_texts[0][0] in response.text and "\\u" not in response.text
    assert recognizer.call_count == 2


def test_blank_image_returns_blank_document_without_model_call() -> None:
    image = io.BytesIO()
    Image.new("RGB", (120, 80), color="white").save(image, format="PNG")
    recognizer = FakeRecognizer()
    response = _post(create_app(recognizer=recognizer), image.getvalue())
    assert response.status_code == 200
    assert response.json()["pages"][0]["is_blank"] is True
    assert recognizer.call_count == 0


def test_http_error_policy_and_safe_schema(tmp_path: Path) -> None:
    app = create_app(recognizer=FakeRecognizer())
    unsupported = _post(app, b"GIF89a", mime="image/png")
    assert unsupported.status_code == 415
    assert unsupported.json()["error"]["code"] == "unsupported_document_format"

    corrupt = _post(app, b"\x89PNG\r\n\x1a\ncorrupt")
    assert corrupt.status_code == 422
    assert corrupt.json()["error"]["code"] == "document_load_error"

    invalid_dpi = _post(app, _pdf_bytes((SAMPLE_LINE,)), query="?pdf_dpi=500")
    assert invalid_dpi.status_code == 413
    assert invalid_dpi.json()["error"]["code"] == "resource_limit_exceeded"

    invalid_query = _post(app, SAMPLE_LINE.read_bytes(), query="?pdf_dpi=not-an-int")
    assert invalid_query.status_code == 422
    assert invalid_query.json() == {
        "error": {"code": "invalid_request", "message": "request parameters are invalid"}
    }

    text = " ".join(response.text for response in (unsupported, corrupt, invalid_dpi))
    assert str(tmp_path) not in text
    assert "Traceback" not in text and "tensor(" not in text


def test_multipart_is_required_and_bounded_at_exact_file_limit() -> None:
    limits = InputLimitsConfig(max_input_bytes=64)
    app = create_app(recognizer=FakeRecognizer(), input_limits=limits)
    malformed = _request(app, "POST", "/ocr", content=b"raw", headers={"content-type": "image/png"})
    assert malformed.status_code == 422
    assert malformed.json()["error"]["code"] == "invalid_multipart"

    oversized = _post(app, b"x" * 65)
    assert oversized.status_code == 413
    assert oversized.json() == {
        "error": {
            "code": "resource_limit_exceeded",
            "message": "uploaded document exceeds the configured byte limit",
        }
    }


def test_checkpoint_loader_called_once_and_recognizer_reused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import urdu_document_ocr.api as api

    recognizer = FakeRecognizer()
    loads: list[tuple[str, str]] = []

    def fake_loader(path: str, *, device: str):
        loads.append((path, device))
        return recognizer

    monkeypatch.setattr(api, "_load_service_recognizer", fake_loader)
    app = create_app(checkpoint_directory="fixed-checkpoint", device="cpu")
    assert _post(app, SAMPLE_LINE.read_bytes()).status_code == 200
    assert _post(app, SAMPLE_LINE.read_bytes()).status_code == 200
    assert loads == [("fixed-checkpoint", "cpu")]
    assert recognizer.call_count == 2


@pytest.mark.skipif(not HAS_ML, reason="ML extra is not installed")
def test_real_temporary_checkpoint_runs_through_api(tmp_path: Path) -> None:
    from urdu_document_ocr import CRNNRecognizer, TrainingConfig
    from urdu_document_ocr.training import save_checkpoint

    vocabulary = load_vocabulary(SAMPLE_ROOT / "synthetic-fixture-vocabulary.json")
    checkpoint = tmp_path / "best"
    save_checkpoint(
        checkpoint,
        CRNNRecognizer(vocabulary),
        TrainingConfig(batch_size=1, epochs=1),
        epoch=1,
        validation_loss=1.0,
        train_dataset_fingerprint=DATASET_FINGERPRINT,
        validation_dataset_fingerprint="b" * 64,
    )
    response = _post(
        create_app(checkpoint_directory=str(checkpoint), batch_size=1),
        SAMPLE_LINE.read_bytes(),
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["schema_version"] == 1 and len(payload["pages"]) == 1
    _assert_forbidden_fields_absent(payload)
    assert str(tmp_path) not in response.text


def test_factory_validation_and_internal_failure_mapping() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        create_app()
    with pytest.raises(ValueError, match="exactly one"):
        create_app(checkpoint_directory="x", recognizer=FakeRecognizer())
    with pytest.raises(TypeError, match="service contract"):
        create_app(recognizer=object())

    recognizer = FakeRecognizer()
    recognizer.recognize_batch = lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError())
    response = _post(create_app(recognizer=recognizer), SAMPLE_LINE.read_bytes())
    assert response.status_code == 500
    assert response.json() == {
        "error": {"code": "ocr_processing_error", "message": "OCR processing failed"}
    }
