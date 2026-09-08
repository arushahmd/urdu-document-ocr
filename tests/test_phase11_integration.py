from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from urdu_document_ocr import (
    CRNNRecognizer,
    OCRPrediction,
    TrainingConfig,
    document_to_json,
    document_to_text,
    load_recognizer,
    load_vocabulary,
    recognize_document,
    recognize_line,
)
from urdu_document_ocr.training import save_checkpoint

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_ROOT = REPOSITORY_ROOT / "data" / "sample"
DATASET_FINGERPRINT = "73be137ce8e62843354fa4699c54f61f62ff1364d554562c4a5018bd58c4fc6d"


class FixtureTextRecognizer:
    fingerprint = "d" * 64

    def __init__(self, page_texts: tuple[tuple[str, ...], ...]) -> None:
        self.page_texts = list(page_texts)
        self.call_count = 0

    def recognize_batch(self, images, *, batch_size):
        texts = self.page_texts[self.call_count]
        self.call_count += 1
        assert len(images) == len(texts)
        return tuple(OCRPrediction(text) for text in texts)


def _page_record(page_id: str) -> dict[str, object]:
    manifest = json.loads((SAMPLE_ROOT / "page-ground-truth.json").read_text(encoding="utf-8"))
    return next(page for page in manifest["pages"] if page["page_id"] == page_id)


def test_real_safetensors_checkpoint_runs_real_phase8_line(tmp_path: Path) -> None:
    vocabulary = load_vocabulary(SAMPLE_ROOT / "synthetic-fixture-vocabulary.json")
    model = CRNNRecognizer(vocabulary)
    save_checkpoint(
        tmp_path / "best",
        model,
        TrainingConfig(batch_size=1, epochs=1),
        epoch=1,
        validation_loss=1.0,
        train_dataset_fingerprint=DATASET_FINGERPRINT,
        validation_dataset_fingerprint="b" * 64,
    )
    recognizer = load_recognizer(tmp_path / "best", device="cpu")
    with Image.open(SAMPLE_ROOT / "lines" / "syn-line-000001.png") as source:
        line = source.convert("L")
        try:
            pixels = np.asarray(line, dtype=np.uint8).copy()
        finally:
            line.close()

    prediction = recognize_line(pixels, recognizer)

    assert isinstance(prediction, OCRPrediction)
    assert prediction.sequence_length is not None and prediction.sequence_length > 0
    assert all(1 <= index <= len(vocabulary.characters) for index in prediction.character_indices)
    assert not recognizer.model.training
    assert recognizer.checkpoint_metadata.model_weights_sha256


def test_phase8_multipage_pdf_runs_full_pipeline_with_exact_controlled_text(
    tmp_path: Path,
) -> None:
    page_ids = (
        "syn-page-003-uneven_two_column",
        "syn-page-008-blank",
        "syn-page-006-spanning_footer",
    )
    records = tuple(_page_record(page_id) for page_id in page_ids)
    images = [
        Image.open(SAMPLE_ROOT / str(record["image_path"])).convert("RGB") for record in records
    ]
    pdf_path = tmp_path / "phase8-pages.pdf"
    try:
        images[0].save(
            pdf_path,
            format="PDF",
            save_all=True,
            append_images=images[1:],
            resolution=72.0,
            quality=100,
        )
    finally:
        for image in images:
            image.close()

    recognized_page_texts = tuple(
        tuple(str(line["text"]) for line in record["lines"])  # type: ignore[index]
        for record in (records[0], records[2])
    )
    recognizer = FixtureTextRecognizer(recognized_page_texts)

    result = recognize_document(pdf_path, recognizer, pdf_dpi=72)
    serialized = document_to_json(result)

    first_text = "\n".join(recognized_page_texts[0])
    third_text = "\n".join(recognized_page_texts[1])
    assert len(result.pages) == 3
    assert [page.page_index for page in result.pages] == [0, 1, 2]
    assert result.pages[1].is_blank and result.pages[1].lines == ()
    assert document_to_text(result) == f"{first_text}\n\n\n\n{third_text}"
    assert recognizer.call_count == 2
    assert [line.region.column_index for line in result.pages[0].lines] == [0, 0, 0, 0, 1, 1, 1]
    assert result.pages[2].lines[-1].region.region_kind.value == "spanning"
    assert json.loads(serialized)["assembled_text"] == result.assembled_text
    assert recognized_page_texts[0][0] in serialized and recognized_page_texts[1][-1] in serialized
    assert "\\u" not in serialized
    assert str(tmp_path) not in serialized
