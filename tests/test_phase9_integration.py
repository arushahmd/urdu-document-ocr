from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from urdu_document_ocr import (
    CRNNRecognizer,
    compute_ctc_loss,
    load_vocabulary,
    prepare_line_image,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_ROOT = REPOSITORY_ROOT / "data" / "sample"


def test_realistic_synthetic_fixture_line_reaches_untrained_crnn_logits() -> None:
    """This verifies mechanics only; random logits make no OCR-quality claim."""

    vocabulary = load_vocabulary(SAMPLE_ROOT / "synthetic-fixture-vocabulary.json")
    with Image.open(SAMPLE_ROOT / "lines" / "syn-line-000015.png") as source:
        line = np.asarray(source.convert("L"), dtype=np.uint8)
    tensor, valid_width = prepare_line_image(line)
    model = CRNNRecognizer(vocabulary).eval()

    with torch.no_grad():
        output = model(tensor.unsqueeze(0), torch.tensor([valid_width]))

    assert tensor.shape == (1, 64, valid_width)
    assert output.logits.shape == (1, valid_width // 4, len(vocabulary.characters) + 1)
    assert output.input_lengths.tolist() == [valid_width // 4]
    assert torch.isfinite(output.logits).all()
    assert (
        vocabulary.fingerprint == "f69608eaaeea48a2ee25ff0427f779a2382d727250181358f4ee50886e92ec98"
    )


def test_model_ctc_backward_path_has_finite_representative_gradients() -> None:
    torch.manual_seed(29)
    vocabulary = load_vocabulary(SAMPLE_ROOT / "synthetic-fixture-vocabulary.json")
    model = CRNNRecognizer(vocabulary)
    images = torch.rand(1, 1, 64, 20).mul(2).sub(1)
    output = model(images, torch.tensor([20]))
    targets = torch.tensor([1, 2], dtype=torch.int64)

    loss = compute_ctc_loss(
        output.logits,
        targets,
        torch.tensor([2], dtype=torch.int64),
        output.input_lengths,
    )
    loss.backward()

    assert torch.isfinite(loss)
    for parameter in (
        model.cnn[0][0].weight,
        model.recurrent.weight_ih_l0,
        model.classifier.weight,
    ):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
