from __future__ import annotations

from itertools import pairwise

import numpy as np
import pytest
import torch
from torch import nn

from urdu_document_ocr import (
    CRNNRecognizer,
    ModelInputError,
    RecognizerConfig,
    Vocabulary,
    input_width_to_timesteps,
    prepare_line_image,
)
from urdu_document_ocr.recognition.base import RecognizerOutput
from urdu_document_ocr.recognition.model import (
    FEATURE_CHANNELS,
    HEIGHT_STRIDE,
    LSTM_DROPOUT,
    LSTM_HIDDEN_SIZE,
    LSTM_LAYERS,
    MAXIMUM_INPUT_WIDTH,
    MINIMUM_INPUT_WIDTH,
    WIDTH_STRIDE,
)


@pytest.fixture(scope="module")
def vocabulary() -> Vocabulary:
    return Vocabulary(("ا", "ب", "ت"))


@pytest.fixture(scope="module")
def model(vocabulary: Vocabulary) -> CRNNRecognizer:
    torch.manual_seed(17)
    return CRNNRecognizer(vocabulary).eval()


@pytest.fixture(scope="module")
def meta_model(vocabulary: Vocabulary) -> CRNNRecognizer:
    return CRNNRecognizer(vocabulary).to(device="meta")


def test_frozen_model_structure(model: CRNNRecognizer) -> None:
    convolutions = [module for module in model.cnn.modules() if isinstance(module, nn.Conv2d)]
    normalizers = [module for module in model.cnn.modules() if isinstance(module, nn.GroupNorm)]
    activations = [module for module in model.cnn.modules() if isinstance(module, nn.SiLU)]
    pools = [module for module in model.cnn.modules() if isinstance(module, nn.MaxPool2d)]

    assert len(model.cnn) == 4
    assert [layer.out_channels for layer in convolutions] == [64, 64, 128, 128, 256, 256, 384, 384]
    assert len(normalizers) == len(activations) == 8
    assert [pool.kernel_size for pool in pools] == [(2, 2), (2, 2), (2, 1), (2, 1)]
    assert (HEIGHT_STRIDE, WIDTH_STRIDE, FEATURE_CHANNELS) == (16, 4, 384)
    assert model.recurrent.input_size == 384
    assert model.recurrent.hidden_size == LSTM_HIDDEN_SIZE == 256
    assert model.recurrent.num_layers == LSTM_LAYERS == 2
    assert model.recurrent.bidirectional
    assert model.recurrent.dropout == LSTM_DROPOUT == 0.2
    assert model.classifier.in_features == 512
    assert model.classifier.out_features == len(model.vocabulary.characters) + 1


@pytest.mark.parametrize(
    "width",
    [4, 5, 6, 7, 8, 9, 15, 16, 17, 63, 64, 65, 511, 512, 513, 2047, 2048],
)
def test_shared_width_formula_matches_actual_cnn_geometry_on_meta(
    meta_model: CRNNRecognizer, width: int
) -> None:
    feature_map = meta_model.cnn(torch.empty(1, 1, 64, width, device="meta"))

    assert feature_map.shape == (1, 384, 4, input_width_to_timesteps(width))


def test_width_formula_is_exact_monotonic_and_bounded() -> None:
    lengths = [input_width_to_timesteps(width) for width in range(4, 2049)]

    assert lengths[0] == 1
    assert lengths[-1] == 512
    assert all(left <= right for left, right in pairwise(lengths))
    assert all(
        length <= width // WIDTH_STRIDE
        for width, length in zip(range(4, 2049), lengths, strict=True)
    )


@pytest.mark.parametrize("width", [True, 3, 2049, 4.0])
def test_width_formula_rejects_unsupported_values(width: object) -> None:
    with pytest.raises(ModelInputError):
        input_width_to_timesteps(width)  # type: ignore[arg-type]


def test_forward_handles_unsorted_variable_widths_and_raw_logits(
    model: CRNNRecognizer,
) -> None:
    images = torch.linspace(-1.0, 1.0, steps=3 * 64 * 36).reshape(3, 1, 64, 36)
    widths = torch.tensor([20, 36, 28], dtype=torch.int64)

    with torch.no_grad():
        output = model(images, widths)

    assert output.logits.shape == (3, 9, 4)
    assert output.input_lengths.tolist() == [5, 9, 7]
    assert torch.isfinite(output.logits).all()
    assert output.logits.softmax(dim=2).shape == output.logits.shape


def test_valid_logits_are_invariant_to_wider_right_padding(model: CRNNRecognizer) -> None:
    generator = torch.Generator().manual_seed(91)
    line = torch.rand((1, 1, 64, 20), generator=generator).mul(2).sub(1)
    alone = model(line, torch.tensor([20], dtype=torch.int64))
    padded_line = torch.nn.functional.pad(line, (0, 16), value=1.0)
    wider_line = torch.rand((1, 1, 64, 36), generator=generator).mul(2).sub(1)
    batch = torch.cat((padded_line, wider_line), dim=0)
    batched = model(batch, torch.tensor([20, 36], dtype=torch.int64))

    torch.testing.assert_close(alone.logits[0, :5], batched.logits[0, :5], atol=1e-6, rtol=1e-5)


def test_eval_forward_is_stable_in_one_environment(model: CRNNRecognizer) -> None:
    image = torch.zeros((1, 1, 64, 20), dtype=torch.float32)
    width = torch.tensor([20], dtype=torch.int64)

    first = model(image, width).logits
    second = model(image, width).logits

    torch.testing.assert_close(first, second, atol=0.0, rtol=0.0)


@pytest.mark.parametrize(
    ("images", "widths"),
    [
        (torch.zeros(1, 64, 20), torch.tensor([20])),
        (torch.zeros(1, 2, 64, 20), torch.tensor([20])),
        (torch.zeros(1, 1, 32, 20), torch.tensor([20])),
        (torch.zeros(1, 1, 64, 20, dtype=torch.float64), torch.tensor([20])),
        (torch.full((1, 1, 64, 20), 1.1), torch.tensor([20])),
        (torch.full((1, 1, 64, 20), float("nan")), torch.tensor([20])),
        (torch.zeros(1, 1, 64, 20), torch.tensor([[20]])),
        (torch.zeros(1, 1, 64, 20), torch.tensor([20], dtype=torch.int32)),
        (torch.zeros(1, 1, 64, 20), torch.tensor([3])),
        (torch.zeros(1, 1, 64, 20), torch.tensor([21])),
        (torch.zeros(2, 1, 64, 20), torch.tensor([20])),
    ],
)
def test_forward_rejects_invalid_tensor_contract(
    images: torch.Tensor, widths: torch.Tensor
) -> None:
    model = CRNNRecognizer(Vocabulary(("ا",)))

    with pytest.raises(ModelInputError):
        model(images, widths)


def test_prepare_line_image_preserves_aspect_ratio_and_normalizes_extremes() -> None:
    image = np.vstack(
        (
            np.zeros((5, 21), dtype=np.uint8),
            np.full((5, 21), 255, dtype=np.uint8),
        )
    )

    tensor, width = prepare_line_image(image)

    assert width == 134  # 21 * 64 / 10 = 134.4, rounded half up.
    assert tensor.shape == (1, 64, 134)
    assert tensor.dtype == torch.float32
    assert tensor.min().item() == pytest.approx(-1.0)
    assert tensor.max().item() == pytest.approx(1.0)


@pytest.mark.parametrize(
    "image",
    [
        np.zeros((4, 4, 1), dtype=np.uint8),
        np.zeros((4, 4), dtype=np.float32),
        np.zeros((0, 4), dtype=np.uint8),
        np.zeros((100, 3), dtype=np.uint8),
        np.zeros((1, 33), dtype=np.uint8),
    ],
)
def test_prepare_line_image_rejects_invalid_or_out_of_bounds_input(image: np.ndarray) -> None:
    with pytest.raises(ModelInputError):
        prepare_line_image(image)


def test_prepare_line_image_honors_lower_configured_maximum() -> None:
    with pytest.raises(ModelInputError, match="exceeds"):
        prepare_line_image(np.zeros((64, 65), dtype=np.uint8), RecognizerConfig(max_width=64))


def test_model_fingerprint_binds_vocabulary_and_config() -> None:
    first = CRNNRecognizer(Vocabulary(("ا",)), RecognizerConfig(max_width=1024))
    second = CRNNRecognizer(Vocabulary(("ا", "ب")), RecognizerConfig(max_width=1024))

    assert first.fingerprint != second.fingerprint
    assert len(first.fingerprint) == 64
    assert (MINIMUM_INPUT_WIDTH, MAXIMUM_INPUT_WIDTH) == (4, 2048)


@pytest.mark.parametrize(
    ("logits", "lengths"),
    [
        (torch.zeros(2, 3), torch.tensor([2])),
        (torch.zeros(1, 3, 2, dtype=torch.int64), torch.tensor([2])),
        (torch.zeros(1, 3, 2), torch.tensor([[2]])),
        (torch.zeros(1, 3, 2), torch.tensor([2], dtype=torch.int32)),
        (torch.zeros(2, 3, 2), torch.tensor([2])),
        (torch.zeros(1, 3, 2), torch.tensor([0])),
        (torch.zeros(1, 3, 2), torch.tensor([4])),
    ],
)
def test_recognizer_output_rejects_inconsistent_shapes_or_lengths(
    logits: torch.Tensor, lengths: torch.Tensor
) -> None:
    with pytest.raises(ModelInputError):
        RecognizerOutput(logits, lengths)
