"""Current-original PyTorch CNN-BiLSTM-CTC recognizer implementation."""

from __future__ import annotations

import json
from collections.abc import Sequence
from hashlib import sha256

import cv2
import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor, nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence

from urdu_document_ocr.config import RecognizerConfig
from urdu_document_ocr.errors import ModelInputError
from urdu_document_ocr.recognition.base import RecognizerOutput
from urdu_document_ocr.types import Vocabulary

NORMALIZED_HEIGHT = 64
MAXIMUM_INPUT_WIDTH = 2_048
MINIMUM_INPUT_WIDTH = 4
HEIGHT_STRIDE = 16
WIDTH_STRIDE = 4
FEATURE_CHANNELS = 384
LSTM_HIDDEN_SIZE = 256
LSTM_LAYERS = 2
LSTM_DROPOUT = 0.2

MODEL_ARCHITECTURE_ID = "crnn-gn-silu-bilstm-ctc-v1"
_CHANNELS = (64, 128, 256, 384)
_GROUPS = (8, 16, 32, 32)
_POOL_SIZES = ((2, 2), (2, 2), (2, 1), (2, 1))


def input_width_to_timesteps(width: int) -> int:
    """Map valid input pixels to CNN timesteps using both floor-mode width pools."""

    if isinstance(width, bool) or not isinstance(width, int):
        raise ModelInputError("input width must be an integer")
    if not MINIMUM_INPUT_WIDTH <= width <= MAXIMUM_INPUT_WIDTH:
        raise ModelInputError(
            f"input width must be between {MINIMUM_INPUT_WIDTH} and {MAXIMUM_INPUT_WIDTH}"
        )
    return (width // 2) // 2


def _rounded_resized_width(width: int, height: int, target_height: int) -> int:
    """Use integer half-up rounding, avoiding platform-dependent banker rounding."""

    numerator = width * target_height
    return (2 * numerator + height) // (2 * height)


def prepare_line_image(
    image: NDArray[np.uint8],
    config: RecognizerConfig | None = None,
) -> tuple[Tensor, int]:
    """Resize one grayscale uint8 line to height 64 and map pixels to ``[-1, 1]``.

    Aspect-preserving width uses deterministic half-up rounding. Over-width and
    technically too-narrow results fail instead of being cropped or distorted.
    """

    active_config = RecognizerConfig() if config is None else config
    if not isinstance(active_config, RecognizerConfig):
        raise TypeError("config must be a RecognizerConfig")
    if not isinstance(image, np.ndarray) or image.ndim != 2:
        raise ModelInputError("line image must be a two-dimensional NumPy array")
    if image.dtype != np.uint8:
        raise ModelInputError("line image must have dtype uint8")
    height, width = image.shape
    if height < 1 or width < 1:
        raise ModelInputError("line image dimensions must be positive")

    resized_width = _rounded_resized_width(width, height, active_config.normalized_height)
    if resized_width < MINIMUM_INPUT_WIDTH:
        raise ModelInputError(
            f"normalized line width must be at least {MINIMUM_INPUT_WIDTH} pixels"
        )
    if resized_width > active_config.max_width:
        raise ModelInputError(
            f"normalized line width exceeds the configured {active_config.max_width}-pixel limit"
        )

    interpolation = cv2.INTER_AREA if height > active_config.normalized_height else cv2.INTER_CUBIC
    resized = cv2.resize(
        image,
        (resized_width, active_config.normalized_height),
        interpolation=interpolation,
    )
    tensor = torch.from_numpy(np.ascontiguousarray(resized)).to(torch.float32)
    tensor = tensor.div(127.5).sub(1.0).unsqueeze(0)
    return tensor, resized_width


def pad_prepared_line_images(
    images: Sequence[Tensor],
    *,
    max_width: int,
) -> tuple[Tensor, Tensor]:
    """Create the shared white-right-padded batch used by training and inference."""

    values = tuple(images)
    if not values:
        raise ModelInputError("cannot pad an empty line-image batch")
    if isinstance(max_width, bool) or not isinstance(max_width, int) or max_width < 4:
        raise ModelInputError("max_width must be an integer of at least 4")

    widths: list[int] = []
    for image in values:
        if (
            not isinstance(image, Tensor)
            or image.dtype != torch.float32
            or image.device.type != "cpu"
            or image.ndim != 3
            or tuple(image.shape[:2]) != (1, NORMALIZED_HEIGHT)
        ):
            raise ModelInputError(
                "prepared line images must be CPU float32 tensors with shape [1,64,width]"
            )
        width = int(image.shape[2])
        if not MINIMUM_INPUT_WIDTH <= width <= max_width:
            raise ModelInputError("prepared line width is outside the recognizer-supported bounds")
        if not torch.isfinite(image).all() or image.min().item() < -1.0 or image.max().item() > 1.0:
            raise ModelInputError("prepared line images must contain finite values within [-1,1]")
        widths.append(width)

    padded_width = ((max(widths) + WIDTH_STRIDE - 1) // WIDTH_STRIDE) * WIDTH_STRIDE
    if padded_width > max_width:
        raise ModelInputError("stride-rounded batch width exceeds the recognizer maximum")
    batch = torch.full(
        (len(values), 1, NORMALIZED_HEIGHT, padded_width),
        1.0,
        dtype=torch.float32,
    )
    for index, image in enumerate(values):
        batch[index, :, :, : widths[index]] = image
    return batch, torch.tensor(widths, dtype=torch.int64)


class _ConvolutionBlock(nn.Sequential):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        groups: int,
        pool_size: tuple[int, int],
    ) -> None:
        super().__init__(
            nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, output_channels),
            nn.SiLU(),
            nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, output_channels),
            nn.SiLU(),
            nn.MaxPool2d(kernel_size=pool_size, stride=pool_size),
        )


class CRNNRecognizer(nn.Module):
    """Untrained variable-width CNN-BiLSTM network that emits raw CTC logits."""

    def __init__(
        self,
        vocabulary: Vocabulary,
        config: RecognizerConfig | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(vocabulary, Vocabulary):
            raise TypeError("vocabulary must be a Vocabulary")
        self.vocabulary = vocabulary
        self.config = RecognizerConfig() if config is None else config
        if not isinstance(self.config, RecognizerConfig):
            raise TypeError("config must be a RecognizerConfig")

        blocks: list[nn.Module] = []
        input_channels = 1
        for output_channels, groups, pool_size in zip(_CHANNELS, _GROUPS, _POOL_SIZES, strict=True):
            blocks.append(_ConvolutionBlock(input_channels, output_channels, groups, pool_size))
            input_channels = output_channels
        self.cnn = nn.Sequential(*blocks)
        self.recurrent = nn.LSTM(
            input_size=FEATURE_CHANNELS,
            hidden_size=LSTM_HIDDEN_SIZE,
            num_layers=LSTM_LAYERS,
            batch_first=True,
            dropout=LSTM_DROPOUT,
            bidirectional=True,
        )
        self.classifier = nn.Linear(2 * LSTM_HIDDEN_SIZE, len(vocabulary.characters) + 1)

    @property
    def fingerprint(self) -> str:
        payload = {
            "architecture": MODEL_ARCHITECTURE_ID,
            "blank_index": self.config.blank_index,
            "config_fingerprint": self.config.fingerprint,
            "vocabulary_fingerprint": self.vocabulary.fingerprint,
        }
        encoded = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
        return sha256(encoded).hexdigest()

    def _validate_inputs(self, images: Tensor, valid_widths: Tensor) -> tuple[int, ...]:
        if not isinstance(images, Tensor) or images.ndim != 4:
            raise ModelInputError("images must be a rank-4 PyTorch tensor")
        if images.dtype != torch.float32:
            raise ModelInputError("images must have dtype float32")
        if images.shape[1] != 1 or images.shape[2] != self.config.normalized_height:
            raise ModelInputError("images must have shape [batch, 1, 64, padded_width]")
        if images.shape[0] < 1:
            raise ModelInputError("images batch dimension must be positive")
        if not MINIMUM_INPUT_WIDTH <= images.shape[3] <= self.config.max_width:
            raise ModelInputError("padded image width is outside the recognizer-supported bounds")
        if not torch.isfinite(images).all():
            raise ModelInputError("images must contain only finite values")
        if torch.amin(images).item() < -1.0 or torch.amax(images).item() > 1.0:
            raise ModelInputError("images values must be within [-1, 1]")
        if not isinstance(valid_widths, Tensor) or valid_widths.ndim != 1:
            raise ModelInputError("valid_widths must be a rank-1 PyTorch tensor")
        if valid_widths.dtype != torch.int64 or valid_widths.device.type != "cpu":
            raise ModelInputError("valid_widths must be a CPU int64 tensor")
        if valid_widths.shape[0] != images.shape[0]:
            raise ModelInputError("valid_widths must match the image batch size")

        widths = tuple(int(width) for width in valid_widths.tolist())
        for width in widths:
            if width > images.shape[3]:
                raise ModelInputError("valid width cannot exceed the padded tensor width")
            if width > self.config.max_width or width < MINIMUM_INPUT_WIDTH:
                raise ModelInputError("valid width is outside the recognizer-supported bounds")
            input_width_to_timesteps(width)
        return widths

    def forward(self, images: Tensor, valid_widths: Tensor) -> RecognizerOutput:
        """Return raw logits ``[B,T,C]`` and CPU int64 valid lengths ``[B]``.

        Each sample is cropped to its explicit valid width before convolution. This
        prevents right-side white batch padding from changing boundary features.
        The resulting unsorted sequences are packed for the recurrent stack.
        """

        widths = self._validate_inputs(images, valid_widths)
        sequences: list[Tensor] = []
        output_lengths = torch.tensor(
            [input_width_to_timesteps(width) for width in widths], dtype=torch.int64
        )
        for batch_index, width in enumerate(widths):
            features = self.cnn(images[batch_index : batch_index + 1, :, :, :width])
            if features.shape[2] != NORMALIZED_HEIGHT // HEIGHT_STRIDE:
                raise RuntimeError("CNN height geometry no longer matches the frozen contract")
            sequence = features.mean(dim=2).transpose(1, 2).squeeze(0)
            if sequence.shape[0] != input_width_to_timesteps(width):
                raise RuntimeError("CNN width geometry no longer matches the shared formula")
            sequences.append(sequence)

        padded_sequence = pad_sequence(sequences, batch_first=True)
        packed = pack_padded_sequence(
            padded_sequence,
            output_lengths,
            batch_first=True,
            enforce_sorted=False,
        )
        packed_output, _ = self.recurrent(packed)
        recurrent_output, _ = pad_packed_sequence(
            packed_output,
            batch_first=True,
            total_length=input_width_to_timesteps(images.shape[3]),
        )
        logits = self.classifier(recurrent_output)
        return RecognizerOutput(logits=logits, input_lengths=output_lengths)
