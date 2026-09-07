from __future__ import annotations

import pytest
import torch

from urdu_document_ocr import DecodingError, Vocabulary, greedy_ctc_decode


def _logits_for_paths(paths: list[list[int]], class_count: int) -> torch.Tensor:
    logits = torch.full((len(paths), len(paths[0]), class_count), -10.0)
    for batch_index, path in enumerate(paths):
        for time_index, class_index in enumerate(path):
            logits[batch_index, time_index, class_index] = 10.0
    return logits


@pytest.mark.parametrize(
    ("path", "expected_text", "expected_indices"),
    [
        ([0, 0, 1, 0, 0], "ا", (1,)),
        ([1, 1, 0, 0, 0], "ا", (1,)),
        ([1, 0, 1, 0, 0], "اا", (1, 1)),
        ([1, 1, 0, 2, 2], "اب", (1, 2)),
        ([0, 0, 0, 0, 0], "", ()),
    ],
)
def test_greedy_ctc_decode_collapses_then_removes_blank(
    path: list[int], expected_text: str, expected_indices: tuple[int, ...]
) -> None:
    vocabulary = Vocabulary(("ا", "ب"))
    prediction = greedy_ctc_decode(
        _logits_for_paths([path], 3), torch.tensor([len(path)]), vocabulary
    )[0]

    assert prediction.text == expected_text
    assert prediction.character_indices == expected_indices
    assert prediction.sequence_length == len(path)


def test_decoder_ignores_predictions_after_valid_length() -> None:
    vocabulary = Vocabulary(("ا", "ب"))
    predictions = greedy_ctc_decode(
        _logits_for_paths([[1, 0, 2, 2], [2, 2, 0, 1]], 3),
        torch.tensor([2, 4]),
        vocabulary,
    )

    assert [prediction.text for prediction in predictions] == ["ا", "با"]
    assert [prediction.sequence_length for prediction in predictions] == [2, 4]


def test_decoder_rejects_vocabulary_classifier_mismatch() -> None:
    with pytest.raises(DecodingError, match="class count"):
        greedy_ctc_decode(
            _logits_for_paths([[0, 2]], 3),
            torch.tensor([2]),
            Vocabulary(("ا",)),
        )


@pytest.mark.parametrize(
    ("logits", "lengths"),
    [
        (torch.zeros(2, 3), torch.tensor([2])),
        (torch.zeros(1, 2, 3, dtype=torch.int64), torch.tensor([2])),
        (torch.full((1, 2, 3), float("nan")), torch.tensor([2])),
        (torch.zeros(1, 2, 3), torch.tensor([[2]])),
        (torch.zeros(1, 2, 3), torch.tensor([2], dtype=torch.int32)),
        (torch.zeros(1, 2, 3), torch.tensor([3])),
        (torch.zeros(2, 2, 3), torch.tensor([2])),
    ],
)
def test_decoder_rejects_invalid_contract(logits: torch.Tensor, lengths: torch.Tensor) -> None:
    with pytest.raises(DecodingError):
        greedy_ctc_decode(logits, lengths, Vocabulary(("ا", "ب")))
