from __future__ import annotations

import pytest
import torch

from urdu_document_ocr import CRNNRecognizer, Vocabulary
from urdu_document_ocr.errors import TrainingDataError
from urdu_document_ocr.training import OCRLineItem, collate_ocr_batch


def _item(name: str, width: int, targets: tuple[int, ...]) -> OCRLineItem:
    return OCRLineItem(
        image=torch.zeros((1, 64, width), dtype=torch.float32),
        valid_width=width,
        target=torch.tensor(targets, dtype=torch.int64),
        sample_id=name,
    )


def test_collator_pads_right_with_white_and_concatenates_targets() -> None:
    batch = collate_ocr_batch([_item("a", 7, (1, 2)), _item("b", 12, (2,))], class_count=3)

    assert batch.images.shape == (2, 1, 64, 12)
    assert torch.all(batch.images[0, :, :, 7:] == 1.0)
    assert torch.all(batch.images[1] == 0.0)
    assert batch.valid_widths.tolist() == [7, 12]
    assert batch.targets.tolist() == [1, 2, 2]
    assert batch.target_lengths.tolist() == [2, 1]
    assert batch.sample_ids == ("a", "b")
    assert batch.batch_size == 2


def test_collator_one_sample_rounds_to_width_stride() -> None:
    batch = collate_ocr_batch([_item("a", 9, (1,))], class_count=2)

    assert batch.images.shape[-1] == 12
    assert torch.all(batch.images[..., 9:] == 1.0)


@pytest.mark.parametrize(
    ("items", "kwargs", "message"),
    [
        ([], {"class_count": 2}, "empty"),
        ([_item("a", 4, (0,))], {"class_count": 2}, "target indexes"),
        ([_item("a", 4, ())], {"class_count": 2}, "nonempty"),
        ([_item("a", 5, (1,))], {"class_count": 2, "max_width": 4}, "valid width"),
    ],
)
def test_collator_rejects_invalid_values(items, kwargs, message) -> None:
    with pytest.raises(TrainingDataError, match=message):
        collate_ocr_batch(items, **kwargs)


def test_collator_rejects_invalid_tensor_and_duplicate_ids() -> None:
    wrong = OCRLineItem(torch.zeros((1, 63, 4)), 4, torch.tensor([1]), "a")
    with pytest.raises(TrainingDataError, match="shape"):
        collate_ocr_batch([wrong], class_count=2)
    with pytest.raises(TrainingDataError, match="unique"):
        collate_ocr_batch([_item("a", 4, (1,)), _item("a", 4, (1,))], class_count=2)


def test_collator_padding_does_not_change_shared_eval_logits() -> None:
    torch.manual_seed(5)
    model = CRNNRecognizer(Vocabulary(("ا",))).eval()
    short = _item("short", 8, (1,))
    long = _item("long", 16, (1,))
    alone = collate_ocr_batch([short], class_count=2)
    together = collate_ocr_batch([short, long], class_count=2)

    with torch.inference_mode():
        alone_output = model(alone.images, alone.valid_widths)
        together_output = model(together.images, together.valid_widths)

    length = int(alone_output.input_lengths[0])
    assert torch.equal(alone_output.input_lengths, together_output.input_lengths[:1])
    assert torch.allclose(
        alone_output.logits[0, :length], together_output.logits[0, :length], atol=1e-6
    )
