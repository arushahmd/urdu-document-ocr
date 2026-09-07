from __future__ import annotations

import json
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest
from fixtures.data_samples import sample

from urdu_document_ocr import (
    DatasetSample,
    ReviewDecision,
    ReviewOverlayError,
    ReviewStatus,
    apply_review_overlay,
    dataset_fingerprint,
    read_review_overlay,
    review_overlay_fingerprint,
    write_review_overlay,
)


def decision(
    item: DatasetSample,
    source_fingerprint: str,
    status: ReviewStatus = ReviewStatus.VALID,
    *,
    tags: tuple[str, ...] = (),
) -> ReviewDecision:
    return ReviewDecision(
        sample_id=item.sample_id,
        status=status,
        source_dataset_fingerprint=source_fingerprint,
        tags=tags,
    )


def test_review_decision_is_strict_immutable_and_safe() -> None:
    item = ReviewDecision(
        "sample-1",
        ReviewStatus.VALID,
        "a" * 64,
        tags=("other-font", "heading"),
        note="Checked visually",
        reviewer="reviewer-1",
    )

    assert item.tags == ("heading", "other-font")
    assert item.to_public_dict()["status"] == "valid"
    with pytest.raises(FrozenInstanceError):
        item.status = ReviewStatus.INVALID  # type: ignore[misc]


@pytest.mark.parametrize(
    "arguments",
    [
        {"sample_id": ""},
        {"status": "valid"},
        {"source_dataset_fingerprint": "bad"},
        {"source_dataset_fingerprint": 1},
        {"tags": ("Heading",)},
        {"tags": ("heading", "heading")},
        {"tags": "heading"},
        {"note": "line\nbreak"},
        {"note": "x" * 241},
        {"reviewer": "person@example.invalid"},
        {"schema_version": 2},
    ],
)
def test_review_decision_rejects_invalid_fields(arguments: dict[str, object]) -> None:
    values: dict[str, object] = {
        "sample_id": "sample-1",
        "status": ReviewStatus.VALID,
        "source_dataset_fingerprint": "a" * 64,
    }
    values.update(arguments)

    with pytest.raises((TypeError, ValueError)):
        ReviewDecision(**values)  # type: ignore[arg-type]


def test_overlay_roundtrip_is_sorted_utf8_and_fingerprinted(tmp_path: Path) -> None:
    samples = (sample(1), sample(2))
    fingerprint = dataset_fingerprint(samples)
    decisions = (
        decision(samples[1], fingerprint, ReviewStatus.NEEDS_REVIEW),
        decision(samples[0], fingerprint, tags=("heading",)),
    )
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"

    write_review_overlay(decisions, first)
    write_review_overlay(reversed(decisions), second)

    assert first.read_bytes() == second.read_bytes()
    assert read_review_overlay(first)[0].sample_id == "sample-001"
    assert review_overlay_fingerprint(decisions) == review_overlay_fingerprint(reversed(decisions))
    assert len(review_overlay_fingerprint(decisions)) == 64


def test_apply_overlay_excludes_invalid_and_unresolved_and_merges_tags() -> None:
    samples = (sample(1, tags=("existing",)), sample(2), sample(3), sample(4))
    original = tuple(samples)
    fingerprint = dataset_fingerprint(samples)
    overlay = (
        decision(samples[0], fingerprint, tags=("heading",)),
        decision(samples[1], fingerprint, ReviewStatus.INVALID),
        decision(samples[2], fingerprint, ReviewStatus.NEEDS_REVIEW, tags=("other-font",)),
    )

    training_ready = apply_review_overlay(samples, overlay)
    including_unresolved = apply_review_overlay(samples, overlay, include_needs_review=True)

    assert [item.sample_id for item in training_ready] == ["sample-001", "sample-004"]
    assert training_ready[0].tags == ("existing", "heading")
    assert [item.sample_id for item in including_unresolved] == [
        "sample-001",
        "sample-003",
        "sample-004",
    ]
    assert including_unresolved[1].tags == ("other-font",)
    assert samples == original
    assert samples[0].tags == ("existing",)


def test_apply_overlay_rejects_unknown_duplicate_stale_and_duplicate_sources() -> None:
    samples = (sample(1), sample(2))
    fingerprint = dataset_fingerprint(samples)
    known = decision(samples[0], fingerprint)
    unknown = ReviewDecision("unknown", ReviewStatus.VALID, fingerprint)
    stale = decision(samples[0], "f" * 64)

    with pytest.raises(ReviewOverlayError, match="unknown"):
        apply_review_overlay(samples, (unknown,))
    with pytest.raises(ReviewOverlayError, match="duplicate decisions"):
        apply_review_overlay(samples, (known, known))
    with pytest.raises(ReviewOverlayError, match="stale"):
        apply_review_overlay(samples, (stale,))
    with pytest.raises(ReviewOverlayError, match="duplicate sample_id"):
        apply_review_overlay((samples[0], samples[0]), ())
    with pytest.raises(ReviewOverlayError):
        apply_review_overlay((object(),), ())  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        apply_review_overlay(samples, (), include_needs_review=1)  # type: ignore[arg-type]


def test_overlay_writer_rejects_duplicates_and_requires_overwrite(tmp_path: Path) -> None:
    item = ReviewDecision("sample-1", ReviewStatus.VALID, "a" * 64)
    path = tmp_path / "review.jsonl"
    write_review_overlay((item,), path)

    with pytest.raises(ReviewOverlayError, match="overwrite=True"):
        write_review_overlay((item,), path)
    with pytest.raises(ReviewOverlayError, match="duplicate"):
        write_review_overlay((item, item), tmp_path / "duplicate.jsonl")
    write_review_overlay((item,), path, overwrite=True)


@pytest.mark.parametrize(
    "record",
    [
        [],
        {"schema_version": 1},
        {
            "schema_version": 1,
            "sample_id": "sample-1",
            "status": "future",
            "tags": [],
            "source_dataset_fingerprint": "a" * 64,
        },
        {
            "schema_version": 1,
            "sample_id": "sample-1",
            "status": "valid",
            "tags": "heading",
            "source_dataset_fingerprint": "a" * 64,
        },
    ],
)
def test_overlay_reader_rejects_invalid_records(tmp_path: Path, record: object) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps(record), encoding="utf-8")

    with pytest.raises(ReviewOverlayError):
        read_review_overlay(path)


def test_overlay_reader_rejects_duplicate_json_keys_and_decisions(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text('{"schema_version":1,"schema_version":1}\n', encoding="utf-8")
    with pytest.raises(ReviewOverlayError):
        read_review_overlay(path)

    item = ReviewDecision("sample-1", ReviewStatus.VALID, "a" * 64)
    encoded = json.dumps(item.to_public_dict())
    path.write_text(encoded + "\n" + encoded + "\n", encoding="utf-8")
    with pytest.raises(ReviewOverlayError, match="duplicate"):
        read_review_overlay(path)
