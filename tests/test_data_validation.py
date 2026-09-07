from __future__ import annotations

import os
from pathlib import Path

import pytest
from fixtures.data_samples import sample, write_image

from urdu_document_ocr import (
    DatasetSample,
    DatasetValidationError,
    ValidationSeverity,
    Vocabulary,
    normalize_transcription,
    validate_dataset,
)


def issue_codes(report: object) -> list[str]:
    return [issue.code for issue in report.issues]  # type: ignore[attr-defined]


def test_explicit_normalization_preserves_urdu_distinctions_and_zwnj() -> None:
    text = "  \u06cc\u200c\u064a  \u06f1\u061f\u0650  "

    normalized = normalize_transcription(text)

    assert normalized == "\u06cc\u200c\u064a \u06f1\u061f\u0650"
    assert "\u06cc" in normalized and "\u064a" in normalized
    assert "\u200c" in normalized
    assert "\u06f1" in normalized and "\u061f" in normalized and "\u0650" in normalized


def test_normalization_composes_nfc_without_character_folding() -> None:
    decomposed = "\u0627\u0653"

    assert normalize_transcription(decomposed) == "\u0622"
    with pytest.raises(ValueError, match="control"):
        normalize_transcription("ا\tب")
    with pytest.raises(ValueError, match="empty"):
        normalize_transcription("   ")
    with pytest.raises(TypeError):
        normalize_transcription(1)  # type: ignore[arg-type]


def test_valid_png_and_jpeg_dataset_has_deterministic_statistics(tmp_path: Path) -> None:
    write_image(tmp_path, "images/a.png", size=(20, 8))
    write_image(tmp_path, "images/b.jpg", size=(40, 10), image_format="JPEG")
    samples = (
        sample(1, document_id="document-a", image_path="images/a.png", text="اردو متن"),
        sample(2, document_id="document-a", image_path="images/b.jpg", text="یاد"),
    )

    report = validate_dataset(samples, dataset_root=tmp_path)

    assert report.is_valid
    assert report.error_count == 0
    assert report.warning_count == 0
    assert report.statistics.sample_count == 2
    assert report.statistics.document_count == 1
    assert report.statistics.image_width.to_public_dict() == {
        "minimum": 20,
        "median": 30.0,
        "maximum": 40,
    }
    assert report.statistics.image_height.median == 9.0
    assert report.statistics.samples_per_document.minimum == 2
    assert report.statistics.character_frequency == tuple(
        sorted(report.statistics.character_frequency, key=lambda item: ord(item[0]))
    )
    assert report.to_public_dict()["dataset_fingerprint"] == report.dataset_fingerprint


def test_lightweight_validation_does_not_require_or_decode_images() -> None:
    item = sample(1, image_path="images/missing.png")

    report = validate_dataset((item,), inspect_images=False)

    assert report.is_valid
    assert report.statistics.image_width.minimum is None
    assert report.statistics.image_error_count == 0


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("\u0627\u0653", "non_nfc_text"),
        (" اردو", "leading_or_trailing_whitespace"),
        ("اردو ", "leading_or_trailing_whitespace"),
        ("اردو  متن", "noncanonical_whitespace"),
        ("اردو\tمتن", "invalid_text_control"),
        ("اردو\nمتن", "invalid_text_control"),
    ],
)
def test_validation_reports_noncanonical_text_without_mutation(text: str, expected: str) -> None:
    item = sample(1, text=text)

    report = validate_dataset((item,), inspect_images=False)

    assert expected in issue_codes(report)
    assert not report.is_valid
    assert item.text == text


def test_validation_reports_training_vocabulary_unknowns_as_codepoints() -> None:
    vocabulary = Vocabulary(("ا", " "))
    item = sample(1, text="ا ب")

    report = validate_dataset((item,), vocabulary=vocabulary, inspect_images=False)

    issue = next(issue for issue in report.issues if issue.code == "unsupported_character")
    assert issue.severity is ValidationSeverity.WARNING
    assert "U+0628" in issue.message
    assert "ا ب" not in issue.message


def test_missing_directory_and_unreadable_or_wrong_image_are_errors(tmp_path: Path) -> None:
    (tmp_path / "images").mkdir()
    (tmp_path / "images" / "broken.png").write_bytes(b"not an image")
    write_image(tmp_path, "images/line.gif", image_format="GIF")
    (tmp_path / "images" / "folder.png").mkdir()
    samples = (
        sample(1, image_path="images/missing.png"),
        sample(2, image_path="images/broken.png"),
        sample(3, image_path="images/line.gif"),
        sample(4, image_path="images/folder.png"),
    )

    report = validate_dataset(samples, dataset_root=tmp_path)

    assert set(issue_codes(report)) >= {
        "missing_image",
        "unreadable_image",
        "unsupported_image_format",
        "not_regular_file",
    }
    assert report.statistics.image_error_count == 4


def test_unusual_and_over_limit_dimensions_are_distinct(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_image(tmp_path, "images/thin.png", size=(100, 1))
    write_image(tmp_path, "images/large.png", size=(20, 20))
    monkeypatch.setattr("urdu_document_ocr.data.validation._MAXIMUM_IMAGE_PIXELS", 200)

    report = validate_dataset(
        (sample(1, image_path="images/thin.png"), sample(2, image_path="images/large.png")),
        dataset_root=tmp_path,
    )

    assert "unusual_dimensions" in issue_codes(report)
    assert "invalid_dimensions" in issue_codes(report)


def test_over_limit_image_bytes_fail_before_decode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_image(tmp_path, "images/large-file.png")
    monkeypatch.setattr("urdu_document_ocr.data.validation._MAXIMUM_IMAGE_BYTES", 1)

    report = validate_dataset(
        (sample(1, image_path="images/large-file.png"),), dataset_root=tmp_path
    )

    assert issue_codes(report) == ["image_too_large"]
    assert report.statistics.image_error_count == 1


def test_duplicate_ids_paths_images_pairs_and_conflicts_are_reported(tmp_path: Path) -> None:
    write_image(tmp_path, "images/a.png")
    (tmp_path / "images" / "copy.png").write_bytes((tmp_path / "images" / "a.png").read_bytes())
    samples = (
        sample(1, image_path="images/a.png", text="الف"),
        DatasetSample(1, "sample-001", "images/copy.png", "ب", "document-2"),
        sample(3, image_path="images/a.png", text="الف"),
        sample(4, image_path="images/copy.png", text="ب"),
    )

    report = validate_dataset(samples, dataset_root=tmp_path)

    assert set(issue_codes(report)) >= {
        "duplicate_sample_id",
        "duplicate_image_path",
        "duplicate_image",
        "duplicate_pair",
        "conflicting_image_text",
    }
    assert not report.is_valid


def test_dataset_root_validation_is_typed_and_path_safe(tmp_path: Path) -> None:
    unavailable = tmp_path / "private-root"

    with pytest.raises(DatasetValidationError) as captured:
        validate_dataset((sample(1),), dataset_root=unavailable)

    assert str(tmp_path) not in str(captured.value.to_public_dict())
    with pytest.raises(DatasetValidationError):
        validate_dataset((object(),), inspect_images=False)  # type: ignore[arg-type]
    with pytest.raises(DatasetValidationError):
        validate_dataset((sample(1),), vocabulary=object(), inspect_images=False)  # type: ignore[arg-type]
    with pytest.raises(DatasetValidationError):
        validate_dataset((sample(1),), inspect_images=1)  # type: ignore[arg-type]


@pytest.mark.skipif(not hasattr(os, "symlink"), reason="symlink support unavailable")
def test_symlink_escape_is_rejected_when_platform_permits_it(tmp_path: Path) -> None:
    outside = tmp_path.parent / f"{tmp_path.name}-outside"
    outside.mkdir(exist_ok=True)
    write_image(outside, "line.png")
    (tmp_path / "images").mkdir()
    link = tmp_path / "images" / "escape.png"
    try:
        link.symlink_to(outside / "line.png")
    except OSError:
        pytest.skip("symlink creation is not permitted")

    report = validate_dataset((sample(1, image_path="images/escape.png"),), dataset_root=tmp_path)

    assert issue_codes(report) == ["invalid_path"]
