from __future__ import annotations

import json

import numpy as np
import pytest

from urdu_document_ocr.types import (
    BoundingBox,
    DatasetSample,
    DocumentOCRResult,
    EditCounts,
    EvaluationResult,
    LineOCRResult,
    LineRegion,
    OCRPrediction,
    PageImage,
    PageOCRResult,
    PreprocessedPage,
    RegionKind,
    SourceMetadata,
    SourceType,
    Vocabulary,
    normalize_relative_dataset_path,
)


def source_metadata() -> SourceMetadata:
    return SourceMetadata(
        source_type=SourceType.IMAGE,
        display_name="page.png",
        input_sha256="a" * 64,
        byte_size=128,
    )


def page_image(page_index: int = 0) -> PageImage:
    return PageImage(
        page_index=page_index,
        source_page_number=page_index + 1,
        image=np.zeros((4, 6, 3), dtype=np.uint8),
        source=source_metadata(),
    )


def line_region(
    *,
    reading_order_index: int = 0,
    region_kind: RegionKind = RegionKind.LINE,
    column_index: int | None = None,
) -> LineRegion:
    return LineRegion(
        region_id=f"region-{reading_order_index}",
        page_index=0,
        bounding_box=BoundingBox(1, 2, 5, 4),
        column_index=column_index,
        reading_order_index=reading_order_index,
        region_kind=region_kind,
    )


def test_bounding_box_uses_half_open_geometry() -> None:
    box = BoundingBox(x=2, y=3, width=5, height=7)

    assert (box.x_min, box.y_min, box.x_max, box.y_max) == (2, 3, 7, 10)
    assert (box.width, box.height, box.area) == (5, 7, 35)
    assert box.contains(2, 3)
    assert box.contains(6, 9)
    assert not box.contains(7, 9)
    assert not box.contains(6, 10)


@pytest.mark.parametrize(
    ("arguments", "exception"),
    [
        ({"x": -1, "y": 0, "width": 1, "height": 1}, ValueError),
        ({"x": 0, "y": -1, "width": 1, "height": 1}, ValueError),
        ({"x": 0, "y": 0, "width": 0, "height": 1}, ValueError),
        ({"x": 0, "y": 0, "width": 1, "height": 0}, ValueError),
        ({"x": 0.5, "y": 0, "width": 1, "height": 1}, TypeError),
        ({"x": True, "y": 0, "width": 1, "height": 1}, TypeError),
    ],
)
def test_bounding_box_rejects_invalid_coordinates(
    arguments: dict[str, int | float], exception: type[Exception]
) -> None:
    with pytest.raises(exception):
        BoundingBox(**arguments)  # type: ignore[arg-type]


def test_bounding_box_intersection_iou_and_touching_boundary() -> None:
    first = BoundingBox(0, 0, 4, 4)
    second = BoundingBox(2, 1, 4, 3)

    assert first.intersection(second) == BoundingBox(2, 1, 2, 3)
    assert first.intersection_over_union(second) == pytest.approx(6 / 22)
    assert first.intersection(BoundingBox(4, 0, 2, 2)) is None
    assert first.intersection_over_union(BoundingBox(4, 0, 2, 2)) == 0.0


def test_bounding_box_clips_to_image_bounds() -> None:
    assert BoundingBox(3, 4, 5, 6).clipped(6, 7) == BoundingBox(3, 4, 3, 3)
    assert BoundingBox(8, 8, 2, 2).clipped(6, 7) is None


def test_source_metadata_accepts_only_public_safe_basename() -> None:
    source = source_metadata()

    assert source.to_public_dict()["display_name"] == "page.png"
    with pytest.raises(ValueError, match="basename"):
        SourceMetadata(SourceType.IMAGE, display_name=r"C:\Users\example\private\page.png")
    with pytest.raises(ValueError, match="basename"):
        SourceMetadata(SourceType.IMAGE, display_name="/home/example/private/page.png")


def test_source_metadata_validates_digest_and_size() -> None:
    with pytest.raises(ValueError, match="64 hexadecimal"):
        SourceMetadata(SourceType.IMAGE, input_sha256="not-a-digest")
    with pytest.raises(ValueError, match="at least 0"):
        SourceMetadata(SourceType.IMAGE, byte_size=-1)


def test_page_image_accepts_canonical_rgb_uint8() -> None:
    page = page_image()

    assert (page.height, page.width) == (4, 6)
    assert page.to_public_dict()["image"] == {
        "height": 4,
        "width": 6,
        "channels": 3,
        "dtype": "uint8",
    }


@pytest.mark.parametrize(
    "image",
    [
        np.zeros((4, 6, 3), dtype=np.float32),
        np.zeros((4, 6), dtype=np.uint8),
        np.zeros((4, 6, 4), dtype=np.uint8),
        np.zeros((0, 6, 3), dtype=np.uint8),
    ],
)
def test_page_image_rejects_invalid_dtype_or_shape(image: np.ndarray) -> None:
    with pytest.raises((TypeError, ValueError)):
        PageImage(0, 1, image, source_metadata())


def test_page_image_enforces_page_number_invariant() -> None:
    with pytest.raises(ValueError, match=r"page_index \+ 1"):
        PageImage(1, 1, np.zeros((2, 2, 3), dtype=np.uint8), source_metadata())


def test_preprocessed_page_validates_array_contracts() -> None:
    page = page_image()
    processed = PreprocessedPage(
        page=page,
        grayscale=np.zeros((4, 6), dtype=np.uint8),
        foreground_mask=np.zeros((4, 6), dtype=np.bool_),
        is_blank=True,
    )

    assert processed.to_public_dict()["is_blank"] is True
    assert processed.to_public_dict()["deskew_applied"] is False
    assert not processed.deskew_applied
    with pytest.raises(ValueError, match="match"):
        PreprocessedPage(
            page=page,
            grayscale=np.zeros((3, 6), dtype=np.uint8),
            foreground_mask=np.zeros((4, 6), dtype=np.bool_),
            is_blank=False,
        )


def test_line_region_validates_column_and_spanning_semantics() -> None:
    assert line_region(column_index=0).column_index == 0
    assert line_region(region_kind=RegionKind.SPANNING).column_index is None

    with pytest.raises(ValueError, match="cannot be assigned"):
        line_region(region_kind=RegionKind.SPANNING, column_index=1)
    with pytest.raises(ValueError, match="0, 1, or None"):
        line_region(column_index=2)
    with pytest.raises(ValueError, match="at least 0"):
        LineRegion("bad", -1, BoundingBox(0, 0, 1, 1), None, 0)


def test_ocr_prediction_has_no_confidence_and_validates_sequence() -> None:
    prediction = OCRPrediction("اب", (1, 2), sequence_length=4)

    assert not hasattr(prediction, "confidence")
    with pytest.raises(ValueError, match="shorter"):
        OCRPrediction("اب", (1, 2), sequence_length=1)
    with pytest.raises(ValueError, match="at least 1"):
        OCRPrediction("", (0,))


def test_line_result_requires_prediction_or_failure_exclusively() -> None:
    region = line_region()
    result = LineOCRResult(region, prediction=OCRPrediction("متن"))

    assert result.error_code is None
    with pytest.raises(ValueError, match="exactly one"):
        LineOCRResult(region)
    with pytest.raises(ValueError, match="exactly one"):
        LineOCRResult(region, OCRPrediction("متن"), "failed")


def test_page_and_document_results_require_ordered_indices() -> None:
    first = LineOCRResult(line_region(reading_order_index=0), OCRPrediction("الف"))
    second = LineOCRResult(line_region(reading_order_index=1), OCRPrediction("ب"))
    page = PageOCRResult(0, 1, (first, second), "الف\nب")
    document = DocumentOCRResult((page,), "الف\nب")

    assert document.to_public_dict()["pages"][0]["lines"][1]["prediction"]["text"] == "ب"  # type: ignore[index]
    with pytest.raises(ValueError, match="ascending"):
        PageOCRResult(0, 1, (second, first), "")


@pytest.mark.parametrize(
    "unsafe_path",
    [
        r"C:\Users\example\secret\image.png",
        "/home/example/private/image.png",
        "../../outside.png",
        r"\\server\share\image.png",
        r"\rooted\image.png",
        "https://example.invalid/image.png",
        "",
    ],
)
def test_dataset_sample_rejects_unsafe_paths(unsafe_path: str) -> None:
    with pytest.raises((TypeError, ValueError)):
        DatasetSample(1, "sample-1", unsafe_path, "متن", "document-1")


def test_dataset_path_normalizes_relative_separators() -> None:
    assert normalize_relative_dataset_path(r"images\page\line.png") == "images/page/line.png"
    sample = DatasetSample(1, "sample-1", "images/./line.png", "متن", "document-1")

    assert sample.image_path == "images/line.png"


def test_dataset_sample_validates_identifiers_text_and_tags() -> None:
    with pytest.raises(ValueError, match="sample_id"):
        DatasetSample(1, "", "images/line.png", "متن", "document-1")
    with pytest.raises(ValueError, match="text"):
        DatasetSample(1, "sample-1", "images/line.png", " ", "document-1")
    with pytest.raises(ValueError, match="duplicates"):
        DatasetSample(1, "sample-1", "images/line.png", "متن", "document-1", tags=("a", "a"))


def test_vocabulary_is_deterministic_and_blank_is_separate() -> None:
    forward = Vocabulary(("ب", "ا", " "))
    reverse = Vocabulary((" ", "ا", "ب"))

    assert forward.characters == (" ", "ا", "ب")
    assert forward.blank_index == 0
    assert forward.fingerprint == reverse.fingerprint
    assert forward.encode("اب") == (2, 3)
    assert forward.decode((2, 3)) == "اب"


def test_vocabulary_fingerprint_has_stable_known_value() -> None:
    vocabulary = Vocabulary(("ا", "ب"))

    assert (
        vocabulary.fingerprint == "6f37556a577c825b102a87611528d7524781197c2ed537707068591364ce00a9"
    )


def test_vocabulary_rejects_duplicates_unknowns_and_blank_index_changes() -> None:
    with pytest.raises(ValueError, match="unique"):
        Vocabulary(("ا", "ا"))
    with pytest.raises(ValueError, match="blank_index"):
        Vocabulary(("ا",), blank_index=1)
    with pytest.raises(ValueError, match=r"U\+"):
        Vocabulary(("ا",)).encode("ب")
    with pytest.raises(ValueError, match="outside"):
        Vocabulary(("ا",)).decode((2,))


def test_edit_and_evaluation_contracts_are_structural() -> None:
    characters = EditCounts(1, 2, 3, 10)
    words = EditCounts(0, 1, 0, 4)
    result = EvaluationResult(5, 2, characters, words, "data-hash", "config-hash")

    assert characters.total_errors == 6
    assert result.to_public_dict()["character_edits"]["reference_length"] == 10  # type: ignore[index]
    with pytest.raises(ValueError, match="cannot exceed"):
        EvaluationResult(1, 2, characters, words, "data-hash", "config-hash")


def test_public_serialization_omits_arrays_and_absolute_paths() -> None:
    serialized = json.dumps(page_image().to_public_dict(), sort_keys=True)

    assert "page.png" in serialized
    assert "[[[" not in serialized
    assert "C:" not in serialized
    assert "/home/" not in serialized
