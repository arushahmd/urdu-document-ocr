from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from fixtures.page_patterns import preprocessed_from_mask

from urdu_document_ocr import (
    DocumentOCRResult,
    LineOCRResult,
    OCRPrediction,
    PageOCRResult,
    write_json_result,
    write_text_result,
)
from urdu_document_ocr.document.assembly import (
    assemble_document,
    assemble_page,
    document_to_dict,
    document_to_json,
    document_to_text,
)
from urdu_document_ocr.errors import AssemblyError
from urdu_document_ocr.types import BoundingBox, LineRegion, RegionKind


def _line(
    text: str,
    index: int,
    *,
    page_index: int = 0,
    column: int | None = None,
    kind: RegionKind = RegionKind.LINE,
) -> LineOCRResult:
    region = LineRegion(
        region_id=f"page-{page_index}-line-{index}",
        page_index=page_index,
        bounding_box=BoundingBox(10 + index, 20 + index * 10, 50, 8),
        column_index=column,
        reading_order_index=index,
        region_kind=kind,
    )
    return LineOCRResult(region, OCRPrediction(text, (1,), sequence_length=2))


@pytest.mark.parametrize(
    ("lines", "expected"),
    [
        (("line1", "line2", "line3"), "line1\nline2\nline3"),
        (("right1", "right2", "left1", "left2"), "right1\nright2\nleft1\nleft2"),
        (("heading", "right1", "left1", "footer"), "heading\nright1\nleft1\nfooter"),
    ],
)
def test_page_assembly_uses_only_frozen_reading_order(
    lines: tuple[str, ...], expected: str
) -> None:
    page = preprocessed_from_mask(np.zeros((100, 200), dtype=np.bool_))
    results = tuple(_line(text, index) for index, text in enumerate(lines))

    assembled = assemble_page(page, results)

    assert assembled.lines == results
    assert assembled.assembled_text == expected


def test_blank_page_and_explicit_line_error_remain_structured() -> None:
    blank = preprocessed_from_mask(
        np.zeros((100, 200), dtype=np.bool_), page_index=1, is_blank=True
    )
    page = assemble_page(blank, ())
    failed_region = LineRegion("line-error", 0, BoundingBox(0, 0, 10, 5), None, 0)
    populated = assemble_page(
        preprocessed_from_mask(np.zeros((100, 200), dtype=np.bool_)),
        (LineOCRResult(failed_region, error_code="model_input_error"),),
    )

    assert page.is_blank and page.lines == () and page.assembled_text == ""
    assert populated.assembled_text == ""
    assert populated.to_public_dict()["lines"][0]["error_code"] == "model_input_error"  # type: ignore[index]


def test_multipage_text_and_json_preserve_blank_page_and_unicode() -> None:
    page0 = PageOCRResult(0, 1, (_line("نیم‌خودکار، ۱۲۳؟", 0),), "نیم‌خودکار، ۱۲۳؟")
    page1 = PageOCRResult(1, 2, (), "", is_blank=True)
    page2 = PageOCRResult(
        2,
        3,
        (_line("آخری صفحہ۔", 0, page_index=2),),
        "آخری صفحہ۔",
    )

    result = assemble_document((page0, page1, page2))
    payload = document_to_json(result)
    projection = document_to_dict(result)

    assert document_to_text(result) == "نیم‌خودکار، ۱۲۳؟\n\n\n\nآخری صفحہ۔"
    assert payload.endswith("\n")
    assert "نیم‌خودکار، ۱۲۳؟" in payload
    assert "\\u" not in payload
    assert json.loads(payload) == projection
    assert [page["page_index"] for page in projection["pages"]] == [0, 1, 2]  # type: ignore[index]
    line = projection["pages"][0]["lines"][0]  # type: ignore[index]
    assert set(line) == {"region", "prediction", "error_code"}
    assert set(line["region"]) == {  # type: ignore[index]
        "region_id",
        "page_index",
        "bounding_box",
        "column_index",
        "reading_order_index",
        "region_kind",
    }
    assert set(line["prediction"]) == {  # type: ignore[index]
        "text",
        "character_indices",
        "sequence_length",
    }


def test_result_writers_are_atomic_utf8_and_require_explicit_overwrite(tmp_path: Path) -> None:
    page = PageOCRResult(0, 1, (_line("اردو، ۱۲۳۔", 0),), "اردو، ۱۲۳۔")
    result = DocumentOCRResult((page,), page.assembled_text)
    text_path = tmp_path / "result.txt"
    json_path = tmp_path / "result.json"

    write_text_result(result, text_path)
    write_json_result(result, json_path)

    assert text_path.read_bytes().decode("utf-8") == result.assembled_text
    assert "اردو، ۱۲۳۔" in json_path.read_text(encoding="utf-8")
    with pytest.raises(AssemblyError, match="overwrite=True"):
        write_text_result(result, text_path)
    write_text_result(result, text_path, overwrite=True)
    with pytest.raises(AssemblyError, match="parent"):
        write_json_result(result, tmp_path / "missing" / "result.json")


def test_assembly_rejects_reordering_cross_page_and_blank_content() -> None:
    page = preprocessed_from_mask(np.zeros((100, 200), dtype=np.bool_))
    with pytest.raises(AssemblyError, match="reading-order"):
        assemble_page(page, (_line("second", 1), _line("first", 0)))
    with pytest.raises(AssemblyError, match="source page"):
        assemble_page(page, (_line("wrong", 0, page_index=1),))
    blank = preprocessed_from_mask(np.zeros((100, 200), dtype=np.bool_), is_blank=True)
    with pytest.raises(AssemblyError, match="blank"):
        assemble_page(blank, (_line("impossible", 0),))
    with pytest.raises(AssemblyError, match="zero-based"):
        assemble_document((PageOCRResult(1, 2, (), ""),))


def test_assembly_rejects_wrong_contract_types() -> None:
    page = preprocessed_from_mask(np.zeros((20, 20), dtype=np.bool_))
    result = assemble_document(())

    with pytest.raises(TypeError, match="PreprocessedPage"):
        assemble_page(object(), ())  # type: ignore[arg-type]
    with pytest.raises(AssemblyError, match="LineOCRResult"):
        assemble_page(page, (object(),))  # type: ignore[arg-type]
    with pytest.raises(AssemblyError, match="PageOCRResult"):
        assemble_document((object(),))  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="DocumentOCRResult"):
        document_to_text(object())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="DocumentOCRResult"):
        document_to_dict(object())  # type: ignore[arg-type]
    assert document_to_text(result) == ""


def test_result_writer_rejects_invalid_destination_contracts(tmp_path: Path) -> None:
    result = assemble_document(())

    with pytest.raises(AssemblyError, match="overwrite"):
        write_text_result(result, tmp_path / "result.txt", overwrite=1)  # type: ignore[arg-type]
    with pytest.raises(AssemblyError, match="invalid"):
        write_text_result(result, None)  # type: ignore[arg-type]
    directory_target = tmp_path / "directory"
    directory_target.mkdir()
    with pytest.raises(AssemblyError, match="regular file"):
        write_text_result(result, directory_target)
