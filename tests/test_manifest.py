from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

import pytest
from fixtures.data_samples import sample

from urdu_document_ocr import (
    ManifestError,
    dataset_fingerprint,
    read_manifest,
    write_manifest,
)


def test_read_manifest_preserves_file_order_and_ignores_blank_lines(tmp_path: Path) -> None:
    path = tmp_path / "manifest.jsonl"
    records = [sample(2).to_public_dict(), sample(1).to_public_dict()]
    path.write_text(
        json.dumps(records[0], ensure_ascii=False)
        + "\n   \n"
        + json.dumps(records[1], ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )

    loaded = read_manifest(path)

    assert [item.sample_id for item in loaded] == ["sample-002", "sample-001"]
    assert loaded[0].text == "اردو متن"


@pytest.mark.parametrize(
    "payload",
    [
        "{broken\n",
        "[]\n",
        '{"schema_version":1}\n',
        '{"schema_version":2,"sample_id":"a","image_path":"i.png","text":"ا","document_id":"d"}\n',
        '{"schema_version":true,"sample_id":"a","image_path":"i.png","text":"ا","document_id":"d"}\n',
        '{"schema_version":1,"sample_id":"a","image_path":"i.png","text":"ا","document_id":"d","typo":1}\n',
        '{"schema_version":1,"sample_id":1,"image_path":"i.png","text":"ا","document_id":"d"}\n',
        '{"schema_version":1,"sample_id":"a","image_path":"i.png","text":"ا","document_id":"d","line_index":true}\n',
        '{"schema_version":1,"sample_id":"a","image_path":"i.png","text":"ا","document_id":"d","tags":"x"}\n',
        '{"schema_version":1,"sample_id":"a","sample_id":"b","image_path":"i.png","text":"ا","document_id":"d"}\n',
        '{"schema_version":NaN,"sample_id":"a","image_path":"i.png","text":"ا","document_id":"d"}\n',
    ],
)
def test_read_manifest_rejects_malformed_or_nonconforming_records(
    tmp_path: Path, payload: str
) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text(payload, encoding="utf-8")

    with pytest.raises(ManifestError) as captured:
        read_manifest(path)

    assert captured.value.context.get("line_number") == 1
    assert str(tmp_path) not in str(captured.value.to_public_dict())


def test_read_manifest_reports_physical_line_number_after_blank_line(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text("\n\nnot-json\n", encoding="utf-8")

    with pytest.raises(ManifestError) as captured:
        read_manifest(path)

    assert captured.value.context == {"line_number": 3}


def test_read_manifest_rejects_duplicate_sample_id(tmp_path: Path) -> None:
    record = {
        "schema_version": 1,
        "sample_id": "same",
        "image_path": "images/line.png",
        "text": "متن",
        "document_id": "document",
    }
    path = tmp_path / "duplicate.jsonl"
    path.write_text(
        json.dumps(record, ensure_ascii=False) + "\n" + json.dumps(record, ensure_ascii=False),
        encoding="utf-8",
    )

    with pytest.raises(ManifestError) as captured:
        read_manifest(path)

    assert captured.value.context == {"line_number": 2, "sample_id": "same"}


def test_write_manifest_is_canonical_sorted_utf8_and_order_independent(tmp_path: Path) -> None:
    first = sample(2, text="یاد")
    second = sample(1, text="اردو")
    forward = tmp_path / "forward.jsonl"
    reverse = tmp_path / "reverse.jsonl"

    write_manifest((first, second), forward)
    write_manifest((second, first), reverse)

    assert forward.read_bytes() == reverse.read_bytes()
    assert forward.read_bytes().endswith(b"\n")
    assert "اردو" in forward.read_text(encoding="utf-8")
    assert b"\\u" not in forward.read_bytes()
    assert [item.sample_id for item in read_manifest(forward)] == ["sample-001", "sample-002"]
    assert dataset_fingerprint((first, second)) == dataset_fingerprint((second, first))
    assert dataset_fingerprint((first, second)) == sha256(forward.read_bytes()).hexdigest()


def test_manifest_omits_absent_optional_fields(tmp_path: Path) -> None:
    item = sample(1)
    item = type(item)(1, item.sample_id, item.image_path, item.text, item.document_id)
    path = tmp_path / "manifest.jsonl"

    write_manifest((item,), path)

    record = json.loads(path.read_text(encoding="utf-8"))
    assert set(record) == {"schema_version", "sample_id", "image_path", "text", "document_id"}


def test_write_manifest_requires_explicit_overwrite_and_existing_parent(tmp_path: Path) -> None:
    path = tmp_path / "manifest.jsonl"
    write_manifest((sample(1),), path)

    with pytest.raises(ManifestError, match="overwrite=True"):
        write_manifest((sample(2),), path)
    write_manifest((sample(2),), path, overwrite=True)
    assert read_manifest(path)[0].sample_id == "sample-002"
    with pytest.raises(ManifestError, match="parent directory"):
        write_manifest((sample(1),), tmp_path / "missing" / "manifest.jsonl")


def test_manifest_io_rejects_duplicate_or_non_sample_values(tmp_path: Path) -> None:
    with pytest.raises(ManifestError):
        write_manifest((sample(1), sample(1)), tmp_path / "duplicate.jsonl")
    with pytest.raises(ManifestError):
        write_manifest((object(),), tmp_path / "invalid.jsonl")  # type: ignore[arg-type]


def test_manifest_open_failures_do_not_expose_absolute_path(tmp_path: Path) -> None:
    missing = tmp_path / "private" / "missing.jsonl"

    with pytest.raises(ManifestError) as captured:
        read_manifest(missing)

    assert str(tmp_path) not in str(captured.value.to_public_dict())


def test_manifest_rejects_invalid_utf8_and_unsafe_record_path(tmp_path: Path) -> None:
    invalid_utf8 = tmp_path / "invalid-utf8.jsonl"
    invalid_utf8.write_bytes(b"\xff\n")
    with pytest.raises(ManifestError, match="UTF-8"):
        read_manifest(invalid_utf8)

    unsafe = tmp_path / "unsafe.jsonl"
    unsafe.write_text(
        '{"document_id":"d","image_path":"../outside.png","sample_id":"s",'
        '"schema_version":1,"text":"ا"}\n',
        encoding="utf-8",
    )
    with pytest.raises(ManifestError, match="constraints"):
        read_manifest(unsafe)
