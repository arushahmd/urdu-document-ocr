from __future__ import annotations

import json
from pathlib import Path

import pytest
from fixtures.data_samples import sample

from urdu_document_ocr import (
    Vocabulary,
    VocabularyError,
    build_vocabulary,
    find_unseen_characters,
    load_vocabulary,
    save_vocabulary,
)


def test_training_vocabulary_is_codepoint_sorted_and_preserves_output_symbols() -> None:
    text = "\u06cc\u200c\u064a \u06f1\u061f\u0650\u06a9\u0643"

    vocabulary = build_vocabulary((sample(1, text=text),))

    assert vocabulary.characters == tuple(sorted(set(text), key=ord))
    assert " " in vocabulary.characters
    assert "\u200c" in vocabulary.characters
    assert "\u06cc" in vocabulary.characters and "\u064a" in vocabulary.characters
    assert "\u06a9" in vocabulary.characters and "\u0643" in vocabulary.characters
    assert "\u06f1" in vocabulary.characters
    assert "\u061f" in vocabulary.characters
    assert "\u0650" in vocabulary.characters
    assert vocabulary.blank_index == 0
    assert not hasattr(vocabulary, "unknown_index")
    assert vocabulary.decode(vocabulary.encode(text)) == text


def test_vocabulary_is_deterministic_across_sample_order() -> None:
    first = sample(1, text="اب ج")
    second = sample(2, text="د ا")

    forward = build_vocabulary((first, second))
    reverse = build_vocabulary((second, first))

    assert forward == reverse
    assert forward.fingerprint == reverse.fingerprint


def test_vocabulary_is_train_only_and_reports_unseen_validation_characters() -> None:
    vocabulary = build_vocabulary((sample(1, text="اب"),))

    unseen = find_unseen_characters((sample(2, text="اب پ؟"),), vocabulary)

    assert unseen == tuple(sorted((" ", "پ", "؟"), key=ord))
    with pytest.raises(ValueError, match=r"U\+"):
        vocabulary.encode("پ")


def test_vocabulary_build_rejects_empty_or_noncanonical_sources() -> None:
    with pytest.raises(VocabularyError, match="at least one"):
        build_vocabulary(())
    with pytest.raises(VocabularyError, match="normalization"):
        build_vocabulary((sample(1, text=" اردو"),))
    with pytest.raises(VocabularyError):
        build_vocabulary((object(),))  # type: ignore[arg-type]
    with pytest.raises(VocabularyError):
        find_unseen_characters((object(),), Vocabulary(("ا",)))  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        find_unseen_characters((sample(1),), object())  # type: ignore[arg-type]


def test_vocabulary_save_load_roundtrip_is_byte_stable(tmp_path: Path) -> None:
    vocabulary = build_vocabulary((sample(1, text="اردو متن"),))
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"

    save_vocabulary(vocabulary, first)
    save_vocabulary(vocabulary, second)

    assert first.read_bytes() == second.read_bytes()
    assert first.read_bytes().endswith(b"\n")
    assert b"\\u" not in first.read_bytes()
    assert load_vocabulary(first) == vocabulary
    assert load_vocabulary(first).fingerprint == vocabulary.fingerprint


def test_vocabulary_save_requires_explicit_overwrite(tmp_path: Path) -> None:
    path = tmp_path / "vocabulary.json"
    vocabulary = Vocabulary(("ا",))
    save_vocabulary(vocabulary, path)

    with pytest.raises(VocabularyError, match="overwrite=True"):
        save_vocabulary(vocabulary, path)
    save_vocabulary(vocabulary, path, overwrite=True)
    with pytest.raises(TypeError):
        save_vocabulary(object(), tmp_path / "bad.json")  # type: ignore[arg-type]


def vocabulary_record(vocabulary: Vocabulary) -> dict[str, object]:
    return vocabulary.to_public_dict()


@pytest.mark.parametrize(
    "mutation",
    [
        lambda record: record.update(fingerprint="0" * 64),
        lambda record: record.update(characters=["ب", "ا"]),
        lambda record: record.update(characters=["ا", "ا"]),
        lambda record: record.update(characters=["اب"]),
        lambda record: record.update(blank_index=1),
        lambda record: record.update(schema_version=2),
        lambda record: record.update(normalization_policy_version="future"),
        lambda record: record.update(extra=True),
    ],
)
def test_load_vocabulary_rejects_corruption_and_schema_drift(
    tmp_path: Path, mutation: object
) -> None:
    vocabulary = Vocabulary(("ا", "ب"))
    record = vocabulary_record(vocabulary)
    mutation(record)  # type: ignore[operator]
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")

    with pytest.raises(VocabularyError):
        load_vocabulary(path)


def test_load_vocabulary_rejects_duplicate_keys_malformed_json_and_missing_file(
    tmp_path: Path,
) -> None:
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"schema_version":1,"schema_version":1}', encoding="utf-8")
    malformed = tmp_path / "malformed.json"
    malformed.write_text("{", encoding="utf-8")

    with pytest.raises(VocabularyError):
        load_vocabulary(duplicate)
    with pytest.raises(VocabularyError):
        load_vocabulary(malformed)
    with pytest.raises(VocabularyError) as captured:
        load_vocabulary(tmp_path / "private" / "missing.json")
    assert str(tmp_path) not in str(captured.value.to_public_dict())
