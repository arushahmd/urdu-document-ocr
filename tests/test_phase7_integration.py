from __future__ import annotations

from pathlib import Path

from fixtures.data_samples import sample, write_image

from urdu_document_ocr import (
    ReviewDecision,
    ReviewStatus,
    apply_review_overlay,
    build_vocabulary,
    dataset_fingerprint,
    find_unseen_characters,
    read_manifest,
    read_review_overlay,
    save_vocabulary,
    split_dataset,
    validate_dataset,
    write_manifest,
    write_review_overlay,
    write_split_manifests,
)


def test_labeled_lines_cross_manifest_validation_split_vocabulary_and_review(
    tmp_path: Path,
) -> None:
    samples = []
    texts = ("اردو متن", "یاد", "کتاب", "صفحہ", "علم", "قلم")
    documents = ("document-a", "document-a", "document-b", "document-b", "document-c", "document-d")
    for index, (text, document_id) in enumerate(zip(texts, documents, strict=True)):
        item = sample(index, text=text, document_id=document_id)
        write_image(tmp_path, item.image_path, size=(40 + index, 8))
        samples.append(item)

    source_manifest = tmp_path / "manifest.jsonl"
    write_manifest(samples, source_manifest)
    original_manifest_bytes = source_manifest.read_bytes()
    loaded = read_manifest(source_manifest)
    report = validate_dataset(loaded, dataset_root=tmp_path)
    split = split_dataset(loaded)
    split_dir = tmp_path / "splits"
    split_dir.mkdir()
    write_split_manifests(split, split_dir)
    vocabulary = build_vocabulary(split.train)
    save_vocabulary(vocabulary, tmp_path / "vocabulary.json")
    held_out = split.validation + split.test

    assert report.is_valid
    assert split.achieved_sample_counts.total == len(loaded)
    assert set(find_unseen_characters(held_out, vocabulary)) <= set(
        character for item in held_out for character in item.text
    )

    source_fingerprint = dataset_fingerprint(loaded)
    rejected = loaded[0]
    overlay = (
        ReviewDecision(
            rejected.sample_id,
            ReviewStatus.INVALID,
            source_fingerprint,
            tags=("other-font",),
        ),
    )
    overlay_path = tmp_path / "review.jsonl"
    write_review_overlay(overlay, overlay_path)
    derived = apply_review_overlay(loaded, overlay)
    derived_manifest = tmp_path / "reviewed.jsonl"
    write_manifest(derived, derived_manifest)

    assert source_manifest.read_bytes() == original_manifest_bytes
    assert rejected.sample_id not in {item.sample_id for item in read_manifest(derived_manifest)}
    assert len(read_review_overlay(overlay_path)) == 1
