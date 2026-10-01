from __future__ import annotations

import json
from dataclasses import replace
from hashlib import sha256
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import urdu_document_ocr.data.synthetic as synthetic_module
from urdu_document_ocr import (
    DatasetSplitConfig,
    PageLayoutFamily,
    PreprocessingConfig,
    SyntheticDataConfig,
    SyntheticDataError,
    assert_shaping_available,
    build_vocabulary,
    bundled_font_path,
    find_unseen_characters,
    generate_fixture_dataset,
    generate_line_dataset,
    generate_line_sample,
    generate_page_fixture,
    load_document,
    load_font_provenance,
    load_vocabulary,
    preprocess_page,
    read_manifest,
    segment_page,
    split_dataset,
    validate_dataset,
)
from urdu_document_ocr.config import ConfigurationError
from urdu_document_ocr.types import BoundingBox

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_ROOT = REPOSITORY_ROOT / "data" / "sample"
EXPECTED_FONT_SHA256 = "eff3a48f588f599f98e98350f1107e2e492edefbe864c8b61c73f2d605f1dce4"


def _without_rendered_image_hashes(value: object, *, parent_key: str | None = None) -> object:
    if isinstance(value, dict):
        return {
            key: _without_rendered_image_hashes(item, parent_key=key)
            for key, item in value.items()
            if not (key == "image_sha256" or (key == "sha256" and parent_key == "image"))
        }
    if isinstance(value, list):
        return [_without_rendered_image_hashes(item, parent_key=parent_key) for item in value]
    return value


def _portable_artifact_identity(manifest: dict[str, object]) -> dict[str, object]:
    renderer_dependent = {
        "generation-manifest.json",
        *(path for path in manifest_artifact_paths(manifest) if path.endswith(".png")),
    }
    artifacts = []
    for entry in manifest["artifacts"]:
        assert isinstance(entry, dict)
        path = entry["path"]
        assert isinstance(path, str)
        if path in renderer_dependent:
            artifacts.append(
                {key: value for key, value in entry.items() if key not in {"sha256", "size_bytes"}}
            )
        else:
            artifacts.append(entry)
    return {key: value for key, value in manifest.items() if key != "artifacts"} | {
        "artifacts": artifacts
    }


def manifest_artifact_paths(manifest: dict[str, object]) -> tuple[str, ...]:
    entries = manifest["artifacts"]
    assert isinstance(entries, list)
    paths = []
    for entry in entries:
        assert isinstance(entry, dict)
        path = entry["path"]
        assert isinstance(path, str)
        paths.append(path)
    return tuple(paths)


def _generated_png_hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*.png"))
    }


def test_font_provenance_and_direct_shaping_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    font = bundled_font_path()
    provenance = load_font_provenance()
    capabilities = assert_shaping_available()

    assert font.is_file()
    assert font.stat().st_size == 325_092
    assert sha256(font.read_bytes()).hexdigest() == EXPECTED_FONT_SHA256
    assert provenance["artifact_sha256"] == EXPECTED_FONT_SHA256
    assert provenance["license_identifier"] == "OFL-1.1"
    assert provenance["modification_status"] == "unmodified"
    assert (font.parent / provenance["license_file"]).is_file()
    assert capabilities.engine == "uharfbuzz-freetype-direct-v1"
    assert capabilities.rtl_strategy == "explicit-rtl-with-numeric-ltr-runs"

    monkeypatch.setattr(synthetic_module, "hb", None)
    with pytest.raises(SyntheticDataError, match="requires the uharfbuzz"):
        assert_shaping_available()


def test_line_generation_is_deterministic_bounded_and_variable_width() -> None:
    text = "یہ ایک سادہ آزمائشی سطر ہے۔"
    original = text
    first = generate_line_sample(
        text, sample_id="syn-line-test-0001", document_id="synthetic-document-test"
    )
    second = generate_line_sample(
        text, sample_id="syn-line-test-0001", document_id="synthetic-document-test"
    )
    changed = generate_line_sample(
        text,
        sample_id="syn-line-test-0001",
        document_id="synthetic-document-test",
        config=replace(SyntheticDataConfig(), seed=240_818),
    )
    short = generate_line_sample(
        "صاف متن۔", sample_id="syn-line-test-0002", document_id="synthetic-document-test"
    )

    assert text == original == first.sample.text
    assert first.record == second.record
    assert first.png_bytes == second.png_bytes
    assert first.record != changed.record
    assert first.record.width != short.record.width
    assert first.image.shape == (first.record.height, first.record.width)
    assert int(first.image.min()) < first.record.background_intensity - 64
    assert 0.0 <= first.record.blur_sigma <= 0.65
    assert 0.0 <= first.record.noise_std <= 1.8
    assert abs(first.record.skew_degrees) <= 2.5
    assert first.record.width <= 1_800


def test_line_paths_and_output_directory_fail_closed(tmp_path: Path) -> None:
    with pytest.raises(SyntheticDataError, match="identity or path"):
        generate_line_sample(
            "صاف متن۔",
            sample_id="syn-line-test",
            document_id="synthetic-document-test",
            image_path="../escape.png",
        )

    generated = generate_line_dataset(tmp_path, 7)
    assert len(generated) == 7
    assert len(tuple((tmp_path / "lines").glob("*.png"))) == 7
    assert len(read_manifest(tmp_path / "manifest.jsonl")) == 7
    with pytest.raises(SyntheticDataError, match="overwrite=True"):
        generate_line_dataset(tmp_path, 7)
    (tmp_path / "unrelated.txt").write_text("owner data", encoding="utf-8")
    with pytest.raises(SyntheticDataError, match="outside this generation plan"):
        generate_line_dataset(tmp_path, 7, overwrite=True)


@pytest.mark.parametrize("family", tuple(PageLayoutFamily))
def test_page_families_have_deterministic_geometry(family: PageLayoutFamily) -> None:
    page_id = f"syn-page-test-{family.value}"
    first = generate_page_fixture(family, page_id=page_id)
    second = generate_page_fixture(family, page_id=page_id)

    assert first.record == second.record
    assert first.png_bytes == second.png_bytes
    assert first.image.shape == (1_200, 1_000)
    assert [line.reading_order_index for line in first.record.lines] == list(
        range(first.record.expected_line_count)
    )
    assert all(
        line.output_box.clipped(1_000, 1_200) == line.output_box for line in first.record.lines
    )
    if family in {PageLayoutFamily.BLANK, PageLayoutFamily.NEAR_BLANK}:
        assert first.record.expected_blank
        assert first.record.lines == ()
    else:
        assert not first.record.expected_blank
        assert first.record.expected_line_count >= 4
        assert int(first.image.min()) < first.record.background_intensity - 64
    if "two_column" in family.value or family.value.startswith("spanning_"):
        columns = {line.column_index for line in first.record.lines}
        assert {0, 1}.issubset(columns)
    if family.value.startswith("spanning_"):
        assert any(line.region_kind.value == "spanning" for line in first.record.lines)


def test_committed_fixture_manifest_split_vocabulary_and_hashes() -> None:
    samples = read_manifest(FIXTURE_ROOT / "manifest.jsonl")
    vocabulary = load_vocabulary(FIXTURE_ROOT / "synthetic-fixture-vocabulary.json")
    split = split_dataset(
        samples,
        DatasetSplitConfig(train_ratio=2 / 3, validation_ratio=1 / 6, test_ratio=1 / 6, seed=1337),
    )
    report = validate_dataset(samples, dataset_root=FIXTURE_ROOT, vocabulary=vocabulary)
    generation = json.loads((FIXTURE_ROOT / "generation-manifest.json").read_text(encoding="utf-8"))
    artifacts = json.loads((FIXTURE_ROOT / "artifact-manifest.json").read_text(encoding="utf-8"))

    assert len(samples) == 24
    assert report.error_count == report.warning_count == 0
    assert build_vocabulary(split.train) == vocabulary
    assert find_unseen_characters(split.validation + split.test, vocabulary) == ()
    assert generation["dataset_fingerprint"] == report.dataset_fingerprint
    assert generation["split"]["split_fingerprint"] == split.fingerprint
    assert generation["vocabulary_fingerprint"] == vocabulary.fingerprint
    assert split.achieved_sample_counts.to_public_dict() == {
        "train": 16,
        "validation": 4,
        "test": 4,
    }
    for entry in artifacts["artifacts"]:
        payload = (FIXTURE_ROOT / entry["path"]).read_bytes()
        assert len(payload) == entry["size_bytes"]
        assert sha256(payload).hexdigest() == entry["sha256"]


def test_authored_text_provenance_is_canonical_and_complete() -> None:
    payload = json.loads((FIXTURE_ROOT / "text-provenance.json").read_text(encoding="utf-8"))
    assert payload["phrase_count"] == 13
    assert payload["all_nfc"] is True
    assert payload["unique_character_count"] == len(payload["character_frequency"])
    assert payload["missing_from_fixture_vocabulary"] == []
    assert all(source["origin"] == "project-authored" for source in payload["sources"])
    for source in payload["sources"]:
        assert sha256(source["text"].encode()).hexdigest() == source["sha256"]


def test_realistic_pages_follow_current_vision_contracts() -> None:
    ground_truth = json.loads(
        (FIXTURE_ROOT / "page-ground-truth.json").read_text(encoding="utf-8")
    )["pages"]
    for expected_page in ground_truth:
        page = load_document(FIXTURE_ROOT / expected_page["image_path"])[0]
        processed = preprocess_page(page, PreprocessingConfig(deskew_enabled=True))
        regions = segment_page(processed)
        assert processed.is_blank is expected_page["expected_blank"]
        assert len(regions) == expected_page["expected_line_count"]
        for detected, expected in zip(regions, expected_page["lines"], strict=True):
            assert detected.reading_order_index == expected["reading_order_index"]
            assert detected.region_kind.value == expected["region_kind"]
            assert detected.column_index == expected["column_index"]
            key = (
                "composition_box"
                if expected_page["layout_family"] == PageLayoutFamily.MILD_SKEW.value
                else "output_box"
            )
            assert (
                detected.bounding_box.intersection_over_union(BoundingBox(**expected[key])) >= 0.75
            )


def test_fixture_regeneration_matches_frozen_records(tmp_path: Path) -> None:
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    first_root.mkdir()
    second_root.mkdir()
    result = generate_fixture_dataset(first_root)
    generate_fixture_dataset(second_root)
    committed_generation = json.loads(
        (FIXTURE_ROOT / "generation-manifest.json").read_text(encoding="utf-8")
    )
    regenerated_generation = json.loads(
        (first_root / "generation-manifest.json").read_text(encoding="utf-8")
    )
    committed_artifacts = json.loads(
        (FIXTURE_ROOT / "artifact-manifest.json").read_text(encoding="utf-8")
    )
    regenerated_artifacts = json.loads(
        (first_root / "artifact-manifest.json").read_text(encoding="utf-8")
    )

    assert result.line_count == 24
    assert result.page_count == 10
    for key in (
        "config_fingerprint",
        "dataset_fingerprint",
        "line_records",
        "page_records",
        "split",
        "vocabulary_fingerprint",
    ):
        assert _without_rendered_image_hashes(regenerated_generation[key]) == (
            _without_rendered_image_hashes(committed_generation[key])
        )
    for key in ("engine", "font_identifier", "font_sha256", "rtl_strategy"):
        assert regenerated_generation["shaping"][key] == committed_generation["shaping"][key]
    assert _portable_artifact_identity(regenerated_artifacts) == _portable_artifact_identity(
        committed_artifacts
    )

    # PNG bytes are deterministic within one renderer stack, but native FreeType/Pillow
    # rasterization is not byte-portable across the Windows reference and Linux CI stacks.
    assert _generated_png_hashes(first_root) == _generated_png_hashes(second_root)


def test_committed_pngs_contain_no_metadata() -> None:
    for path in sorted(FIXTURE_ROOT.rglob("*.png")):
        with Image.open(path) as image:
            assert image.info == {}
            assert image.getexif() == {}
            assert image.size[0] > 0 and image.size[1] > 0


def test_synthetic_config_rejects_invalid_ranges() -> None:
    with pytest.raises(ConfigurationError, match="contrast"):
        SyntheticDataConfig(background_intensity_min=80)
    with pytest.raises(ConfigurationError, match="at most 5"):
        SyntheticDataConfig(skew_max_abs_degrees=5.1)
    with pytest.raises(ConfigurationError, match="leave no column"):
        SyntheticDataConfig(page_width=800, page_margin=390, column_gutter=40)
    with pytest.raises(ConfigurationError, match="integer"):
        SyntheticDataConfig(seed=True)


def test_near_blank_has_only_bounded_nonsemantic_noise() -> None:
    page = generate_page_fixture(PageLayoutFamily.NEAR_BLANK, page_id="syn-page-near-blank-test")
    differences = page.record.background_intensity - page.image.astype(np.int16)
    assert np.count_nonzero(differences) == 8
    assert int(differences.max()) == 6
