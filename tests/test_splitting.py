from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
from fixtures.data_samples import sample

from urdu_document_ocr import (
    DatasetPartition,
    DatasetSample,
    DatasetSplitConfig,
    DatasetSplitError,
    DatasetSplitResult,
    SplitCounts,
    dataset_fingerprint,
    read_manifest,
    split_dataset,
    write_split_manifests,
)
from urdu_document_ocr.config import ConfigurationError


def grouped_samples(group_sizes: dict[str, int]) -> tuple[DatasetSample, ...]:
    result = []
    index = 0
    for document_id, size in group_sizes.items():
        for _ in range(size):
            result.append(sample(index, document_id=document_id))
            index += 1
    return tuple(result)


def assignments(result: object) -> dict[str, str]:
    return {
        document_id: partition.value
        for document_id, partition in result.document_assignments  # type: ignore[attr-defined]
    }


def test_split_config_defaults_and_strict_validation() -> None:
    config = DatasetSplitConfig()

    assert (config.train_ratio, config.validation_ratio, config.test_ratio, config.seed) == (
        0.8,
        0.1,
        0.1,
        1337,
    )
    assert DatasetSplitConfig.from_mapping(config.to_dict()).fingerprint == config.fingerprint
    for overrides in (
        {"train_ratio": -0.1, "validation_ratio": 0.2, "test_ratio": 0.9},
        {"train_ratio": 0.8, "validation_ratio": 0.1, "test_ratio": 0.2},
        {"train_ratio": True, "validation_ratio": 0.0, "test_ratio": 0.0},
        {"seed": -1},
    ):
        with pytest.raises(ConfigurationError):
            DatasetSplitConfig(**overrides)  # type: ignore[arg-type]
    with pytest.raises(ConfigurationError, match="unknown dataset split"):
        DatasetSplitConfig.from_mapping({"future": 1})


def test_group_greedy_v1_has_locked_assignment_and_no_document_leakage() -> None:
    values = grouped_samples({"a": 4, "b": 3, "c": 2, "d": 1, "e": 1, "f": 1})

    result = split_dataset(values)

    assert assignments(result) == {
        "a": "train",
        "b": "train",
        "c": "train",
        "d": "train",
        "e": "validation",
        "f": "test",
    }
    split_documents = [
        {item.document_id for item in partition}
        for partition in (result.train, result.validation, result.test)
    ]
    assert split_documents[0].isdisjoint(split_documents[1])
    assert split_documents[0].isdisjoint(split_documents[2])
    assert split_documents[1].isdisjoint(split_documents[2])
    assert result.achieved_sample_counts.total == len(values)
    assert result.achieved_document_counts.total == 6
    assert sum(result.achieved_ratios.values()) == pytest.approx(1.0)


def test_split_is_stable_when_manifest_order_changes() -> None:
    values = grouped_samples({"alpha": 3, "beta": 3, "gamma": 2, "delta": 2, "epsilon": 1})

    forward = split_dataset(values)
    reverse = split_dataset(reversed(values))

    assert forward == reverse
    assert forward.fingerprint == reverse.fingerprint
    assert forward.dataset_fingerprint == dataset_fingerprint(values)


def test_different_seed_changes_equal_group_tie_assignments() -> None:
    values = grouped_samples({f"document-{index}": 1 for index in range(12)})

    first = split_dataset(values, DatasetSplitConfig(seed=1))
    second = split_dataset(values, DatasetSplitConfig(seed=2))

    assert assignments(first) != assignments(second)
    assert first.fingerprint != second.fingerprint


@pytest.mark.parametrize(
    "group_sizes",
    [
        {"only": 5},
        {"first": 1, "second": 1},
        {"large": 20, "small-a": 1, "small-b": 1, "small-c": 1},
        {f"equal-{index}": 2 for index in range(10)},
    ],
)
def test_small_and_imbalanced_group_sets_assign_every_sample_once(
    group_sizes: dict[str, int],
) -> None:
    values = grouped_samples(group_sizes)

    result = split_dataset(values)

    output = result.train + result.validation + result.test
    assert {item.sample_id for item in output} == {item.sample_id for item in values}
    assert len(output) == len(values)


def test_zero_ratio_partition_stays_empty() -> None:
    values = grouped_samples({"a": 2, "b": 2, "c": 2})
    config = DatasetSplitConfig(train_ratio=0.75, validation_ratio=0.25, test_ratio=0.0)

    result = split_dataset(values, config)

    assert result.test == ()
    assert all(
        partition is not DatasetPartition.TEST for _, partition in result.document_assignments
    )


def test_empty_dataset_has_stable_zero_ratios() -> None:
    result = split_dataset(())

    assert result.train == result.validation == result.test == ()
    assert result.achieved_ratios == {"train": 0.0, "validation": 0.0, "test": 0.0}
    assert len(result.fingerprint) == 64


def test_split_rejects_duplicate_ids_noncanonical_text_and_bad_config() -> None:
    duplicate = (sample(1), sample(1, document_id="other"))

    with pytest.raises(DatasetSplitError, match="unique"):
        split_dataset(duplicate)
    with pytest.raises(DatasetSplitError, match="normalization"):
        split_dataset((sample(1, text=" اردو"),))
    with pytest.raises(DatasetSplitError):
        split_dataset((object(),))  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        split_dataset((sample(1),), config="default")  # type: ignore[arg-type]


def test_write_split_manifests_writes_canonical_metadata_and_protects_outputs(
    tmp_path: Path,
) -> None:
    result = split_dataset(grouped_samples({"a": 4, "b": 2, "c": 1, "d": 1}))

    write_split_manifests(result, tmp_path)

    assert set(path.name for path in tmp_path.iterdir()) == {
        "train.jsonl",
        "validation.jsonl",
        "test.jsonl",
        "split_metadata.json",
    }
    assert tuple(read_manifest(tmp_path / "train.jsonl")) == result.train
    metadata = json.loads((tmp_path / "split_metadata.json").read_text(encoding="utf-8"))
    assert metadata["algorithm"] == "group-greedy-v1"
    assert metadata["split_fingerprint"] == result.fingerprint
    assert metadata["normalization_policy_version"] == "nfc-v1"
    assert set(metadata["output_sha256"]) == {
        "train.jsonl",
        "validation.jsonl",
        "test.jsonl",
    }
    with pytest.raises(DatasetSplitError, match="overwrite=True"):
        write_split_manifests(result, tmp_path)
    write_split_manifests(result, tmp_path, overwrite=True)
    with pytest.raises(DatasetSplitError, match="already exist"):
        write_split_manifests(result, tmp_path / "missing")


def test_split_outputs_are_byte_identical_across_reordered_inputs(tmp_path: Path) -> None:
    values = grouped_samples({"a": 3, "b": 3, "c": 2, "d": 1})
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    first_dir.mkdir()
    second_dir.mkdir()

    write_split_manifests(split_dataset(values), first_dir)
    write_split_manifests(split_dataset(reversed(values)), second_dir)

    for name in ("train.jsonl", "validation.jsonl", "test.jsonl", "split_metadata.json"):
        assert (first_dir / name).read_bytes() == (second_dir / name).read_bytes()


def test_split_result_contract_rejects_inconsistent_external_construction() -> None:
    values = grouped_samples({"a": 2, "b": 1})
    result = split_dataset(values)

    with pytest.raises(ValueError, match="nonnegative"):
        SplitCounts(-1, 0, 0)
    with pytest.raises(TypeError, match="config"):
        replace(result, config=object())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="algorithm"):
        replace(result, algorithm="future")
    with pytest.raises(TypeError, match="tuples"):
        replace(result, train=list(result.train))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="at most once"):
        replace(
            result,
            train=(values[0], values[0]),
            validation=(),
            test=(),
            document_assignments=(("a", DatasetPartition.TRAIN),),
            dataset_fingerprint=dataset_fingerprint((values[0], values[0])),
        )
    with pytest.raises(ValueError, match="unique"):
        replace(
            result,
            document_assignments=result.document_assignments + result.document_assignments[:1],
        )
    with pytest.raises(ValueError, match="match partition"):
        replace(result, document_assignments=())
    with pytest.raises(ValueError, match="fingerprint"):
        replace(result, dataset_fingerprint="0" * 64)


def test_split_result_rejects_document_crossing_partitions() -> None:
    first = sample(1, document_id="same")
    second = sample(2, document_id="same")

    with pytest.raises(ValueError, match="cannot cross"):
        DatasetSplitResult(
            train=(first,),
            validation=(),
            test=(second,),
            document_assignments=(("same", DatasetPartition.TEST),),
            config=DatasetSplitConfig(),
            dataset_fingerprint=dataset_fingerprint((first, second)),
        )


def test_split_writer_validates_result_and_destination(tmp_path: Path) -> None:
    with pytest.raises(TypeError):
        write_split_manifests(object(), tmp_path)  # type: ignore[arg-type]
    with pytest.raises(DatasetSplitError, match="invalid"):
        write_split_manifests(split_dataset(()), object())  # type: ignore[arg-type]
