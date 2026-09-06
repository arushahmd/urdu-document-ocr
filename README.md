# Urdu Document OCR

Foundation for a reproducible Urdu document OCR system with typed document, vision,
recognition, training, and evaluation contracts.

## Status

This is an active, clean public reimplementation built in incremental verified phases. Phase 4
implements the package foundation, domain contracts, configuration validation, path/privacy
safeguards, provenance records, tests, and continuous integration. It does **not** yet perform
OCR, image preprocessing, PDF rasterization, segmentation, training, inference, evaluation, or
HTTP serving.

## Planned architecture

```text
document
  -> validated ingestion
  -> preprocessing and blank-page detection
  -> line segmentation and one/two-column RTL ordering
  -> line crops
  -> CNN-BiLSTM-CTC recognition
  -> ordered line and page results
  -> document assembly
  -> evaluation, Python API, CLI, and reference HTTP API
```

Only the contracts shared by those layers exist today. Planned modules are added in their owning
phase, with tested behavior, rather than created as empty scaffolding.

## Current implemented scope

- Half-open bounding-box geometry with intersection, containment, clipping, and IoU.
- Privacy-aware source, page, preprocessing, line-region, OCR-result, and dataset contracts.
- Deterministic CTC vocabulary indexing and SHA-256 fingerprints.
- Structural edit/evaluation result records; metric algorithms are not implemented yet.
- Strict input-limit, recognizer, and training configuration records.
- Portable relative dataset-path validation.
- Safe JSON-compatible projections that exclude image pixel arrays and local filesystem paths.
- Automated tests, Ruff checks, and a Python 3.11 CI workflow.

## Provenance boundary

This repository is a clean, self-contained public reimplementation informed by earlier
professional work on Urdu/Nastaliq OCR. Historical datasets, trained weights, internal source
code, private infrastructure, and deployment assets are not included. Current code is newly
implemented; external algorithms, libraries, fonts, and public assets are cited and licensed as
applicable.

See the [source ledger](docs/provenance/source-ledger.md),
[behavioral reference record](docs/provenance/behavioral-references.md), and
[security policy](docs/security.md).

## Development setup

Python 3.11 or newer is required. The current dependency constraint intentionally retains Python
3.11 compatibility.

```console
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python -m ruff check .
python -m ruff format --check .
python -m pytest
```

## Roadmap

1. Document ingestion and preprocessing.
2. Line segmentation and RTL layout ordering.
3. Data manifests, validation, grouped splitting, and vocabulary workflows.
4. Provenance-cleared synthetic Urdu fixtures.
5. Trainable CNN-BiLSTM-CTC recognition and safe checkpoints.
6. End-to-end inference, evaluation, CLI, and a thin reference API.
7. Reproducible synthetic benchmarks and final publication audit.

No production or accuracy claim is made at this stage. Repository licensing is intentionally
unresolved; public visibility does not itself grant reuse rights.
