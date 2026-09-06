# Urdu Document OCR

Foundation for a reproducible Urdu document OCR system with typed document, vision,
recognition, training, and evaluation contracts.

## Status

This is an active, clean public reimplementation built in incremental verified phases. Phase 4
implements the package foundation plus bounded PNG/JPEG/PDF ingestion and deterministic
classical preprocessing. It does **not** yet perform segmentation, layout analysis, OCR,
training, inference, evaluation, CLI business commands, or HTTP serving.

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

- Bounded filesystem-path and in-memory byte ingestion for PNG, JPEG, and PDF documents.
- EXIF-aware image decoding into owned RGB `uint8[height,width,3]` arrays without embedded
  metadata or absolute source paths.
- In-memory pypdfium2 rasterization with page-count, DPI, per-page pixel, and cumulative-pixel
  preflight checks.
- Deterministic grayscale, Otsu or Sauvola thresholding, and True-is-foreground polarity.
- Optional CLAHE, small median denoise, 3x3/5x5 opening or closing, and confidence-gated deskew.
- Three-signal blank-page classification that preserves every page and its original position.
- Half-open bounding-box geometry with intersection, containment, clipping, and IoU.
- Privacy-aware source, page, preprocessing, line-region, OCR-result, and dataset contracts.
- Deterministic CTC vocabulary indexing and SHA-256 fingerprints.
- Structural edit/evaluation result records; metric algorithms are not implemented yet.
- Strict input-limit, recognizer, and training configuration records.
- Portable relative dataset-path validation.
- Safe JSON-compatible projections that exclude image pixel arrays and local filesystem paths.
- Automated tests, Ruff checks, and a Python 3.11 CI workflow.

## Core API

```python
from urdu_document_ocr import PreprocessingConfig, load_document, preprocess_page

pages = load_document("scan.pdf")
processed = tuple(
    preprocess_page(page, PreprocessingConfig(threshold_method="sauvola")) for page in pages
)
```

The default preprocessing path uses Otsu and does not enable enhancement, morphology, or
deskew. See [architecture details](docs/architecture.md) for exact limits and decision rules.

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

1. Line segmentation and RTL layout ordering.
2. Data manifests, validation, grouped splitting, and vocabulary workflows.
3. Provenance-cleared synthetic Urdu fixtures.
4. Trainable CNN-BiLSTM-CTC recognition and safe checkpoints.
5. End-to-end inference, evaluation, CLI, and a thin reference API.
6. Reproducible synthetic benchmarks and final publication audit.

No production or accuracy claim is made at this stage. Repository licensing is intentionally
unresolved; public visibility does not itself grant reuse rights.
