# Urdu Document OCR

A reproducible Urdu document OCR system with typed document vision, CRNN training, safe
checkpoints, and structured checkpoint-backed inference.

## Status

This is an active, clean public reimplementation built in incremental verified phases. Through
Phase 11 it implements the package foundation, bounded PNG/JPEG/PDF ingestion, deterministic
classical preprocessing, classical-CV line segmentation, conservative one/two-column layout,
Urdu RTL reading order, reproducible OCR dataset engineering, and provenance-cleared synthetic
Urdu fixtures, a trainable PyTorch CNN-BiLSTM-CTC recognizer, a validated AdamW training path,
validation-loss selection, safetensors checkpoints, checkpoint-backed line recognition, and
document-level OCR with text/geometry output. No canonical trained model is published. The
project does **not** yet provide OCR quality results, CER/WER, CLI business commands, or HTTP
serving.

## Architecture

```text
document
  -> validated ingestion
  -> preprocessing and blank-page detection
  -> line segmentation and one/two-column RTL ordering
  -> ordered grayscale line crops
  -> checkpoint-backed CNN-BiLSTM-CTC recognition
  -> greedy CTC decode
  -> ordered line and page results
  -> document TXT/JSON assembly
  -> evaluation, CLI, and reference HTTP API (planned)
```

The recognizer, training, inference, and assembly paths are implemented. Evaluation and the
CLI/HTTP adapters remain planned. Modules are added in their owning phase with tested behavior
rather than created as empty scaffolding.

## Current implemented scope

- Bounded filesystem-path and in-memory byte ingestion for PNG, JPEG, and PDF documents.
- EXIF-aware image decoding into owned RGB `uint8[height,width,3]` arrays without embedded
  metadata or absolute source paths.
- In-memory pypdfium2 rasterization with page-count, DPI, per-page pixel, and cumulative-pixel
  preflight checks.
- Deterministic grayscale, Otsu or Sauvola thresholding, and True-is-foreground polarity.
- Optional CLAHE, small median denoise, 3x3/5x5 opening or closing, and confidence-gated deskew.
- Three-signal blank-page classification that preserves every page and its original position.
- Eight-connected foreground-component analysis with robust, scale-relative line grouping.
- Conservative attachment of nearby dot/diacritic-like marks and filtering of isolated specks.
- One-column and persistent-gutter two-column layout inference with right-column-first ordering.
- Geometric spanning regions placed in vertical reading bands, including headings and footers.
- Owned grayscale or foreground line-crop extraction from immutable `LineRegion` geometry.
- Strict canonical UTF-8 JSONL manifests with root-contained PNG/JPEG validation.
- Deterministic document-grouped train/validation/test splits with achieved-ratio evidence.
- Train-derived Unicode vocabulary construction, persistence, and unseen-character reporting.
- Immutable, source-fingerprinted review overlays with no image relocation or deletion.
- Half-open bounding-box geometry with intersection, containment, clipping, and IoU.
- Privacy-aware source, page, preprocessing, line-region, OCR-result, and dataset contracts.
- Deterministic CTC vocabulary indexing and SHA-256 fingerprints.
- Structural edit/evaluation result records; metric algorithms are not implemented yet.
- Strict input-limit, recognizer, and training configuration records.
- Portable relative dataset-path validation.
- Safe JSON-compatible projections that exclude image pixel arrays and local filesystem paths.
- Automated tests, Ruff checks, and a Python 3.11 CI workflow.
- Deterministic variable-width Urdu line rendering through reviewed HarfBuzz/FreeType shaping.
- Ten synthetic page/layout families with half-open ground truth, including RTL columns,
  spanning bands, blank pages, near-blank pages, and mild skew.
- A small safe public fixture package with project-authored text, grouped splits, a train-only
  synthetic vocabulary, provenance records, and SHA-256 artifact inventory.
- A trainable, randomly initialized PyTorch CNN-BiLSTM-CTC model accepting explicit valid widths.
- Four GroupNorm/SiLU CNN blocks with exact width-to-timestep geometry, mean height collapse,
  packed two-layer bidirectional LSTM sequences, and a vocabulary-bound character classifier.
- Strict model-input normalization to grayscale `float32[1,64,W]` in `[-1,+1]`, with no silent
  squeezing, cropping, or truncation of over-width lines.
- Mean CTC loss with explicit repeated-label feasibility checks and vocabulary-bound greedy CTC
  decoding that emits no fabricated confidence.
- A root-contained `OCRLineDataset`, dynamic white-padded collation, deterministic DataLoaders,
  fixed-rate AdamW optimization, mutation-free validation, and exact early stopping.
- Best-validation-loss checkpoints with safetensors weights, strict JSON metadata/vocabulary,
  SHA-256 integrity checks, atomic replacement, and no serialized optimizer state.
- Reusable checkpoint loading with exact model/configuration/vocabulary validation, explicit CPU
  or available-CUDA selection, evaluation mode, and inference mode.
- Single and dynamically batched grayscale line recognition using the same normalization and
  white-right-padding contract as training, with order-preserving greedy CTC decoding.
- End-to-end image/PDF OCR orchestration that preserves Phase 6 RTL/spanning order and retains
  blank pages at their original indexes.
- Immutable `LineOCRResult`, `PageOCRResult`, and `DocumentOCRResult` assembly with one newline
  between lines and two newline separators between every adjacent page.
- Deterministic Unicode-preserving TXT and JSON projections plus explicit, atomic, no-overwrite
  output helpers that serialize no images, tensors, local paths, or model internals.

## Core API

```python
from urdu_document_ocr import (
    PreprocessingConfig,
    SegmentationConfig,
    load_document,
    preprocess_page,
    segment_page,
)

pages = load_document("scan.pdf")
processed = tuple(
    preprocess_page(page, PreprocessingConfig(threshold_method="sauvola")) for page in pages
)
regions = tuple(segment_page(page, SegmentationConfig()) for page in processed)
```

The default preprocessing path uses Otsu and does not enable enhancement, morphology, or
deskew. Segmentation consumes only the normalized foreground mask and returns ordered geometry,
not recognized text. See [architecture details](docs/architecture.md) for exact limits and
decision rules, [recognizer details](docs/recognizer.md) for the model contract, and
[OCR data format](docs/data-format.md) for labeled-data contracts.

The optional ML extra also provides the Python-only training API documented in
[training and checkpoints](docs/training.md). It requires explicit manifests, vocabulary, dataset
root, and output directory; no user-facing training CLI exists yet.

Checkpoint-backed inference is also a Python API and loads the model once for reuse:

```python
from urdu_document_ocr import (
    document_to_json,
    load_recognizer,
    recognize_document,
)

recognizer = load_recognizer("run-output/best", device="cpu")
result = recognize_document("scan.pdf", recognizer)
json_text = document_to_json(result)
```

The caller must supply a compatible checkpoint; the repository does not bundle or download one.
See [checkpoint-backed inference](docs/inference.md) for batching, blank-page, failure, text, and
JSON contracts. The examples under `examples/` call the same public API and require explicit input,
checkpoint, and output paths.

Dataset operations are ordinary library APIs:

```python
from urdu_document_ocr import (
    build_vocabulary,
    read_manifest,
    split_dataset,
    validate_dataset,
)

samples = read_manifest("manifest.jsonl")
report = validate_dataset(samples, dataset_root="dataset")
split = split_dataset(samples)
training_vocabulary = build_vocabulary(split.train)
```

Segmented document lines do not acquire transcriptions automatically. Dataset manifests describe
separately labeled line images supplied by the caller.

Synthetic fixtures can be regenerated without network access:

```console
python scripts/generate_sample_data.py --output data/sample --overwrite
```

The command writes only to the explicit destination and rejects unrelated existing files. See
[synthetic data](docs/synthetic-data.md) and the [fixture package](data/sample/README.md).

![Synthetic Urdu layout overlay](data/sample/visual/synthetic-layout-overlay.png)

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

Core document/data functionality intentionally does not require PyTorch. Install the reviewed ML
extra to use and test recognition and training:

```console
python -m pip install -e ".[ml,dev]"
```

CPU-only CI first resolves the official PyTorch CPU wheel channel and then installs the extra.
Runtime code never downloads weights. Training and inference default to CPU and accept an explicit
CUDA device; an unavailable configured CUDA device fails without fallback.

## Roadmap

1. CER/WER evaluation, error analysis, and reproducible benchmark freeze.
2. CLI and thin reference API adapters over the existing Python pipeline.
3. Final recruiter documentation and publication audit.

No OCR accuracy, CER, WER, benchmark, production, or performance claim is made at this stage.
Evaluation remains Phase 12 and CLI/API adapters remain Phase 13. Repository licensing is
intentionally unresolved; public visibility does not itself grant reuse rights.
