# Urdu Document OCR

A reproducible Urdu document OCR system with typed document vision, CRNN training, safe
checkpoints, checkpoint-backed inference, exact evaluation, and frozen synthetic benchmarks.

## Status

This is an active, clean public reimplementation built in incremental verified phases. Through
Phase 13 it implements the package foundation, bounded PNG/JPEG/PDF ingestion, deterministic
classical preprocessing, classical-CV line segmentation, conservative one/two-column layout,
Urdu RTL reading order, reproducible OCR dataset engineering, and provenance-cleared synthetic
Urdu fixtures, a trainable PyTorch CNN-BiLSTM-CTC recognizer, a validated AdamW training path,
validation-loss selection, safetensors checkpoints, checkpoint-backed line recognition, and
document-level OCR with text/geometry output, standard CER/WER and exact-match evaluation,
deterministic error analysis, two frozen synthetic benchmark layers, an installed `urdu-ocr`
command, and a thin local/reference FastAPI adapter. No canonical trained model is published.

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
  -> exact evaluation and frozen synthetic benchmarks
  -> CLI and thin reference HTTP API
```

The recognizer, training, inference, assembly, evaluation, benchmark, CLI, and HTTP paths are
implemented. Modules are added in their owning phase with tested behavior rather than created as
empty scaffolding.

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
- Standard code-point CER, ASCII-space WER, exact match, deterministic edit alignment, corpus
  aggregation, ID-complete prediction evaluation, and aligned character error analysis.
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
- New benchmark-only synthetic document/line generation with frozen seeds, group-disjoint
  train/validation/test identities, train-only vocabulary, one-time test evaluation, and compact
  SHA-256-verifiable results without committed images or weights.
- A separately versioned `recognition-synthetic-v2` benchmark selected from external
  development-only evidence, frozen before one held-out test run, and reproducible without a
  committed checkpoint.
- Eight `argparse` commands for segmentation, data engineering, synthetic generation, training,
  checkpoint-backed recognition/evaluation, and local API serving, with shell-safe output and
  explicit exit behavior.
- A thin FastAPI factory with one fixed reusable recognizer, bounded in-memory multipart parsing,
  content-based document detection, thread-pool inference, safe errors, and `/health` plus `/ocr`.

## Quick start

Install the core package for document vision and data commands:

```console
python -m pip install -e .
urdu-ocr segment data/sample/pages/syn-page-002-balanced_two_column.png
```

Install ML support to train, recognize, and evaluate:

```console
python -m pip install -e ".[ml]"
urdu-ocr train --train-manifest data/sample/splits/train.jsonl \
  --validation-manifest data/sample/splits/validation.jsonl --dataset-root data/sample \
  --vocabulary data/sample/synthetic-fixture-vocabulary.json --output-dir run-output
urdu-ocr recognize data/sample/pages/syn-page-001-one_column.png \
  --checkpoint run-output/best --format json --output recognized.json
urdu-ocr evaluate --manifest data/sample/splits/test.jsonl --dataset-root data/sample \
  --checkpoint run-output/best --output evaluation.json
```

Install the local reference service dependencies alongside ML support:

```console
python -m pip install -e ".[ml,api]"
urdu-ocr serve --checkpoint run-output/best
```

No pretrained or canonical OCR checkpoint is bundled. Recognition, evaluation, and serving need
a compatible model trained by the user or supplied from a separately trusted source. The
committed benchmark numbers are synthetic engineering evidence, not real-document accuracy.
See the complete [CLI reference](docs/cli.md) and [API boundary](docs/api.md).

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

The optional ML extra also provides the training API documented in
[training and checkpoints](docs/training.md). It requires explicit manifests, vocabulary, dataset
root, and output directory; the CLI calls this same API.

Checkpoint-backed inference loads the model once for reuse:

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

## Synthetic benchmarks

These are frozen engineering benchmarks over deterministic repository-generated data. They do
not establish accuracy on historical material, scanned books, handwriting, arbitrary real
documents, or production workloads. No trained weights are bundled.

**Synthetic document-vision benchmark (`vision-synthetic-v1`)**

| Metric | Frozen result |
|---|---:|
| Pages | 13 |
| Blank-state classifications | 13/13 |
| Exact line-count pages | 12/13 |
| Matched regions | 46/70 expected; 66 detected |
| Region precision | 0.696969696969697 |
| Region recall | 0.6571428571428571 |
| Region F1 | 0.676470588235294 |
| Mean matched IoU | 0.8171916765145802 |
| Exact kind assignments | 46/46 matched |
| Exact column assignments | 46/46 matched |
| Exact reading-order pages | 3/13 |
| Pages retaining failures | 10/13 |

**Synthetic OCR recognition benchmark (`recognition-synthetic-v1`)**

| Metric | Frozen held-out test result |
|---|---:|
| Test lines | 32 |
| Corpus CER | 1.0 (100.00%) over 1,256 reference code points |
| Corpus WER | 1.0 (100.00%) over 232 reference words |
| Exact line match | 0/32 (0.00%) |
| Character S/D/I | 0 / 1,256 / 0 |
| Word S/D/I | 0 / 232 / 0 |

The recognition model selected at epoch 3 by minimum validation CTC loss decoded every frozen
test line as empty after the deliberately bounded three-epoch CPU training protocol. That poor
result is retained without post-test tuning.

**Current synthetic OCR recognition benchmark (`recognition-synthetic-v2`)**

| Metric | Frozen held-out test result |
|---|---:|
| Test lines | 16 |
| Corpus CER | 0.5686274509803921 (56.86%) over 306 reference code points |
| Corpus WER | 0.7777777777777778 (77.78%) over 54 reference words |
| Exact line match | 0/16 (0.00%) |
| Character S/D/I | 16 / 158 / 0 |
| Word S/D/I | 20 / 22 / 0 |
| Empty predictions | 0/16 (0.00%) |

V2 retains the same recognizer, CTC/decoder semantics, validation-loss selection, and
`ocr-edit-v1` metrics. A development-only study established tiny-set exact matches and selected
the 30-epoch, 240-update protocol before V2 identities were generated. V2 was then independently
frozen and evaluated once; its checkpoint was deleted. The known aggregate V1 failure motivated
the study, but no V1 sample-specific test prediction and no V2 test result was used for tuning.
The comparison demonstrates learnability only within these repository-generated synthetic
domains—it is not a real-document accuracy claim. See the
[V2 benchmark record](benchmark/recognition-synthetic-v2/README.md), the byte-preserved
[initial benchmark record](benchmark/README.md), and [metric definitions](docs/evaluation.md).

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
python -m pip install -e ".[ml,api,dev]"
python -m ruff check .
python -m ruff format --check .
python -m pytest
```

Core document/data functionality intentionally does not require PyTorch or FastAPI. Smaller
development matrices remain supported:

```console
python -m pip install -e ".[dev]"
python -m pip install -e ".[ml,dev]"
python -m pip install -e ".[api,dev]"
```

CPU-only CI first resolves the official PyTorch CPU wheel channel and then installs the full
extras. Runtime code never downloads weights. Training and inference default to CPU and accept an
explicit CUDA device; an unavailable configured CUDA device fails without fallback.

## Roadmap

1. Final recruiter documentation and publication audit.

Only the explicitly labeled synthetic benchmark results above are claimed; no real-world,
production, or external-comparison claim is made. Repository licensing is intentionally
unresolved; public visibility does not itself grant reuse rights.
