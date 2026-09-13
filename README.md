# Urdu Document OCR

End-to-end Urdu document OCR with classical-CV RTL layout analysis, a trainable PyTorch
CNN-BiLSTM-CTC recognizer, reproducible data and training, checkpoint-backed inference,
evaluation, a command-line interface, and a thin FastAPI adapter.

![Synthetic Urdu document with detected reading order](data/sample/visual/synthetic-layout-overlay.png)

*Synthetic Urdu document/layout example with detected reading order; not a production scan.*

## What this demonstrates

This repository implements the engineering path around an OCR model, not only the neural network:
bounded document ingestion, deterministic vision and RTL layout, Unicode-aware dataset contracts,
leakage-safe splitting, variable-width CTC training, verified checkpoints, document assembly,
evaluation, reproducible synthetic evidence, packaging, CLI/API boundaries, and isolated CI.

The implementation is a clean public reimplementation. It contains no historical source, private
datasets, pretrained weights, or deployment infrastructure.

## Core capabilities

- Bounded in-memory PNG, JPEG, and PDF ingestion with page, DPI, and pixel limits.
- Deterministic preprocessing, blank-page detection, connected-component line segmentation, and
  conservative one/two-column RTL reading order with spanning regions.
- Canonical UTF-8 JSONL manifests, root-contained image validation, document-grouped splits, and
  train-only Unicode vocabularies.
- Provenance-cleared synthetic Urdu rendered with reviewed HarfBuzz/FreeType Nastaliq shaping.
- A variable-width PyTorch CNN-BiLSTM-CTC recognizer with explicit valid widths, dynamic padding,
  repeated-label feasibility checks, and greedy CTC decoding.
- AdamW training, validation-loss model selection, early stopping, deterministic loaders, and
  safetensors checkpoints with strict JSON metadata and SHA-256 verification.
- Checkpoint-backed line, page, and document inference with ordered TXT and JSON assembly.
- Standard code-point CER, space-tokenized WER, exact match, edit alignment, and error analysis.
- Frozen synthetic vision and recognition benchmarks whose failures remain part of the record.
- An installed `urdu-ocr` CLI and a bounded local/reference FastAPI adapter.

## Architecture

```text
Image/PDF
  -> bounded ingestion -> preprocessing -> line segmentation + RTL layout
  -> variable-width line crops -> CNN -> BiLSTM -> CTC decode
  -> page/document assembly -> TXT/JSON

Manifest
  -> validation -> document-grouped split -> train-only vocabulary
  -> AdamW training -> safetensors checkpoint -> inference/evaluation
```

The CLI and HTTP API are adapters over these same library contracts; neither contains a second OCR,
segmentation, training, or metric implementation. See the [architecture](docs/architecture.md) for
the exact component and failure boundaries.

## Quick start

Core document vision and data tooling:

```console
python -m pip install -e .
urdu-ocr segment data/sample/pages/syn-page-002-balanced_two_column.png
```

Training, recognition, and evaluation:

```console
python -m pip install -e ".[ml]"
urdu-ocr train --train-manifest data/sample/splits/train.jsonl --validation-manifest data/sample/splits/validation.jsonl --dataset-root data/sample --vocabulary data/sample/synthetic-fixture-vocabulary.json --output-dir run-output
urdu-ocr recognize data/sample/pages/syn-page-001-one_column.png --checkpoint run-output/best --format json --output recognized.json
urdu-ocr evaluate --manifest data/sample/splits/test.jsonl --dataset-root data/sample --checkpoint run-output/best --output evaluation.json
```

Local API service:

```console
python -m pip install -e ".[ml,api]"
urdu-ocr serve --checkpoint run-output/best
```

No pretrained or canonical OCR checkpoint is bundled. Recognition, evaluation, and serving require
a compatible checkpoint trained by the user or supplied from a separately trusted source. See the
[CLI reference](docs/cli.md), [training guide](docs/training.md),
[inference contract](docs/inference.md), and [API boundary](docs/api.md).

## Training and checkpoints

Training consumes explicit train/validation manifests, a dataset root, and a vocabulary. It checks
dataset separation, image safety, vocabulary coverage, CTC feasibility, and model-width limits
before optimization. The trainer uses fixed-rate AdamW, sample-weighted validation loss, gradient
clipping, and deterministic best-validation-loss selection.

Checkpoints contain safetensors model weights plus validated JSON metadata and vocabulary. They do
not contain executable pickle data or optimizer state, so exact training resume is not advertised.
No checkpoint is committed to this repository.

## Recognition and outputs

`load_recognizer` validates a compatible checkpoint once for reuse. `recognize_document` composes
the existing ingestion, preprocessing, segmentation, RTL ordering, batched recognition, and
assembly stages. Blank pages retain their original indexes. TXT uses one newline between lines and
two between pages; JSON preserves geometry, order, text, and public result fields without serializing
pixels, tensors, local paths, or uncalibrated confidence.

The public Python examples are [image recognition](examples/recognize_image.py) and
[PDF recognition](examples/recognize_pdf.py).

## Evaluation and synthetic benchmarks

Reproducible synthetic benchmarks exercise the complete vision and train/evaluate workflows. The
initial recognition V1 exposed early CTC blank collapse; the independently frozen V2 demonstrated
synthetic-domain learnability without changing the recognizer, trainer, decoder, or metric policy.
Failures and denominators are retained rather than filtered.

These are engineering checks on repository-generated synthetic data—not evidence of accuracy on
scanned books, handwriting, historical material, or arbitrary real documents. Detailed metrics,
predictions, configurations, hashes, and reproduction instructions remain in the
[V1 benchmark record](benchmark/README.md),
[V2 benchmark record](benchmark/recognition-synthetic-v2/README.md), and
[evaluation documentation](docs/evaluation.md). V2's `56.86%` value is CER, not accuracy.

## CLI and local API

The `argparse` CLI exposes eight focused commands:

```text
recognize  segment  train  evaluate
validate-data  split-data  generate-synthetic  serve
```

Successful result data goes to stdout, diagnostics go to stderr, and documented exit codes support
shell composition. The FastAPI adapter exposes `GET /health` and multipart `POST /ocr`, loads one
fixed recognizer at startup, bounds uploads in chunks, distrusts declared MIME type, performs OCR in
a worker thread, and retains nothing after the request.

The service binds to `127.0.0.1:8000` by default. It has no authentication, permissive CORS, TLS,
rate limiting, malware isolation, database, or job system and is not a hardened public deployment.

## Repository structure

| Path | Purpose |
|---|---|
| `src/urdu_document_ocr/` | Typed document, vision, data, recognition, training, evaluation, CLI, and API modules |
| `data/sample/` | Small reviewed synthetic fixture package with hashes and provenance |
| `benchmark/` | Frozen compact benchmark inputs, results, predictions, and verification records |
| `examples/` | Checkpoint-backed image and PDF inference examples |
| `scripts/` | Fixture generation and explicit benchmark reproduction/verification entry points |
| `tests/` | Contract, resource, Unicode, layout, ML, API, and integration tests |
| `docs/` | Detailed architecture, data, model, security, and interface documentation |
| `.github/workflows/` | Read-only Python 3.11 core, API-only, and full CPU-ML CI matrices |

## Testing and reproducibility

```console
python -m pip install -e ".[ml,api,dev]"
python -m ruff check .
python -m ruff format --check .
python -m pytest --cov=urdu_document_ocr --cov-fail-under=85
python scripts/run_benchmark.py verify
python scripts/run_recognition_benchmark_v2.py verify
```

CI separates core, API-without-PyTorch, and full CPU-ML validation. Ordinary verification
recalculates benchmark metrics from committed predictions and does not repeat the longer V2
training run. Runtime code downloads neither fonts nor model weights.

Regenerate the reviewed public fixture package without network access:

```console
python scripts/generate_sample_data.py --output data/sample --overwrite
```

## Limitations

- No pretrained or canonical OCR checkpoint is bundled.
- The committed benchmarks are synthetic; real scanned-book/document generalization is not
  established.
- Supported layout scope is one/two-column text with geometric spanning regions.
- Handwriting, tables, forms, and arbitrary complex layouts are outside V1.
- Recognition has no calibrated confidence output.
- The FastAPI adapter is local/reference software, not a hardened public deployment.
- Checkpoints intentionally omit optimizer state and therefore do not support exact resume.

## Documentation

- [Architecture](docs/architecture.md)
- [Data format](docs/data-format.md)
- [Synthetic data and shaping](docs/synthetic-data.md)
- [Recognizer](docs/recognizer.md)
- [Training and checkpoints](docs/training.md)
- [Inference](docs/inference.md)
- [Evaluation and benchmarks](docs/evaluation.md)
- [CLI](docs/cli.md)
- [API](docs/api.md)
- [Security and privacy boundary](docs/security.md)
- [Dependency and license review](docs/provenance/dependency-review.md)
- [Source provenance ledger](docs/provenance/source-ledger.md)

## Provenance and licensing

This repository is a clean, self-contained public reimplementation informed by earlier
professional work on Urdu/Nastaliq OCR. Historical datasets, trained weights, internal source,
private infrastructure, and deployment assets are not included. Current code is newly implemented;
external algorithms, libraries, and assets are documented in the
[source ledger](docs/provenance/source-ledger.md).

No project-wide source-code license is currently provided. The bundled Noto Nastaliq Urdu font is
separately licensed under OFL-1.1; its unmodified asset, license, and provenance record are packaged
together.
