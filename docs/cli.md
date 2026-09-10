# Command-line interface

The `urdu-ocr` entry point is a compact adapter over the package APIs. It performs no model,
font, or dataset downloads. Core commands work without PyTorch or FastAPI; model commands need
the `ml` extra, and `serve` needs both `ml` and `api`.

```console
python -m pip install -e .
python -m pip install -e ".[ml]"
python -m pip install -e ".[api]"
python -m pip install -e ".[ml,api]"
```

`urdu-ocr --help` and `urdu-ocr --version` require only the core installation. Requested result
data goes to standard output. Diagnostics go to standard error. Exit code `0` means success,
`2` means an argument, configuration, input, checkpoint, or dataset error, and `1` means an
unexpected runtime failure. Ordinary expected failures do not print tracebacks.

## Commands

### `recognize`

Recognize a bounded PNG, JPEG, or PDF with an existing compatible checkpoint.

```console
urdu-ocr recognize scan.pdf --checkpoint run-output/best --format json --output result.json
```

The default format is `text`. `--format json` uses the same safe document projection as the
Python API. With no `--output`, the selected projection goes to standard output. File output is
atomic and refuses replacement unless `--overwrite` is explicit. `--device`, `--batch-size`, and
`--pdf-dpi` map directly to existing inference and ingestion contracts. No checkpoint path,
pixels, tensors, or confidence value appears in result JSON.

### `segment`

Run bounded ingestion, default preprocessing, and line segmentation without a model.

```console
urdu-ocr segment data/sample/pages/syn-page-002-balanced_two_column.png
```

The JSON contains page indices, one-based source page numbers, blank decisions, deskew angle,
line boxes, region kinds, column assignments, and reading-order indices. It contains no image or
mask arrays. Use `--output`, `--pdf-dpi`, and explicit `--overwrite` when needed.

### `train`

Train the current CNN-BiLSTM-CTC model through the Phase 10 trainer.

```console
urdu-ocr train --train-manifest splits/train.jsonl \
  --validation-manifest splits/validation.jsonl --dataset-root dataset \
  --vocabulary vocabulary.json --output-dir run-output --epochs 20
```

Required inputs are the train and validation manifests, dataset root, vocabulary, and new/empty
output directory. Options expose only current `TrainingConfig` fields: device, epochs, batch
size, learning rate, weight decay, gradient clipping, seed, patience, minimum improvement,
worker count, maximum image width, and relative checkpoint directory. The JSON summary includes
loss history, best epoch/loss, optimization steps, early-stopping state, safe checkpoint
directory name, and dataset fingerprints. Checkpoints contain weights and validated metadata,
not optimizer state; exact resume is not supported.

### `evaluate`

Evaluate a checkpoint against the caller-supplied manifest with the existing CER, WER,
exact-match, and error-analysis implementation.

```console
urdu-ocr evaluate --manifest splits/test.jsonl --dataset-root dataset \
  --checkpoint run-output/best --output evaluation.json
```

The deterministic JSON includes per-sample evidence, corpus character and word edit counts and
rates, exact-match count/rate, identities, and error analysis. It does not infer whether a
dataset is synthetic or real. Output replacement is opt-in.

### `validate-data`

```console
urdu-ocr validate-data --manifest manifest.jsonl --dataset-root dataset
```

The validator reports sample/document statistics and every error or warning. Validation errors
still produce the JSON report but return exit code `2`; warnings alone return `0`.

### `split-data`

```console
mkdir splits
urdu-ocr split-data --manifest manifest.jsonl --output-dir splits --seed 1337
```

The existing document-grouped splitter writes `train.jsonl`, `validation.jsonl`, `test.jsonl`,
and `split_metadata.json` into an existing directory. Ratios, seed, and explicit overwrite are
configurable. The standard-output summary reports achieved—not assumed—sample/document counts,
ratios, and the split fingerprint.

### `generate-synthetic`

```console
mkdir generated-lines
urdu-ocr generate-synthetic --output-dir generated-lines --count 32 --seed 240817
```

This uses the reviewed bundled Noto Nastaliq Urdu font, current project-authored text, shaping
gate, and Phase 8 renderer to produce line images, `manifest.jsonl`, and
`generation-manifest.json`. The directory must already exist, be non-root, and contain no
unrelated files. No network access occurs. This command does not generate or train the frozen
recognition benchmark.

### `serve`

```console
urdu-ocr serve --checkpoint run-output/best
```

The default bind is `127.0.0.1:8000`. `--host`, `--port`, `--device`, and `--batch-size` are
startup settings. The checkpoint is verified and loaded once before Uvicorn starts. Missing
optional packages produce installation guidance. See [the API boundary](api.md) before choosing
an explicitly network-visible host.

No pretrained or canonical OCR checkpoint is bundled. Recognition, evaluation, and serving
require a compatible checkpoint trained by the user or supplied from a separately trusted
source.
