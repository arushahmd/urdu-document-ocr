# Synthetic OCR recognition benchmark V2

`recognition-synthetic-v2` is the current, independently frozen synthetic recognition benchmark.
It coexists with the byte-preserved initial `recognition-synthetic-v1` benchmark at the parent
`benchmark/` path. The parent benchmark README is itself a V1 hashed artifact and therefore is
intentionally not edited in Phase 12B; this version-local document supplies the required V2
benchmark documentation without invalidating history.

## Selection and firewall

The unchanged `crnn-gn-silu-bilstm-ctc-v1` architecture was first tested only on newly generated
development train/validation data. Those experiments showed a temporary early CTC blank-collapse
phase followed by substantial learning over hundreds of updates. The selected protocol was
written to `provenance.json` before V2 logical identities were generated. No V2 test prediction,
metric, or image existed during selection, and V1 individual test identities/predictions were not
used for tuning.

V2 has 160 deterministic lines grouped into 20 synthetic documents. Each document contains eight
new, concise, programmatically composed Urdu templates spanning joining, spaces, punctuation,
Urdu/ASCII digits, ZWNJ, and selected diacritics. The grouped split targets 80% train, 10%
validation, and 10% test with zero cross-partition document, sample, or exact-text overlap. The
vocabulary is derived from training text only.

## Frozen training protocol

- Architecture: `crnn-gn-silu-bilstm-ctc-v1`, normalized height 64, maximum width 1024, blank 0.
- Generator: deterministic `synthetic-v1` shaping with a mild degradation profile.
- Optimizer: existing fixed-rate AdamW trainer; LR 0.001, batch 16, weight decay 0.0001,
  gradient clipping at 5.0.
- Budget: at most 30 epochs / 240 optimizer updates; patience 8; CPU; zero workers.
- Selection: minimum validation CTC loss only.
- Evaluation: one test inference after freeze, using unchanged `ocr-edit-v1` CER/WER/exact-match
  metrics and the existing deterministic error analyzer.

The compact committed results include predictions and sufficient identities to recalculate every
metric. Generated images and the best safetensors checkpoint are temporary and are not published.

## Explicit reproduction

To verify the committed compact artifacts from a fresh CPython 3.11 environment with the reviewed
ML and development extras installed, run:

```console
python scripts/run_recognition_benchmark_v2.py verify
```

For full reproduction, use a separate checkout/copy with V2 results absent, create an empty
external work directory, and run exactly:

```console
python scripts/run_recognition_benchmark_v2.py run --work-directory ../empty-v2-work
```

The `run` command is deliberately guarded against an existing results directory, so the committed
benchmark cannot be rerun in place. Ordinary CI uses `verify`: it checks hashes and recalculates
results from committed predictions without retraining. The reviewed run used CPython 3.11.9,
PyTorch 2.14.0+cpu, NumPy 2.3.5, OpenCV 4.14.0, Pillow 12.3.0, uharfbuzz 0.56.1, and freetype-py
2.5.1. Training took 1,767.43 seconds and test inference plus metrics took 2.17 seconds.

## Frozen result

- Test lines: 16
- CER: 0.5686274509803921 (56.86%); character S/D/I: 16 / 158 / 0
- WER: 0.7777777777777778 (77.78%); word S/D/I: 20 / 22 / 0
- Exact line matches: 0/16
- Empty predictions: 0/16
- Mean prediction/reference lengths: 9.25 / 19.125 Unicode code points

This is a **SYNTHETIC OCR RECOGNITION BENCHMARK** only. It is not evidence of accuracy on scanned
books, historical sources, handwriting, arbitrary documents, or production inputs.
