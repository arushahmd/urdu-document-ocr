# Frozen synthetic benchmarks

This directory contains the compact, reviewed inputs and outputs for two explicitly synthetic
benchmarks:

- `vision-synthetic-v1`: deterministic document preprocessing, line segmentation, supported
  one/two-column layout, and Urdu RTL reading order.
- `recognition-synthetic-v1`: deterministic rendered Urdu/Nastaliq line recognition with a
  CNN-BiLSTM-CTC model trained from scratch.

These results are engineering evidence for the repository's reproducible workflow. They are not
real-document, historical-scan, handwriting, production, or independent external accuracy
claims.

## Frozen data

All source text is newly project-authored or programmatically composed. The vision set covers 13
fixed families: one column, unequal-width one column, balanced and uneven two columns, a small
valid gutter, three spanning-region positions, mild skew, blur/noise, low contrast, blank, and
near blank. Images are regenerated from the reviewed bundled font and recorded by SHA-256; they
are not committed.

The recognition set has 64 synthetic document groups with four unique lines each (256 lines).
Document-grouped splitting produces 192 training, 32 validation, and 32 held-out test lines in
48/8/8 document groups. No sample ID, document ID, or exact transcription crosses partitions.
Vocabulary is derived only from training text, with zero unseen validation/test characters. The
shared bounded generator family varies margins, font size, foreground/background, mild blur,
Gaussian noise, and modest skew; it adds no perspective or mobile-camera distortion.

Training uses the existing Phase 10 AdamW trainer. The best checkpoint is selected only by minimum
validation CTC loss. Test predictions had no role in thresholds, hyperparameters, vocabulary,
selection, or regeneration. The temporary checkpoint is removed after final result serialization
and is never public model weight.

## Freeze and interpretation

`freeze.json` was written after source text, seeds, partitions, font hash, configs, logical
manifests, vocabulary, and evaluation policy were final. Its source hashes must verify before a
final run starts. A frozen version cannot be changed because results are disappointing; an
infrastructure defect requires an explicit invalidation and a new version/freeze.

Raw rates and counts are stored without presentation rounding. Vision failures remain in
per-case output, and all recognition references/hypotheses remain in per-sample output because
the text is public-safe. `artifact_manifest.json` verifies all reviewed configs, manifests,
provenance, results, summaries, and the reproduction script. See `results/summary.json` for a
compact result and `../docs/evaluation.md` for exact metric denominators.

## Frozen results

The document-vision suite classified blank state correctly on 13/13 pages and found exact line
counts on 12/13. It matched 46 of 70 expected regions against 66 detections: precision
0.696969696969697, recall 0.6571428571428571, F1 0.676470588235294, and mean matched IoU
0.8171916765145802. Kind and column were correct on all 46 matches; reading order was exact on
3/13 pages. Ten pages retain one or more explicit failures.

The recognition suite evaluated 32 held-out lines (1,256 reference code points and 232 reference
words). It produced CER 1.0, WER 1.0, and 0/32 exact matches, with character S/D/I of
0/1,256/0 and word S/D/I of 0/232/0. The three-epoch model emitted empty text for every test
line. This poor synthetic result is preserved as generated and did not trigger retuning.

## Reproduction

Install the ML and development extras, create two existing empty directories for generated work
and reproduced results, then run explicitly:

```console
python scripts/run_benchmark.py run --work-directory build/benchmark-work --results-directory build/benchmark-results
```

The command verifies the freeze, regenerates all images, verifies their logical identities,
runs the vision suite, trains once on CPU, selects by validation loss, evaluates the held-out test
set, writes compact JSON, and deletes generated images/checkpoints from the work directory. Allow
roughly nine minutes on the reference CPU; hardware-dependent timing is engineering context, not
a latency guarantee.

Ordinary CI does not retrain. It uses tiny logic fixtures, recalculates all committed recognition
metrics/error analysis from saved safe predictions, verifies every artifact/source hash, and
cheaply regenerates the vision result:

```console
python scripts/run_benchmark.py verify
```

To intentionally prepare or freeze a new benchmark version, use the script's `prepare`,
`preflight`, and `freeze` modes before any final test evaluation. Preflight loads every training
and validation image and checks CTC feasibility without producing test predictions. `manifest`
writes the final reviewed-artifact inventory after result generation. These maintenance modes are
not a user-facing OCR CLI.
