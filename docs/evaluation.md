# Evaluation and frozen synthetic benchmarks

## Metric policy

Evaluation compares the reference and hypothesis exactly as supplied. It performs no Unicode
normalization, case conversion, punctuation removal, Arabic/Urdu character folding, diacritic
removal, or whitespace repair. Canonical project manifests already require NFC text; callers of
the general API remain responsible for any preprocessing they intend.

`edit_distance` implements unit-cost Levenshtein distance for generic sequences. The companion
alignment uses the stable backtrace priority match, substitution, deletion, insertion whenever
multiple predecessor cells are optimal. That priority freezes attribution for error analysis;
it cannot change the minimum distance. For every selected alignment, distance equals
substitutions plus deletions plus insertions.

Character error rate is

```text
(character substitutions + deletions + insertions) / reference Unicode code points
```

Word error rate uses the same formula over nonempty tokens separated by the ASCII space U+0020.
Punctuation remains attached to its token. Neither rate is capped, so insertion-heavy CER or WER
can exceed 1.0. Exact match is exact code-point equality of the complete supplied strings.

For an empty reference, edit counts and exact match remain defined but that sample's CER and WER
are `None`. A corpus rate is likewise `None` when its summed reference denominator is zero. Corpus
CER and WER divide summed edit counts by summed reference lengths; they are never averages of
per-sample rates. Corpus output also includes sample count, exact-match count and rate, and
character and word substitution/deletion/insertion totals.

## APIs and error analysis

`evaluate_text` evaluates one pair. `evaluate_predictions` evaluates ID-keyed collections and
rejects duplicate IDs, missing predictions, and extra predictions rather than silently
intersecting inputs. `evaluate_checkpoint` uses the existing line-image loader and batched
checkpoint recognizer; it does not create another inference path.

`analyze_errors` derives evidence only from the deterministic optimal character alignment. It
reports the worst samples by CER, complete character S/D/I totals, reference and hypothesis
character frequencies, actual substitution pairs, and inserted/deleted character frequencies.
Ties use count-descending then code-point ordering; worst samples use CER, total edits, and sample
ID deterministically. Characters are accompanied by explicit Unicode code points in public JSON.

## Benchmark protocol

The repository freezes two independent suites:

- `vision-synthetic-v1` regenerates benchmark-only pages, runs the normal preprocessing and
  segmentation/layout pipeline, and scores deterministic one-to-one maximum-IoU matches at the
  frozen threshold. Detection, matched geometry, kind, column, layout, and page-level reading
  order have separate denominators.
- `recognition-synthetic-v1` regenerates 256 line images, applies document-grouped train,
  validation, and test splits, builds vocabulary from training text only, trains the existing
  CNN-BiLSTM-CTC path, selects the checkpoint by validation CTC loss only, and evaluates the held-
  out test partition once.

The source corpus, seeds, configs, font identity, logical image/manifests, partitions,
vocabulary, model and training fingerprints, metric policy, and source hashes are recorded in
`benchmark/freeze.json` before final result generation. Once frozen, changing a source, seed,
threshold, model setting, training setting, or test member requires a new benchmark version or a
documented invalidation. Final test output is not tuning evidence. Compact results retain all
cases and sample predictions so metrics can be recalculated without retraining.

Both suites contain only deterministic, provenance-cleared synthetic data. They do not establish
performance on historical industrial data, scanned books, handwriting, arbitrary real documents,
or production deployment. No benchmark checkpoint is retained or published.
