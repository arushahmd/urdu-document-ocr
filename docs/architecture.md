# Architecture

## Vertical design

```text
Document
  -> ingestion
  -> preprocessing
  -> segmentation and one/two-column layout
  -> ordered Urdu line crops
  -> CNN-BiLSTM-CTC recognizer
  -> ordered line results
  -> page and document assembly
  -> evaluation and public interfaces
```

The intended V1 is a single-process Python package. Vision, recognition, evaluation, and
interface layers remain separable so each can be tested without inventing service
infrastructure.

## Implemented in Phase 4

- `urdu_document_ocr.types`: immutable structural records, validated image-array contracts,
  half-open geometry, vocabulary indexing/fingerprinting, and privacy-safe public projections.
- `urdu_document_ocr.config`: strict resource, recognizer, and training configuration with
  canonical SHA-256 fingerprints.
- Repository provenance, dependency-review, security, test, packaging, and CI controls.

RGB page images are `uint8[height,width,3]`. Preprocessed grayscale pages are
`uint8[height,width]`; foreground masks are `bool[height,width]`. Page and reading-order indices
are zero-based, while source-facing page numbers are one-based.

## Planned, not implemented

- Phase 5: bounded image/PDF ingestion, preprocessing, blank detection, and modest deskew.
- Phase 6: scale-derived line segmentation and deterministic RTL layout ordering.
- Phase 7: JSONL manifests, validation, grouped splitting, and manifest-derived vocabulary.
- Phase 8: provenance-cleared shaped Urdu synthetic fixtures.
- Phases 9–10: PyTorch CNN-BiLSTM-CTC recognition, training, and tensor-only checkpoints.
- Phases 11–13: inference, assembly, metrics, benchmarks, CLI, and reference HTTP adapter.

Future directories and imports do not exist until their phase supplies meaningful tested code.
