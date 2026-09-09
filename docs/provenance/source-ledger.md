# Current implementation source ledger

Review date: 2026-09-09
Implementation rule: `REIMPLEMENT_FROM_CONCEPT`

This ledger describes the provenance of the current public implementation. It records evidence
and engineering decisions; it does not offer legal conclusions.

| Component | Status | Source category | Evidence or influence | Historical source copied? |
|---|---|---|---|---|
| Repository and package foundation | Implemented in Phase 4 | `CURRENT_ORIGINAL_IMPLEMENTATION` | Newly designed from the frozen public architecture and Python packaging conventions | No |
| Domain types and geometry | Implemented in Phase 4 | `CURRENT_ORIGINAL_IMPLEMENTATION` | Newly written value records; half-open rectangle arithmetic is a general convention | No |
| Path and public-serialization safeguards | Implemented in Phase 4 | `CURRENT_ORIGINAL_IMPLEMENTATION` | Newly written validation against portable relative-path and privacy requirements | No |
| Configuration validation and fingerprints | Implemented in Phase 4 | `CURRENT_ORIGINAL_IMPLEMENTATION` | Newly written strict dataclasses; SHA-256 and canonical JSON are standard-library mechanisms | No |
| Vocabulary index/fingerprint contract | Implemented in Phase 4 | `GENERAL_ALGORITHM` | Deterministic code-point ordering and SHA-256 over canonical JSON; no external code incorporated | No |
| Phase 4 tests and hygiene checks | Implemented in Phase 4 | `CURRENT_ORIGINAL_IMPLEMENTATION` | Newly written against current public contracts | No |
| NumPy | Used in Phase 4 | `EXTERNAL_LIBRARY` | Array type and shape/dtype validation; [official project](https://numpy.org/) | No |
| setuptools | Build dependency | `EXTERNAL_LIBRARY` | Standard `pyproject.toml` build backend; [packaging guide](https://packaging.python.org/) | No |
| pytest and pytest-cov | Development dependencies | `EXTERNAL_LIBRARY` | Test execution and coverage reporting; [pytest documentation](https://docs.pytest.org/) | No |
| Ruff | Development dependency | `EXTERNAL_LIBRARY` | Linting and formatting; [Ruff documentation](https://docs.astral.sh/ruff/) | No |
| GitHub Actions | CI configuration | `EXTERNAL_LIBRARY` | Stable checkout/setup actions and documented Python workflow conventions | No |
| Earlier OCR capability set | Requirements context only | `BEHAVIORAL_REFERENCE` | High-level workflow recorded separately in `behavioral-references.md` | No |
| PNG/JPEG ingestion | Implemented in Phase 5 | `CURRENT_ORIGINAL_IMPLEMENTATION` | Newly written bounded orchestration around documented Pillow decode, verify, EXIF-transpose, and RGB conversion APIs | No |
| PDF preflight and rasterization | Implemented in Phase 5 | `CURRENT_ORIGINAL_IMPLEMENTATION` | Newly written resource accounting around documented pypdfium2 document, page-size, render, Pillow-conversion, and close APIs | No |
| Otsu thresholding and morphology | Implemented in Phase 5 | `EXTERNAL_LIBRARY` | OpenCV threshold and small morphology primitives; project ordering, polarity, defaults, and validation are newly written | No |
| Sauvola thresholding | Implemented in Phase 5 | `GENERAL_ALGORITHM` | Standard local mean/standard-deviation formula independently implemented with NumPy/OpenCV float64 statistics; no historical or third-party implementation copied | No |
| Polarity and blank decision | Implemented in Phase 5 | `CURRENT_ORIGINAL_IMPLEMENTATION` | Newly designed border-deviation rule and three-signal conjunction from the frozen architecture | No |
| Projection-profile deskew | Implemented in Phase 5 | `GENERAL_ALGORITHM` | Newly written transparent small-angle search, normalized row-profile score, confidence gate, and same-size transform | No |
| Phase 5 tests | Implemented in Phase 5 | `CURRENT_ORIGINAL_IMPLEMENTATION` | Deterministic programmatically generated pixels, geometric marks, and in-memory PDFs; no external fixture or font content | No |
| Pillow | Used in Phase 5 | `EXTERNAL_LIBRARY` | Image decoding and EXIF orientation from [official documentation](https://pillow.readthedocs.io/en/stable/) | No |
| OpenCV headless | Used in Phase 5 | `EXTERNAL_LIBRARY` | Grayscale, threshold, local statistics, connected components, morphology, and affine transforms from [official documentation](https://docs.opencv.org/) | No |
| pypdfium2 / PDFium | Used in Phase 5 | `EXTERNAL_LIBRARY` | In-memory PDF inspection and rasterization from [official documentation](https://pypdfium2.readthedocs.io/) | No |
| Eight-connected component extraction | Implemented in Phase 6 | `EXTERNAL_LIBRARY` | OpenCV component statistics API from [official documentation](https://docs.opencv.org/4.13.0/d3/dc0/group__imgproc__shape.html); orchestration and retained records are newly written | No |
| Robust component-scale estimation | Implemented in Phase 6 | `GENERAL_ALGORITHM` | Newly written area-weighted median-height statistic with relative outlier exclusion | No |
| Geometry helpers and component-to-line grouping | Implemented in Phase 6 | `CURRENT_ORIGINAL_IMPLEMENTATION` | Newly written half-open interval operations, x-sweep/union-find grouping, mark attachment, and conservative fragment merging | No |
| Spanning-region and persistent-gutter inference | Implemented in Phase 6 | `CURRENT_ORIGINAL_IMPLEMENTATION` | Newly designed central occupancy, bilateral support, paired-line, vertical-persistence, and gutter-crossing rules | No |
| Urdu RTL reading-order assignment | Implemented in Phase 6 | `CURRENT_ORIGINAL_IMPLEMENTATION` | Newly written vertical-band ordering with explicit right-column-first semantics | No |
| Phase 6 tests | Implemented in Phase 6 | `CURRENT_ORIGINAL_IMPLEMENTATION` | Deterministic generated geometric masks and RGB pages; no external scan, font, transcription, or benchmark fixture | No |
| JSONL manifest I/O and logical dataset fingerprints | Implemented in Phase 7 | `CURRENT_ORIGINAL_IMPLEMENTATION` | Newly written strict schema parsing, canonical UTF-8 writing, atomic replacement, and SHA-256 identity using Python standard-library APIs | No |
| Unicode normalization and transcription validation | Implemented in Phase 7 | `GENERAL_ALGORITHM` | Unicode NFC through Python `unicodedata`; explicit whitespace/control policy independently implemented from the frozen design | No |
| Dataset containment, image validation, and statistics | Implemented in Phase 7 | `CURRENT_ORIGINAL_IMPLEMENTATION` | Newly written resolved-root containment, bounded Pillow inspection, duplicate hashing, structured findings, and deterministic summaries | No |
| `group-greedy-v1` document split | Implemented in Phase 7 | `CURRENT_ORIGINAL_IMPLEMENTATION` | Newly written frozen group-size ordering, seeded SHA-256 tie-break, global target-deviation assignment, and reproducibility metadata | No |
| Dataset-derived vocabulary persistence | Implemented in Phase 7 | `CURRENT_ORIGINAL_IMPLEMENTATION` | Newly written training-text character collection and strict JSON wrapper around the Phase 4 `Vocabulary` contract | No |
| Immutable review overlays | Implemented in Phase 7 | `CURRENT_ORIGINAL_IMPLEMENTATION` | Newly written source-fingerprint validation, decision filtering, tag merging, and canonical JSONL persistence without filesystem mutation | No |
| Phase 7 tests | Implemented in Phase 7 | `CURRENT_ORIGINAL_IMPLEMENTATION` | Project-authored short Unicode strings and programmatically generated tiny geometric images only; no historical labels, vocabulary, or images | No |
| Safe Urdu fixture text | Implemented in Phase 8 | `CURRENT_ORIGINAL_IMPLEMENTATION` | Thirteen newly authored neutral phrases with per-source SHA-256 and code-point coverage in `data/sample/text-provenance.json` | No |
| Synthetic line/page generator | Implemented in Phase 8 | `CURRENT_ORIGINAL_IMPLEMENTATION` | Newly written seeded composition, controlled degradation, safe-write, geometry, manifest, and artifact-hash orchestration | No |
| Urdu complex-script shaping route | Used in Phase 8 | `EXTERNAL_LIBRARY` | uharfbuzz/HarfBuzz explicit RTL shaping and freetype-py/FreeType rasterization through documented public APIs | No |
| Noto Nastaliq Urdu | Bundled in Phase 8 | `EXTERNAL_ASSET` | Unmodified `NotoNastaliqUrdu-v4.000` variable TTF from the authoritative release; exact archive/inner-file hashes and OFL-1.1 evidence stored at asset level | No |
| Public synthetic fixtures and visual | Generated in Phase 8 | `CURRENT_GENERATED_ARTIFACT` | Deterministic output solely from the Phase 8 authored text, reviewed font, `synthetic-v1`, and frozen configuration | No |
| Phase 8 tests | Implemented in Phase 8 | `CURRENT_ORIGINAL_IMPLEMENTATION` | Newly written provenance, shaping, line/page, regeneration, safe-output, metadata, and realistic vision integration checks | No |
| CNN-BiLSTM-CTC recognizer source | Implemented in Phase 9 | `CURRENT_ORIGINAL_IMPLEMENTATION` | Independently written PyTorch modules from the frozen public contract; no historical model/config/comment/test source consulted or copied | No |
| CRNN and CTC model family | Used in Phase 9 | `GENERAL_ALGORITHM` | Standard convolutional sequence recognition, bidirectional recurrence, and connectionist temporal classification concepts | No |
| CNN width geometry and model-input normalization | Implemented in Phase 9 | `CURRENT_ORIGINAL_IMPLEMENTATION` | One exact pooling-length helper plus deterministic aspect-preserving grayscale conversion and `x / 127.5 - 1` mapping | No |
| GroupNorm, SiLU, packed BiLSTM, and CTCLoss primitives | Used in Phase 9 | `EXTERNAL_LIBRARY` | PyTorch 2.14 public APIs; project composition, validation, and contracts are current-original | No |
| Greedy CTC decoder and alignment feasibility checks | Implemented in Phase 9 | `CURRENT_ORIGINAL_IMPLEMENTATION` | Newly written collapse-before-blank-removal decoder and adjacent-repeat minimum-timestep validation | No |
| Phase 9 tests | Implemented in Phase 9 | `CURRENT_ORIGINAL_IMPLEMENTATION` | Programmatic tensors plus the provenance-cleared Phase 8 synthetic line/vocabulary; no expected recognition text or accuracy claim | No |
| OCR line Dataset and dynamic collator | Implemented in Phase 10 | `CURRENT_ORIGINAL_IMPLEMENTATION` | Newly written wrapper around current manifest, image-containment, vocabulary, and recognizer-input contracts; variable-width white padding follows the frozen design | No |
| Training preflight and deterministic DataLoaders | Implemented in Phase 10 | `CURRENT_ORIGINAL_IMPLEMENTATION` | Newly written split-leakage, vocabulary, image-conflict, width, and exact CTC-feasibility checks with explicit seeded generators/workers | No |
| AdamW trainer, validation, and early stopping | Implemented in Phase 10 | `CURRENT_ORIGINAL_IMPLEMENTATION` | Newly written transparent epoch loops, finite-value guards, sample-weighted loss aggregation, gradient clipping, and validation-loss selection | No |
| Safe checkpoint metadata and atomic save/load | Implemented in Phase 10 | `CURRENT_ORIGINAL_IMPLEMENTATION` | Newly written canonical JSON identity, SHA-256 verification, temporary-directory replacement, and strict pre-mutation tensor checks | No |
| AdamW and PyTorch DataLoader | Used in Phase 10 | `EXTERNAL_LIBRARY` | PyTorch 2.14 public optimizer, DataLoader, generator, and gradient-clipping APIs; orchestration and policy are current-original | No |
| safetensors | Used in Phase 10 | `EXTERNAL_LIBRARY` | safetensors 0.8 public PyTorch save/load APIs for tensor-only model weights; Apache-2.0 | No |
| Phase 10 tests | Implemented in Phase 10 | `CURRENT_ORIGINAL_IMPLEMENTATION` | Programmatic tensors/images and the provenance-cleared Phase 8 fixtures; all checkpoints are temporary and no quality assertion is made | No |
| Checkpoint-backed recognizer loading | Implemented in Phase 11 | `CURRENT_ORIGINAL_IMPLEMENTATION` | Newly written lifecycle wrapper around the current strict Phase 10 loader, explicit device policy, and evaluation/inference mode | No |
| Ordered line batching | Implemented in Phase 11 | `CURRENT_ORIGINAL_IMPLEMENTATION` | Newly written inference batching that shares the current normalization and white-right-padding primitive and never width-sorts inputs | No |
| Page OCR orchestration | Implemented in Phase 11 | `CURRENT_ORIGINAL_IMPLEMENTATION` | Newly written composition of current segmentation, grayscale crop, recognizer protocol, result, and assembly contracts | No |
| Document OCR orchestration | Implemented in Phase 11 | `CURRENT_ORIGINAL_IMPLEMENTATION` | Newly written sequential page loop over bounded ingestion and preprocessing with stable blank-page retention | No |
| Deterministic result assembly and TXT/JSON projections | Implemented in Phase 11 | `CURRENT_ORIGINAL_IMPLEMENTATION` | Newly written newline policy, privacy-safe typed projection, UTF-8 serialization, and explicit atomic output helpers | No |
| Phase 11 tests and examples | Implemented in Phase 11 | `CURRENT_ORIGINAL_IMPLEMENTATION` | Newly written programmatic fakes, current Phase 8 fixtures, temporary safetensors checkpoints, and public-API-only examples; no text-quality assertion | No |
| Edit distance and deterministic alignment | Implemented in Phase 12 | `CURRENT_ORIGINAL_IMPLEMENTATION` / `GENERAL_ALGORITHM` | Transparent current-original implementation of unit-cost Levenshtein distance with a frozen backtrace tie policy; hand-calculated tests | No |
| CER, WER, exact match, and corpus aggregation | Implemented in Phase 12 | `CURRENT_ORIGINAL_IMPLEMENTATION` / `GENERAL_ALGORITHM` | Standard edit-rate definitions independently implemented with exact Unicode input and explicit zero-denominator behavior | No |
| OCR error analysis | Implemented in Phase 12 | `CURRENT_ORIGINAL_IMPLEMENTATION` | Newly written deterministic aggregation from actual selected alignments, including safe code-point records | No |
| Synthetic benchmark harness | Implemented in Phase 12 | `CURRENT_ORIGINAL_IMPLEMENTATION` | Newly written regeneration, checkpoint evaluation, one-to-one IoU scoring, result serialization, and compact verification around existing public package APIs | No |
| Benchmark freeze and integrity logic | Implemented in Phase 12 | `CURRENT_ORIGINAL_IMPLEMENTATION` | Newly written canonical JSON, SHA-256 source/artifact records, frozen-identity checks, and result recalculation | No |
| Levenshtein distance, CER, WER, and IoU | Used in Phase 12 | `GENERAL_ALGORITHM` | Standard mathematical definitions; no third-party metric implementation incorporated | No |
| Phase 12 benchmark data and tests | Implemented in Phase 12 | `CURRENT_ORIGINAL_IMPLEMENTATION` / `CURRENT_GENERATED_ARTIFACT` | Newly authored neutral Urdu composition, benchmark-only seeds, programmatic generation, and hand-verifiable cases; no historical assets or metrics | No |
| Phase 12B external learnability study | Executed outside the repository | `CURRENT_ORIGINAL_IMPLEMENTATION` / `CURRENT_GENERATED_ARTIFACT` | Newly authored development-only Urdu compositions, deterministic synthetic images, and instrumentation around current public APIs; only the report SHA-256 and aggregate selection evidence enter V2 provenance | No |
| Synthetic recognition benchmark V2 harness | Implemented in Phase 12B | `CURRENT_ORIGINAL_IMPLEMENTATION` | Newly written recognition-only prepare/freeze/preflight/run/verify orchestration reusing the current renderer, trainer, recognizer, evaluator, and error analyzer | No |
| Synthetic recognition benchmark V2 data/results | Implemented in Phase 12B | `CURRENT_ORIGINAL_IMPLEMENTATION` / `CURRENT_GENERATED_ARTIFACT` | Eight newly authored concise templates, deterministic numeric composition, new seeds, grouped identities, train-only vocabulary, and one post-freeze test run | No |

Planned libraries and their licensing/package implications are recorded in
`dependency-review.md`. Listing a dependency does not incorporate its source into this repository
or select a license for repository-owned code. Phases 5 through 9 were implemented without
consulting or copying historical preprocessing, conversion, segmentation, layout, data-processing,
split, vocabulary, review, synthetic generation, recognizer, CTC, decoder, notebook, or test
source. Phase 10 training, optimizer, batching, early-stopping, and checkpoint source was also
written without consulting or copying historical trainers, notebooks, loaders, or utilities.
Phase 11 recognizer loading, line batching, page/document orchestration, assembly, serialization,
tests, and examples were implemented from the frozen current-project contracts without inspecting
or copying historical prediction, document OCR, output, model-loading, web-handler, notebook, or
path code. Phase 12 metric, alignment, error-analysis, benchmark, freeze, integrity, and test source
was independently written from the frozen current-project specification. The existing library
stack is reused and no edit-distance, metrics, matching, dataframe, or scientific-optimization
dependency was added. No historical text, font, image, crop, annotation, vocabulary, generated sample,
checkpoint, or model artifact is present.

Phase 12B did not inspect or copy historical OCR code, data, text, checkpoints, metrics, or model
artifacts. Its external development study used only new project-authored/programmatically
composed text and current public package APIs. The V2 corpus is also newly authored and has zero
exact-text overlap with the development corpus. Model, trainer, CTC, decoder, and `ocr-edit-v1`
implementations remain unchanged. No trained checkpoint is committed.
