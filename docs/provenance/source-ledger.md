# Current implementation source ledger

Review date: 2026-09-07
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

Planned libraries and their licensing/package implications are recorded in
`dependency-review.md`. Listing a dependency does not incorporate its source into this repository
or select a license for repository-owned code. Phases 5 through 9 were implemented without
consulting or copying historical preprocessing, conversion, segmentation, layout, data-processing,
split, vocabulary, review, synthetic generation, recognizer, CTC, decoder, notebook, or test
source. No historical text, font, image, crop, annotation, vocabulary, or generated sample is
present in the Phase 8 fixtures, and no historical model artifact is present in Phase 9.
