# Current implementation source ledger

Review date: 2026-09-06
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

Planned libraries and their licensing/package implications are recorded in
`dependency-review.md`. Listing a dependency does not incorporate its source into this repository
or select a license for repository-owned code. Phases 5 and 6 were implemented without consulting
or copying historical preprocessing, conversion, segmentation, layout, notebook, or test source.
