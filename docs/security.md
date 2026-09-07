# Security and privacy boundary

This document records engineering controls, not a security certification.

## Repository controls

- Do not commit credentials, environment files, private URLs, internal identifiers, machine
  paths, runtime logs, databases, model checkpoints, or private datasets.
- Never reuse historical secrets, source files, trained weights, transcriptions, fonts, or
  deployment assets.
- Generated checkpoints, bulk data, local outputs, and environment files are ignored by default.
- Project-owned domain data must use deterministic JSON-compatible structures, not pickle,
  joblib, or arbitrary executable serialization.
- Historical repository and preservation material are not part of this source tree.

## Runtime boundary

Documents and images are treated as untrusted input. Phase 5 enforces the frozen input-byte, PDF
page, per-page pixel, total-pixel, and DPI limits. Files are size-checked before reading; byte
buffers are checked before parsing; PDF page count and every page geometry are checked before any
page is rendered. Pillow's decompression-bomb protection remains enabled, and its warning is
handled locally as a resource failure.

Image and PDF resources are decoded in memory and closed deterministically. Returned NumPy arrays
own their data. Image metadata is discarded, and public source records never include an absolute
path. Typed public errors contain stable stage codes and content-free messages; underlying
decoder exceptions remain chained for controlled diagnostics but are not serialized.

Library logging must not include OCR document text by default. Diagnostic logging should record
safe stage names, counts, fingerprints, and machine-safe error codes.

The current loader does not accept PDF passwords, write temporary page files, fetch remote
content, auto-download assets, execute document content, or silently drop blank pages.

Phase 6 segmentation stays inside the already bounded page arrays and makes no network calls,
data-service calls, or external model downloads. It processes only geometry derived from the
foreground mask. `LineRegion` records do not contain or serialize raw page data, and crop
extraction is explicit and returns an owned in-memory array. No recognized document text exists
at this phase. Segmentation and layout emit no content logs; typed failure context contains only a
source page number and stage name.

## Reporting concerns

Potential vulnerabilities or accidental private-data inclusions should be reported privately to
the repository owner before public disclosure. No public reporting address is declared while the
repository remains local.
