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

Documents and images will be treated as untrusted input. Later ingestion phases must enforce the
frozen input byte, PDF page, per-page pixel, total pixel, and DPI limits before expensive work.
Errors and serialized results must omit absolute paths and document content unless content is the
explicit requested result.

Library logging must not include OCR document text by default. Diagnostic logging should record
safe stage names, counts, fingerprints, and machine-safe error codes.

## Reporting concerns

Potential vulnerabilities or accidental private-data inclusions should be reported privately to
the repository owner before public disclosure. No public reporting address is declared while the
repository remains local.
