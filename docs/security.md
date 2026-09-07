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

Phase 7 treats dataset manifests and line images as untrusted input. Manifest parsing is strict
UTF-8/JSONL with fixed fields and no arbitrary metadata object. Portable relative paths are
resolved beneath a resolved dataset root; containment is checked after symlink resolution. Public
errors and reports use issue codes, physical line numbers, and sample IDs without serializing the
absolute dataset root.

Line-image validation retains Pillow decompression-bomb protection, accepts decoded PNG/JPEG only,
checks a 100-MiB encoded-file ceiling before decoding, and enforces a 50-million-pixel ceiling.
Dataset fingerprints use logical records rather than filesystem paths or timestamps. Vocabulary,
split metadata, and review overlays use JSON/JSONL and SHA-256; pickle and executable
serialization remain prohibited.

Writers require explicit overwrite and never create an unrequested output directory. Validation,
splitting, vocabulary construction, and review application do not move, edit, or delete source
images. Review application creates new sample values, and writing a derived manifest remains a
separate explicit call. No malware-scanning claim is made.

Phase 8 synthetic generation reads only explicit text, the reviewed bundled font, and local
configuration. It performs no font/data download and makes no network request. HarfBuzz/FreeType
capability, the bundled font hash, its asset provenance, canonical Unicode, glyph coverage, and
output bounds fail closed. Generation requires an existing explicit non-root output directory;
all generated paths are portable relative paths resolved beneath it. Traversal and symlink escape
are rejected, unrelated existing files block generation, overwrite is explicit, and no directory
or file is deleted automatically.

The committed Urdu source phrases are current neutral fixture content, not private/historical
transcriptions. Generated PNGs are minimal grayscale/RGB images without EXIF, GPS, usernames,
paths, or text metadata. Public generation records contain stable IDs, relative paths, dimensions,
parameters, and hashes—not machine paths. The bundled font is isolated with its own OFL-1.1 text
and provenance; that license does not license repository-owned source code.

## Reporting concerns

Potential vulnerabilities or accidental private-data inclusions should be reported privately to
the repository owner before public disclosure. No public reporting address is declared while the
repository remains local.
