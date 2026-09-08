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

## Implemented through Phase 10

- `urdu_document_ocr.types`: immutable structural records, validated image-array contracts,
  half-open geometry, vocabulary indexing/fingerprinting, and privacy-safe public projections.
- `urdu_document_ocr.config`: strict resource, recognizer, and training configuration with
  canonical SHA-256 fingerprints.
- Repository provenance, dependency-review, security, test, packaging, and CI controls.
- `urdu_document_ocr.document`: bounded PNG/JPEG decoding and PDF preflight/rasterization into
  owned RGB page arrays.
- `urdu_document_ocr.vision`: grayscale enhancement, foreground thresholding, blank detection,
  optional modest deskew, connected-component line segmentation, one/two-column layout, and Urdu
  RTL reading order.
- `urdu_document_ocr.data`: strict JSONL manifests, non-mutating dataset/image validation,
  deterministic document-grouped splitting, train-derived vocabulary, and immutable review
  overlays.
- `urdu_document_ocr.recognition`: optional PyTorch CNN-BiLSTM-CTC logits, input normalization,
  explicit alignment validation and loss, and greedy vocabulary decoding.
- `urdu_document_ocr.training`: validated line datasets, variable-width batching, deterministic
  DataLoaders, AdamW optimization, validation/early stopping, and strict safetensors checkpoints.

RGB page images are `uint8[height,width,3]`. Preprocessed grayscale pages are
`uint8[height,width]`; foreground masks are `bool[height,width]`. Page and reading-order indices
are zero-based, while source-facing page numbers are one-based.

## Ingestion behavior

`load_document` accepts a filesystem path or an in-memory `bytes`, `bytearray`, or `memoryview`.
The file suffix is not authoritative: PDF recognition requires a PDF header near the beginning,
and Pillow must identify a decoded image as PNG or JPEG. Other decodable image formats are
rejected. Image orientation is applied from EXIF, transparency is flattened onto white, and the
result is copied to RGB `uint8`. No EXIF, GPS, color-profile, comment, or application metadata is
placed in domain objects. Public source metadata has only an optional caller-safe basename,
SHA-256, byte size, and source kind.

PDF rendering uses pypdfium2 entirely in memory. Page points convert to pixels as
`ceil(points * dpi / 72)`, the same scale passed to the renderer. The loader checks the 100 MiB
input limit, allowed 72–400 DPI, 100-page count, 50,000,000 pixels per page, and 500,000,000
cumulative document pixels. It inspects all page geometries and total pixels before rendering
the first page. Password-required PDFs are rejected without prompting. All native page, bitmap,
and document handles are closed deterministically.

## Preprocessing behavior

The exact operation order is RGB-to-grayscale, optional CLAHE, optional 3x3 or 5x5 median
filter, thresholding, polarity normalization, optional 3x3 or 5x5 opening/closing, blank
decision, and optional deskew. The default is Otsu with every optional operation disabled.
Sauvola uses replicated-border local mean and standard deviation in float64:
`mean * (1 + k * (std / R - 1))`, with defaults window 31, `k=0.2`, and `R=128`.

For polarity, the outer 5% frame supplies a median background estimate. The implementation
counts pixels at least two intensity levels lighter and darker than that median. More lighter
pixels imply a dark background; otherwise more darker pixels imply a light background. A tie
uses median below 127.5 as dark. The final boolean invariant is always `True = likely ink`.

A page is blank only when all three conditions hold:

- foreground fraction is below `0.001`;
- grayscale population standard deviation is below `3.0`;
- no 8-connected component reaches `max(2, ceil(page_pixels * 0.00001))` pixels.

Blank pages remain in the output. These thresholds are configurable with strict validation.
The component fraction scales with resolution while the two-pixel floor makes tiny synthetic
tests well-defined.

Deskew is opt-in and preserves dimensions. A projection-profile search scores corrections from
-5° through +5° in 0.25° steps using normalized squared row sums. The selected score must improve
over zero and separate from candidates at least 1° away by the configured minimum confidence
(default `0.01` for both, using their minimum). A boundary optimum, blank page, zero baseline,
or low-confidence result is a no-op. A returned positive angle means the counter-clockwise
correction passed to OpenCV. Applied grayscale rotation uses linear interpolation and the border
median as fill; the foreground mask and blank decision are then regenerated.

## Segmentation behavior

`segment_page` consumes `PreprocessedPage.foreground_mask` directly, preserving the invariant
`True = likely ink`; it never thresholds the RGB image again. A blank flag or empty mask returns
an empty tuple. Otherwise OpenCV extracts 8-connected components and the implementation retains
only each component's half-open box, area, and centroid after discarding the label image.

The component floor is the greater of 2 pixels and `ceil(page_area * 0.000001)`. Text scale is
the area-weighted median component height after excluding components larger than 5% of the page
when smaller alternatives exist. This robust height statistic drives all grouping distances.
Core components have height at least 40% of scale or area at least `ceil(0.5 * scale^2)`.
Remaining small components are not discarded indiscriminately: each is attached to the nearest
core group when both its horizontal and vertical interval gaps are at most 1.25 scale units.

Core grouping is a deterministic x-sweep plus union-find. Candidate pairs must be no more than
4 scale units apart horizontally and must either overlap vertically by at least 25% of the
smaller box or have centers within 0.65 scale units. A conservative second pass joins fragments
within 8 scale units only with 55% vertical overlap or centers within 0.35 scale units. On pages
with gutter evidence, that pass runs separately for spanning, right, and left pools so a line
cannot bridge the columns. Phase 6 applies no morphology.

Final groups must contain at least the greater of five foreground pixels (with the default
component floor) and `ceil(page_area * 0.00001)`. Their boxes are exact unions of retained
components, padded on every side by `round(0.2 * scale)` and clipped to the page. These permissive
area rules retain short lines, punctuation, and page numbers while rejecting unsupported isolated
specks. Candidate identifiers are assigned from deterministic geometric order; final reading
indices are contiguous.

## Layout and Urdu reading order

V1 supports one column, two columns, and geometric spanning lines only. Gutter detection excludes
preliminary wide-centered candidates, then searches the central 50% of the content x-range for a
zero-occupancy run. A viable gutter is at least the greater of 6% of content width and two text
scale units. Each side must have at least two body lines, their aggregate vertical spans must
overlap by at least 35% of the smaller span, and at least two left/right line centers must pair
within 1.5 scale units. Candidate gutters are ranked by paired lines, vertical support, width,
then proximity to content center. If no candidate passes every check, the page is one-column.

Given a valid gutter, a candidate is `RegionKind.SPANNING` only when it is at least 65% of content
width, its center is within 15% of the content width from content center, and its box crosses the
entire inferred gutter. Other candidates are assigned by center: `column_index=0` is the right
column and `column_index=1` is the left column. One-column and spanning lines use
`column_index=None`.

Reading order is divided at the vertical center of each spanning line. In every intervening body
band, right-column lines are read top-to-bottom before left-column lines; the spanning line then
follows. This places a full-width heading before its body, a mid-page separator between sections,
and a footer after the preceding columns. Within vertical ordering, starts separated by at most
25% of text scale share a tolerance band and use rightmost-first, then stable geometric and
identifier tie-breaks.

The optional `extract_line_crop` helper validates page identity and bounds, then returns an owned
copy from either grayscale pixels or the boolean foreground mask. `LineRegion` itself contains no
page arrays and its public projection serializes geometry only.

## ML data pipeline

The implemented system now has three independent tracks:

```text
DOCUMENT: image/PDF -> PageImage -> PreprocessedPage -> ordered LineRegion geometry

DATA: labeled PNG/JPEG line images -> JSONL manifest -> validation
      -> document-grouped train/validation/test split -> train vocabulary -> review overlay

SAFE REPRODUCIBILITY: authored Urdu text + reviewed local font -> shaped lines/pages
                      -> controlled degradations -> public fixtures and provenance
```

Manifest schema v1 requires `schema_version`, `sample_id`, `image_path`, `text`, and
`document_id`; `page_id`, `line_index`, and `tags` are optional. Unknown top-level fields fail.
Manifest parsing preserves record order and labels exactly. Canonical writing sorts by sample ID,
uses compact readable UTF-8 with stable key order and a final newline, and requires explicit
overwrite. Logical dataset identity is SHA-256 over those canonical records and is independent of
input order, absolute paths, and filesystem timestamps.

Full validation resolves each portable relative image path beneath a resolved dataset root,
rejecting symlink escape. It safely verifies PNG/JPEG images with existing Pillow protections and
reports rather than repairs Unicode, duplicate, conflict, dimension, and vocabulary findings.
Statistics cover documents, image geometry, transcription/word lengths, samples per document,
and exact Unicode code-point frequencies. Lightweight validation can skip filesystem decoding.

The `nfc-v1` canonical contract preserves ZWNJ, punctuation, digits, diacritics, and distinct
Urdu/Arabic letter forms. Training text must already be NFC, single-line, free of surrounding or
noncanonical whitespace, and free of control/format characters other than ZWNJ. An explicit
normalizer can produce NFC/single-space text for a new manifest; no reader silently invokes it.

`group-greedy-v1` groups by `document_id`, sorts groups by descending size and seeded SHA-256
tie-break, and assigns each indivisible group to the positive-ratio split minimizing total absolute
deviation from requested sample targets. Exact assignment ties prefer train, validation, then test.
Defaults are 80/10/10 with seed 1337. Results expose requested and achieved counts/ratios,
document assignments, dataset identity, and a split fingerprint.

Vocabulary is built from canonical training labels, includes spaces present in those labels, sorts
characters by Unicode code point, reserves CTC index 0 for blank, and has no UNK. Validation/test
unseen code points are reported instead of substituted. Strict JSON persistence revalidates the
normalization policy, ordering, uniqueness, blank index, and fingerprint.

Review decisions use `valid`, `invalid`, or `needs_review`, source-dataset hashes, safe tags, and
optional short notes/generic reviewer labels. Application returns a derived tuple: invalid and,
by default, unresolved samples are excluded; retained tags are merged. It never edits a source
manifest or moves/deletes images. Exact schemas are documented in `data-format.md`.

## Recognition core

Phase 9 implements the trainable but untrained line recognizer. Its fixed geometry is:

```text
[B,1,64,W]
   ↓ four CNN blocks (two 3x3 Conv + GroupNorm + SiLU per block)
[B,384,4,T]
   ↓ mean(H), transpose
[B,T,384]
   ↓ packed 2-layer bidirectional LSTM
[B,T,512]
   ↓ Linear(512, C)
[B,T,C]
   ↓ CTC loss or greedy CTC decoding
```

The CNN channels are 64, 128, 256, and 384. Pooling is `(2,2)`, `(2,2)`, `(2,1)`,
and `(2,1)`, so height stride is 16 and width stride is 4. The authoritative valid-length
formula is `floor(floor(W / 2) / 2)`. Each sample is cropped to its declared valid width before
convolution, and unsorted feature sequences are packed before the BiLSTM; right-side batch padding
therefore cannot alter a sample's valid CNN boundary features or recurrent context.

The model accepts only finite float32 tensors in `[-1,+1]` with white `+1`, black `-1`, height 64,
minimum width 4, and configured maximum width 2048 by default. It emits raw logits rather than
probabilities. Class zero is CTC blank; vocabulary characters occupy `1..N`; no UNK, PAD, BOS, or
EOS class exists. See `recognizer.md` for the complete contract and limitations.

## Training pipeline

```text
labeled manifest + vocabulary
  -> root-contained OCRLineDataset
  -> seeded dynamic-width DataLoader
  -> CRNN logits + validated CTC loss
  -> fixed-rate AdamW optimization
  -> sample-weighted validation loss
  -> best/ safetensors + strict JSON identity
```

All image preparation and width geometry are shared with recognition. Preflight rejects unseen
characters, sample/document overlap, conflicting prepared images, unsafe or invalid images, model
width violations, and impossible CTC alignments before output creation. Only the lowest qualifying
validation loss is persisted. Optimizer state is deliberately absent, so exact resume is not
claimed.

## Planned, not implemented

- Phases 11–13: inference, assembly, metrics, benchmarks, CLI, and reference HTTP adapter.

Future directories and imports do not exist until their phase supplies meaningful tested code.

## Synthetic fixture layer

`data.synthetic` is an implemented data-generation layer, not recognition. `synthetic-v1`
combines the frozen project-authored text set, exact reviewed Noto Nastaliq Urdu font hash,
strict `SyntheticDataConfig`, stable IDs, and per-ID derived seeds. HarfBuzz performs contextual
RTL shaping and FreeType rasterizes glyphs directly; missing shaping dependencies, font drift,
missing glyphs, or noncanonical text fail before generation.

Line output stays grayscale and naturally sized. Page composition adds only the layout families
needed to exercise existing contracts: one column, balanced/uneven two columns, spanning bands,
wide lines, blank/near-blank, and mild skew. Ground truth stores tight composition and transformed
half-open boxes. This data feeds the existing preprocessing and segmentation layers for
development acceptance checks; it is not a Phase 12 benchmark and no aggregate accuracy is
published.
