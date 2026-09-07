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

## Implemented through Phase 6

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

## Planned, not implemented

- Phase 7: JSONL manifests, validation, grouped splitting, and manifest-derived vocabulary.
- Phase 8: provenance-cleared shaped Urdu synthetic fixtures.
- Phases 9–10: PyTorch CNN-BiLSTM-CTC recognition, training, and tensor-only checkpoints.
- Phases 11–13: inference, assembly, metrics, benchmarks, CLI, and reference HTTP adapter.

Future directories and imports do not exist until their phase supplies meaningful tested code.
