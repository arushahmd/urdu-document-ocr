# Synthetic Urdu data

Phase 8 adds `synthetic-v1`, a small provenance-cleared data generator. It exists to exercise
data, shaping, preprocessing, segmentation, and RTL layout contracts before a recognizer exists.
It is not representative real-world OCR data and supports no recognition-quality claim.

## Reviewed inputs

The text set consists of 13 short neutral Urdu phrases authored for this repository. It includes
ordinary joining and non-joining letters, spaces, Urdu and ASCII digits, Urdu punctuation, one
selected diacritic, and a ZWNJ example. Text is kept exactly as NFC under the Phase 7 `nfc-v1`
policy; no folding or silent label conversion occurs. `data/sample/text-provenance.json` records
each phrase, its origin and hash, and a deterministic code-point frequency report.

The bundled font is the unmodified variable TTF from Noto Nastaliq Urdu release
`NotoNastaliqUrdu-v4.000`. Its reviewed SHA-256 is
`eff3a48f588f599f98e98350f1107e2e492edefbe864c8b61c73f2d605f1dce4`. The asset is distributed
under OFL-1.1 with its copyright and complete license text beside it. OFL-1.1 applies only to the
font, not to repository-owned source code; the project source license remains unresolved.

## Shaping and RTL scope

Generation fails closed unless `uharfbuzz` and `freetype-py` can contextually shape and rasterize
the reviewed font. The frozen route uses HarfBuzz with direction `rtl`, script `arab`, language
`ur`, and `calt`, `kern`, and `liga`, followed by direct FreeType glyph rasterization. Numeric runs
are explicitly shaped LTR and placed in RTL run order. This controlled strategy is sufficient for
the authored single-script fixtures. It is not a general Unicode Bidirectional Algorithm
implementation for arbitrary mixed-language paragraphs; callers needing that must supply a
reviewed full-bidi shaping layer in a later version. Pillow RAQM is reported as an environment
capability but is not required by this justified direct shaping route.

## Deterministic rendering

All randomness comes from the explicit master seed `240817`. Each stable sample/page ID derives
an isolated 64-bit seed with SHA-256, so iteration order does not perturb another item. Line images
retain natural width and vary font size and four margins. Light background and dark foreground
intensities maintain at least 64 levels of contrast.

The exact operation order is:

```text
render -> contrast/background -> Gaussian blur -> Gaussian noise -> rotation
```

Applied values, zero included when an operation is skipped, are recorded per line. Blur, Gaussian
noise, and rotation are deliberately mild; skew never exceeds 2.5 degrees in the reference
configuration. No perspective transform or colorful document simulation is used.

Logical records are deterministic for the same source set, configuration, seed, exact font, and
`synthetic-v1` implementation. PNG bytes are frozen and verified for the exact dependency versions
recorded in `generation-manifest.json`; other compliant platforms should rely on logical identity
unless byte equality has also been demonstrated there.

## Pages and public fixture package

Page generation supports one column, balanced and uneven two columns, top/middle/footer spanning
bands, a wide one-column line, blank, near-blank, and mild-skew families. Urdu lines are
right-aligned. In two-column regions the right column is index 0 and is read before left column 1
within a band. A spanning region has no column index and is ordered by its vertical band; it is not
automatically called a heading.

`data/sample/` contains 24 line images and 10 pages. Line metadata uses the existing Phase 7
JSONL schema. Documents remain intact across 16/4/4 train/validation/test manifests. The vocabulary
is derived from training lines only and is explicitly a synthetic fixture vocabulary; validation
and test contain no unseen characters. Page ground truth records transcription, pre-rotation
composition boxes, output boxes, region kind, column, and exact reading-order index using half-open
coordinates.

The technical overlay is produced only from the spanning-top safe fixture and ground truth. Red
marks spanning regions, blue marks right-column lines, green marks left-column lines, and labels
show one-based reading order. Its source and process are recorded in the artifact manifest.

## Safe regeneration

`generate_line_sample`, `generate_line_dataset`, `generate_page_fixture`, and
`generate_fixture_dataset` are library APIs. Dataset writers require an existing explicit output
directory, normalize every relative path, reject roots and traversal, require explicit overwrite,
never delete a directory, reject unrelated existing files, never contact the network, and never
read the historical repository. The script in `scripts/generate_sample_data.py` delegates to the
public API and is a reproduction utility, not the final business CLI.
