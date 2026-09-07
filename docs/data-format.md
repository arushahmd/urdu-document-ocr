# OCR data format

Phase 7 defines a transparent data-engineering boundary for caller-supplied labeled OCR line
images. It does not create labels from segmented documents: each image must already have an
explicit transcription.

## Manifest schema v1

Manifests are UTF-8 JSON Lines without a BOM. Each nonblank line is one object. Whitespace-only
lines are ignored while physical line numbers remain available in diagnostics.

Required fields:

| Field | Type | Rule |
|---|---|---|
| `schema_version` | integer | Exactly `1`; booleans are not integers here. |
| `sample_id` | string | Nonempty and unique within the manifest. |
| `image_path` | string | Portable relative path with POSIX separators after parsing. |
| `text` | string | Nonempty single-line transcription satisfying the Unicode policy below. |
| `document_id` | string | Nonempty group key used to prevent split leakage. |

Optional fields:

| Field | Type | Rule |
|---|---|---|
| `page_id` | string | Nonempty stable page identifier. |
| `line_index` | integer | Nonnegative reading-order position. |
| `tags` | string array | Nonempty, unique values; stored in sorted order. |

Unknown top-level fields, duplicate JSON keys, nonstandard numeric constants, and malformed or
non-object JSON fail closed. There is no arbitrary `metadata` field in schema v1.

```json
{"document_id":"document-001","image_path":"images/line-0001.png","line_index":0,"page_id":"page-001","sample_id":"line-0001","schema_version":1,"text":"محفوظ متن"}
```

`read_manifest` preserves file order and never requires a dataset root. `write_manifest` emits
compact readable Unicode, stable key order, records sorted by `sample_id`, and a final newline.
Absent optional fields are omitted. Existing files are rejected unless `overwrite=True` is
explicit, and replacement uses a complete temporary file in the destination directory.

## Paths and images

Lexical validation rejects absolute POSIX paths, Windows drives, UNC/rooted paths, URI-like
values, NUL, and parent traversal. Full validation resolves both the declared dataset root and
`dataset_root / image_path`; the result must remain beneath the resolved root, including through
symlinks. Serialized errors and reports contain sample IDs and issue codes, never the resolved
absolute root.

Line samples must decode as PNG or JPEG through Pillow, be regular files, occupy no more than 100
MiB, have positive dimensions, and contain no more than 50 million pixels. The byte-size bound is
checked before decoding, and Pillow decompression-bomb protection remains active. A dimension
below 4 pixels, width above 4096 pixels, or aspect ratio above 50 is a review warning rather than
automatic rejection. `inspect_images=False` provides lightweight structure/text/vocabulary
validation without filesystem access.

## Unicode and whitespace

The normalization policy identifier is `nfc-v1`. Manifest reading and dataset validation never
modify a label. Canonical training text must:

- be Unicode NFC;
- be nonempty;
- contain no leading or trailing whitespace;
- use one ASCII space between words, without repeated spaces;
- contain no NUL, tab, line/paragraph separator, surrogate, or other Unicode control/format
  character except ZWNJ (`U+200C`).

ZWNJ, spaces, punctuation, Urdu/Arabic digits, diacritics, and distinct Urdu/Arabic letter forms
remain separate code points. There is no folding between Yeh variants, Kaf variants, digit
families, punctuation, or diacritics.

`normalize_transcription` is an explicit transformation utility. It rejects forbidden controls,
applies NFC, trims surrounding Unicode whitespace, collapses internal Unicode whitespace runs to
one ASCII space, and rejects an empty result. Callers must deliberately write the returned value
to a new manifest; readers never invoke it silently.

## Validation and statistics

Errors prevent training-ready use: duplicate sample IDs, root escape, missing/non-file/unreadable
or unsupported images, excessive image byte size, unsafe dimensions, conflicting transcriptions for identical image bytes,
and noncanonical transcription text. Warnings identify duplicate paths, duplicate image bytes,
duplicate image/transcription pairs, unusual dimensions, labels longer than 512 code points, and
characters outside a supplied vocabulary.

The immutable report contains deterministic issues sorted by severity, code, and sample ID, plus
sample/document counts; minimum/median/maximum width, height, character length, word count, and
samples per document; exact code-point frequencies; unique-character count; and image/error
counts. It does not repair files or labels.

## Dataset identity

`dataset_fingerprint` is SHA-256 over canonical JSONL logical records, including present optional
fields. Records are sorted by `sample_id` with canonical bytes as a duplicate tie-break, so
manifest input order, absolute roots, timestamps, and filesystem metadata do not affect identity.
For a valid canonically written manifest, the logical fingerprint equals the file SHA-256.

## Document-grouped splits

`DatasetSplitConfig` defaults to train `0.80`, validation `0.10`, test `0.10`, and seed `1337`.
Ratios are nonnegative, must sum to `1.0` within `1e-12`, and zero-ratio partitions are ineligible.

Algorithm `group-greedy-v1` is frozen as follows:

1. Validate sample identity and canonical transcription without changing labels.
2. Group every sample by `document_id`.
3. Sort groups by descending sample count, then the bytewise SHA-256 digest of
   `str(seed) + NUL + document_id`, then UTF-8 `document_id` bytes.
4. For each group, tentatively add it to every positive-ratio partition and calculate the sum of
   absolute deviations from all three requested sample-count targets.
5. Choose the smallest deviation, breaking exact ties as train, validation, then test.
6. Sort records inside each output by `sample_id`.

Groups are indivisible, so achieved ratios may differ from requested ratios. `DatasetSplitResult`
records sample/document counts, achieved ratios, document assignments, source dataset fingerprint,
algorithm, seed, and split fingerprint. The split fingerprint hashes the source identity,
algorithm, seed, requested ratios, and sorted document assignments. `write_split_manifests` writes
`train.jsonl`, `validation.jsonl`, `test.jsonl`, and `split_metadata.json` into an existing explicit
directory without silent overwrite.

## Vocabulary

`build_vocabulary` should consume the training partition. It requires canonical labels, includes
ordinary space when present, and sorts unique Unicode code points by numeric value. CTC blank is
index `0`; characters are indices `1..N`. There is no PAD or UNK output class.

`find_unseen_characters` reports code points appearing in validation/test data but absent from the
training vocabulary. Encoding an unseen character fails explicitly. Vocabulary JSON stores schema
version, `nfc-v1`, blank index, character array, and the existing canonical fingerprint. Loading
rechecks exact fields, code-point order, uniqueness, single-code-point entries, policy, and hash.

## Review overlays

Review decisions are immutable JSONL records with `schema_version`, `sample_id`, `status`, `tags`,
and the source dataset fingerprint, plus optional short `note` and generic `reviewer` label. Status
is `valid`, `invalid`, or `needs_review`; `heading`, `other-font`, and `miscellaneous` are tags, not
statuses.

Applying an overlay returns new sample records. `valid` remains, `invalid` is excluded, and
`needs_review` is excluded unless `include_needs_review=True`. Tags are merged on retained derived
records; unmentioned samples remain. Duplicate decisions, unknown sample IDs, and stale source
fingerprints fail. Source manifests and images are never moved, edited, or deleted.

## Canonical synthetic fixtures

`data/sample/manifest.jsonl` uses the schema above without extensions. Every record is tagged
`synthetic`, uses a stable `syn-line-NNNNNN` ID and a neutral `synthetic-document-NNN` group, and
points once to a variable-width PNG beneath `lines/`. The grouped manifests under `splits/` use
the same schema. `synthetic-fixture-vocabulary.json` uses the existing vocabulary schema and is
derived only from the committed training split.

`generation-manifest.json` is a separate generation-evidence document. It records
`synthetic-v1`, frozen configuration and fingerprint, shaping capabilities, line records, page
records, validation report, split evidence, and vocabulary fingerprint. A line record includes
text/source hash, font identity/hash/size, master and derived seeds, dimensions, asymmetric
padding, intensity values, blur/noise/skew values, operation order, relative image path, and PNG
hash. It does not alter the training manifest schema.

`page-ground-truth.json` uses schema version 1 and explicit half-open coordinates. Each page has
a stable ID, layout family, relative path and image hash, dimensions, seeds, blank expectation,
and ordered lines. Each line stores exact text/hash, its pre-rotation `composition_box`, final
`output_box`, `line` or `spanning` kind, right/left column index (`0`/`1`) or `null`, and contiguous
reading-order index.

`text-provenance.json` records project-authored source records and code-point coverage.
`provenance.json` records the font boundary and fixture/visual origins. `artifact-manifest.json`
hashes every other file in the package by relative path, size, SHA-256, and logical role; it omits
itself to avoid a recursive hash.
