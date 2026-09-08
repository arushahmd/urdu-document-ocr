# Checkpoint-backed OCR inference

Phase 11 connects the implemented ingestion, vision, CRNN, decoder, and result contracts into a
reusable Python pipeline. No canonical trained model is bundled. A caller must train or otherwise
provide a checkpoint that satisfies the exact Phase 10 format. Randomly initialized and smoke-test
checkpoints validate plumbing only and have no OCR-quality meaning.

## Load once and reuse

Install the optional ML dependencies and load one checkpoint before processing documents:

```python
from urdu_document_ocr import load_recognizer, recognize_document

recognizer = load_recognizer("run-output/best", device="cpu")
first = recognize_document("first.png", recognizer)
second = recognize_document("second.pdf", recognizer, pdf_dpi=200)
```

`load_recognizer` delegates to the Phase 10 loader. The checkpoint directory must contain exactly
`metadata.json`, `model.safetensors`, and `vocabulary.json`. Strict JSON, architecture and
configuration fingerprints, vocabulary content/hash, class count, blank index, tensor names,
shapes, dtypes, finite values, and the safetensors file hash are checked before strict state
loading. The wrapper exposes the validated metadata and safe model fingerprint, never its local
checkpoint path.

The model is moved to the explicit device and set to evaluation mode. `cpu`, `cuda`, and
`cuda:<index>` are accepted. CUDA is never auto-selected and an unavailable requested device
fails without falling back. Each call runs under `torch.inference_mode()`; the same in-memory
weights serve every line and page.

## Line and batch recognition

`recognize_line(image, recognizer)` accepts one grayscale NumPy `uint8[height,width]` crop.
`recognize_lines(images, recognizer, batch_size=16)` accepts several in fixed input order. The
loaded recognizer reuses `prepare_line_image`: aspect ratio is preserved at height 64, pixels map
to `[-1,+1]`, and an over-wide normalized line fails rather than being truncated.

Prepared lines use the same batching primitive as training. Each batch is padded only on the
right with white `+1` to its widest line rounded to width stride four. CPU `int64` valid widths
remain explicit. Batches are never sorted by width. Raw model logits and valid output lengths go
directly to the vocabulary-bound greedy CTC decoder; decoded text receives no spell correction,
language model, variant normalization, or fabricated confidence.

`InferenceConfig` deliberately contains only `batch_size`. Device choice belongs to recognizer
loading, and document resource limits remain in `InputLimitsConfig`.

## Page and document pipeline

`recognize_page` accepts a `PreprocessedPage`. It calls `segment_page`, takes the returned
contiguous `reading_order_index` values as authoritative, extracts owned grayscale crops through
`extract_line_crop`, recognizes them in stable batches, and pairs each prediction with its exact
`LineRegion` in a `LineOCRResult`. It never re-sorts by x-coordinate, width, or model output.

A blank page or a page with no regions returns a valid `PageOCRResult` with zero lines and empty
text; the recognizer is not called. Successful line text is joined with one newline. No column
labels, punctuation inference, or linguistic merging is added: one-column, right-before-left
two-column, and spanning-band text all follow the Phase 6 flattened geometry.

`recognize_document` performs bounded `load_document`, preprocessing, page recognition, and
assembly sequentially. It retains every zero-based page index and one-based source page number,
including blank pages. Adjacent page texts are joined with exactly two newline characters. Thus a
blank middle page contributes an empty page between both separators, producing four consecutive
newlines between the populated page texts. The result stores structure and text, not images,
tensors, logits, or model internals.

Expected ingestion, preprocessing, segmentation, checkpoint, and model failures propagate as
typed errors. Any line-recognition failure fails the page and document operation; the pipeline
does not substitute empty text or return an apparently successful partial result. `LineOCRResult`
retains its existing typed `error_code` representation for explicit downstream assembly, but the
Phase 11 high-level pipeline is fail-closed.

## TXT and JSON output

The pure projections are:

```python
from urdu_document_ocr import document_to_dict, document_to_json, document_to_text

text = document_to_text(first)
public_record = document_to_dict(first)
json_text = document_to_json(first)
```

TXT is exactly `DocumentOCRResult.assembled_text`: UTF-8 Urdu with one newline between lines and
two newline separators between every adjacent page. It has no metadata header or implicit final
newline. JSON is deterministic, sorted-key, indented UTF-8 with `ensure_ascii=False` and one final
newline. It contains:

- document `schema_version`, `pages`, and `assembled_text`;
- page index, source page number, blank flag, assembled text, and ordered lines;
- line region ID, half-open bounding box, region kind, column index, reading-order index, and
  prediction text, emitted character indices, and sequence length or an explicit error code.

It excludes raw images, masks, crops, tensors, logits, checkpoint/source filesystem paths, host or
device information, model weights, private URLs, and machine metadata.

`write_text_result` and `write_json_result` write only to a caller-selected path under an existing
regular directory. They use a same-directory temporary file plus atomic replacement. Existing
files require `overwrite=True`; symbolic-link destinations and implicit directory creation fail.

## Limitations

- No model checkpoint is bundled or downloaded.
- Greedy CTC is the only decoder; there is no language model or postprocessor.
- No calibrated confidence is exposed.
- No OCR accuracy, CER, WER, benchmark, throughput, or production-readiness claim exists yet.
- Phase 12 owns evaluation and benchmark freeze; Phase 13 owns CLI and HTTP adapters.
