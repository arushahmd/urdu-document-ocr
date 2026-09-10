# Thin FastAPI adapter

`urdu_document_ocr.api.create_app` exposes the existing document OCR pipeline as a small local
reference service. Install both optional groups for checkpoint-backed serving:

```console
python -m pip install -e ".[ml,api]"
urdu-ocr serve --checkpoint run-output/best
```

No pretrained or canonical OCR checkpoint is bundled. The service must receive one fixed,
compatible Phase 10 checkpoint at startup. It never accepts a checkpoint path from a request.

## Lifecycle and execution

`create_app(checkpoint_directory=..., device="cpu", batch_size=16)` verifies and loads the
recognizer during app construction. Failure prevents creation of a partially initialized app.
Tests can supply the same recognizer protocol through `recognizer=...`; that path does not create
a competing application design and lets the HTTP boundary run without PyTorch.

Every request reuses the initialized recognizer and an `InferenceConfig` created from the
startup batch size. Multipart input is read asynchronously. The synchronous, CPU-bound document
pipeline then runs in Starlette's thread pool. A minimal process-local lock serializes use of the
shared PyTorch recognizer; the implementation does not add queues or distributed concurrency.

## `GET /health`

Health reports initialized state without running inference:

```json
{
  "status": "ok",
  "model_fingerprint": "<sha256>",
  "vocabulary_fingerprint": "<sha256>"
}
```

It does not expose a checkpoint path, device, hostname, hardware inventory, or document content.

## `POST /ocr`

Send exactly one multipart field named `file`; `pdf_dpi` is an optional integer query parameter.

```console
curl -F "file=@scan.pdf" "http://127.0.0.1:8000/ocr?pdf_dpi=200"
```

The outer multipart type is required, but the file part's filename, extension, and declared MIME
type are untrusted. Existing ingestion identifies actual PDF content from its header and accepts
only images Pillow decodes as PNG or JPEG. The response is exactly the existing
`document_to_dict` projection: ordered pages and lines, geometry, blank status, and UTF-8 text,
with no raw images, masks, logits, weights, confidence, model path, or device information.

## Limits and upload handling

The adapter feeds request-body chunks into `python-multipart` directly and accumulates only the
single document field in memory. It stops when the configured file-byte limit is crossed and
also caps total multipart framing overhead. It does not use an unlimited upload read, Starlette's
temporary-file upload object, or application persistence. The byte array is released after the
request; uploads and OCR results are not archived.

`InputLimitsConfig` remains authoritative. Defaults are 100 MiB input, 100 PDF pages, 50 million
pixels per page, 500 million document pixels, and PDF DPI from 72 through 400 (200 default). An
early `Content-Length` check is only an optimization; streamed byte accounting is always
enforced.

## Error responses

All adapter errors use a fixed safe shape:

```json
{"error": {"code": "unsupported_document_format", "message": "..."}}
```

| Status | Meaning |
|---:|---|
| 413 | Upload, PDF DPI, page, or pixel resource limit exceeded |
| 415 | Actual content is not PNG, JPEG, or PDF |
| 422 | Invalid multipart/query input or a malformed supported document |
| 500 | Unexpected or non-request-correctable OCR processing failure |

Responses contain no decoder tracebacks, filesystem paths, checkpoint metadata, tensor details,
or private machine state.

## Security and deployment boundary

The command binds to `127.0.0.1` by default. The app configures no CORS middleware and includes
no authentication, TLS termination, rate limiting, malware scanning, process isolation, durable
storage, or multi-tenant controls. It does not log filenames, bytes, or OCR text; structural byte
counts and safe fingerprints are the only intended diagnostics.

This is a reference/local adapter, not a hardened internet-facing service. A real deployment
would need an independently designed authenticated reverse proxy, TLS, request/rate controls,
worker and memory isolation, monitoring, retention/privacy policy, and document-security review.
Those controls are intentionally outside V1.
