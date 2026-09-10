"""Thin, in-memory FastAPI adapter for checkpoint-backed document OCR."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from threading import Lock
from typing import Any

from fastapi import FastAPI, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from python_multipart import MultipartParser
from python_multipart.exceptions import MultipartParseError
from python_multipart.multipart import parse_options_header
from starlette.concurrency import run_in_threadpool

from urdu_document_ocr import __version__
from urdu_document_ocr.config import InferenceConfig, InputLimitsConfig
from urdu_document_ocr.document import document_to_dict
from urdu_document_ocr.errors import (
    DocumentLoadError,
    ModelInputError,
    PreprocessingError,
    ResourceLimitError,
    SegmentationError,
    UnsupportedDocumentFormatError,
    UrduOCRError,
)
from urdu_document_ocr.pipeline import recognize_document

_LOGGER = logging.getLogger(__name__)
_MULTIPART_OVERHEAD_BYTES = 64 * 1024


class _UploadError(ValueError):
    pass


class _UploadTooLarge(_UploadError):
    pass


@dataclass(slots=True)
class _MultipartUploadState:
    max_file_bytes: int
    body: bytearray = field(default_factory=bytearray)
    headers: dict[bytes, bytes] = field(default_factory=dict)
    header_field: bytearray = field(default_factory=bytearray)
    header_value: bytearray = field(default_factory=bytearray)
    part_count: int = 0
    accepting_file_data: bool = False
    file_complete: bool = False
    message_complete: bool = False

    def on_part_begin(self) -> None:
        self.part_count += 1
        if self.part_count > 1:
            raise _UploadError("exactly one multipart file field is required")
        self.headers.clear()
        self.accepting_file_data = False

    def on_header_begin(self) -> None:
        self.header_field.clear()
        self.header_value.clear()

    def on_header_field(self, data: bytes, start: int, end: int) -> None:
        self.header_field.extend(data[start:end])

    def on_header_value(self, data: bytes, start: int, end: int) -> None:
        self.header_value.extend(data[start:end])

    def on_header_end(self) -> None:
        name = bytes(self.header_field).strip().lower()
        if not name or name in self.headers:
            raise _UploadError("multipart headers are invalid")
        self.headers[name] = bytes(self.header_value).strip()

    def on_headers_finished(self) -> None:
        disposition, options = parse_options_header(self.headers.get(b"content-disposition"))
        if disposition != b"form-data" or options.get(b"name") != b"file":
            raise _UploadError("multipart input must contain one field named file")
        self.accepting_file_data = True

    def on_part_data(self, data: bytes, start: int, end: int) -> None:
        if not self.accepting_file_data:
            raise _UploadError("multipart file headers are invalid")
        incoming = end - start
        if len(self.body) + incoming > self.max_file_bytes:
            raise _UploadTooLarge("uploaded document exceeds the configured byte limit")
        self.body.extend(data[start:end])

    def on_part_end(self) -> None:
        if not self.accepting_file_data:
            raise _UploadError("multipart file field is invalid")
        self.file_complete = True
        self.accepting_file_data = False

    def on_end(self) -> None:
        self.message_complete = True


def _multipart_callbacks(state: _MultipartUploadState) -> dict[str, Callable[..., None]]:
    return {
        "on_part_begin": state.on_part_begin,
        "on_header_begin": state.on_header_begin,
        "on_header_field": state.on_header_field,
        "on_header_value": state.on_header_value,
        "on_header_end": state.on_header_end,
        "on_headers_finished": state.on_headers_finished,
        "on_part_data": state.on_part_data,
        "on_part_end": state.on_part_end,
        "on_end": state.on_end,
    }


async def _read_multipart_document(request: Request, *, max_file_bytes: int) -> bytes:
    content_type, options = parse_options_header(request.headers.get("content-type"))
    boundary = options.get(b"boundary")
    if content_type != b"multipart/form-data" or not boundary or len(boundary) > 200:
        raise _UploadError("content type must be multipart/form-data with a valid boundary")

    maximum_body_bytes = max_file_bytes + _MULTIPART_OVERHEAD_BYTES
    raw_length = request.headers.get("content-length")
    if raw_length is not None:
        try:
            declared_length = int(raw_length)
        except ValueError as error:
            raise _UploadError("content-length must be a nonnegative integer") from error
        if declared_length < 0:
            raise _UploadError("content-length must be a nonnegative integer")
        if declared_length > maximum_body_bytes:
            raise _UploadTooLarge("multipart request exceeds the configured byte limit")

    state = _MultipartUploadState(max_file_bytes)
    parser = MultipartParser(
        boundary,
        _multipart_callbacks(state),
        max_size=maximum_body_bytes,
        max_header_count=8,
        max_header_size=16_384,
    )
    received = 0
    try:
        async for chunk in request.stream():
            received += len(chunk)
            if received > maximum_body_bytes:
                raise _UploadTooLarge("multipart request exceeds the configured byte limit")
            if parser.write(chunk) != len(chunk):
                raise _UploadTooLarge("multipart request exceeds the configured byte limit")
        parser.finalize()
    except _UploadError:
        raise
    except (MultipartParseError, ValueError) as error:
        raise _UploadError("multipart request could not be parsed") from error
    if state.part_count != 1 or not state.file_complete or not state.message_complete:
        raise _UploadError("multipart request is incomplete or missing the file field")
    return bytes(state.body)


def _safe_error(status_code: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"error": {"code": code, "message": message}},
    )


def _domain_error_response(error: UrduOCRError) -> JSONResponse:
    if isinstance(error, ResourceLimitError):
        return _safe_error(413, error.code, str(error))
    if isinstance(error, UnsupportedDocumentFormatError):
        return _safe_error(415, error.code, str(error))
    if isinstance(
        error,
        (DocumentLoadError, ModelInputError, PreprocessingError, SegmentationError),
    ):
        return _safe_error(422, error.code, str(error))
    return _safe_error(500, "ocr_processing_error", "OCR processing failed")


def _recognize_locked(
    lock: Lock,
    document: bytes,
    recognizer: Any,
    *,
    input_limits: InputLimitsConfig,
    pdf_dpi: int | None,
    inference_config: InferenceConfig,
) -> dict[str, object]:
    with lock:
        result = recognize_document(
            document,
            recognizer,
            input_limits=input_limits,
            pdf_dpi=pdf_dpi,
            inference_config=inference_config,
        )
    return document_to_dict(result)


def _load_service_recognizer(checkpoint_directory: str, *, device: str) -> Any:
    from urdu_document_ocr.recognition import load_recognizer

    return load_recognizer(checkpoint_directory, device=device)


def create_app(
    *,
    checkpoint_directory: str | None = None,
    recognizer: Any | None = None,
    device: str = "cpu",
    batch_size: int = 16,
    input_limits: InputLimitsConfig | None = None,
) -> FastAPI:
    """Create an initialized local/reference app around one reusable recognizer."""

    if (checkpoint_directory is None) == (recognizer is None):
        raise ValueError("supply exactly one of checkpoint_directory or recognizer")
    limits = InputLimitsConfig() if input_limits is None else input_limits
    if not isinstance(limits, InputLimitsConfig):
        raise TypeError("input_limits must be an InputLimitsConfig or None")
    inference = InferenceConfig(batch_size=batch_size)
    active_recognizer = (
        _load_service_recognizer(checkpoint_directory, device=device)
        if recognizer is None and checkpoint_directory is not None
        else recognizer
    )
    method = getattr(active_recognizer, "recognize_batch", None)
    fingerprint = getattr(active_recognizer, "fingerprint", None)
    vocabulary = getattr(active_recognizer, "vocabulary", None)
    vocabulary_fingerprint = getattr(vocabulary, "fingerprint", None)
    if (
        not callable(method)
        or not isinstance(fingerprint, str)
        or not isinstance(vocabulary_fingerprint, str)
    ):
        raise TypeError("recognizer does not satisfy the service contract")

    app = FastAPI(
        title="Urdu Document OCR API",
        version=__version__,
        description="Reference API for checkpoint-backed Urdu document OCR.",
    )
    inference_lock = Lock()

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(
        _request: Request, _error: RequestValidationError
    ) -> JSONResponse:
        return _safe_error(422, "invalid_request", "request parameters are invalid")

    @app.get("/health", summary="Report initialized service state")
    def health() -> dict[str, str]:
        return {
            "status": "ok",
            "model_fingerprint": fingerprint,
            "vocabulary_fingerprint": vocabulary_fingerprint,
        }

    @app.post(
        "/ocr",
        summary="Recognize one uploaded PNG, JPEG, or PDF",
        response_model=None,
        openapi_extra={
            "requestBody": {
                "required": True,
                "content": {
                    "multipart/form-data": {
                        "schema": {
                            "type": "object",
                            "required": ["file"],
                            "properties": {"file": {"type": "string", "format": "binary"}},
                        }
                    }
                },
            }
        },
    )
    async def ocr(
        request: Request,
        pdf_dpi: int | None = Query(default=None, ge=1),
    ) -> dict[str, object] | JSONResponse:
        try:
            document = await _read_multipart_document(
                request,
                max_file_bytes=limits.max_input_bytes,
            )
        except _UploadTooLarge as error:
            return _safe_error(413, "resource_limit_exceeded", str(error))
        except _UploadError as error:
            return _safe_error(422, "invalid_multipart", str(error))

        _LOGGER.info("OCR request accepted", extra={"byte_count": len(document)})
        try:
            return await run_in_threadpool(
                _recognize_locked,
                inference_lock,
                document,
                active_recognizer,
                input_limits=limits,
                pdf_dpi=pdf_dpi,
                inference_config=inference,
            )
        except UrduOCRError as error:
            return _domain_error_response(error)
        except Exception:
            _LOGGER.error("OCR request failed unexpectedly")
            return _safe_error(500, "internal_error", "OCR processing failed unexpectedly")

    return app


__all__ = ["create_app"]
