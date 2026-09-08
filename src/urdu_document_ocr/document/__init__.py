"""Bounded document ingestion and deterministic result assembly."""

from urdu_document_ocr.document.assembly import (
    assemble_document,
    assemble_page,
    document_to_dict,
    document_to_json,
    document_to_text,
    write_json_result,
    write_text_result,
)
from urdu_document_ocr.document.ingestion import load_document

__all__ = [
    "assemble_document",
    "assemble_page",
    "document_to_dict",
    "document_to_json",
    "document_to_text",
    "load_document",
    "write_json_result",
    "write_text_result",
]
