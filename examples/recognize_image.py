"""Recognize one supported image through the public checkpoint-backed API."""

from __future__ import annotations

import argparse
from pathlib import Path

from urdu_document_ocr import (
    load_recognizer,
    recognize_document,
    write_json_result,
    write_text_result,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="PNG or JPEG input path")
    parser.add_argument("checkpoint", type=Path, help="Phase 10 checkpoint directory")
    parser.add_argument("text_output", type=Path, help="explicit UTF-8 TXT destination")
    parser.add_argument("--json-output", type=Path, help="optional structured JSON destination")
    parser.add_argument("--device", default="cpu", help="cpu, cuda, or cuda:<index>")
    parser.add_argument("--overwrite", action="store_true", help="replace existing outputs")
    args = parser.parse_args()

    recognizer = load_recognizer(args.checkpoint, device=args.device)
    result = recognize_document(args.input, recognizer)
    write_text_result(result, args.text_output, overwrite=args.overwrite)
    if args.json_output is not None:
        write_json_result(result, args.json_output, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
