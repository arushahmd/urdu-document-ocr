"""Command-line adapters over the public Urdu OCR library contracts."""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
from collections.abc import Callable, Sequence
from dataclasses import replace

from urdu_document_ocr import __version__
from urdu_document_ocr.config import (
    ConfigurationError,
    DatasetSplitConfig,
    InferenceConfig,
    PreprocessingConfig,
    RecognizerConfig,
    SegmentationConfig,
    SyntheticDataConfig,
    TrainingConfig,
)
from urdu_document_ocr.data import (
    dataset_fingerprint,
    generate_line_dataset,
    load_vocabulary,
    read_manifest,
    split_dataset,
    validate_dataset,
    write_split_manifests,
)
from urdu_document_ocr.document import load_document
from urdu_document_ocr.errors import UrduOCRError
from urdu_document_ocr.evaluation import write_canonical_json
from urdu_document_ocr.vision import preprocess_page, segment_page

_PROGRAM = "urdu-ocr"


class CLIInputError(ValueError):
    """An expected user-facing CLI boundary failure."""


def _configure_utf8_stream(stream: object) -> None:
    reconfigure = getattr(stream, "reconfigure", None)
    if callable(reconfigure):
        with contextlib.suppress(OSError, ValueError):
            reconfigure(encoding="utf-8")


def _positive_integer(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a positive integer") from error
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _port(value: str) -> int:
    parsed = _positive_integer(value)
    if parsed > 65_535:
        raise argparse.ArgumentTypeError("must be between 1 and 65535")
    return parsed


def _add_output_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output", help="write JSON to this file instead of stdout")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace the requested output artifact if it already exists",
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the stable Phase 13 argument parser without optional imports."""

    parser = argparse.ArgumentParser(
        prog=_PROGRAM,
        description="Urdu document OCR ingestion, training, inference, and evaluation.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    commands = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    recognize = commands.add_parser("recognize", help="recognize an image or PDF")
    recognize.add_argument("input", help="PNG, JPEG, or PDF input path")
    recognize.add_argument("--checkpoint", required=True, help="compatible checkpoint directory")
    recognize.add_argument("--format", choices=("text", "json"), default="text")
    recognize.add_argument("--output", help="write the selected format to this file")
    recognize.add_argument("--device", default="cpu", help="cpu, cuda, or cuda:N")
    recognize.add_argument("--batch-size", type=_positive_integer, default=16)
    recognize.add_argument("--pdf-dpi", type=_positive_integer)
    recognize.add_argument("--overwrite", action="store_true")
    recognize.set_defaults(handler=_run_recognize, optional_extra="ml")

    segment = commands.add_parser("segment", help="segment document lines without a model")
    segment.add_argument("input", help="PNG, JPEG, or PDF input path")
    segment.add_argument("--pdf-dpi", type=_positive_integer)
    _add_output_arguments(segment)
    segment.set_defaults(handler=_run_segment)

    train = commands.add_parser("train", help="train a new CRNN-CTC model")
    train.add_argument("--train-manifest", required=True)
    train.add_argument("--validation-manifest", required=True)
    train.add_argument("--dataset-root", required=True)
    train.add_argument("--vocabulary", required=True)
    train.add_argument("--output-dir", required=True)
    train.add_argument("--device", default="cpu", help="cpu, cuda, or cuda:N")
    train.add_argument("--epochs", type=_positive_integer, default=20)
    train.add_argument("--batch-size", type=_positive_integer, default=16)
    train.add_argument("--learning-rate", type=float, default=0.0005)
    train.add_argument("--weight-decay", type=float, default=0.0001)
    train.add_argument("--gradient-clip", type=float, default=5.0)
    train.add_argument("--seed", type=int, default=1337)
    train.add_argument("--patience", type=_positive_integer, default=5)
    train.add_argument("--min-delta", type=float, default=0.0)
    train.add_argument("--num-workers", type=int, default=0)
    train.add_argument("--max-image-width", type=_positive_integer, default=2048)
    train.add_argument("--checkpoint-directory", default="best")
    train.epilog = "Checkpoints contain model weights and metadata, not optimizer resume state."
    train.set_defaults(handler=_run_train, optional_extra="ml")

    evaluate = commands.add_parser("evaluate", help="evaluate a checkpoint against a manifest")
    evaluate.add_argument("--manifest", required=True)
    evaluate.add_argument("--dataset-root", required=True)
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument("--device", default="cpu", help="cpu, cuda, or cuda:N")
    evaluate.add_argument("--batch-size", type=_positive_integer, default=16)
    _add_output_arguments(evaluate)
    evaluate.set_defaults(handler=_run_evaluate, optional_extra="ml")

    validate = commands.add_parser("validate-data", help="validate a line-dataset manifest")
    validate.add_argument("--manifest", required=True)
    validate.add_argument("--dataset-root", required=True)
    _add_output_arguments(validate)
    validate.set_defaults(handler=_run_validate_data)

    split = commands.add_parser("split-data", help="write document-grouped dataset splits")
    split.add_argument("--manifest", required=True)
    split.add_argument("--output-dir", required=True)
    split.add_argument("--train-ratio", type=float, default=0.80)
    split.add_argument("--validation-ratio", type=float, default=0.10)
    split.add_argument("--test-ratio", type=float, default=0.10)
    split.add_argument("--seed", type=int, default=1337)
    split.add_argument("--overwrite", action="store_true")
    split.set_defaults(handler=_run_split_data)

    synthetic = commands.add_parser(
        "generate-synthetic", help="generate a provenance-cleared synthetic line dataset"
    )
    synthetic.add_argument("--output-dir", required=True)
    synthetic.add_argument("--count", type=_positive_integer, required=True)
    synthetic.add_argument("--seed", type=int, default=SyntheticDataConfig().seed)
    synthetic.add_argument("--overwrite", action="store_true")
    synthetic.set_defaults(handler=_run_generate_synthetic)

    serve = commands.add_parser("serve", help="run the local reference HTTP API")
    serve.add_argument("--checkpoint", required=True, help="compatible checkpoint directory")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=_port, default=8000)
    serve.add_argument("--device", default="cpu", help="cpu, cuda, or cuda:N")
    serve.add_argument("--batch-size", type=_positive_integer, default=16)
    serve.set_defaults(handler=_run_serve, optional_extra="ml,api")
    return parser


def _json_text(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def _emit_json(value: object, *, output: str | None, overwrite: bool) -> None:
    if output is None:
        sys.stdout.write(_json_text(value))
    else:
        write_canonical_json(value, output, overwrite=overwrite)


def _run_recognize(arguments: argparse.Namespace) -> int:
    from urdu_document_ocr.document import (
        document_to_json,
        document_to_text,
        write_json_result,
        write_text_result,
    )
    from urdu_document_ocr.pipeline import recognize_document
    from urdu_document_ocr.recognition import load_recognizer

    inference = InferenceConfig(batch_size=arguments.batch_size)
    recognizer = load_recognizer(arguments.checkpoint, device=arguments.device)
    result = recognize_document(
        arguments.input,
        recognizer,
        pdf_dpi=arguments.pdf_dpi,
        inference_config=inference,
    )
    if arguments.output is None:
        projection = (
            document_to_json(result) if arguments.format == "json" else document_to_text(result)
        )
        sys.stdout.write(projection)
        return 0
    if arguments.format == "json":
        write_json_result(result, arguments.output, overwrite=arguments.overwrite)
    else:
        write_text_result(result, arguments.output, overwrite=arguments.overwrite)
    return 0


def _run_segment(arguments: argparse.Namespace) -> int:
    pages = load_document(arguments.input, pdf_dpi=arguments.pdf_dpi)
    result_pages: list[dict[str, object]] = []
    for page in pages:
        processed = preprocess_page(page, PreprocessingConfig())
        regions = segment_page(processed, SegmentationConfig())
        result_pages.append(
            {
                "page_index": page.page_index,
                "source_page_number": page.source_page_number,
                "is_blank": processed.is_blank,
                "deskew_angle_degrees": float(processed.deskew_angle_degrees),
                "regions": [region.to_public_dict() for region in regions],
            }
        )
    _emit_json(
        {"schema_version": 1, "page_count": len(result_pages), "pages": result_pages},
        output=arguments.output,
        overwrite=arguments.overwrite,
    )
    return 0


def _run_train(arguments: argparse.Namespace) -> int:
    from urdu_document_ocr.recognition import CRNNRecognizer
    from urdu_document_ocr.training import train_model

    config = TrainingConfig(
        seed=arguments.seed,
        batch_size=arguments.batch_size,
        epochs=arguments.epochs,
        learning_rate=arguments.learning_rate,
        weight_decay=arguments.weight_decay,
        gradient_clip=arguments.gradient_clip,
        device=arguments.device,
        checkpoint_directory=arguments.checkpoint_directory,
        num_workers=arguments.num_workers,
        early_stopping_patience=arguments.patience,
        min_delta=arguments.min_delta,
        max_image_width=arguments.max_image_width,
    )
    vocabulary = load_vocabulary(arguments.vocabulary)
    model = CRNNRecognizer(vocabulary, RecognizerConfig(max_width=config.max_image_width))
    result = train_model(
        model,
        read_manifest(arguments.train_manifest),
        read_manifest(arguments.validation_manifest),
        vocabulary,
        dataset_root=arguments.dataset_root,
        output_directory=arguments.output_dir,
        config=config,
    )
    sys.stdout.write(_json_text(result.to_public_dict()))
    return 0


def _run_evaluate(arguments: argparse.Namespace) -> int:
    from urdu_document_ocr.evaluation import analyze_errors, evaluate_checkpoint
    from urdu_document_ocr.recognition import load_recognizer

    samples = read_manifest(arguments.manifest)
    inference = InferenceConfig(batch_size=arguments.batch_size)
    recognizer = load_recognizer(arguments.checkpoint, device=arguments.device)
    report = evaluate_checkpoint(
        samples,
        dataset_root=arguments.dataset_root,
        recognizer=recognizer,
        batch_size=inference.batch_size,
        config_fingerprint=inference.fingerprint,
    )
    payload = {
        "evaluation": report.to_public_dict(),
        "error_analysis": analyze_errors(report).to_public_dict(),
    }
    _emit_json(payload, output=arguments.output, overwrite=arguments.overwrite)
    return 0


def _run_validate_data(arguments: argparse.Namespace) -> int:
    samples = read_manifest(arguments.manifest)
    report = validate_dataset(samples, dataset_root=arguments.dataset_root)
    _emit_json(
        report.to_public_dict(),
        output=arguments.output,
        overwrite=arguments.overwrite,
    )
    return 0 if report.is_valid else 2


def _run_split_data(arguments: argparse.Namespace) -> int:
    config = DatasetSplitConfig(
        train_ratio=arguments.train_ratio,
        validation_ratio=arguments.validation_ratio,
        test_ratio=arguments.test_ratio,
        seed=arguments.seed,
    )
    result = split_dataset(read_manifest(arguments.manifest), config)
    write_split_manifests(result, arguments.output_dir, overwrite=arguments.overwrite)
    sys.stdout.write(_json_text(result.to_public_dict()))
    return 0


def _run_generate_synthetic(arguments: argparse.Namespace) -> int:
    config = replace(SyntheticDataConfig(), seed=arguments.seed)
    generated = generate_line_dataset(
        arguments.output_dir,
        arguments.count,
        config=config,
        overwrite=arguments.overwrite,
    )
    samples = tuple(item.sample for item in generated)
    sys.stdout.write(
        _json_text(
            {
                "schema_version": 1,
                "sample_count": len(samples),
                "dataset_fingerprint": dataset_fingerprint(samples),
                "config_fingerprint": config.fingerprint,
                "manifest": "manifest.jsonl",
                "generation_manifest": "generation-manifest.json",
            }
        )
    )
    return 0


def _run_serve(arguments: argparse.Namespace) -> int:
    import uvicorn

    from urdu_document_ocr.api import create_app

    app = create_app(
        checkpoint_directory=arguments.checkpoint,
        device=arguments.device,
        batch_size=arguments.batch_size,
    )
    uvicorn.run(app, host=arguments.host, port=arguments.port)
    return 0


def _optional_dependency_message(extra: str) -> str:
    return f"optional dependencies are missing; install urdu-document-ocr[{extra}]"


def main(argv: Sequence[str] | None = None) -> int:
    """Parse one command, translate safe failures, and return a process exit code."""

    _configure_utf8_stream(sys.stdout)
    _configure_utf8_stream(sys.stderr)
    parser = build_parser()
    arguments = parser.parse_args(argv)
    handler: Callable[[argparse.Namespace], int] = arguments.handler
    try:
        return handler(arguments)
    except (ModuleNotFoundError, ImportError) as error:
        extra = getattr(arguments, "optional_extra", None)
        if extra is None:
            sys.stderr.write(f"error: {error}\n")
        else:
            sys.stderr.write(f"error: {_optional_dependency_message(extra)}\n")
        return 2
    except (CLIInputError, ConfigurationError, UrduOCRError) as error:
        sys.stderr.write(f"error: {error}\n")
        return 2
    except Exception:
        sys.stderr.write("error: operation failed unexpectedly\n")
        return 1


if __name__ == "__main__":  # pragma: no cover - exercised by the installed entry point
    raise SystemExit(main())
