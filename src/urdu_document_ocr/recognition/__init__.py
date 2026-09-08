"""Optional PyTorch recognition API."""

from importlib.util import find_spec

if find_spec("torch") is None:
    raise ImportError("PyTorch recognition support is optional; install urdu-document-ocr[ml]")

from urdu_document_ocr.recognition.base import OCRRecognizer, RecognizerOutput
from urdu_document_ocr.recognition.ctc import (
    CTCAlignmentReport,
    compute_ctc_loss,
    minimum_ctc_timesteps,
    validate_ctc_alignment,
)
from urdu_document_ocr.recognition.decoding import greedy_ctc_decode
from urdu_document_ocr.recognition.inference import (
    LoadedRecognizer,
    load_recognizer,
    recognize_line,
    recognize_lines,
)
from urdu_document_ocr.recognition.model import (
    CRNNRecognizer,
    input_width_to_timesteps,
    prepare_line_image,
)

__all__ = [
    "CRNNRecognizer",
    "CTCAlignmentReport",
    "LoadedRecognizer",
    "OCRRecognizer",
    "RecognizerOutput",
    "compute_ctc_loss",
    "greedy_ctc_decode",
    "input_width_to_timesteps",
    "load_recognizer",
    "minimum_ctc_timesteps",
    "prepare_line_image",
    "recognize_line",
    "recognize_lines",
    "validate_ctc_alignment",
]
