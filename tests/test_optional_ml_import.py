from __future__ import annotations

import subprocess
import sys


def test_core_import_does_not_require_torch_and_ml_access_is_actionable() -> None:
    script = r"""
import sys

class TorchBlocker:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "torch" or fullname.startswith("torch."):
            raise ModuleNotFoundError("blocked for core-only test", name="torch")
        return None

sys.meta_path.insert(0, TorchBlocker())
import urdu_document_ocr
assert "torch" not in sys.modules
assert "safetensors" not in sys.modules
assert urdu_document_ocr.RecognizerConfig().normalized_height == 64
assert callable(urdu_document_ocr.recognize_document)
assert callable(urdu_document_ocr.document_to_json)
try:
    urdu_document_ocr.CRNNRecognizer
except ImportError as error:
    assert "urdu-document-ocr[ml]" in str(error)
else:
    raise AssertionError("ML API access unexpectedly succeeded without torch")
try:
    urdu_document_ocr.load_recognizer
except ImportError as error:
    assert "urdu-document-ocr[ml]" in str(error)
else:
    raise AssertionError("checkpoint inference API access unexpectedly succeeded without torch")
try:
    urdu_document_ocr.OCRLineDataset
except ImportError as error:
    assert "urdu-document-ocr[ml]" in str(error)
else:
    raise AssertionError("training API access unexpectedly succeeded without ML dependencies")
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
