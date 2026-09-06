"""Public package metadata for the Urdu document OCR reimplementation."""

import logging as _logging

__version__ = "0.1.0"

_logging.getLogger(__name__).addHandler(_logging.NullHandler())

__all__ = ["__version__"]
