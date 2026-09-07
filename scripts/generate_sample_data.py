"""Regenerate the reviewed small public synthetic fixture package."""

from __future__ import annotations

import argparse
from pathlib import Path

from urdu_document_ocr import generate_fixture_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    arguments = parser.parse_args()
    result = generate_fixture_dataset(arguments.output, overwrite=arguments.overwrite)
    print(result.to_public_dict())


if __name__ == "__main__":
    main()
