from __future__ import annotations

import ast
import logging
import re
import subprocess
from pathlib import Path

import urdu_document_ocr

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

FORBIDDEN_SUFFIXES = {
    ".ckpt",
    ".db",
    ".env",
    ".ipynb",
    ".joblib",
    ".log",
    ".pickle",
    ".pkl",
    ".pt",
    ".pth",
    ".safetensors",
    ".sqlite",
    ".sqlite3",
}

CONTENT_SUFFIXES = {".md", ".py", ".toml", ".yaml", ".yml"}
HISTORICAL_SCAN_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".pdf", ".png", ".tif", ".tiff"}
DATASET_ARTIFACT_SUFFIXES = {".csv", ".jsonl", ".tsv"}
HISTORICAL_VOCABULARY_STEMS = {"charset", "vocab", "vocabulary"}
MAXIMUM_EXPECTED_FILE_BYTES = 1_000_000
WINDOWS_ABSOLUTE_PATH = re.compile(r"(?<![A-Za-z0-9])[A-Za-z]:[\\/](?:[^\s`\"']+[\\/])+[^\s`\"']+")
PRIVATE_HOME_PATH = re.compile(r"/home/[A-Za-z0-9._-]+/")
INTERNAL_IPV4 = re.compile(
    r"(?<!\d)(?:10(?:\.\d{1,3}){3}|192\.168(?:\.\d{1,3}){2}|"
    r"172\.(?:1[6-9]|2\d|3[01])(?:\.\d{1,3}){2})(?!\d)"
)
PERSONAL_EMAIL = re.compile(r"(?i)(?<![\w.+-])[\w.+-]+@[a-z0-9-]+(?:\.[a-z0-9-]+)+")
CREDENTIAL_ASSIGNMENT = re.compile(
    r"(?i)(?:password|api[_-]?key|secret|access[_-]?token)\s*[:=]\s*[\"'][^\"']+[\"']"
)
PRIVATE_KEY_HEADER = "BEGIN " + "PRIVATE KEY"
HISTORICAL_ORGANIZATION_MARKERS = ("Digi" + "tho", "Cyg" + "nus")


def candidate_files() -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
    )
    return sorted(
        (REPOSITORY_ROOT / item.decode()).resolve() for item in result.stdout.split(b"\0") if item
    )


def public_text_files() -> list[Path]:
    return [
        path
        for path in candidate_files()
        if path.suffix.lower() in CONTENT_SUFFIXES
        and "tests" not in path.relative_to(REPOSITORY_ROOT).parts
    ]


def test_candidate_tree_has_no_forbidden_artifacts() -> None:
    forbidden = [
        path.relative_to(REPOSITORY_ROOT).as_posix()
        for path in candidate_files()
        if path.suffix.lower() in FORBIDDEN_SUFFIXES
    ]

    assert forbidden == []


def test_candidate_tree_has_no_historical_scans_or_unexpected_large_files() -> None:
    scans = [
        path.relative_to(REPOSITORY_ROOT).as_posix()
        for path in candidate_files()
        if path.suffix.lower() in HISTORICAL_SCAN_SUFFIXES
    ]
    large = [
        path.relative_to(REPOSITORY_ROOT).as_posix()
        for path in candidate_files()
        if path.stat().st_size > MAXIMUM_EXPECTED_FILE_BYTES
    ]

    assert scans == []
    assert large == []


def test_candidate_tree_has_no_historical_dataset_or_vocabulary_artifacts() -> None:
    findings: list[str] = []
    for path in candidate_files():
        relative = path.relative_to(REPOSITORY_ROOT)
        is_dataset_artifact = (
            relative.parts[0].casefold() in {"data", "datasets"}
            or path.suffix.casefold() in DATASET_ARTIFACT_SUFFIXES
        )
        is_vocabulary_artifact = (
            path.stem.casefold() in HISTORICAL_VOCABULARY_STEMS
            and path.suffix.casefold() in {".json", ".txt"}
        )
        if is_dataset_artifact or is_vocabulary_artifact:
            findings.append(relative.as_posix())

    assert findings == []


def test_repository_has_no_license_or_premature_modules() -> None:
    assert not any(
        (REPOSITORY_ROOT / name).exists() for name in ("LICENSE", "LICENSE.md", "COPYING")
    )
    package = REPOSITORY_ROOT / "src" / "urdu_document_ocr"
    assert not any((package / name).exists() for name in ("recognition", "training", "evaluation"))
    assert not any((package / name).exists() for name in ("segmentation.py", "layout.py", "api.py"))
    assert not (package / "data" / "synthetic.py").exists()


def test_public_files_have_no_private_paths_or_credential_values() -> None:
    findings: list[str] = []
    for path in public_text_files():
        text = path.read_text(encoding="utf-8")
        relative = path.relative_to(REPOSITORY_ROOT).as_posix()
        if WINDOWS_ABSOLUTE_PATH.search(text):
            findings.append(f"absolute Windows path in {relative}")
        if PRIVATE_HOME_PATH.search(text):
            findings.append(f"private home path in {relative}")
        if INTERNAL_IPV4.search(text):
            findings.append(f"internal IPv4 address in {relative}")
        if PERSONAL_EMAIL.search(text):
            findings.append(f"personal email in {relative}")
        if CREDENTIAL_ASSIGNMENT.search(text):
            findings.append(f"credential-like assignment in {relative}")
        if PRIVATE_KEY_HEADER in text:
            findings.append(f"private-key header in {relative}")
        for marker in HISTORICAL_ORGANIZATION_MARKERS:
            if marker.casefold() in text.casefold():
                findings.append(f"historical organization marker in {relative}")

    assert findings == []


def test_source_does_not_import_unsafe_serialization_or_future_frameworks() -> None:
    forbidden_imports = {"joblib", "pickle", "torch"}
    findings: list[str] = []
    source_root = REPOSITORY_ROOT / "src"
    for path in source_root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [alias.name.split(".", maxsplit=1)[0] for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module.split(".", maxsplit=1)[0]]
            for name in names:
                if name in forbidden_imports:
                    findings.append(f"{name} imported by {path.relative_to(REPOSITORY_ROOT)}")

    assert findings == []


def test_gitignore_covers_private_runtime_and_model_products() -> None:
    ignored = (REPOSITORY_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()

    for required in (".env", "*.log", "checkpoints/", "*.pkl", "*.safetensors", "data/private/"):
        if required == "*.pkl":
            assert "*.pkl" not in ignored  # Pickle files are rejected rather than merely ignored.
        else:
            assert required in ignored


def test_package_logging_is_quiet_by_default() -> None:
    package_logger = logging.getLogger(urdu_document_ocr.__name__)

    assert any(isinstance(handler, logging.NullHandler) for handler in package_logger.handlers)
