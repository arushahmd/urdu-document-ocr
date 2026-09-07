from __future__ import annotations

from pathlib import Path

from PIL import Image

from urdu_document_ocr.types import DatasetSample


def sample(
    index: int,
    *,
    document_id: str = "document-1",
    text: str = "اردو متن",
    image_path: str | None = None,
    tags: tuple[str, ...] = (),
) -> DatasetSample:
    return DatasetSample(
        schema_version=1,
        sample_id=f"sample-{index:03d}",
        image_path=image_path or f"images/line-{index:03d}.png",
        text=text,
        document_id=document_id,
        page_id=f"page-{index // 10:03d}",
        line_index=index,
        tags=tags,
    )


def write_image(
    root: Path,
    relative_path: str,
    *,
    size: tuple[int, int] = (32, 8),
    image_format: str | None = None,
) -> Path:
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("L", size, color=255).save(path, format=image_format)
    return path
