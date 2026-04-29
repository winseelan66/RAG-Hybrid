from __future__ import annotations

from pathlib import Path
from uuid import UUID, uuid4


ASSET_ROOT = Path(__file__).resolve().parent.parent / "extracted_assets"


def save_binary_asset(document_id: UUID, source_name: str, extension: str, payload: bytes) -> str:
    source_stem = Path(source_name).stem
    safe_stem = "".join(character if character.isalnum() else "_" for character in source_stem).strip("_") or "document"
    asset_dir = ASSET_ROOT / str(document_id) / safe_stem
    asset_dir.mkdir(parents=True, exist_ok=True)

    normalized_extension = extension if extension.startswith(".") else f".{extension}"
    file_name = f"{uuid4().hex}{normalized_extension.lower()}"
    target = asset_dir / file_name
    target.write_bytes(payload)
    return str(target)
