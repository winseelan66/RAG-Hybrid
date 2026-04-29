from __future__ import annotations

import re
from uuid import UUID, uuid4

from rag_hybrid.app_logging import get_logger
from rag_hybrid.config import get_settings
from rag_hybrid.embeddings import embed_text
from rag_hybrid.models import ExtractedChunk
from rag_hybrid.qdrant_store import initialize_qdrant_collection, upsert_points

logger = get_logger()


class TextIngestionHandler:
    def ingest(self, document_id: UUID, chunks: list[ExtractedChunk]) -> int:
        text_chunks = [chunk for chunk in chunks if chunk.content_type == "text"]
        if not text_chunks:
            logger.info("No text chunks found for text ingestion.")
            return 0

        settings = get_settings()
        initialize_qdrant_collection(settings.qdrant.text_collection)
        points = []
        for chunk in text_chunks:
            points.append(
                {
                    "id": str(uuid4()),
                    "vector": embed_text(chunk.content),
                    "payload": {
                        "content": chunk.content,
                        "content_type": "text",
                        "document_id": str(document_id),
                        "file_name": chunk.source,
                        "source": chunk.source,
                        "page_number": chunk.page_number,
                        "section_title": chunk.section_title or chunk.section,
                        "section": chunk.section,
                        "document_version": chunk.document_version,
                        "chunk_id": chunk.chunk_id,
                    },
                }
            )

        return upsert_points(settings.qdrant.text_collection, points)


class ImageIngestionHandler:
    def ingest(self, document_id: UUID, chunks: list[ExtractedChunk]) -> int:
        image_chunks = [chunk for chunk in chunks if chunk.content_type == "image"]
        if not image_chunks:
            logger.info("No images found for image ingestion.")
            return 0

        settings = get_settings()
        initialize_qdrant_collection(settings.qdrant.image_collection)
        points = []
        for chunk in image_chunks:
            image_id = str(uuid4())
            caption, topic, image_type = infer_image_metadata(chunk.content)
            embedding_text = " ".join(part for part in [caption, topic, image_type] if part)
            points.append(
                {
                    "id": image_id,
                    "vector": embed_text(embedding_text),
                    "payload": {
                        "content": embedding_text,
                        "content_type": "image",
                        "image_id": image_id,
                        "document_id": str(document_id),
                        "file_name": chunk.source,
                        "source": chunk.source,
                        "page_number": chunk.page_number,
                        "section_title": chunk.section_title or chunk.section,
                        "section": chunk.section,
                        "image_caption": caption,
                        "image_type": image_type,
                        "topic": topic,
                        "storage_url": chunk.metadata.get("asset_path", ""),
                        "metadata_json": chunk.metadata,
                        "document_version": chunk.document_version,
                        "chunk_id": chunk.chunk_id,
                    },
                }
            )

        return upsert_points(settings.qdrant.image_collection, points)


def infer_image_metadata(text: str) -> tuple[str, str, str]:
    lowered = text.lower()
    topic = _topic_from_text(lowered)
    image_type = _image_type_from_text(lowered)

    if image_type == "dimension_diagram" and topic:
        caption = f"Dimension of {topic}"
    elif image_type == "map" and topic:
        caption = f"Map of {topic.title()}"
    elif topic:
        caption = f"Image of {topic}"
    else:
        caption = text.strip() or "Extracted image"

    return caption, topic, image_type


def _topic_from_text(text: str) -> str:
    known_topics = ["portugal", "machine", "network", "coolant", "modbus"]
    for topic in known_topics:
        if topic in text:
            return topic

    match = re.search(r"(?:map of|image of|picture of|dimension of)\s+([a-zA-Z0-9 _-]+)", text)
    if match:
        topic = match.group(1).strip(" .")
        words = [word for word in topic.split() if word not in {"the", "below", "picture", "image"}]
        return " ".join(words[:3])
    return ""


def _image_type_from_text(text: str) -> str:
    if "dimension" in text:
        return "dimension_diagram"
    if "diagram" in text:
        return "diagram"
    if "map" in text:
        return "map"
    if "photo" in text or "picture" in text or "image" in text:
        return "photo"
    return "image"
