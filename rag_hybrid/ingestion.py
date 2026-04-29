from __future__ import annotations

from uuid import uuid4

from rag_hybrid.app_logging import get_logger
from rag_hybrid.db import count_document_chunks
from rag_hybrid.db import insert_chunks as insert_pgvector_chunks
from rag_hybrid.embeddings import embed_text
from rag_hybrid.extractor import extract_file
from rag_hybrid.graph import store_tables
from rag_hybrid.ingestion_handlers import ImageIngestionHandler, TextIngestionHandler

logger = get_logger()


def ingest_uploaded_file(file_name: str, file_bytes: bytes, vector_store: str = "pgVector") -> dict[str, int | str]:
    document_id = uuid4()
    vector_backend = _vector_backend(vector_store)
    logger.info("Starting ingestion for file '%s' with document_id=%s using %s.", file_name, document_id, vector_store)
    chunks, tables = extract_file(file_name, file_bytes, document_id)
    logger.info(
        "Extraction completed for file '%s': %s chunks, %s tables.",
        file_name,
        len(chunks),
        len(tables),
    )
    vector_chunks = [chunk for chunk in chunks if chunk.content_type in {"text", "image"}]
    if vector_backend == "Qdrant":
        text_count = TextIngestionHandler().ingest(document_id, chunks)
        image_count = ImageIngestionHandler().ingest(document_id, chunks)
        stored_chunks = text_count + image_count
        verified_vector_rows = stored_chunks
        vector_storage = "Qdrant"
    else:
        embeddings = [embed_text(chunk.content) for chunk in vector_chunks]
        logger.info("Generated %s embeddings for file '%s'.", len(embeddings), file_name)
        stored_chunks = insert_pgvector_chunks(document_id, vector_chunks, embeddings)
        verified_vector_rows = count_document_chunks(document_id)
        text_count = sum(1 for chunk in vector_chunks if chunk.content_type == "text")
        image_count = sum(1 for chunk in vector_chunks if chunk.content_type == "image")
        vector_storage = "pgVector"

    stored_tables = store_tables(tables)
    logger.info(
        "Ingestion finished for file '%s': vector_store=%s, stored_chunks=%s, stored_tables=%s.",
        file_name,
        vector_store,
        stored_chunks,
        stored_tables,
    )

    return {
        "document_id": str(document_id),
        "stored_chunks": stored_chunks,
        "stored_text_chunks": text_count,
        "stored_image_chunks": image_count,
        "verified_vector_rows": verified_vector_rows,
        "stored_tables": stored_tables,
        "table_storage": "Neo4j",
        "vector_storage": vector_storage,
        "vector_store": vector_store,
        "source": file_name,
    }


def _vector_backend(vector_store: str) -> str:
    return "Qdrant" if "qdrant" in vector_store.lower() else "pgVector"
