from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime
import json
import re
from typing import Iterator
from uuid import UUID, uuid4

import psycopg
from psycopg.rows import dict_row

from rag_hybrid.app_logging import get_logger
from rag_hybrid.config import get_settings
from rag_hybrid.models import ExtractedChunk, SearchChunkResult, StoredChunk

logger = get_logger()
EMBEDDING_DIMENSION = get_settings().extraction.embedding_dimension


CREATE_TABLE_SQL = f"""
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents_embeddings (
    id UUID PRIMARY KEY,
    embedding VECTOR({EMBEDDING_DIMENSION}) NOT NULL,
    content TEXT NOT NULL,
    content_type VARCHAR(32) NOT NULL,
    document_id UUID NOT NULL,
    chunk_id INTEGER NOT NULL,
    section VARCHAR(255),
    source TEXT NOT NULL,
    metadata_json JSONB NOT NULL DEFAULT '{{}}'::jsonb,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

ALTER TABLE documents_embeddings
    ADD COLUMN IF NOT EXISTS metadata_json JSONB NOT NULL DEFAULT '{{}}'::jsonb;

CREATE INDEX IF NOT EXISTS idx_documents_embeddings_document_id
    ON documents_embeddings (document_id);

CREATE INDEX IF NOT EXISTS idx_documents_embeddings_content_type
    ON documents_embeddings (content_type);

CREATE TABLE IF NOT EXISTS interaction_feedback (
    id UUID PRIMARY KEY,
    message_id UUID NOT NULL UNIQUE,
    user_prompt TEXT NOT NULL,
    assistant_response TEXT NOT NULL,
    csat_rating INTEGER NOT NULL CHECK (csat_rating BETWEEN 1 AND 5),
    is_incorrect BOOLEAN NOT NULL DEFAULT FALSE,
    is_incomplete BOOLEAN NOT NULL DEFAULT FALSE,
    is_unclear BOOLEAN NOT NULL DEFAULT FALSE,
    comments TEXT,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_interaction_feedback_created_at
    ON interaction_feedback (created_at);
"""


@contextmanager
def get_connection() -> Iterator[psycopg.Connection]:
    settings = get_settings()
    last_error: Exception | None = None

    for host in settings.postgres.hosts:
        connection: psycopg.Connection | None = None
        try:
            connection = psycopg.connect(settings.postgres.dsn(host), row_factory=dict_row)
        except psycopg.OperationalError as error:
            last_error = error
            continue

        try:
            yield connection
        finally:
            connection.close()
        return

    if last_error is not None:
        raise last_error
    raise RuntimeError("No PostgreSQL hosts configured.")


def initialize_pgvector_schema() -> None:
    logger.info("Initializing pgVector schema.")
    with get_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(CREATE_TABLE_SQL)
        connection.commit()
    logger.info("pgVector schema initialization completed.")


def vector_literal(values: list[float]) -> str:
    return "[" + ",".join(f"{value:.8f}" for value in values) + "]"


def insert_chunks(document_id: UUID, chunks: list[ExtractedChunk], embeddings: list[list[float]]) -> int:
    now = datetime.utcnow()
    logger.info("Preparing %s chunk record(s) for document_id=%s.", len(chunks), document_id)
    records = []
    for chunk, embedding in zip(chunks, embeddings, strict=True):
        records.append(
            {
                "id": uuid4(),
                "embedding": vector_literal(embedding),
                "content": chunk.content,
                "content_type": chunk.content_type,
                "document_id": document_id,
                "chunk_id": chunk.chunk_id,
                "section": chunk.section,
                "source": chunk.source,
                "metadata_json": json.dumps(chunk.metadata),
                "created_at": now,
            }
        )

    with get_connection() as connection:
        with connection.cursor() as cursor:
            for record in records:
                cursor.execute(
                    """
                    INSERT INTO documents_embeddings (
                        id,
                        embedding,
                        content,
                        content_type,
                        document_id,
                        chunk_id,
                        section,
                        source,
                        metadata_json,
                        created_at
                    )
                    VALUES (
                        %(id)s,
                        %(embedding)s::vector,
                        %(content)s,
                        %(content_type)s,
                        %(document_id)s,
                        %(chunk_id)s,
                        %(section)s,
                        %(source)s,
                        %(metadata_json)s::jsonb,
                        %(created_at)s
                    )
                    """,
                    record,
                )
        connection.commit()

    logger.info("Inserted %s chunk record(s) for document_id=%s.", len(records), document_id)
    return len(records)


def count_document_chunks(document_id: UUID) -> int:
    with get_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT COUNT(*) AS count
                FROM documents_embeddings
                WHERE document_id = %(document_id)s
                """,
                {"document_id": document_id},
            )
            row = cursor.fetchone()
    return int(row["count"]) if row else 0


def list_documents() -> list[dict]:
    logger.info("Loading document summary list from pgVector.")
    with get_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT
                    document_id,
                    source,
                    MIN(created_at) AS created_at,
                    COUNT(*) AS total_chunks,
                    COUNT(*) FILTER (WHERE content_type = 'text') AS text_chunks,
                    COUNT(*) FILTER (WHERE content_type = 'table') AS table_chunks,
                    COUNT(*) FILTER (WHERE content_type = 'image') AS image_chunks
                FROM documents_embeddings
                GROUP BY document_id, source
                ORDER BY MIN(created_at) DESC
                """
            )
            rows = list(cursor.fetchall())
    logger.info("Loaded %s document summary row(s).", len(rows))
    return rows


def get_document_chunks(document_id: UUID) -> list[StoredChunk]:
    logger.info("Loading chunks for document_id=%s.", document_id)
    with get_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT
                    id,
                    content,
                    content_type,
                    document_id,
                    chunk_id,
                    section,
                    source,
                    metadata_json,
                    created_at
                FROM documents_embeddings
                WHERE document_id = %(document_id)s
                ORDER BY chunk_id
                """,
                {"document_id": document_id},
            )
            rows = cursor.fetchall()

    chunks = [
        StoredChunk(
            id=row["id"],
            embedding=[],
            content=row["content"],
            content_type=row["content_type"],
            document_id=row["document_id"],
            chunk_id=row["chunk_id"],
            section=row["section"] or "",
            source=row["source"],
            created_at=row["created_at"],
            metadata=row["metadata_json"] or {},
        )
        for row in rows
    ]
    logger.info("Loaded %s chunk row(s) for document_id=%s.", len(chunks), document_id)
    return chunks


def search_similar_chunks(query_embedding: list[float], limit: int = 8, sources: list[str] | None = None) -> list[SearchChunkResult]:
    logger.info("Running pgVector similarity search with limit=%s.", limit)
    where_clause = ""
    params: dict[str, object] = {
        "embedding": vector_literal(query_embedding),
        "limit": limit,
    }
    if sources:
        where_clause = "WHERE source = ANY(%(sources)s)"
        params["sources"] = sources

    with get_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                f"""
                SELECT
                    content,
                    content_type,
                    document_id,
                    chunk_id,
                    section,
                    source,
                    metadata_json,
                    1 - (embedding <=> %(embedding)s::vector) AS score
                FROM documents_embeddings
                {where_clause}
                ORDER BY embedding <=> %(embedding)s::vector
                LIMIT %(limit)s
                """,
                params,
            )
            rows = cursor.fetchall()

    results = [
        SearchChunkResult(
            content=row["content"],
            content_type=row["content_type"],
            document_id=row["document_id"],
            chunk_id=row["chunk_id"],
            section=row["section"] or "",
            source=row["source"],
            score=float(row["score"]),
            metadata=row["metadata_json"] or {},
        )
        for row in rows
    ]
    logger.info("pgVector similarity search returned %s row(s).", len(results))
    return results


def _normalize_search_tokens(query_text: str) -> list[str]:
    raw_tokens = [token.lower() for token in re.findall(r"[a-zA-Z0-9]+", query_text)]
    normalized_tokens: list[str] = []
    for token in raw_tokens:
        if len(token) < 3:
            continue
        normalized_tokens.append(token)
        if token.endswith("s") and len(token) > 3:
            normalized_tokens.append(token[:-1])
        else:
            normalized_tokens.append(f"{token}s")
    return list(dict.fromkeys(normalized_tokens))


def search_keyword_chunks(query_text: str, limit: int = 8, sources: list[str] | None = None) -> list[SearchChunkResult]:
    tokens = _normalize_search_tokens(query_text)
    logger.info("Running keyword search on pgVector chunks with %s token(s).", len(tokens))
    if not tokens:
        return []

    score_parts = []
    params: dict[str, object] = {"limit": limit}
    for index, token in enumerate(tokens):
        key = f"token_{index}"
        score_parts.append(
            f"CASE WHEN position(%({key})s in lower(content)) > 0 THEN 1 ELSE 0 END"
        )
        params[key] = token

    score_sql = " + ".join(score_parts)
    source_clause = ""
    if sources:
        source_clause = " AND source = ANY(%(sources)s)"
        params["sources"] = sources
    query = f"""
        SELECT
            content,
            content_type,
            document_id,
            chunk_id,
            section,
            source,
            metadata_json,
            ({score_sql})::float AS score
        FROM documents_embeddings
        WHERE ({score_sql}) > 0{source_clause}
        ORDER BY
            CASE WHEN content_type = 'table' THEN 0 ELSE 1 END,
            ({score_sql}) DESC,
            created_at DESC
        LIMIT %(limit)s
    """

    with get_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(query, params)
            rows = cursor.fetchall()

    results = [
        SearchChunkResult(
            content=row["content"],
            content_type=row["content_type"],
            document_id=row["document_id"],
            chunk_id=row["chunk_id"],
            section=row["section"] or "",
            source=row["source"],
            score=float(row["score"]),
            metadata=row["metadata_json"] or {},
        )
        for row in rows
    ]
    logger.info("Keyword search returned %s row(s).", len(results))
    return results


def save_interaction_feedback(
    message_id: UUID,
    user_prompt: str,
    assistant_response: str,
    csat_rating: int,
    is_incorrect: bool,
    is_incomplete: bool,
    is_unclear: bool,
    comments: str,
) -> None:
    logger.info("Saving interaction feedback for message_id=%s.", message_id)
    with get_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO interaction_feedback (
                    id,
                    message_id,
                    user_prompt,
                    assistant_response,
                    csat_rating,
                    is_incorrect,
                    is_incomplete,
                    is_unclear,
                    comments
                )
                VALUES (
                    %(id)s,
                    %(message_id)s,
                    %(user_prompt)s,
                    %(assistant_response)s,
                    %(csat_rating)s,
                    %(is_incorrect)s,
                    %(is_incomplete)s,
                    %(is_unclear)s,
                    %(comments)s
                )
                ON CONFLICT (message_id) DO UPDATE
                SET
                    user_prompt = EXCLUDED.user_prompt,
                    assistant_response = EXCLUDED.assistant_response,
                    csat_rating = EXCLUDED.csat_rating,
                    is_incorrect = EXCLUDED.is_incorrect,
                    is_incomplete = EXCLUDED.is_incomplete,
                    is_unclear = EXCLUDED.is_unclear,
                    comments = EXCLUDED.comments
                """,
                {
                    "id": uuid4(),
                    "message_id": message_id,
                    "user_prompt": user_prompt,
                    "assistant_response": assistant_response,
                    "csat_rating": csat_rating,
                    "is_incorrect": is_incorrect,
                    "is_incomplete": is_incomplete,
                    "is_unclear": is_unclear,
                    "comments": comments.strip(),
                },
            )
        connection.commit()
    logger.info("Feedback saved for message_id=%s.", message_id)
