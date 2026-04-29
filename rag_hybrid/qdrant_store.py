from __future__ import annotations

from datetime import datetime
import json
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from uuid import UUID, uuid4

from rag_hybrid.app_logging import get_logger
from rag_hybrid.config import get_settings
from rag_hybrid.models import ExtractedChunk, SearchChunkResult

logger = get_logger()


def _request(method: str, path: str, body: dict[str, Any] | None = None) -> dict[str, Any]:
    settings = get_settings()
    data = json.dumps(body).encode("utf-8") if body is not None else None
    request = Request(
        f"{settings.qdrant.url}{path}",
        data=data,
        method=method,
        headers={"Content-Type": "application/json"},
    )

    try:
        with urlopen(request, timeout=30) as response:
            raw_body = response.read().decode("utf-8")
    except HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Qdrant request failed: {error.code} {detail}") from error
    except URLError as error:
        raise RuntimeError(f"Qdrant is unavailable at {settings.qdrant.url}: {error.reason}") from error

    if not raw_body:
        return {}
    return json.loads(raw_body)


def verify_qdrant_connectivity() -> None:
    settings = get_settings()
    logger.info("Checking Qdrant connectivity at %s.", settings.qdrant.url)
    _request("GET", "/")
    logger.info("Qdrant connectivity check completed.")


def initialize_qdrant_collection(collection_name: str | None = None) -> None:
    settings = get_settings()
    collection = collection_name or settings.qdrant.collection
    vector_size = settings.extraction.embedding_dimension
    logger.info("Initializing Qdrant collection '%s'.", collection)
    try:
        _request("GET", f"/collections/{collection}")
        logger.info("Qdrant collection '%s' already exists.", collection)
        return
    except RuntimeError as error:
        if "404" not in str(error):
            raise

    _request(
        "PUT",
        f"/collections/{collection}",
        {
            "vectors": {
                "size": vector_size,
                "distance": "Cosine",
            }
        },
    )
    logger.info("Qdrant collection initialization completed.")


def initialize_smartcoolant_collections() -> None:
    settings = get_settings()
    initialize_qdrant_collection(settings.qdrant.text_collection)
    initialize_qdrant_collection(settings.qdrant.image_collection)


def upsert_points(collection_name: str, points: list[dict[str, Any]]) -> int:
    if not points:
        return 0

    _request("PUT", f"/collections/{collection_name}/points?wait=true", {"points": points})
    logger.info("Inserted %s point(s) into Qdrant collection '%s'.", len(points), collection_name)
    return len(points)


def insert_chunks(document_id: UUID, chunks: list[ExtractedChunk], embeddings: list[list[float]], collection_name: str | None = None) -> int:
    if not chunks:
        return 0

    settings = get_settings()
    collection = collection_name or settings.qdrant.collection
    now = datetime.utcnow().isoformat()
    points = []
    for chunk, embedding in zip(chunks, embeddings, strict=True):
        points.append(
            {
                "id": str(uuid4()),
                "vector": embedding,
                "payload": {
                    "content": chunk.content,
                    "content_type": chunk.content_type,
                    "document_id": str(document_id),
                    "chunk_id": chunk.chunk_id,
                    "section": chunk.section,
                    "source": chunk.source,
                    "metadata_json": chunk.metadata,
                    "created_at": now,
                },
            }
        )

    _request(
        "PUT",
        f"/collections/{collection}/points?wait=true",
        {"points": points},
    )
    logger.info("Inserted %s chunk point(s) into Qdrant for document_id=%s.", len(points), document_id)
    return len(points)


def search_collection(
    collection_name: str,
    query_embedding: list[float],
    limit: int = 8,
    sources: list[str] | None = None,
    extra_filter: list[dict[str, Any]] | None = None,
) -> list[SearchChunkResult]:
    settings = get_settings()
    body: dict[str, Any] = {
        "vector": query_embedding,
        "limit": limit,
        "with_payload": True,
    }
    filters = list(extra_filter or [])
    if sources:
        filters.append({"key": "source", "match": {"any": sources}})
    if filters:
        body["filter"] = {"must": filters}

    response = _request("POST", f"/collections/{collection_name}/points/search", body)
    results = [_point_to_search_result(point) for point in response.get("result", [])]
    logger.info("Qdrant similarity search returned %s row(s) from '%s'.", len(results), collection_name)
    return results


def search_similar_chunks(query_embedding: list[float], limit: int = 8, sources: list[str] | None = None) -> list[SearchChunkResult]:
    return search_collection(get_settings().qdrant.collection, query_embedding, limit, sources)


def scroll_collection(collection_name: str, limit: int = 256, sources: list[str] | None = None) -> list[dict[str, Any]]:
    body: dict[str, Any] = {
        "limit": limit,
        "with_payload": True,
    }
    if sources:
        body["filter"] = {"must": [{"key": "source", "match": {"any": sources}}]}

    response = _request("POST", f"/collections/{collection_name}/points/scroll", body)
    return list(response.get("result", {}).get("points", []))


def search_keyword_collection(collection_name: str, query_text: str, limit: int = 8, sources: list[str] | None = None) -> list[SearchChunkResult]:
    from rag_hybrid.db import _normalize_search_tokens

    tokens = _normalize_search_tokens(query_text)
    if not tokens:
        return []

    scored_results: list[SearchChunkResult] = []
    for point in scroll_collection(collection_name, sources=sources):
        result = _point_to_search_result(point)
        content = result.content.lower()
        score = float(sum(1 for token in tokens if token in content))
        if score <= 0:
            continue
        result.score = score
        scored_results.append(result)

    scored_results.sort(key=lambda item: (item.content_type != "table", -item.score))
    logger.info("Qdrant keyword search returned %s row(s) from '%s'.", len(scored_results[:limit]), collection_name)
    return scored_results[:limit]


def search_keyword_chunks(query_text: str, limit: int = 8, sources: list[str] | None = None) -> list[SearchChunkResult]:
    return search_keyword_collection(get_settings().qdrant.collection, query_text, limit, sources)


def _point_to_search_result(point: dict[str, Any]) -> SearchChunkResult:
    payload = point.get("payload") or {}
    metadata = dict(payload)
    metadata.update(payload.get("metadata_json") or {})
    return SearchChunkResult(
        content=str(payload.get("content", "")),
        content_type=str(payload.get("content_type", "")),
        document_id=UUID(str(payload["document_id"])),
        chunk_id=int(payload.get("chunk_id", 0)),
        section=str(payload.get("section") or ""),
        source=str(payload.get("source", "")),
        score=float(point.get("score", 0.0)),
        metadata=metadata,
    )
