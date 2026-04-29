from __future__ import annotations

import re

from rag_hybrid.app_logging import get_logger
from rag_hybrid.config import get_settings
from rag_hybrid.embeddings import embed_text
from rag_hybrid.graph import search_tables
from rag_hybrid.models import RetrievalItem, RetrievalResult, SearchChunkResult
from rag_hybrid.qdrant_store import search_collection, search_keyword_collection
from rag_hybrid.query_classification import QueryType, classify_query
from rag_hybrid.llm import FALLBACK_RESPONSE

logger = get_logger()


class TextRetriever:
    def retrieve(self, query: str, sources: list[str] | None = None) -> list[RetrievalItem]:
        settings = get_settings()
        vector_results = search_collection(
            settings.qdrant.text_collection,
            embed_text(query),
            settings.retrieval.vector_search_limit,
            sources,
        )
        keyword_results = search_keyword_collection(
            settings.qdrant.text_collection,
            query,
            settings.retrieval.vector_search_limit,
            sources,
        )
        return [_to_retrieval_item(item) for item in _merge_chunk_results(keyword_results + vector_results)]


class ImageRetriever:
    def retrieve(self, query: str, sources: list[str] | None = None) -> list[RetrievalItem]:
        settings = get_settings()
        results = search_collection(
            settings.qdrant.image_collection,
            embed_text(query),
            settings.retrieval.vector_search_limit,
            sources,
        )
        topic, image_type = _image_query_filters(query)
        filtered = []
        for item in results:
            item_topic = str(item.metadata.get("topic", "")).lower()
            item_type = str(item.metadata.get("image_type", "")).lower()
            if topic and topic not in item_topic:
                continue
            if image_type and image_type != item_type:
                continue
            filtered.append(_to_retrieval_item(item))
        return filtered or [_to_retrieval_item(item) for item in results]


class TableRetriever:
    def retrieve(self, query: str, sources: list[str] | None = None) -> list[RetrievalItem]:
        settings = get_settings()
        tables = search_tables(query, settings.retrieval.graph_search_limit, sources)
        return [
            RetrievalItem(
                content_type="table",
                document_id=item.document_id,
                file_name=item.source,
                section_title=item.section,
                text=item.summary,
                metadata={
                    "table_id": item.table_id,
                    "table_name": item.section,
                    "headers": item.headers,
                    "matched_rows": item.rows[1:] if item.headers and item.rows and item.rows[0] == item.headers else item.rows,
                    "rows": item.rows,
                },
            )
            for item in tables
        ]


def retrieve(query: str, sources: list[str] | None = None) -> RetrievalResult:
    query_type = classify_query(query)
    logger.info("Selected query type '%s'.", query_type.value)

    result = RetrievalResult(query=query, query_type=query_type.value)
    text_retriever = TextRetriever()
    image_retriever = ImageRetriever()
    table_retriever = TableRetriever()

    if query_type == QueryType.OUT_OF_SCOPE:
        result.controlled_response = FALLBACK_RESPONSE
        result.retrieval_paths.append("none")
        return result

    if query_type == QueryType.IMAGE_REQUEST:
        result.image_results = image_retriever.retrieve(query, sources)
        result.retrieval_paths.append("smartcoolant_images")
        return result

    if query_type == QueryType.TABLE_LOOKUP:
        result.table_results = table_retriever.retrieve(query, sources)
        result.retrieval_paths.append("neo4j_tables")
        return result

    if query_type == QueryType.TEXT_QA:
        result.text_results = text_retriever.retrieve(query, sources)
        result.retrieval_paths.append("smartcoolant_text")
        return result

    if query_type == QueryType.TROUBLESHOOTING:
        result.text_results = text_retriever.retrieve(query, sources)
        result.table_results = table_retriever.retrieve(query, sources)
        result.image_results = image_retriever.retrieve(query, sources)
        result.retrieval_paths.extend(["smartcoolant_text", "neo4j_tables", "smartcoolant_images"])
        return result

    result.text_results = text_retriever.retrieve(query, sources)
    result.image_results = image_retriever.retrieve(query, sources)
    result.table_results = table_retriever.retrieve(query, sources)
    result.retrieval_paths.extend(["smartcoolant_text", "smartcoolant_images", "neo4j_tables"])
    return result


def _to_retrieval_item(item: SearchChunkResult) -> RetrievalItem:
    return RetrievalItem(
        content_type=item.content_type,
        document_id=str(item.document_id),
        file_name=str(item.metadata.get("file_name") or item.source),
        page_number=item.metadata.get("page_number"),
        section_title=str(item.metadata.get("section_title") or item.section),
        score=item.score,
        text=item.content,
        metadata=item.metadata,
    )


def _merge_chunk_results(results: list[SearchChunkResult]) -> list[SearchChunkResult]:
    merged: list[SearchChunkResult] = []
    seen: set[tuple[str, int, str]] = set()
    for item in results:
        key = (str(item.document_id), item.chunk_id, item.content_type)
        if key in seen:
            continue
        seen.add(key)
        merged.append(item)
    return merged


def _image_query_filters(query: str) -> tuple[str, str]:
    lowered = query.lower()
    topic = ""
    for candidate in ["portugal", "machine", "network", "coolant", "modbus"]:
        if candidate in lowered:
            topic = candidate
            break

    image_type = ""
    if "dimension" in lowered:
        image_type = "dimension_diagram"
    elif "map" in lowered:
        image_type = "map"
    elif re.search(r"\bdiagram\b", lowered):
        image_type = "diagram"

    return topic, image_type
