from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import re

from rag_hybrid.app_logging import get_logger
from rag_hybrid.config import get_settings
from rag_hybrid.db import search_keyword_chunks, search_similar_chunks
from rag_hybrid.embeddings import embed_text
from rag_hybrid.graph import search_tables
from rag_hybrid.models import GraphSearchResult, SearchChunkResult
from rag_hybrid.retrievers import retrieve

logger = get_logger()
settings = get_settings()
IMAGE_QUERY_TOKENS = {"image", "images", "map", "maps", "photo", "photos", "picture", "pictures", "diagram", "diagrams", "show"}
STOP_TOKENS = {
    "tell",
    "about",
    "what",
    "which",
    "who",
    "is",
    "are",
    "the",
    "a",
    "an",
    "me",
    "please",
    "list",
    "show",
    "give",
    "can",
    "get",
    "on",
    "each",
    "every",
}
COUNT_QUERY_TOKENS = {"count", "counts", "number", "numbers", "total", "totals"}
INCHARGE_QUERY_TOKENS = {"incharge", "incharges", "owner", "owners", "responsible", "manager", "managers"}
TABLE_INTENT_TOKENS = {
    "available",
    "availability",
    "country",
    "countries",
    "countrywise",
    "list",
    "age",
    "stage",
    "row",
    "column",
    *COUNT_QUERY_TOKENS,
    *INCHARGE_QUERY_TOKENS,
}


def _query_tokens(prompt: str) -> tuple[list[str], list[str]]:
    tokens = [token.lower() for token in re.findall(r"[a-zA-Z0-9]+", prompt)]
    numeric_tokens = [token for token in tokens if token.isdigit()]
    word_tokens = [token for token in tokens if not token.isdigit() and token not in STOP_TOKENS and len(token) > 1]
    return word_tokens, numeric_tokens


def _exact_tokens(text: str) -> set[str]:
    return {token.lower() for token in re.findall(r"[a-zA-Z0-9]+", text)}


def _compound_match(text: str, query_words: list[str], numeric_tokens: list[str]) -> bool:
    lowered = text.lower()
    for word in query_words:
        for number in numeric_tokens:
            forward = re.search(rf"\b{re.escape(word)}\b(?:\W+\w+){{0,2}}\W+\b{re.escape(number)}\b", lowered)
            backward = re.search(rf"\b{re.escape(number)}\b(?:\W+\w+){{0,2}}\W+\b{re.escape(word)}\b", lowered)
            if forward or backward:
                return True
    return False


def _filter_vector_results(prompt: str, results: list[SearchChunkResult]) -> list[SearchChunkResult]:
    query_words, numeric_tokens = _query_tokens(prompt)
    wants_image = any(token in IMAGE_QUERY_TOKENS for token in query_words)
    filtered: list[SearchChunkResult] = []

    for item in results:
        combined_text = f"{item.section} {item.content}"
        exact_tokens = _exact_tokens(combined_text)
        if item.content_type == "image" and not wants_image:
            continue

        word_match = any(token in exact_tokens for token in query_words) if query_words else True
        numeric_match = any(token in exact_tokens for token in numeric_tokens) if numeric_tokens else True

        if query_words and numeric_tokens:
            if item.content_type == "text" and not _compound_match(combined_text, query_words, numeric_tokens):
                continue
            if not (word_match and numeric_match):
                continue
        elif numeric_tokens and not numeric_match:
            continue
        elif query_words and not word_match:
            continue

        if not query_words and not numeric_tokens:
            filtered.append(item)
        else:
            filtered.append(item)

    return filtered


def _filter_graph_results(prompt: str, results: list[GraphSearchResult]) -> list[GraphSearchResult]:
    query_words, numeric_tokens = _query_tokens(prompt)
    filtered: list[GraphSearchResult] = []
    is_count_query = bool(set(query_words) & COUNT_QUERY_TOKENS)
    is_incharge_query = bool(set(query_words) & INCHARGE_QUERY_TOKENS)

    for item in results:
        headers = item.headers or (item.rows[0] if item.rows else [])
        header_tokens = _exact_tokens(" ".join(headers) + " " + item.section + " " + item.summary)
        matched_rows: list[list[str]] = []

        for row in item.rows[1:]:
            row_tokens = _exact_tokens(" ".join(row) + " " + " ".join(headers))
            word_match = any(token in row_tokens or token in header_tokens for token in query_words) if query_words else True
            numeric_match = any(token in row_tokens for token in numeric_tokens) if numeric_tokens else True

            if numeric_tokens:
                if numeric_match and word_match:
                    matched_rows.append(row)
            else:
                if word_match:
                    matched_rows.append(row)

        if matched_rows:
            filtered.append(
                GraphSearchResult(
                    document_id=item.document_id,
                    source=item.source,
                    table_id=item.table_id,
                    section=item.section,
                    summary=item.summary,
                    rows=[headers, *matched_rows],
                    headers=headers,
                )
            )
            continue

        table_level_match = any(token in header_tokens for token in query_words) if query_words else False
        if table_level_match and not numeric_tokens:
            filtered.append(item)

    subject_tokens = [token for token in query_words if token not in TABLE_INTENT_TOKENS]
    if subject_tokens and not is_incharge_query and any(_table_contains_any(item, subject_tokens) for item in filtered):
        filtered = [item for item in filtered if _table_contains_any(item, subject_tokens)]

    if is_count_query and not is_incharge_query and any(_table_has_count_measure(item) for item in filtered):
        filtered = [item for item in filtered if _table_has_count_measure(item)]

    return _dedupe_graph_results(filtered)


def _table_has_count_measure(item: GraphSearchResult) -> bool:
    headers = item.headers or (item.rows[0] if item.rows else [])
    header_tokens = _exact_tokens(" ".join(headers))
    return bool(header_tokens & COUNT_QUERY_TOKENS)


def _table_contains_any(item: GraphSearchResult, tokens: list[str]) -> bool:
    table_text = " ".join(
        [
            item.section,
            item.summary,
            " ".join(item.headers),
            " ".join(" ".join(row) for row in item.rows),
        ]
    ).lower()
    normalized_tokens = []
    for token in tokens:
        normalized_tokens.append(token)
        if token.endswith("s") and len(token) > 3:
            normalized_tokens.append(token[:-1])
    return any(token in table_text for token in normalized_tokens)


def _dedupe_graph_results(results: list[GraphSearchResult]) -> list[GraphSearchResult]:
    deduped: list[GraphSearchResult] = []
    seen: set[tuple[str, str, str]] = set()
    for item in results:
        key = (item.source, item.section, "|".join(item.headers))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def retrieve_chat_context(prompt: str, sources: list[str] | None = None, vector_store: str = "pgVector") -> dict[str, object]:
    logger.info("Conversation retrieval started using %s.", vector_store)
    if _vector_backend(vector_store) == "Qdrant":
        routed_result = retrieve(prompt, sources)
        vector_results = [_retrieval_item_to_chunk(item) for item in routed_result.text_results + routed_result.image_results]
        graph_results = _filter_graph_results(
            prompt,
            [_retrieval_item_to_graph(item) for item in routed_result.table_results],
        )
        logger.info(
            "Qdrant routed retrieval completed. query_type=%s paths=%s text=%s image=%s table=%s.",
            routed_result.query_type,
            routed_result.retrieval_paths,
            len(routed_result.text_results),
            len(routed_result.image_results),
            len(routed_result.table_results),
        )
        return {
            "vector_results": vector_results,
            "graph_results": graph_results,
            "query_type": routed_result.query_type,
            "retrieval_paths": routed_result.retrieval_paths,
            "controlled_response": routed_result.controlled_response,
        }

    query_embedding = embed_text(prompt)
    similar_search = search_similar_chunks
    keyword_search = search_keyword_chunks

    with ThreadPoolExecutor(max_workers=3) as executor:
        vector_future = executor.submit(similar_search, query_embedding, settings.retrieval.vector_search_limit, sources)
        keyword_future = executor.submit(keyword_search, prompt, settings.retrieval.vector_search_limit, sources)
        graph_future = executor.submit(search_tables, prompt, settings.retrieval.graph_search_limit, sources)
        vector_results = vector_future.result()
        keyword_results = keyword_future.result()
        graph_results = graph_future.result()

    merged_vector_results: list[SearchChunkResult] = []
    seen_keys: set[tuple[str, int, str]] = set()
    for item in keyword_results + vector_results:
        key = (str(item.document_id), item.chunk_id, item.content_type)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        merged_vector_results.append(item)
        if len(merged_vector_results) >= settings.retrieval.vector_search_limit:
            break

    filtered_vector_results = _filter_vector_results(prompt, merged_vector_results)
    filtered_graph_results = _filter_graph_results(prompt, graph_results)

    logger.info(
        "Conversation retrieval completed with %s vector result(s) and %s graph result(s).",
        len(filtered_vector_results),
        len(filtered_graph_results),
    )
    return {
        "vector_results": filtered_vector_results,
        "graph_results": filtered_graph_results,
    }


def _vector_backend(vector_store: str) -> str:
    return "Qdrant" if "qdrant" in vector_store.lower() else "pgVector"


def _retrieval_item_to_chunk(item) -> SearchChunkResult:
    metadata = dict(item.metadata)
    if item.content_type == "image" and metadata.get("storage_url"):
        metadata["asset_path"] = metadata["storage_url"]
    return SearchChunkResult(
        content=item.text or metadata.get("image_caption", ""),
        content_type=item.content_type,
        document_id=item.document_id,
        chunk_id=int(metadata.get("chunk_id") or 0),
        section=item.section_title,
        source=item.file_name,
        score=item.score,
        metadata=metadata,
    )


def _retrieval_item_to_graph(item) -> GraphSearchResult:
    metadata = item.metadata
    headers = metadata.get("headers") or []
    rows = metadata.get("matched_rows") or []
    return GraphSearchResult(
        document_id=item.document_id,
        source=item.file_name,
        table_id=str(metadata.get("table_id", "")),
        section=item.section_title,
        summary=item.text,
        rows=[headers, *rows] if headers else rows,
        headers=headers,
    )


def build_graphviz(graph_results: list[GraphSearchResult]) -> str:
    if not graph_results:
        return ""

    lines = ["digraph G {", 'rankdir="LR";', 'node [shape=box, style="rounded"];']
    for table in graph_results:
        document_node = f'doc_{table.document_id.replace("-", "_")}'
        table_node = f'table_{table.table_id.replace("-", "_")}'
        lines.append(f'{document_node} [label="Document\\n{table.source}"];')
        lines.append(f'{table_node} [label="Table\\n{table.section}"];')
        lines.append(f"{document_node} -> {table_node} [label=\"HAS_TABLE\"];")
        for index, header in enumerate(table.headers, start=1):
            column_node = f'column_{table.table_id.replace("-", "_")}_{index}'
            safe_header = header.replace('"', "'")
            lines.append(f'{column_node} [label="Column\\n{safe_header}"];')
            lines.append(f"{table_node} -> {column_node} [label=\"HAS_COLUMN\"];")
        row_count_node = f'rowcount_{table.table_id.replace("-", "_")}'
        lines.append(f'{row_count_node} [label="Rows\\n{max(len(table.rows) - 1, 0)}"];')
        lines.append(f"{table_node} -> {row_count_node} [label=\"HAS_ROWS\"];")
    lines.append("}")
    return "\n".join(lines)


def serialize_vector_results(results: list[SearchChunkResult]) -> list[dict[str, object]]:
    return [
        {
            "source": item.source,
            "section": item.section,
            "content_type": item.content_type,
            "score": round(item.score, 4),
            "content": item.content,
        }
        for item in results
    ]
