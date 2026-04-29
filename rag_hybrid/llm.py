from __future__ import annotations

from collections.abc import Iterator

from openai import OpenAI

from rag_hybrid.app_logging import get_logger
from rag_hybrid.config import get_settings
from rag_hybrid.models import GraphSearchResult, SearchChunkResult

logger = get_logger()


def _no_evidence_response(prompt: str) -> str:
    return f"I could not find matching information in the uploaded documents for: {prompt}"


def _format_table_for_context(rows: list[list[str]]) -> str:
    if not rows:
        return "No rows available."

    header = rows[0]
    body = rows[1:]
    lines = [" | ".join(header), " | ".join(["---"] * len(header))]
    for row in body:
        padded = row + [""] * max(0, len(header) - len(row))
        lines.append(" | ".join(padded[: len(header)]))
    return "\n".join(lines)


def build_context(vector_results: list[SearchChunkResult], graph_results: list[GraphSearchResult]) -> str:
    vector_sections = []
    for item in vector_results:
        metadata_lines = []
        if item.content_type == "image" and item.metadata.get("asset_path"):
            metadata_lines.append(f"Image Path: {item.metadata['asset_path']}")
        vector_sections.append(
            "\n".join(
                [
                    f"Source: {item.source}",
                    f"Section: {item.section}",
                    f"Type: {item.content_type}",
                    f"Score: {item.score:.4f}",
                    f"Content: {item.content}",
                    *metadata_lines,
                ]
            )
        )

    graph_sections = []
    for item in graph_results:
        graph_sections.append(
            "\n".join(
                [
                    f"Document: {item.source}",
                    f"Table Section: {item.section}",
                    f"Summary: {item.summary}",
                    "Structured Table:",
                    _format_table_for_context(item.rows),
                ]
            )
        )

    return "\n\n".join(
        [
            "Vector Results:",
            "\n\n".join(vector_sections) if vector_sections else "None",
            "Graph Results:",
            "\n\n".join(graph_sections) if graph_sections else "None",
        ]
    )


def generate_answer(
    prompt: str,
    vector_results: list[SearchChunkResult],
    graph_results: list[GraphSearchResult],
    chat_history: list[dict[str, str]],
) -> str:
    settings = get_settings()
    if not vector_results and not graph_results:
        logger.info("No grounded evidence found; returning not-found response.")
        return _no_evidence_response(prompt)
    if not settings.openai.api_key:
        logger.warning("OpenAI API key is not configured; returning retrieval-only message.")
        return (
            "OpenAI response generation is unavailable because `OPENAI_API_KEY` is not configured. "
            "Retrieved vector and graph results are shown below."
        )

    context_text = build_context(vector_results, graph_results)
    history_text = "\n".join(
        f"{item['role']}: {item['content']}"
        for item in chat_history[-6:]
        if item["role"] == "user"
    )

    client = OpenAI(api_key=settings.openai.api_key)
    logger.info("Requesting OpenAI response with model '%s'.", settings.openai.model)
    response = client.responses.create(
        model=settings.openai.model,
        input=[
            {
                "role": "system",
                "content": (
                    f"{settings.openai.system_prompt} "
                    "When the retrieved context contains table rows and the user asks for a list, states, names, populations, or tabular data, "
                    "return the answer as a markdown table using the retrieved rows. "
                    "Do not use outside knowledge. If the context does not contain an exact matching entity, id, or fact for the question, say it was not found in the uploaded documents."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Conversation history:\n{history_text}\n\n"
                    f"Retrieved context:\n{context_text}\n\n"
                    f"User question:\n{prompt}"
                ),
            },
        ],
    )
    logger.info("OpenAI response received successfully.")
    return response.output_text


def stream_answer(
    prompt: str,
    vector_results: list[SearchChunkResult],
    graph_results: list[GraphSearchResult],
    chat_history: list[dict[str, str]],
) -> Iterator[str]:
    settings = get_settings()
    if not vector_results and not graph_results:
        logger.info("No grounded evidence found; returning streamed not-found response.")
        yield _no_evidence_response(prompt)
        return
    if not settings.openai.api_key:
        logger.warning("OpenAI API key is not configured; returning retrieval-only message.")
        yield (
            "OpenAI response generation is unavailable because `OPENAI_API_KEY` is not configured. "
            "Retrieved table data is displayed below."
        )
        return

    context_text = build_context(vector_results, graph_results)
    history_text = "\n".join(
        f"{item['role']}: {item['content']}"
        for item in chat_history[-6:]
        if item["role"] == "user"
    )

    client = OpenAI(api_key=settings.openai.api_key)
    logger.info("Requesting streamed OpenAI response with model '%s'.", settings.openai.model)
    stream = client.responses.create(
        model=settings.openai.model,
        input=[
            {
                "role": "system",
                "content": (
                    f"{settings.openai.system_prompt} "
                    "When the retrieved context contains table rows and the user asks for a list, states, names, populations, or tabular data, "
                    "return the answer as a markdown table using the retrieved rows. "
                    "Do not use outside knowledge. If the context does not contain an exact matching entity, id, or fact for the question, say it was not found in the uploaded documents."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Conversation history:\n{history_text}\n\n"
                    f"Retrieved context:\n{context_text}\n\n"
                    f"User question:\n{prompt}"
                ),
            },
        ],
        stream=True,
    )

    for event in stream:
        if event.type == "response.output_text.delta":
            yield event.delta

    logger.info("Streamed OpenAI response completed successfully.")
