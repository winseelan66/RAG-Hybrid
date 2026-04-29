from __future__ import annotations

from enum import StrEnum
import re

from rag_hybrid.app_logging import get_logger

logger = get_logger()


class QueryType(StrEnum):
    TEXT_QA = "text_qa"
    IMAGE_REQUEST = "image_request"
    TABLE_LOOKUP = "table_lookup"
    TROUBLESHOOTING = "troubleshooting"
    MIXED_QUERY = "mixed_query"
    OUT_OF_SCOPE = "out_of_scope"


IMAGE_TERMS = {"show", "display", "image", "picture", "diagram", "map", "photo"}
TABLE_TERMS = {
    "table",
    "register",
    "address",
    "parameter",
    "value",
    "specification",
    "specifications",
    "age",
    "stage",
    "row",
    "column",
    "country",
    "countries",
    "count",
    "counts",
    "available",
    "availability",
    "list",
}
TROUBLESHOOTING_TERMS = {"not working", "error", "failure", "troubleshoot", "how to fix", "not connecting"}
TEXT_TERMS = {
    "explain",
    "what",
    "how",
    "why",
    "when",
    "where",
    "tell",
    "act",
    "law",
    "implemented",
    "implementation",
    "protected",
    "protection",
    "wildlife",
    "schedule",
}
DOMAIN_TERMS = {
    "coolant",
    "machine",
    "modbus",
    "register",
    "portugal",
    "network",
    "dimension",
    "map",
    "table",
    "document",
    "peacock",
    "peacocks",
    "country",
    "countries",
    "wildlife",
    "protection",
    "act",
    "schedule",
    "india",
}


def classify_query(query: str) -> QueryType:
    lowered = query.lower()
    tokens = set(re.findall(r"[a-zA-Z0-9]+", lowered))

    has_image = bool(tokens & IMAGE_TERMS)
    has_table = bool(tokens & TABLE_TERMS)
    has_text = bool(tokens & TEXT_TERMS)
    has_troubleshooting = any(term in lowered for term in TROUBLESHOOTING_TERMS)
    in_scope = bool(tokens & DOMAIN_TERMS) or has_image or has_table or has_troubleshooting or has_text

    if not in_scope:
        logger.info("Query classified as out_of_scope.")
        return QueryType.OUT_OF_SCOPE
    if has_troubleshooting:
        logger.info("Query classified as troubleshooting.")
        return QueryType.TROUBLESHOOTING
    if has_image and (has_text or has_table):
        logger.info("Query classified as mixed_query.")
        return QueryType.MIXED_QUERY
    if has_image:
        logger.info("Query classified as image_request.")
        return QueryType.IMAGE_REQUEST
    if has_table:
        logger.info("Query classified as table_lookup.")
        return QueryType.TABLE_LOOKUP
    if has_text:
        logger.info("Query classified as text_qa.")
        return QueryType.TEXT_QA

    logger.info("Query classified as text_qa by fallback.")
    return QueryType.TEXT_QA
