from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from uuid import UUID
from pydantic import BaseModel, Field


@dataclass
class ExtractedChunk:
    content: str
    content_type: str
    section: str
    chunk_id: int
    source: str
    page_number: int | None = None
    section_title: str = ""
    document_version: str = "v1"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExtractedTable:
    table_id: str
    document_id: UUID
    source: str
    section: str
    rows: list[list[str]]
    summary: str
    headers: list[str]
    page_number: int | None = None
    section_title: str = ""
    table_name: str = ""
    document_version: str = "v1"


@dataclass
class StoredChunk:
    id: UUID
    embedding: list[float]
    content: str
    content_type: str
    document_id: UUID
    chunk_id: int
    section: str
    source: str
    created_at: datetime
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchChunkResult:
    content: str
    content_type: str
    document_id: UUID
    chunk_id: int
    section: str
    source: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class GraphSearchResult:
    document_id: str
    source: str
    table_id: str
    section: str
    summary: str
    rows: list[list[str]]
    headers: list[str]


class RetrievalItem(BaseModel):
    content_type: str
    document_id: str
    file_name: str
    page_number: int | None = None
    section_title: str = ""
    score: float = 0.0
    text: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievalResult(BaseModel):
    query: str
    query_type: str
    retrieval_paths: list[str] = Field(default_factory=list)
    text_results: list[RetrievalItem] = Field(default_factory=list)
    image_results: list[RetrievalItem] = Field(default_factory=list)
    table_results: list[RetrievalItem] = Field(default_factory=list)
    controlled_response: str = ""
