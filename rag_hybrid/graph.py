from __future__ import annotations

import json
import re

from neo4j import GraphDatabase

from rag_hybrid.app_logging import get_logger
from rag_hybrid.config import get_settings
from rag_hybrid.models import ExtractedTable, GraphSearchResult

logger = get_logger()


def _write(session, query: str, **params: object) -> None:
    session.run(query, **params).consume()


def verify_neo4j_connectivity() -> None:
    settings = get_settings()
    driver = GraphDatabase.driver(
        settings.neo4j.uri,
        auth=(settings.neo4j.user, settings.neo4j.password),
    )
    try:
        with driver.session(database=settings.neo4j.database) as session:
            session.run("RETURN 1").consume()
    finally:
        driver.close()


def store_tables(tables: list[ExtractedTable]) -> int:
    if not tables:
        logger.info("No tables found for Neo4j storage.")
        return 0

    settings = get_settings()
    driver = GraphDatabase.driver(
        settings.neo4j.uri,
        auth=(settings.neo4j.user, settings.neo4j.password),
    )

    with driver.session(database=settings.neo4j.database) as session:
        logger.info("Writing %s extracted tables to Neo4j.", len(tables))
        _write(session, "CREATE CONSTRAINT document_id_unique IF NOT EXISTS FOR (d:Document) REQUIRE d.document_id IS UNIQUE")
        _write(session, "CREATE CONSTRAINT table_id_unique IF NOT EXISTS FOR (t:Table) REQUIRE t.table_id IS UNIQUE")
        _write(session, "CREATE CONSTRAINT row_id_unique IF NOT EXISTS FOR (r:Row) REQUIRE r.row_id IS UNIQUE")
        _write(session, "CREATE CONSTRAINT column_id_unique IF NOT EXISTS FOR (c:Column) REQUIRE c.column_id IS UNIQUE")
        _write(session, "CREATE CONSTRAINT cell_id_unique IF NOT EXISTS FOR (c:Cell) REQUIRE c.cell_id IS UNIQUE")

        for table in tables:
            headers = table.headers or (table.rows[0] if table.rows else [])
            body_rows = table.rows[1:] if len(table.rows) > 1 else []

            _write(
                session,
                """
                MERGE (d:Document {document_id: $document_id})
                SET d.source = $source
                MERGE (t:Table {table_id: $table_id})
                SET
                    t.section = $section,
                    t.summary = $summary,
                    t.rows_json = $rows_json,
                    t.headers_json = $headers_json,
                    t.source = $source,
                    t.row_count = $row_count,
                    t.column_count = $column_count
                MERGE (d)-[:HAS_TABLE]->(t)
                """,
                document_id=str(table.document_id),
                table_id=table.table_id,
                source=table.source,
                section=table.section,
                summary=table.summary,
                rows_json=json.dumps(table.rows),
                headers_json=json.dumps(headers),
                row_count=len(body_rows),
                column_count=len(headers),
            )

            _write(
                session,
                """
                MATCH (t:Table {table_id: $table_id})
                OPTIONAL MATCH (t)-[:HAS_COLUMN]->(c:Column)
                DETACH DELETE c
                """,
                table_id=table.table_id,
            )
            _write(
                session,
                """
                MATCH (t:Table {table_id: $table_id})
                OPTIONAL MATCH (t)-[:HAS_ROW]->(r:Row)-[:HAS_CELL]->(cell:Cell)
                DETACH DELETE cell, r
                """,
                table_id=table.table_id,
            )
            _write(
                session,
                """
                MATCH (t:Table {table_id: $table_id})
                OPTIONAL MATCH (t)-[:HAS_ROW]->(r:Row)
                DETACH DELETE r
                """,
                table_id=table.table_id,
            )

            for column_index, header in enumerate(headers, start=1):
                _write(
                    session,
                    """
                    MATCH (t:Table {table_id: $table_id})
                    MERGE (c:Column {column_id: $column_id})
                    SET c.name = $name, c.position = $position
                    MERGE (t)-[:HAS_COLUMN]->(c)
                    """,
                    table_id=table.table_id,
                    column_id=f"{table.table_id}-column-{column_index}",
                    name=header,
                    position=column_index,
                )

            for row_index, row in enumerate(body_rows, start=1):
                row_id = f"{table.table_id}-row-{row_index}"
                _write(
                    session,
                    """
                    MATCH (t:Table {table_id: $table_id})
                    MERGE (r:Row {row_id: $row_id})
                    SET r.position = $position, r.preview = $preview
                    MERGE (t)-[:HAS_ROW]->(r)
                    """,
                    table_id=table.table_id,
                    row_id=row_id,
                    position=row_index,
                    preview=" | ".join(row[:4]),
                )

                for column_index, cell_value in enumerate(row, start=1):
                    _write(
                        session,
                        """
                        MATCH (t:Table {table_id: $table_id})-[:HAS_ROW]->(r:Row {row_id: $row_id})
                        MATCH (t)-[:HAS_COLUMN]->(col:Column {column_id: $column_id})
                        MERGE (cell:Cell {cell_id: $cell_id})
                        SET cell.value = $value, cell.row_position = $row_position, cell.column_position = $column_position
                        MERGE (r)-[:HAS_CELL]->(cell)
                        MERGE (cell)-[:IN_COLUMN]->(col)
                        """,
                        table_id=table.table_id,
                        row_id=row_id,
                        column_id=f"{table.table_id}-column-{column_index}",
                        cell_id=f"{row_id}-cell-{column_index}",
                        value=cell_value,
                        row_position=row_index,
                        column_position=column_index,
                    )

    driver.close()
    logger.info("Stored %s tables in Neo4j.", len(tables))
    return len(tables)


def search_tables(query_text: str, limit: int = 5, sources: list[str] | None = None) -> list[GraphSearchResult]:
    settings = get_settings()
    driver = GraphDatabase.driver(
        settings.neo4j.uri,
        auth=(settings.neo4j.user, settings.neo4j.password),
    )
    raw_tokens = [token.lower() for token in re.findall(r"[a-zA-Z0-9]+", query_text) if len(token) > 2]
    tokens: list[str] = []
    for token in raw_tokens:
        tokens.append(token)
        if token.endswith("s") and len(token) > 3:
            tokens.append(token[:-1])
        else:
            tokens.append(f"{token}s")
    tokens = list(dict.fromkeys(tokens))
    logger.info("Running Neo4j table search with %s token(s).", len(tokens))

    source_filter = "WHERE d.source IN $sources" if sources else ""
    query = f"""
    MATCH (d:Document)-[:HAS_TABLE]->(t:Table)
    {source_filter}
    OPTIONAL MATCH (t)-[:HAS_ROW]->(r:Row)
    WITH d, t, collect(r) AS rows, $tokens AS tokens
    WITH
        d,
        t,
        rows,
        reduce(score = 0, token IN tokens |
            score +
            CASE
                WHEN
                    toLower(coalesce(t.summary, '')) CONTAINS token OR
                    toLower(coalesce(t.rows_json, '')) CONTAINS token OR
                    toLower(coalesce(t.section, '')) CONTAINS token OR
                    toLower(coalesce(d.source, '')) CONTAINS token
                THEN 1
                ELSE 0
            END
        ) AS score
    WHERE size(tokens) = 0 OR score > 0
    RETURN
        d.document_id AS document_id,
        d.source AS source,
        t.table_id AS table_id,
        t.section AS section,
        t.summary AS summary,
        t.rows_json AS rows_json,
        t.headers_json AS headers_json,
        score
    ORDER BY score DESC, table_id
    LIMIT $limit
    """

    with driver.session(database=settings.neo4j.database) as session:
        rows = list(session.run(query, tokens=tokens, limit=limit, sources=sources or []))

    driver.close()
    results = [
        GraphSearchResult(
            document_id=row["document_id"],
            source=row["source"] or "",
            table_id=row["table_id"],
            section=row["section"] or "",
            summary=row["summary"] or "",
            rows=json.loads(row["rows_json"] or "[]"),
            headers=json.loads(row["headers_json"] or "[]"),
        )
        for row in rows
    ]
    logger.info("Neo4j table search returned %s row(s).", len(results))
    return results
