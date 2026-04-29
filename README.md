# RAG-Hybrid

Hybrid RAG starter project with:

- `pgVector` for text chunks, table summaries, and image captions
- `Neo4j` for extracted table structures and relationships
- `Streamlit` UI with two pages:
  - `Upload Files`
  - `Documents`

## Start Infra

```powershell
docker compose up -d
```

## Install App Dependencies

```powershell
.venv\Scripts\python -m pip install -e .
```

## Run The UI

```powershell
.venv\Scripts\streamlit run app.py
```

## Database Connections

- PostgreSQL host: `db` inside Docker, `127.0.0.1` from the local app
- PostgreSQL database: `vector_db`
- PostgreSQL user/password: `postgres` / `postgres`
- pgAdmin: `http://localhost:5050`
- Neo4j URI: `neo4j://127.0.0.1:7687`
- Neo4j user/password: `neo4j` / `jey-test2`

## Notes

- The ingestion pipeline creates `documents_embeddings` automatically if it is missing.
- Embeddings currently use a deterministic local hashing embedder so the pipeline runs without an external model service.
