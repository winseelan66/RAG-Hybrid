CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents_embeddings (
    id UUID PRIMARY KEY,
    embedding VECTOR(384) NOT NULL,
    content TEXT NOT NULL,
    content_type VARCHAR(32) NOT NULL,
    document_id UUID NOT NULL,
    chunk_id INTEGER NOT NULL,
    section VARCHAR(255),
    source TEXT NOT NULL,
    metadata_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

ALTER TABLE documents_embeddings
    ADD COLUMN IF NOT EXISTS metadata_json JSONB NOT NULL DEFAULT '{}'::jsonb;

CREATE INDEX IF NOT EXISTS idx_documents_embeddings_document_id
    ON documents_embeddings (document_id);

CREATE INDEX IF NOT EXISTS idx_documents_embeddings_content_type
    ON documents_embeddings (content_type);
