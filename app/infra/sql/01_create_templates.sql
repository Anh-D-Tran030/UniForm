CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS templates (
    id BIGSERIAL PRIMARY KEY,
    template_id TEXT UNIQUE NOT NULL,
    display_name TEXT,
    image_path TEXT,
    ocr_json JSONB,
    embedding VECTOR(128)
);

CREATE INDEX IF NOT EXISTS templates_embedding_hnsw_idx
ON templates
USING hnsw (embedding vector_cosine_ops);
