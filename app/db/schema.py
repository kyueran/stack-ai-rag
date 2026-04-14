SCHEMA_STATEMENTS: tuple[str, ...] = (
    """
    CREATE TABLE IF NOT EXISTS documents (
        document_id TEXT PRIMARY KEY,
        filename TEXT NOT NULL,
        byte_size INTEGER NOT NULL,
        page_count INTEGER NOT NULL,
        chunk_count INTEGER NOT NULL,
        text_char_count INTEGER NOT NULL,
        ingested_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS chunks (
        chunk_id TEXT PRIMARY KEY,
        document_id TEXT NOT NULL,
        page_start INTEGER NOT NULL,
        page_end INTEGER NOT NULL,
        char_count INTEGER NOT NULL,
        text TEXT NOT NULL,
        FOREIGN KEY(document_id) REFERENCES documents(document_id) ON DELETE CASCADE
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS embeddings (
        chunk_id TEXT PRIMARY KEY,
        model TEXT NOT NULL,
        dimension INTEGER NOT NULL,
        vector_json TEXT NOT NULL,
        created_at TEXT NOT NULL,
        FOREIGN KEY(chunk_id) REFERENCES chunks(chunk_id) ON DELETE CASCADE
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS terms (
        term TEXT NOT NULL,
        chunk_id TEXT NOT NULL,
        tf INTEGER NOT NULL,
        field TEXT NOT NULL DEFAULT 'body',
        PRIMARY KEY(term, chunk_id),
        FOREIGN KEY(chunk_id) REFERENCES chunks(chunk_id) ON DELETE CASCADE
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS retrieval_logs (
        retrieval_id INTEGER PRIMARY KEY AUTOINCREMENT,
        query_text TEXT NOT NULL,
        transformed_query TEXT NOT NULL,
        intent TEXT NOT NULL,
        top_k INTEGER NOT NULL,
        results_json TEXT NOT NULL,
        created_at TEXT NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_chunks_document ON chunks(document_id)",
    "CREATE INDEX IF NOT EXISTS idx_terms_term ON terms(term)",
    "CREATE INDEX IF NOT EXISTS idx_embeddings_model ON embeddings(model)",
    "CREATE INDEX IF NOT EXISTS idx_documents_ingested_at ON documents(ingested_at)",
)
