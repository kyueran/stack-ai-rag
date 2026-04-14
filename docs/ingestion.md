# Ingestion and Chunking Considerations

This project ingests PDFs through `POST /api/v1/ingest`, extracts page text, and chunks it for retrieval.

## Extraction Considerations

- Text quality depends on PDF internals. Image-only/scanned files often have no extractable text without OCR.
- Page-level extraction failures are captured as warnings instead of failing the entire document by default.
- Encrypted PDFs are accepted, but extraction quality can degrade and is flagged in warnings.

## Chunking Strategy

- Sentence-aware splitting: chunks are formed by sentence boundaries to preserve semantic coherence.
- Fixed-size cap (`CHUNK_SIZE`): keeps chunks small enough for good retrieval precision.
- Character-overlap (`CHUNK_OVERLAP`): preserves continuity so boundary facts are less likely to be lost.
- Deterministic chunk IDs: generated from document/page/offset/text hash, enabling idempotent re-ingestion.

## Tradeoffs

- Larger chunks increase context coverage but can reduce retrieval precision.
- Smaller chunks increase precision but may fragment concepts and require stronger reranking.
- Higher overlap improves recall near boundaries but increases index size and duplicate evidence.

## Hard Cases

- Tables and forms: linearized text can lose row/column semantics.
- Repeated headers/footers: can add noise to keyword matching unless filtered in post-processing.
- Multi-column layouts: extraction order may interleave unrelated text segments.

## Default Tuning Guidance

- Start with `CHUNK_SIZE=900` and `CHUNK_OVERLAP=150`.
- For short documents with dense facts, reduce chunk size to increase precision.
- For narrative documents, increase chunk size modestly to retain discourse context.
