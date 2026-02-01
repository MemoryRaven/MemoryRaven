# Vector Knowledge Indexing Pipeline

Production-oriented, multi-source indexing pipeline for building a high-quality vector knowledge base from:
- Web pages (with caching + change detection)
- Research papers (PDF/HTML)
- Tool outputs (logs, JSON, artifacts)
- Conversations (chat transcripts / memory files)

## Goals
- High quality chunks (type-specific chunking)
- Strong metadata & enrichment
- Deduplication + incremental updates
- Namespace routing (multi-tenant / multi-domain separation)
- Batch and real-time indexing modes

## Quickstart

### 1) Create a venv and install (optional) extras
This project is dependency-light by default.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
# optional extras
pip install chromadb sentence-transformers pypdf beautifulsoup4 lxml
```

### 2) Run a demo batch index
```bash
python -m vector_indexing.cli batch \
  --db vector_indexing/data/index.db \
  --vector-dir vector_indexing/data/chroma \
  --source conversations:/home/bigphoot/memory \
  --source web:https://example.com \
  --namespace default
```

### 3) Run streaming mode (tail a file)
```bash
python -m vector_indexing.cli stream \
  --db vector_indexing/data/index.db \
  --watch /home/bigphoot/logs/service.log \
  --namespace logs
```

## Architecture

Pipeline stages:
1. **Ingestion** → `Document` items from multiple `Source` drivers
2. **Normalization** → text cleanup + structural hints
3. **Chunking** → content-type chunkers (web, pdf/paper, code/log, conversation)
4. **Metadata extraction** → URL/domain, title, headings, authors, timestamps, language, etc.
5. **Enrichment** → quality scoring, entity/keyword extraction (optional), topic tags
6. **Fingerprinting** → exact hash + simhash for near-dup
7. **Routing** → namespace + collection selection
8. **Embedding** → multiple embeddings per chunk (semantic + contextual)
9. **Storage** → SQLite manifest + vector store (Chroma optional)
10. **Incremental update** → change detection by source revision (etag/last-mod, file mtime, git hash)

## Storage
- **SQLite**: canonical registry of sources, documents, chunks, fingerprints, embedding jobs
- **Vector store**: Chroma (optional) or JSONL fallback

## Notes
- The embedding layer is provider-agnostic.
- Web caching follows the spirit of `claude-research-team`: persist fetched pages + metadata, avoid refetching when unchanged.
