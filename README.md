# Supabase RAG Toolkit

Abstract RAG (Retrieval Augmented Generation) toolkit for any Supabase project. Groq-powered ETL for extraction and enrichment, multi-provider embeddings (Cohere or NVIDIA NIM), and vector search with reranking.

## Features

- **Groq-powered ETL**: Groq Vision extracts text from images/scanned PDFs, Groq LLM enriches chunks with semantic structuring + synthetic questions
- **Multi-provider**: NVIDIA NIM or Cohere for embeddings + reranking (1 env var switch)
- **Multi-project**: Isolate multiple projects in the same table using the `project` parameter
- **Multi-input**: Ingest URLs, base64 images (Groq Vision), or raw text
- **Vector search**: Cosine similarity search with optional reranking
- **Abstract**: Works with any Supabase project, no project-specific code
- **Configurable**: All models, dimensions, chunk sizes via environment variables
- **Graceful fallback**: If Groq is unavailable, falls back to mechanical chunking automatically

## Architecture

```
URLs / Images / Text → rag-ingest → Groq Vision (extract) → Groq LLM (enrich) → Cohere embed → pgvector
                                                                                                        ↓
Query → rag-query → Cohere embed → vector search → Cohere rerank → results
                                                                                                        ↓
                                                                                            Your AI agent (tool calling)
```

## Setup

### 1. Run the SQL migration

```bash
supabase db push
# or manually run supabase/migrations/rag_pgvector.sql in Supabase SQL Editor
```

This creates:
- `rag_documents` table with `vector(1024)` embedding column
- `rag_search` RPC function for cosine similarity search
- RLS policies (service role bypasses, public read)
- IVFFlat index for fast vector search

### 2. Deploy Edge Functions

```bash
supabase functions deploy rag-ingest
supabase functions deploy rag-query
```

### 3. Set environment variables (secrets)

In Supabase Dashboard → Edge Functions → Secrets:

| Secret | Required | Default | Description |
|--------|----------|---------|-------------|
| `SUPABASE_URL` | ✅ | — | Your Supabase project URL |
| `SUPABASE_SERVICE_ROLE_KEY` | ✅ | — | Service role key (bypasses RLS) |
| `GROQ_API_KEY` | ❌ | — | Groq API key for Vision extraction + LLM enrichment |
| `GROQ_VISION_MODEL` | ❌ | `meta-llama/llama-3.2-90b-vision-preview` | Groq Vision model for image extraction |
| `GROQ_LLM_MODEL` | ❌ | `llama-3.3-70b-versatile` | Groq LLM model for chunk enrichment |
| `RAG_ENRICH_ENABLED` | ❌ | `true` | Enable/disable Groq enrichment (set `false` for mechanical chunking only) |
| `RAG_EMBEDDING_PROVIDER` | ❌ | `cohere` | `cohere` or `nvidia` |
| `COHERE_API_KEY` | If Cohere | — | Cohere API key |
| `NVIDIA_NIM_API_KEY` | If NVIDIA | — | NVIDIA NIM API key |
| `RAG_EMBEDDING_MODEL` | ❌ | Auto | Override embedding model |
| `RAG_EMBEDDING_DIMENSIONS` | ❌ | `1024` | Vector dimensions (max 2000 for Supabase pgvector indexes) |
| `RAG_RERANK_MODEL` | ❌ | Auto | Override rerank model |
| `RAG_RERANK_ENABLED` | ❌ | `true` | Enable/disable reranking |
| `RAG_CHUNK_SIZE` | ❌ | `800` | Chunk size in characters |
| `RAG_CHUNK_OVERLAP` | ❌ | `100` | Overlap between chunks |

### Default models per provider

| Provider | Embedding Model | Rerank Model | Dimensions |
|----------|----------------|--------------|------------|
| Cohere | `embed-multilingual-v3.0` | `rerank-multilingual-v3.0` | 1024 |
| NVIDIA | `nvidia/llama-nemotron-embed-1b-v2` | `nvidia/llama-nemotron-rerank-1b-v2` | 1024 |

> **Note**: Supabase pgvector indexes (IVFFlat and HNSW) support up to 2000 dimensions. Both providers default to 1024 for compatibility.

## Usage

### Ingest URLs

```bash
curl -X POST https://YOUR_PROJECT.supabase.co/functions/v1/rag-ingest \
  -H "Authorization: Bearer YOUR_SERVICE_ROLE_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "urls": ["https://example.com/article", "https://example.com/doc.pdf"],
    "project": "my-project",
    "metadata": { "category": "clinical" }
  }'
```

### Ingest Images (Groq Vision OCR)

```bash
curl -X POST https://YOUR_PROJECT.supabase.co/functions/v1/rag-ingest \
  -H "Authorization: Bearer YOUR_SERVICE_ROLE_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "images": ["iVBORw0KGgo...base64..."],
    "project": "my-project",
    "metadata": { "source_label": "historia_clinica_001.jpg" }
  }'
```

### Ingest Raw Text (Groq LLM enrichment only)

```bash
curl -X POST https://YOUR_PROJECT.supabase.co/functions/v1/rag-ingest \
  -H "Authorization: Bearer YOUR_SERVICE_ROLE_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "texts": ["Texto crudo a procesar y enriquecer..."],
    "project": "my-project",
    "metadata": { "source_label": "manual_enfermeria.txt" }
  }'
```

Response:
```json
{
  "success": true,
  "project": "my-project",
  "results": [
    { "source": "https://example.com/article", "status": "success", "chunks": 12, "source_type": "html", "enriched": true }
  ],
  "total_inputs": 1,
  "total_success": 1,
  "total_chunks": 12,
  "enrichment": "groq"
}
```

### Query (vector search + reranking)

```bash
curl -X POST https://YOUR_PROJECT.supabase.co/functions/v1/rag-query \
  -H "Authorization: Bearer YOUR_SERVICE_ROLE_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are palliative care?",
    "project": "my-project",
    "top_k": 5,
    "rerank": true
  }'
```

Response:
```json
{
  "query": "What are palliative care?",
  "project": "my-project",
  "results": [
    {
      "chunk_text": "Palliative care is...",
      "source_url": "https://example.com/article",
      "source_type": "html",
      "chunk_index": 0,
      "metadata": { "category": "clinical", "ingested_at": "2024-01-01T00:00:00.000Z" },
      "score": 0.99
    }
  ],
  "count": 5,
  "reranked": true
}
```

### Integrate with your AI agent

Add a tool to your AI agent that calls `rag-query`:

```typescript
async function ragKnowledgeSearch(args: { query: string; top_k?: number }) {
  const res = await fetch(`${Deno.env.get("SUPABASE_URL")}/functions/v1/rag-query`, {
    method: "POST",
    headers: {
      "Authorization": `Bearer ${Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      query: args.query,
      project: "my-project",
      top_k: args.top_k || 5,
      rerank: true,
    }),
  });
  const data = await res.json();
  return data.results.map((r: any) => ({
    text: r.chunk_text,
    source: r.source_url,
    score: r.score,
  }));
}
```

## Switching providers

Change one environment variable and re-ingest:

```
# Cohere → NVIDIA
RAG_EMBEDDING_PROVIDER=nvidia
NVIDIA_NIM_API_KEY=your_key

# NVIDIA → Cohere
RAG_EMBEDDING_PROVIDER=cohere
COHERE_API_KEY=your_key
```

> **Important**: Embeddings from different providers are incompatible. Always re-ingest all URLs after switching providers.

## Multi-project support

Use the `project` parameter to isolate different projects in the same table:

```bash
# Project A
curl -X POST .../rag-ingest -d '{"urls": [...], "project": "project-a"}'
curl -X POST .../rag-query -d '{"query": "...", "project": "project-a"}'

# Project B
curl -X POST .../rag-ingest -d '{"urls": [...], "project": "project-b"}'
curl -X POST .../rag-query -d '{"query": "...", "project": "project-b"}'
```

## File structure

```
supabase-rag-toolkit/
├── README.md
├── supabase/
│   ├── functions/
│   │   ├── rag-ingest/
│   │   │   └── index.ts      # URLs/Images/Text → Groq Vision → Groq LLM enrich → Cohere embed → pgvector
│   │   └── rag-query/
│   │       └── index.ts      # query → Cohere embed → vector search → Cohere rerank
│   └── migrations/
│       └── rag_pgvector.sql  # Table + RPC + indexes + RLS
```

## License

MIT
