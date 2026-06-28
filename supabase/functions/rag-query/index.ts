// @ts-nocheck — Deno Edge Function: RAG Query
// Abstract RAG toolkit: search pgvector for relevant chunks, optional reranking via NVIDIA NIM
// Works with any Supabase project. Configurable via env vars.

import { createClient } from "jsr:@supabase/supabase-js@2";

const SUPABASE_URL = Deno.env.get("SUPABASE_URL");
const SUPABASE_SERVICE_ROLE_KEY = Deno.env.get("SUPABASE_SERVICE_ROLE_KEY");

// Embedding provider: "cohere" (default) | "nvidia"
const EMBEDDING_PROVIDER = Deno.env.get("RAG_EMBEDDING_PROVIDER") || "cohere";
const COHERE_API_KEY = Deno.env.get("COHERE_API_KEY");
const NVIDIA_NIM_API_KEY = Deno.env.get("NVIDIA_NIM_API_KEY");

// Configurable via env vars
const EMBEDDING_MODEL = Deno.env.get("RAG_EMBEDDING_MODEL") || (EMBEDDING_PROVIDER === "nvidia" ? "nvidia/llama-nemotron-embed-1b-v2" : "embed-multilingual-v3.0");
const EMBEDDING_DIMENSIONS = parseInt(Deno.env.get("RAG_EMBEDDING_DIMENSIONS") || "1024");
const RERANK_MODEL = Deno.env.get("RAG_RERANK_MODEL") || (EMBEDDING_PROVIDER === "nvidia" ? "nvidia/llama-nemotron-rerank-1b-v2" : "rerank-multilingual-v3.0");
const RERANK_ENABLED = Deno.env.get("RAG_RERANK_ENABLED") !== "false";
const COHERE_EMBED_URL = "https://api.cohere.com/v2/embed";
const COHERE_RERANK_URL = "https://api.cohere.com/v1/rerank";
const NVIDIA_NIM_EMBED_URL = "https://integrate.api.nvidia.com/v1/embeddings";
const NVIDIA_NIM_RANK_URL = "https://integrate.api.nvidia.com/v1/ranking";

const corsHeaders = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Methods": "POST, OPTIONS",
  "Access-Control-Allow-Headers": "Content-Type, Authorization, apikey",
};

interface QueryRequest {
  query: string;
  project?: string;
  top_k?: number;
  rerank?: boolean;
  filter?: Record<string, any>;
}

// ===== EMBEDDING (NVIDIA NIM) =====

async function generateQueryEmbedding(query: string): Promise<number[]> {
  let res: Response;

  if (EMBEDDING_PROVIDER === "nvidia") {
    res = await fetch(NVIDIA_NIM_EMBED_URL, {
      method: "POST",
      headers: {
        "Authorization": `Bearer ${NVIDIA_NIM_API_KEY}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        model: EMBEDDING_MODEL,
        input: [query],
        input_type: "query",
        encoding_format: "float",
        dimensions: EMBEDDING_DIMENSIONS,
      }),
    });
  } else {
    // Cohere
    res = await fetch(COHERE_EMBED_URL, {
      method: "POST",
      headers: {
        "Authorization": `Bearer ${COHERE_API_KEY}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        model: EMBEDDING_MODEL,
        texts: [query],
        input_type: "search_query",
        embedding_types: ["float"],
      }),
    });
  }

  if (!res.ok) {
    const err = await res.text();
    throw new Error(`${EMBEDDING_PROVIDER} embedding error: ${res.status} — ${err.slice(0, 300)}`);
  }

  const data = await res.json();
  if (EMBEDDING_PROVIDER === "nvidia") {
    return data.data[0].embedding;
  } else {
    return data.embeddings.float[0];
  }
}

// ===== RERANKING (NVIDIA NIM) =====

async function rerankChunks(query: string, chunks: Array<{ id: string; text: string; [key: string]: any }>): Promise<Array<{ id: string; text: string; score: number; [key: string]: any }>> {
  try {
    let res: Response;

    if (EMBEDDING_PROVIDER === "nvidia") {
      res = await fetch(NVIDIA_NIM_RANK_URL, {
        method: "POST",
        headers: {
          "Authorization": `Bearer ${NVIDIA_NIM_API_KEY}`,
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          model: RERANK_MODEL,
          query: query,
          documents: chunks.map((c) => c.text),
          top_n: chunks.length,
          return_documents: false,
        }),
      });
    } else {
      // Cohere
      res = await fetch(COHERE_RERANK_URL, {
        method: "POST",
        headers: {
          "Authorization": `Bearer ${COHERE_API_KEY}`,
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          model: RERANK_MODEL,
          query: query,
          documents: chunks.map((c) => c.text),
          top_n: chunks.length,
          return_documents: false,
        }),
      });
    }

    if (!res.ok) {
      const err = await res.text();
      console.log(`[rag-query] Rerank error: ${res.status} — ${err.slice(0, 200)}`);
      return chunks.map((c, i) => ({ ...c, score: 1 - i * 0.05 }));
    }

    const data = await res.json();
    const rankings = data.rankings || data.results || [];

    return rankings.map((r: any) => ({
      ...chunks[r.index],
      score: r.relevance_score || r.score || 0,
    }));
  } catch (err: any) {
    console.log(`[rag-query] Rerank fallback: ${err.message}`);
    return chunks.map((c, i) => ({ ...c, score: 1 - i * 0.05 }));
  }
}

// ===== MAIN =====

Deno.serve(async (req: Request) => {
  if (req.method === "OPTIONS") {
    return new Response(null, { status: 200, headers: corsHeaders });
  }

  if (req.method !== "POST") {
    return new Response(JSON.stringify({ error: "Method not allowed" }), {
      status: 405,
      headers: { "Content-Type": "application/json", ...corsHeaders },
    });
  }

  try {
    const { query, project = "default", top_k = 5, rerank = RERANK_ENABLED, filter = {} }: QueryRequest = await req.json();

    if (!query || query.trim().length < 3) {
      return new Response(JSON.stringify({ error: "Se requiere una consulta de al menos 3 caracteres" }), {
        status: 400,
        headers: { "Content-Type": "application/json", ...corsHeaders },
      });
    }

    console.log(`[rag-query] START | project: ${project} | query: ${query.slice(0, 80)} | top_k: ${top_k} | rerank: ${rerank}`);

    const supabase = createClient(SUPABASE_URL!, SUPABASE_SERVICE_ROLE_KEY!);

    // 1. Generate query embedding
    const queryEmbedding = await generateQueryEmbedding(query);

    // 2. Vector search in pgvector (fetch more than top_k for reranking)
    const fetchCount = rerank ? Math.max(top_k * 3, 15) : top_k;

    let dbQuery = supabase
      .from("rag_documents")
      .select("id, source_url, source_type, chunk_index, chunk_text, metadata")
      .eq("project", project)
      .limit(fetchCount);

    // Apply optional metadata filters
    for (const [key, value] of Object.entries(filter)) {
      dbQuery = dbQuery.contains(`metadata->${key}`, JSON.stringify(value));
    }

    // Use RPC for vector similarity search
    const { data: vectorResults, error: rpcError } = await supabase.rpc("rag_search", {
      query_embedding: queryEmbedding,
      match_count: fetchCount,
      filter_project: project,
    });

    if (rpcError) {
      // Fallback: use direct query with embedding filter (less efficient)
      console.log(`[rag-query] RPC not available, using fallback: ${rpcError.message}`);
      const { data, error: queryError } = await supabase
        .from("rag_documents")
        .select("id, source_url, source_type, chunk_index, chunk_text, metadata")
        .eq("project", project)
        .limit(fetchCount);

      if (queryError) {
        throw new Error(`Query error: ${queryError.message}`);
      }

      // Client-side cosine similarity (fallback only)
      const scored = (data || []).map((doc: any) => ({
        ...doc,
        score: 0,
      }));

      if (scored.length === 0) {
        return new Response(JSON.stringify({
          query,
          project,
          results: [],
          count: 0,
          reranked: false,
        }), {
          status: 200,
          headers: { "Content-Type": "application/json", ...corsHeaders },
        });
      }

      // If we have results but no RPC, return them without similarity scoring
      const fallbackResults = scored.slice(0, top_k).map((r: any) => ({
        chunk_text: r.chunk_text,
        source_url: r.source_url,
        source_type: r.source_type,
        chunk_index: r.chunk_index,
        metadata: r.metadata,
        score: 0,
      }));

      return new Response(JSON.stringify({
        query,
        project,
        results: fallbackResults,
        count: fallbackResults.length,
        reranked: false,
        warning: "Vector search RPC not available — install rag_search function for similarity search",
      }), {
        status: 200,
        headers: { "Content-Type": "application/json", ...corsHeaders },
      });
    }

    const candidates = vectorResults || [];

    if (candidates.length === 0) {
      console.log(`[rag-query] No results found`);
      return new Response(JSON.stringify({
        query,
        project,
        results: [],
        count: 0,
        reranked: false,
      }), {
        status: 200,
        headers: { "Content-Type": "application/json", ...corsHeaders },
      });
    }

    console.log(`[rag-query] Vector search: ${candidates.length} candidates`);

    // 3. Optional reranking
    let finalResults = candidates;

    if (rerank && candidates.length > 1) {
      const chunksForRerank = candidates.map((c: any) => ({
        id: c.id || c.chunk_index?.toString() || Math.random().toString(),
        text: c.chunk_text,
        chunk_text: c.chunk_text,
        source_url: c.source_url,
        source_type: c.source_type,
        chunk_index: c.chunk_index,
        metadata: c.metadata,
      }));

      const reranked = await rerankChunks(query, chunksForRerank);
      finalResults = reranked.slice(0, top_k);
      console.log(`[rag-query] Reranked → top ${finalResults.length}`);
    } else {
      finalResults = candidates.slice(0, top_k);
    }

    // 4. Format results
    const results = finalResults.map((r: any) => ({
      chunk_text: r.chunk_text,
      source_url: r.source_url,
      source_type: r.source_type,
      chunk_index: r.chunk_index,
      metadata: r.metadata,
      score: r.score || r.similarity || 0,
    }));

    console.log(`[rag-query] END | results: ${results.length} | reranked: ${rerank && candidates.length > 1}`);

    return new Response(JSON.stringify({
      query,
      project,
      results,
      count: results.length,
      reranked: rerank && candidates.length > 1,
    }), {
      status: 200,
      headers: { "Content-Type": "application/json", ...corsHeaders },
    });

  } catch (err) {
    const errMsg = err instanceof Error ? err.message : "Error desconocido";
    console.log(`[rag-query] EXCEPTION: ${errMsg}`);
    return new Response(JSON.stringify({ error: errMsg }), {
      status: 500,
      headers: { "Content-Type": "application/json", ...corsHeaders },
    });
  }
});
