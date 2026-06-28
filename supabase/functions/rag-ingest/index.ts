// @ts-nocheck — Deno Edge Function: RAG Ingest
// Abstract RAG toolkit: ingest URLs/PDFs, chunk, embed with NVIDIA NIM, store in pgvector
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
const CHUNK_SIZE = parseInt(Deno.env.get("RAG_CHUNK_SIZE") || "800");
const CHUNK_OVERLAP = parseInt(Deno.env.get("RAG_CHUNK_OVERLAP") || "100");
const COHERE_EMBED_URL = "https://api.cohere.com/v2/embed";
const NVIDIA_NIM_URL = "https://integrate.api.nvidia.com/v1/embeddings";

const corsHeaders = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Methods": "POST, OPTIONS",
  "Access-Control-Allow-Headers": "Content-Type, Authorization, apikey",
};

interface IngestRequest {
  urls: string[];
  project?: string;
  metadata?: Record<string, any>;
}

// ===== TEXT EXTRACTION =====

async function extractTextFromURL(url: string): Promise<{ text: string; sourceType: string }> {
  const res = await fetch(url, {
    headers: { "User-Agent": "RAG-Toolkit/1.0" },
  });

  if (!res.ok) {
    throw new Error(`Failed to fetch ${url}: ${res.status}`);
  }

  const contentType = res.headers.get("content-type") || "";

  if (contentType.includes("application/pdf")) {
    // For PDFs, we extract text from the raw bytes
    // Deno doesn't have a built-in PDF parser, so we use a simple approach:
    // fetch the PDF and extract text using a basic regex on the raw content
    // For production, consider using pdf-parse or similar
    const arrayBuffer = await res.arrayBuffer();
    const text = extractTextFromPDF(arrayBuffer);
    return { text, sourceType: "pdf" };
  }

  // HTML: fetch and extract text content
  const html = await res.text();
  const text = extractTextFromHTML(html);
  return { text, sourceType: "html" };
}

function extractTextFromHTML(html: string): string {
  // Remove script and style tags
  let cleaned = html
    .replace(/<script[^>]*>[\s\S]*?<\/script>/gi, " ")
    .replace(/<style[^>]*>[\s\S]*?<\/style>/gi, " ")
    .replace(/<nav[^>]*>[\s\S]*?<\/nav>/gi, " ")
    .replace(/<footer[^>]*>[\s\S]*?<\/footer>/gi, " ")
    .replace(/<header[^>]*>[\s\S]*?<\/header>/gi, " ");

  // Convert common HTML elements to text with spacing
  cleaned = cleaned
    .replace(/<br\s*\/?>/gi, "\n")
    .replace(/<\/p>/gi, "\n\n")
    .replace(/<\/div>/gi, "\n")
    .replace(/<\/li>/gi, "\n")
    .replace(/<\/h[1-6]>/gi, "\n\n")
    .replace(/<li[^>]*>/gi, "- ");

  // Remove all remaining HTML tags
  cleaned = cleaned.replace(/<[^>]+>/g, " ");

  // Decode common HTML entities
  cleaned = cleaned
    .replace(/&nbsp;/g, " ")
    .replace(/&amp;/g, "&")
    .replace(/&lt;/g, "<")
    .replace(/&gt;/g, ">")
    .replace(/&quot;/g, '"')
    .replace(/&#39;/g, "'")
    .replace(/&aacute;/g, "á")
    .replace(/&eacute;/g, "é")
    .replace(/&iacute;/g, "í")
    .replace(/&oacute;/g, "ó")
    .replace(/&uacute;/g, "ú")
    .replace(/&ntilde;/g, "ñ")
    .replace(/&Ntilde;/g, "Ñ");

  // Normalize whitespace
  cleaned = cleaned
    .replace(/[ \t]+/g, " ")
    .replace(/\n{3,}/g, "\n\n")
    .trim();

  return cleaned;
}

function extractTextFromPDF(buffer: ArrayBuffer): string {
  // Basic PDF text extraction: find text between BT and ET markers
  // This is a simplified approach — works for many text-based PDFs
  const bytes = new Uint8Array(buffer);
  let text = "";

  // Convert to string for regex (latin1 to preserve byte values)
  const raw = new TextDecoder("latin1").decode(bytes);

  // Extract text from PDF streams (simplified)
  const textMatches = raw.match(/\(([^)]+)\)/g);
  if (textMatches) {
    text = textMatches
      .map((m) => m.slice(1, -1))
      .join(" ")
      .replace(/\\n/g, "\n")
      .replace(/\\t/g, " ");
  }

  // Fallback: if no text found, try Tj/TJ operators
  if (!text.trim()) {
    const tjMatches = raw.match(/\[([^\]]+)\]\s*TJ/g);
    if (tjMatches) {
      text = tjMatches
        .map((m) => {
          const inner = m.match(/\(([^)]+)\)/g);
          return inner ? inner.map((x) => x.slice(1, -1)).join("") : "";
        })
        .join(" ");
    }
  }

  return text.trim() || "[PDF: unable to extract text — may be scanned image]";
}

// ===== CHUNKING =====

function chunkText(text: string, chunkSize: number, overlap: number): string[] {
  // Split by paragraphs first
  const paragraphs = text.split(/\n\n+/);

  const chunks: string[] = [];
  let currentChunk = "";

  for (const para of paragraphs) {
    const paraTrimmed = para.trim();
    if (!paraTrimmed) continue;

    // If paragraph is longer than chunk_size, split by sentences
    if (paraTrimmed.length > chunkSize) {
      if (currentChunk) {
        chunks.push(currentChunk.trim());
        currentChunk = "";
      }
      const sentences = paraTrimmed.match(/[^.!?]+[.!?]+/g) || [paraTrimmed];
      let sentenceChunk = "";
      for (const sentence of sentences) {
        if ((sentenceChunk + " " + sentence).length > chunkSize) {
          if (sentenceChunk) chunks.push(sentenceChunk.trim());
          // Start new chunk with overlap from previous
          if (chunks.length > 0 && overlap > 0) {
            const prev = chunks[chunks.length - 1];
            const overlapText = prev.slice(-overlap);
            sentenceChunk = overlapText + " " + sentence;
          } else {
            sentenceChunk = sentence;
          }
        } else {
          sentenceChunk = (sentenceChunk + " " + sentence).trim();
        }
      }
      if (sentenceChunk) chunks.push(sentenceChunk.trim());
    } else {
      // Paragraph fits in chunk
      if ((currentChunk + "\n\n" + paraTrimmed).length > chunkSize) {
        if (currentChunk) {
          chunks.push(currentChunk.trim());
          // Start new chunk with overlap
          if (overlap > 0) {
            const overlapText = currentChunk.slice(-overlap);
            currentChunk = overlapText + "\n\n" + paraTrimmed;
          } else {
            currentChunk = paraTrimmed;
          }
        } else {
          currentChunk = paraTrimmed;
        }
      } else {
        currentChunk = currentChunk ? currentChunk + "\n\n" + paraTrimmed : paraTrimmed;
      }
    }
  }

  if (currentChunk) chunks.push(currentChunk.trim());

  return chunks.filter((c) => c.length > 50); // Filter out tiny chunks
}

// ===== EMBEDDING (NVIDIA NIM) =====

async function generateEmbeddings(texts: string[]): Promise<number[][]> {
  const allEmbeddings: number[][] = [];
  const batchSize = EMBEDDING_PROVIDER === "nvidia" ? 25 : 96; // Cohere supports 96

  for (let i = 0; i < texts.length; i += batchSize) {
    const batch = texts.slice(i, i + batchSize);

    let res: Response;

    if (EMBEDDING_PROVIDER === "nvidia") {
      res = await fetch(NVIDIA_NIM_URL, {
        method: "POST",
        headers: {
          "Authorization": `Bearer ${NVIDIA_NIM_API_KEY}`,
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          model: EMBEDDING_MODEL,
          input: batch,
          input_type: "passage",
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
          texts: batch,
          input_type: "search_document",
          embedding_types: ["float"],
        }),
      });
    }

    if (!res.ok) {
      const err = await res.text();
      throw new Error(`${EMBEDDING_PROVIDER} embedding error: ${res.status} — ${err.slice(0, 300)}`);
    }

    const data = await res.json();
    let embeddings: number[][];
    if (EMBEDDING_PROVIDER === "nvidia") {
      embeddings = data.data.map((d: any) => d.embedding);
    } else {
      embeddings = data.embeddings.float;
    }
    allEmbeddings.push(...embeddings);

    console.log(`[rag-ingest] Embedded batch ${Math.floor(i / batchSize) + 1}/${Math.ceil(texts.length / batchSize)}`);
  }

  return allEmbeddings;
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
    const { urls, project = "default", metadata = {} }: IngestRequest = await req.json();

    if (!urls || !Array.isArray(urls) || urls.length === 0) {
      return new Response(JSON.stringify({ error: "Se requiere al menos una URL" }), {
        status: 400,
        headers: { "Content-Type": "application/json", ...corsHeaders },
      });
    }

    console.log(`[rag-ingest] START | project: ${project} | urls: ${urls.length}`);

    const supabase = createClient(SUPABASE_URL!, SUPABASE_SERVICE_ROLE_KEY!);

    const results: any[] = [];

    for (const url of urls) {
      try {
        console.log(`[rag-ingest] Processing: ${url}`);

        // 1. Extract text
        const { text, sourceType } = await extractTextFromURL(url);

        if (!text || text.length < 50) {
          console.log(`[rag-ingest] Skipping ${url} — insufficient text`);
          results.push({ url, status: "skipped", reason: "insufficient_text" });
          continue;
        }

        // 2. Chunk
        const chunks = chunkText(text, CHUNK_SIZE, CHUNK_OVERLAP);
        console.log(`[rag-ingest] ${url} → ${chunks.length} chunks`);

        if (chunks.length === 0) {
          results.push({ url, status: "skipped", reason: "no_chunks" });
          continue;
        }

        // 3. Delete existing chunks for this URL (re-ingest support)
        await supabase
          .from("rag_documents")
          .delete()
          .eq("source_url", url)
          .eq("project", project);

        // 4. Generate embeddings
        const embeddings = await generateEmbeddings(chunks);

        // 5. Store in pgvector
        const rows = chunks.map((chunkText, i) => ({
          project,
          source_url: url,
          source_type: sourceType,
          chunk_index: i,
          chunk_text: chunkText,
          embedding: embeddings[i],
          metadata: { ...metadata, ingested_at: new Date().toISOString() },
        }));

        const { error: insertError } = await supabase
          .from("rag_documents")
          .insert(rows);

        if (insertError) {
          console.log(`[rag-ingest] Insert error for ${url}: ${insertError.message}`);
          results.push({ url, status: "error", error: insertError.message });
        } else {
          console.log(`[rag-ingest] ${url} → ${rows.length} chunks stored`);
          results.push({ url, status: "success", chunks: rows.length, source_type: sourceType });
        }
      } catch (err: any) {
        console.log(`[rag-ingest] Error processing ${url}: ${err.message}`);
        results.push({ url, status: "error", error: err.message });
      }
    }

    const successCount = results.filter((r) => r.status === "success").length;
    const totalChunks = results.reduce((sum, r) => sum + (r.chunks || 0), 0);

    console.log(`[rag-ingest] END | success: ${successCount}/${urls.length} | total chunks: ${totalChunks}`);

    return new Response(JSON.stringify({
      success: true,
      project,
      results,
      total_urls: urls.length,
      total_success: successCount,
      total_chunks: totalChunks,
    }), {
      status: 200,
      headers: { "Content-Type": "application/json", ...corsHeaders },
    });

  } catch (err) {
    const errMsg = err instanceof Error ? err.message : "Error desconocido";
    console.log(`[rag-ingest] EXCEPTION: ${errMsg}`);
    return new Response(JSON.stringify({ error: errMsg }), {
      status: 500,
      headers: { "Content-Type": "application/json", ...corsHeaders },
    });
  }
});
