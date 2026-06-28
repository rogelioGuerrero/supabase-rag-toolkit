-- RAG Toolkit: Enable pgvector + create rag_documents table
-- Works with any Supabase project. Uses project field for multi-project isolation.

-- 1. Enable pgvector extension
create extension if not exists vector;

-- 2. Create rag_documents table
create table if not exists rag_documents (
  id uuid primary key default gen_random_uuid(),
  project text not null default 'default',
  source_url text not null,
  source_type text not null default 'html', -- 'html' | 'pdf' | 'text'
  chunk_index int not null,
  chunk_text text not null,
  embedding vector(1024),
  metadata jsonb default '{}'::jsonb,
  created_at timestamptz not null default now()
);

-- 3. Index for fast vector similarity search (cosine)
create index if not exists rag_documents_embedding_idx
  on rag_documents using ivfflat (embedding vector_cosine_ops)
  with (lists = 100);

-- 4. Index for project filtering
create index if not exists rag_documents_project_idx
  on rag_documents (project);

-- 5. Index for source_url (to delete/re-ingest)
create index if not exists rag_documents_source_url_idx
  on rag_documents (source_url);

-- 6. Enable RLS
alter table rag_documents enable row level security;

-- 7. RLS policies (service role bypasses RLS, so Edge Functions work)
-- Public can read (adjust to your needs)
create policy if not exists "rag_documents_read_all"
  on rag_documents for select
  using (true);

-- Only service role can insert/update/delete (Edge Functions use service role)
create policy if not exists "rag_documents_write_service_only"
  on rag_documents for all
  using (auth.role() = 'service_role')
  with check (auth.role() = 'service_role');

-- 8. RPC function for vector similarity search (cosine distance)
-- Used by rag-query Edge Function
create or replace function rag_search(
  query_embedding vector(1024),
  match_count int default 10,
  filter_project text default 'default'
)
returns table (
  id uuid,
  source_url text,
  source_type text,
  chunk_index int,
  chunk_text text,
  metadata jsonb,
  similarity float
)
language sql
stable
as $$
  select
    id,
    source_url,
    source_type,
    chunk_index,
    chunk_text,
    metadata,
    1 - (embedding <=> query_embedding) as similarity
  from rag_documents
  where project = filter_project
    and embedding is not null
  order by embedding <=> query_embedding
  limit match_count;
$$;

-- 9. Grant execute on RPC to authenticated and anon
grant execute on function rag_search to authenticated;
grant execute on function rag_search to anon;
