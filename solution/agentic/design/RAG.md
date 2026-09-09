# RAG design

UDA-Hub retrieval is **lexical TF-IDF cosine similarity**, implemented in
`agentic/tools/rag.py`. It does not call an embedding API, so retrieval works
offline and in CI.

## Corpus

Primary source: `knowledge` rows in `data/core/udahub.db` (loaded from
`cultpass_articles.jsonl`, 16 articles covering reservation, subscription,
login, quota, QR, premium fees, profile, refunds, blocks, transfers, weather
cancellations, guests, billing, human support, and reviews).

Fallback: if the core DB has no articles, the same JSONL file is read directly.

## Indexing

1. Concatenate `title`, `tags`, and `content`
2. Lowercase alphanumeric tokens
3. Per-document TF and smoothed IDF: `log((N+1)/(df+1)) + 1`
4. Cosine similarity against the ticket query vector

## Use at runtime

The Resolver requests the top 3 articles. The highest-scoring article supplies
the suggested phrasing. Score also feeds resolver confidence. If no article
scores above 0, the resolver escalates.

## Why not vector embeddings

The starter stack already includes LangChain and SQLite. A local TF-IDF index
avoids shipping a large `.db` of embeddings and does not require `OPENAI_API_KEY`
for the retrieval path. The optional LLM only rewrites the draft after retrieval.
