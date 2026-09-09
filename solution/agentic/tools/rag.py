"""Lexical RAG over CultPass knowledge articles.

Retrieval is TF-IDF cosine similarity over title, tags, and body. This keeps
the project offline-friendly (no embedding API required) while still ranking
articles by query overlap rather than naive substring match.

If the core database is empty, articles are loaded from cultpass_articles.jsonl.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from typing import Any

from agentic.tools.paths import articles_jsonl
from agentic.tools.udahub_ops import list_knowledge_articles

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall((text or "").lower())


def _load_articles() -> list[dict[str, Any]]:
    articles = list_knowledge_articles()
    if articles:
        return articles
    rows = []
    path = articles_jsonl()
    if path.exists():
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                rows.append(
                    {
                        "article_id": item.get("title"),
                        "title": item["title"],
                        "content": item["content"],
                        "tags": item.get("tags", ""),
                    }
                )
    return rows


def _tfidf_vector(tokens: list[str], idf: dict[str, float]) -> dict[str, float]:
    tf = Counter(tokens)
    length = max(len(tokens), 1)
    return {term: (count / length) * idf.get(term, 0.0) for term, count in tf.items()}


def _cosine(left: dict[str, float], right: dict[str, float]) -> float:
    keys = set(left) | set(right)
    dot = sum(left.get(k, 0.0) * right.get(k, 0.0) for k in keys)
    n1 = math.sqrt(sum(v * v for v in left.values()))
    n2 = math.sqrt(sum(v * v for v in right.values()))
    if n1 == 0 or n2 == 0:
        return 0.0
    return dot / (n1 * n2)


def retrieve_knowledge(query: str, k: int = 3) -> list[dict[str, Any]]:
    articles = _load_articles()
    if not articles:
        return []
    docs = [tokenize(f"{a['title']} {a['tags']} {a['content']}") for a in articles]
    df: Counter[str] = Counter()
    for tokens in docs:
        df.update(set(tokens))
    n = len(docs)
    idf = {term: math.log((n + 1) / (freq + 1)) + 1.0 for term, freq in df.items()}
    query_vec = _tfidf_vector(tokenize(query), idf)
    scored = []
    for article, tokens in zip(articles, docs):
        score = _cosine(query_vec, _tfidf_vector(tokens, idf))
        scored.append((score, article))
    scored.sort(key=lambda item: item[0], reverse=True)
    results = []
    for score, article in scored[:k]:
        if score <= 0:
            continue
        results.append({**article, "score": round(score, 4)})
    return results


def retrieve_knowledge_json(query: str, k: int = 3) -> str:
    return json.dumps(retrieve_knowledge(query, k=k), ensure_ascii=False)
