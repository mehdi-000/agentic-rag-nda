import json
from pathlib import Path
from typing import Any

from langchain_ollama import OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_qdrant.fastembed_sparse import FastEmbedSparse
from langchain_qdrant.qdrant import RetrievalMode

from qdrant_client.http import models as qmodels

from config import (
    PARENTS_DIR, OLLAMA_HOST,
    EMBED_MODEL, CHILD_COLLECTION, TOP_K, SCORE_THRESHOLD,
)
from qdrant_store import qdrant_client

_store: QdrantVectorStore | None = None


def _vector_store() -> QdrantVectorStore:
    global _store
    if _store is None:
        _store = QdrantVectorStore(
            client=qdrant_client(),
            collection_name=CHILD_COLLECTION,
            embedding=OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_HOST),
            sparse_embedding=FastEmbedSparse(model_name="Qdrant/bm25"),
            retrieval_mode=RetrievalMode.HYBRID,
            sparse_vector_name="sparse",
        )
    return _store


def invalidate_store() -> None:
    global _store
    _store = None


def search_chunks(query: str, k: int = TOP_K, doc_id: str | None = None) -> list[dict[str, Any]]:
    kwargs: dict[str, Any] = {"k": k, "score_threshold": SCORE_THRESHOLD}
    if doc_id:
        kwargs["filter"] = qmodels.Filter(
            must=[qmodels.FieldCondition(key="metadata.doc_id", match=qmodels.MatchValue(value=doc_id))]
        )
    results = _vector_store().similarity_search_with_score(query, **kwargs)
    return [
        {
            "chunk_id": doc.metadata.get("chunk_id", ""),
            "parent_id": doc.metadata.get("parent_id", ""),
            "doc_id": doc.metadata.get("doc_id", ""),
            "page_num": doc.metadata.get("page_num"),
            "section": doc.metadata.get("section", ""),
            "snippet": doc.metadata.get("snippet", doc.page_content[:120]),
            "text": doc.page_content,
            "score": round(float(score), 4),
        }
        for doc, score in results
    ]


def fetch_parent(parent_id: str) -> dict[str, Any] | None:
    path = PARENTS_DIR / f"{parent_id}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def fetch_parents_for_chunks(chunks: list[dict]) -> list[dict]:
    seen: set[str] = set()
    parents: list[dict] = []
    for chunk in chunks:
        pid = chunk.get("parent_id", "")
        if pid and pid not in seen:
            parent = fetch_parent(pid)
            if parent:
                parents.append(parent)
            seen.add(pid)
    return parents
