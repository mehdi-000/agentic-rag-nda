import json
import re
import uuid
from pathlib import Path
from typing import Any

import fitz
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_qdrant.fastembed_sparse import FastEmbedSparse
from langchain_qdrant.qdrant import RetrievalMode
from langchain_core.documents import Document
from qdrant_client.http import models as qmodels

from config import (
    DOCS_DIR, INDEX_DIR, PARENTS_DIR, QDRANT_DIR, PARSE_STATUS_FILE,
    OLLAMA_HOST, LLM_MODEL, EMBED_MODEL, EMBED_DIM,
    CHILD_COLLECTION,
    PARENT_SIZE, PARENT_OVERLAP, CHILD_SIZE, CHILD_OVERLAP,
    MIN_BLOCK_CHARS, HEADER_FOOTER_MAX_CHARS, HEADING_FONT_RATIO,
    CONTEXTUAL_ENRICHMENT,
)
from qdrant_store import qdrant_client, reset_qdrant_client


def _ensure_collection(client) -> None:
    if not client.collection_exists(CHILD_COLLECTION):
        client.create_collection(
            collection_name=CHILD_COLLECTION,
            vectors_config=qmodels.VectorParams(size=EMBED_DIM, distance=qmodels.Distance.COSINE),
            sparse_vectors_config={"sparse": qmodels.SparseVectorParams()},
        )


def _extract_pages(doc: fitz.Document) -> list[dict]:
    pages = []
    for page in doc:
        blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)["blocks"]
        pages.append({"page_num": page.number + 1, "blocks": blocks})
    return pages


def _modal_font_size(pages: list[dict]) -> float:
    sizes: list[float] = []
    for p in pages:
        for b in p["blocks"]:
            if b["type"] != 0:
                continue
            for line in b.get("lines", []):
                for span in line.get("spans", []):
                    if span.get("text", "").strip():
                        sizes.append(round(span["size"], 1))
    if not sizes:
        return 11.0
    return max(set(sizes), key=sizes.count)


def _collect_repeated_texts(pages: list[dict]) -> set[str]:
    from collections import Counter
    texts: list[str] = []
    for p in pages:
        for b in p["blocks"]:
            if b["type"] != 0:
                continue
            text = " ".join(
                span["text"]
                for line in b.get("lines", [])
                for span in line.get("spans", [])
            ).strip()
            if text and len(text) <= HEADER_FOOTER_MAX_CHARS:
                texts.append(text)
    counts = Counter(texts)
    return {t for t, c in counts.items() if c >= max(2, len(pages) // 3)}


def _extract_structured_blocks(doc_id: str, pages: list[dict]) -> list[dict]:
    modal_size = _modal_font_size(pages)
    repeated = _collect_repeated_texts(pages)
    heading_threshold = modal_size * HEADING_FONT_RATIO

    blocks_out: list[dict] = []
    current_section = "Preamble"

    for p in pages:
        for b in p["blocks"]:
            if b["type"] != 0:
                continue

            lines_text: list[str] = []
            max_size = 0.0
            is_bold = False

            for line in b.get("lines", []):
                line_text = ""
                for span in line.get("spans", []):
                    t = span.get("text", "")
                    lines_text.append(t)
                    if span.get("size", 0) > max_size:
                        max_size = span["size"]
                    if span.get("flags", 0) & 2**4:
                        is_bold = True
                    line_text += t

            text = " ".join(lines_text).strip()
            text = re.sub(r"\s+", " ", text)

            if len(text) < MIN_BLOCK_CHARS:
                continue
            if text in repeated:
                continue

            if max_size >= heading_threshold or (is_bold and len(text) < 120):
                current_section = text[:80]
                continue

            blocks_out.append({
                "doc_id": doc_id,
                "page_num": p["page_num"],
                "section": current_section,
                "text": text,
            })

    return blocks_out


def _sliding_chunks(text: str, size: int, overlap: int) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        spans.append((start, end))
        if end == len(text):
            break
        start += size - overlap
    return spans


def _build_parents(blocks: list[dict]) -> list[dict]:
    full_text = "\n\n".join(b["text"] for b in blocks)
    if not full_text.strip():
        return []

    char_to_block: list[int] = []
    offset = 0
    for i, b in enumerate(blocks):
        char_to_block.extend([i] * len(b["text"]))
        if i < len(blocks) - 1:
            char_to_block.extend([i] * 2)
            offset += len(b["text"]) + 2
    char_to_block.extend([len(blocks) - 1] * max(0, len(full_text) - len(char_to_block)))

    parents: list[dict] = []
    for start, end in _sliding_chunks(full_text, PARENT_SIZE, PARENT_OVERLAP):
        chunk_text = full_text[start:end].strip()
        if not chunk_text:
            continue
        mid = (start + end) // 2
        block_idx = min(char_to_block[mid], len(blocks) - 1) if char_to_block else 0
        ref_block = blocks[block_idx]
        parent_id = f"{ref_block['doc_id']}_p{ref_block['page_num']}_{uuid.uuid4().hex[:8]}"
        parents.append({
            "parent_id": parent_id,
            "doc_id": ref_block["doc_id"],
            "page_num": ref_block["page_num"],
            "section": ref_block["section"],
            "text": chunk_text,
            "char_start": start,
            "char_end": end,
        })
    return parents


def _build_children(parent: dict) -> list[dict]:
    children: list[dict] = []
    for i, (start, end) in enumerate(_sliding_chunks(parent["text"], CHILD_SIZE, CHILD_OVERLAP)):
        chunk_text = parent["text"][start:end].strip()
        if not chunk_text:
            continue
        children.append({
            "chunk_id": f"{parent['parent_id']}_c{i}",
            "parent_id": parent["parent_id"],
            "doc_id": parent["doc_id"],
            "page_num": parent["page_num"],
            "section": parent["section"],
            "text": chunk_text,
            "char_start": parent["char_start"] + start,
            "char_end": parent["char_start"] + end,
            "snippet": chunk_text[:120],
        })
    return children


def _deterministic_context(child: dict) -> str:
    return f"[{child['doc_id']} · p{child['page_num']} · {child['section']}] {child['text']}"


def _llm_context(child: dict, llm: ChatOllama) -> str:
    prompt = (
        f"Write one sentence (max 20 words) describing what this NDA passage covers. "
        f"Document: {child['doc_id']}, page {child['page_num']}, section '{child['section']}'.\n\n"
        f"Passage: {child['text'][:300]}"
    )
    try:
        response = llm.invoke(prompt)
        context_sentence = response.content.strip().split("\n")[0]
        return f"[Context: {context_sentence}] {child['text']}"
    except Exception:
        return _deterministic_context(child)


def _enrich(children: list[dict], llm: ChatOllama | None) -> list[str]:
    if llm is not None and CONTEXTUAL_ENRICHMENT:
        return [_llm_context(c, llm) for c in children]
    return [_deterministic_context(c) for c in children]


def _delete_old_chunks(client: Any, doc_ids: set[str]) -> None:
    if not doc_ids or not client.collection_exists(CHILD_COLLECTION):
        return
    for doc_id in doc_ids:
        client.delete(
            collection_name=CHILD_COLLECTION,
            points_selector=qmodels.FilterSelector(
                filter=qmodels.Filter(must=[qmodels.FieldCondition(key="doc_id", match=qmodels.MatchValue(value=doc_id))])
            ),
        )
        for f in PARENTS_DIR.glob("*.json"):
            try:
                data = json.loads(f.read_text())
                if data.get("doc_id") == doc_id:
                    f.unlink()
            except Exception:
                pass


def _load_parse_status() -> dict:
    if PARSE_STATUS_FILE.exists():
        return json.loads(PARSE_STATUS_FILE.read_text())
    return {}


def _save_parse_status(status: dict) -> None:
    PARSE_STATUS_FILE.write_text(json.dumps(status, indent=2))


def _ensure_dirs() -> None:
    for d in [DOCS_DIR, INDEX_DIR, PARENTS_DIR, QDRANT_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def index_documents(pdf_paths: list[Path] | None = None) -> dict:
    _ensure_dirs()
    if pdf_paths is None:
        pdf_paths = list(DOCS_DIR.glob("*.pdf"))

    if not pdf_paths:
        return {}

    llm = ChatOllama(model=LLM_MODEL, base_url=OLLAMA_HOST, temperature=0) if CONTEXTUAL_ENRICHMENT else None
    embeddings = OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_HOST)
    sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")

    client = qdrant_client()
    _ensure_collection(client)

    vector_store = QdrantVectorStore(
        client=client,
        collection_name=CHILD_COLLECTION,
        embedding=embeddings,
        sparse_embedding=sparse_embeddings,
        retrieval_mode=RetrievalMode.HYBRID,
        sparse_vector_name="sparse",
    )

    parse_status = _load_parse_status()
    all_child_docs: list[Document] = []

    doc_ids_to_ingest = {p.name for p in pdf_paths}
    _delete_old_chunks(client, doc_ids_to_ingest)

    for pdf_path in pdf_paths:
        doc_id = pdf_path.name
        try:
            fitz_doc = fitz.open(str(pdf_path))
            pages = _extract_pages(fitz_doc)
            fitz_doc.close()

            blocks = _extract_structured_blocks(doc_id, pages)
            if not blocks:
                parse_status[doc_id] = {"status": "partial", "error": "No text blocks extracted", "chunks": 0}
                continue

            parents = _build_parents(blocks)
            if not parents:
                parse_status[doc_id] = {"status": "partial", "error": "No parent chunks built", "chunks": 0}
                continue

            children: list[dict] = []
            for parent in parents:
                children.extend(_build_children(parent))
                parent_path = PARENTS_DIR / f"{parent['parent_id']}.json"
                parent_path.write_text(json.dumps(parent, ensure_ascii=False, indent=2))

            enriched_texts = _enrich(children, llm)

            for child, enriched in zip(children, enriched_texts):
                all_child_docs.append(Document(
                    page_content=enriched,
                    metadata={
                        "chunk_id": child["chunk_id"],
                        "parent_id": child["parent_id"],
                        "doc_id": child["doc_id"],
                        "page_num": child["page_num"],
                        "section": child["section"],
                        "char_start": child["char_start"],
                        "char_end": child["char_end"],
                        "snippet": child["snippet"],
                    },
                ))

            total_pages = max((b["page_num"] for b in blocks), default=0)
            parse_status[doc_id] = {
                "status": "ok",
                "pages": total_pages,
                "parents": len(parents),
                "chunks": len(children),
                "error": None,
            }

        except Exception as e:
            parse_status[doc_id] = {"status": "failed", "error": str(e), "chunks": 0}

    _save_parse_status(parse_status)

    if all_child_docs:
        try:
            vector_store.add_documents(all_child_docs)
        except Exception as e:
            for doc_id in [p.name for p in pdf_paths]:
                if doc_id in parse_status and parse_status[doc_id]["status"] == "ok":
                    parse_status[doc_id]["status"] = "partial"
                    parse_status[doc_id]["error"] = f"Chunks parsed but vector indexing failed: {e}"
            _save_parse_status(parse_status)

    from retriever import invalidate_store
    invalidate_store()

    return parse_status


def reset_index() -> None:
    _ensure_dirs()
    client = qdrant_client()
    if client.collection_exists(CHILD_COLLECTION):
        client.delete_collection(CHILD_COLLECTION)
    for f in PARENTS_DIR.glob("*.json"):
        f.unlink()
    if PARSE_STATUS_FILE.exists():
        PARSE_STATUS_FILE.unlink()
    reset_qdrant_client()


if __name__ == "__main__":
    results = index_documents()
    for doc, info in results.items():
        print(f"{doc}: {info['status']} — {info.get('chunks', 0)} chunks")
