import json
from pathlib import Path

import gradio as gr

from config import DOCS_DIR, PARENTS_DIR, PARSE_STATUS_FILE, INDEX_DIR, QDRANT_DIR

CUSTOM_CSS = """
.gradio-container {
    width: 70% !important;
    margin: 0 auto !important;
}
h1 {
    font-weight: 700 !important;
    letter-spacing: -0.02em !important;
    margin-bottom: 0.25rem !important;
}
.subtitle {
    color: var(--body-text-color-subdued) !important;
    font-size: 0.95rem !important;
    margin-top: 0 !important;
    margin-bottom: 1.5rem !important;
}
footer { display: none !important; }
"""


def _parse_status() -> dict:
    if PARSE_STATUS_FILE.exists():
        return json.loads(PARSE_STATUS_FILE.read_text())
    return {}


def _all_chunks() -> list[dict]:
    chunks: list[dict] = []
    for f in sorted(PARENTS_DIR.glob("*.json")):
        try:
            chunks.append(json.loads(f.read_text()))
        except Exception:
            pass
    return chunks


# ── Document Library ───────────────────────────────────────────────────────


def render_document_library() -> list[list]:
    status = _parse_status()
    rows = []
    for doc_id, info in sorted(status.items()):
        badge = {"ok": "ok", "partial": "partial", "failed": "FAILED"}.get(info["status"], info["status"])
        rows.append([doc_id, badge, info.get("pages", "—"), info.get("parents", "—"), info.get("chunks", 0), info.get("error") or ""])
    return rows


def ingest_uploaded(files) -> tuple[str, list[list]]:
    if not files:
        return "No files selected.", render_document_library()
    from ingest import index_documents
    paths = []
    for f in files:
        dest = DOCS_DIR / Path(f.name).name
        dest.write_bytes(Path(f.name).read_bytes())
        paths.append(dest)
    results = index_documents(paths)
    uploaded = {p.name for p in paths}
    summary = "\n".join(f"{k}: {v['status']} ({v.get('chunks', 0)} chunks)" for k, v in results.items() if k in uploaded)
    return summary, render_document_library()


def reindex_all() -> tuple[str, list[list]]:
    from ingest import index_documents
    results = index_documents()
    summary = "\n".join(f"{k}: {v['status']} ({v.get('chunks', 0)} chunks)" for k, v in results.items())
    return summary, render_document_library()


# ── Chunk Inspector ────────────────────────────────────────────────────────


def render_chunk_table(doc_filter: str) -> list[list]:
    rows = []
    for c in _all_chunks():
        if doc_filter and doc_filter.lower() not in c.get("doc_id", "").lower():
            continue
        rows.append([c.get("parent_id", ""), c.get("doc_id", ""), c.get("page_num", ""), c.get("section", "")[:50], len(c.get("text", "")), c.get("text", "")[:200]])
    return rows


# ── Q&A ────────────────────────────────────────────────────────────────────


def _render_evidence(evidence: list[dict]) -> str:
    if not evidence:
        return "_No evidence returned._"
    labels = {"direct": "✅ DIRECT", "inferred": "🔶 INFERRED", "absent": "❌ ABSENT"}
    parts = []
    for e in evidence:
        label = labels.get(e.get("support", ""), e.get("support", "").upper())
        snippet = e.get("snippet") or "—"
        parts.append(
            f"**{label}** &nbsp; `{e.get('document', '—')}` · page {e.get('page')} · `{e.get('chunk_id') or '—'}`\n\n> {snippet}"
        )
    return "\n\n---\n\n".join(parts)


def _render_correction(steps: list[dict]) -> str:
    if not steps:
        return "_No correction steps._"
    parts = []
    for s in steps:
        missing = ", ".join(s.get("missing_aspects", [])) or "none"
        icon = {"sufficient": "✅", "retry": "🔄", "give_up": "⚠️"}.get(s.get("decision", ""), "")
        parts.append(
            f"**Step {s['step']} — {s['action']}** {icon}\n"
            f"- Query: `{s['query']}`\n"
            f"- Score: `{s['relevance_score']}` &nbsp; Coverage: `{s['coverage']}`\n"
            f"- Missing: {missing}\n"
            f"- Decision: **{s['decision']}**"
        )
    return "\n\n".join(parts)


def _chunk_rows(chunks: list[dict]) -> list[list]:
    return [
        [c.get("doc_id", ""), c.get("page_num", ""), round(c.get("score", 0), 4), c.get("section", "")[:40], c.get("snippet", "")[:120]]
        for c in chunks
    ]


def run_query(query: str):
    if not query.strip():
        yield "", "", [], "", ""
        return

    from agent import run_pipeline_only, stream_answer, classify_evidence

    yield "Retrieving and grading evidence…", "", [], "", ""

    state = run_pipeline_only(query)
    rows = _chunk_rows(state["chunks"])
    correction_md = _render_correction(state["self_correction"])

    yield "Generating answer…", "", rows, correction_md, ""

    answer_text = ""
    for token in stream_answer(state):
        answer_text += token
        yield answer_text, "", rows, correction_md, ""

    confidence, evidence = classify_evidence(state, answer_text)

    final = {
        "answer": answer_text,
        "confidence": confidence,
        "evidence": evidence,
        "self_correction": state["self_correction"],
        "_debug": {"chunks": state["chunks"]},
    }
    yield (
        answer_text,
        _render_evidence(evidence),
        rows,
        correction_md,
        json.dumps(final, indent=2, ensure_ascii=False, default=str),
    )


# ── App layout ─────────────────────────────────────────────────────────────


THEME = gr.themes.Soft(
    primary_hue=gr.themes.colors.orange,
    secondary_hue=gr.themes.colors.orange,
    neutral_hue=gr.themes.colors.gray,
    font=gr.themes.GoogleFont("Inter"),
    font_mono=gr.themes.GoogleFont("JetBrains Mono"),
)


def build_app() -> gr.Blocks:
    with gr.Blocks(title="NDA RAG") as app:
        gr.Markdown("# NDA Agentic RAG")
        gr.Markdown("Ask questions about your NDA documents.", elem_classes=["subtitle"])

        with gr.Tabs():

            with gr.Tab("Q&A"):
                with gr.Row():
                    query_box = gr.Textbox(
                        label="Question",
                        placeholder="e.g. Which parties are involved? What is the term?",
                        scale=5,
                    )
                    btn_ask = gr.Button("Ask", variant="primary", scale=1, min_width=100)

                answer_box = gr.Textbox(label="Answer", lines=5, interactive=False)

                with gr.Accordion("Evidence", open=False):
                    evidence_box = gr.Markdown()

                with gr.Accordion("Pipeline Trace", open=False):
                    gr.Markdown("**Retrieved chunks**")
                    chunk_table = gr.Dataframe(
                        headers=["Document", "Page", "Score", "Section", "Snippet"],
                        datatype=["str", "number", "number", "str", "str"],
                        interactive=False,
                        wrap=True,
                    )
                    gr.Markdown("**Self-correction steps**")
                    correction_box = gr.Markdown()

                with gr.Accordion("Raw JSON", open=False):
                    raw_json_box = gr.Code(language="json", interactive=False)

                btn_ask.click(
                    run_query,
                    inputs=[query_box],
                    outputs=[answer_box, evidence_box, chunk_table, correction_box, raw_json_box],
                )

            with gr.Tab("Document Library"):
                gr.Markdown("Upload NDA PDFs or re-index the `docs/` folder.")
                with gr.Row():
                    upload = gr.File(file_count="multiple", file_types=[".pdf"], label="Upload PDFs")
                    with gr.Column():
                        btn_upload = gr.Button("Ingest uploaded files", variant="primary")
                        btn_reindex = gr.Button("Re-index docs/ folder")
                ingest_log = gr.Textbox(label="Ingest log", lines=3, interactive=False)
                doc_table = gr.Dataframe(
                    headers=["Document", "Status", "Pages", "Parents", "Chunks", "Error"],
                    datatype=["str", "str", "number", "number", "number", "str"],
                    value=render_document_library(),
                    interactive=False,
                )
                btn_upload.click(ingest_uploaded, inputs=[upload], outputs=[ingest_log, doc_table])
                btn_reindex.click(reindex_all, outputs=[ingest_log, doc_table])

            with gr.Tab("Chunk Inspector"):
                gr.Markdown("Browse all indexed parent chunks. Filter by document name.")
                with gr.Row():
                    doc_filter = gr.Textbox(label="Filter by document name", placeholder="partial match, e.g. abc123", scale=4)
                    btn_filter = gr.Button("Filter", scale=1)
                chunk_table_inspector = gr.Dataframe(
                    headers=["Parent ID", "Document", "Page", "Section", "Chars", "Text preview"],
                    datatype=["str", "str", "number", "str", "number", "str"],
                    value=render_chunk_table(""),
                    interactive=False,
                    wrap=True,
                )
                btn_filter.click(render_chunk_table, inputs=[doc_filter], outputs=[chunk_table_inspector])

    return app


if __name__ == "__main__":
    for d in [DOCS_DIR, INDEX_DIR, PARENTS_DIR, QDRANT_DIR]:
        d.mkdir(parents=True, exist_ok=True)
    app = build_app()
    app.launch(server_name="0.0.0.0", server_port=7860, theme=THEME, css=CUSTOM_CSS)
