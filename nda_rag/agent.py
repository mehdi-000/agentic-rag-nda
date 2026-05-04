import json
import re
from collections.abc import Generator
from typing import Any, Literal, TypedDict

from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field

from config import OLLAMA_HOST, LLM_MODEL, FAST_MODEL, MAX_RETRIES
from prompts import sanitize_input, REWRITE_SYSTEM, GRADE_SYSTEM, ANSWER_SYSTEM, ANSWER_TEXT_SYSTEM
from retriever import search_chunks, fetch_parents_for_chunks


class GradeResult(BaseModel):
    relevance_score: float = Field(ge=0.0, le=1.0)
    coverage: Literal["full", "partial", "none"]
    missing_aspects: list[str] = Field(default_factory=list)
    decision: Literal["sufficient", "retry", "give_up"]


class EvidenceItem(BaseModel):
    document: str
    page: int | None
    chunk_id: str | None
    snippet: str | None
    support: Literal["direct", "inferred", "absent"]


class AnswerResult(BaseModel):
    answer: str
    confidence: Literal["high", "medium", "low"]
    evidence: list[EvidenceItem] = Field(default_factory=list)


class CorrectionStep(TypedDict):
    step: int
    action: str
    query: str
    relevance_score: float
    coverage: str
    missing_aspects: list[str]
    decision: str


class AgentState(TypedDict):
    original_query: str
    current_query: str
    attempt: int
    chunks: list[dict]
    parents: list[dict]
    grade: dict
    answer: str
    confidence: str
    evidence: list[dict]
    self_correction: list[CorrectionStep]


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)

_llm: ChatOllama | None = None
_fast_llm: ChatOllama | None = None


def get_llm() -> ChatOllama:
    global _llm
    if _llm is None:
        _llm = ChatOllama(model=LLM_MODEL, base_url=OLLAMA_HOST, temperature=0)
    return _llm


def get_fast_llm() -> ChatOllama:
    global _fast_llm
    if _fast_llm is None:
        _fast_llm = ChatOllama(model=FAST_MODEL, base_url=OLLAMA_HOST, temperature=0)
    return _fast_llm


def _parse_json_response(text: str, model: type[BaseModel]) -> BaseModel:
    match = _JSON_RE.search(text)
    if not match:
        raise ValueError("No JSON object found in response")
    return model.model_validate_json(match.group())


def _build_docs_text(state: AgentState) -> str:
    seen: set[str] = set()
    parts: list[str] = []
    for parent in state["parents"]:
        pid = parent.get("parent_id", "")
        if pid not in seen:
            parts.append(
                f"[Document: {parent['doc_id']} · Page: {parent['page_num']} · Chunk: {parent['parent_id']} · Section: {parent['section']}]\n{parent['text']}"
            )
            seen.add(pid)
    if not parts and state["chunks"]:
        parts = [f"[{c['doc_id']} · p{c['page_num']}]\n{c['snippet']}" for c in state["chunks"]]
    return "\n\n".join(parts) or "No relevant documents found."


def _rewrite(state: AgentState) -> None:
    missing = state.get("grade", {}).get("missing_aspects", [])
    retry_context = ", ".join(missing) if missing and state["attempt"] > 0 else ""
    prompt = REWRITE_SYSTEM.format(
        query=state["current_query"] if state["attempt"] == 0 else state["original_query"],
        retry_context=retry_context,
    )
    response = get_fast_llm().invoke([HumanMessage(content=prompt)])
    state["current_query"] = response.content.strip().split("\n")[0]


def _retrieve(state: AgentState) -> None:
    state["chunks"] = search_chunks(state["current_query"])
    state["parents"] = fetch_parents_for_chunks(state["chunks"])


def _grade(state: AgentState) -> None:
    attempt = state["attempt"]
    passages_text = "\n\n---\n\n".join(
        f"[{c['doc_id']} · p{c['page_num']} · score {c['score']}]\n{c['text']}"
        for c in state["chunks"]
    ) or "No passages retrieved."

    prompt = GRADE_SYSTEM.format(
        query=state["current_query"],
        passages=passages_text,
        attempt=attempt + 1,
        max_attempts=MAX_RETRIES + 1,
    )
    response = get_fast_llm().invoke([HumanMessage(content=prompt)])

    try:
        grade = _parse_json_response(response.content, GradeResult).model_dump()
    except Exception:
        grade = {
            "relevance_score": 0.0,
            "coverage": "none",
            "missing_aspects": [],
            "decision": "give_up" if attempt >= MAX_RETRIES else "retry",
        }

    state["grade"] = grade
    state["self_correction"].append({
        "step": attempt + 1,
        "action": "initial_retrieval" if attempt == 0 else "query_rewrite",
        "query": state["current_query"],
        "relevance_score": grade["relevance_score"],
        "coverage": grade["coverage"],
        "missing_aspects": grade["missing_aspects"],
        "decision": grade["decision"],
    })


def _initial_state(query: str) -> AgentState:
    return {
        "original_query": query,
        "current_query": query,
        "attempt": 0,
        "chunks": [],
        "parents": [],
        "grade": {},
        "answer": "",
        "confidence": "low",
        "evidence": [],
        "self_correction": [],
    }


def run_pipeline_only(query: str) -> AgentState:
    """Run rewrite -> retrieve -> grade (with retries). Returns state ready for answer generation."""
    state = _initial_state(sanitize_input(query))
    for _ in range(MAX_RETRIES + 1):
        _rewrite(state)
        _retrieve(state)
        _grade(state)
        decision = state["grade"].get("decision", "give_up")
        if decision in ("sufficient", "give_up"):
            break
        state["attempt"] += 1
    return state


def stream_answer(state: AgentState) -> Generator[str, None, None]:
    """Stream a user-facing answer from the 7B model."""
    prompt = ANSWER_TEXT_SYSTEM.format(query=state["original_query"], documents=_build_docs_text(state))
    for chunk in get_llm().stream([HumanMessage(content=prompt)]):
        if chunk.content:
            yield chunk.content


def classify_evidence(state: AgentState, answer_text: str) -> tuple[str, list[dict]]:
    """Fast 3B call to extract confidence + evidence from the completed answer."""
    chunks_summary = "\n".join(
        f"- chunk_id={c['chunk_id']} doc={c['doc_id']} page={c['page_num']} snippet={c['snippet']}"
        for c in state["chunks"]
    )
    prompt = (
        f"Given this answer to an NDA question and the retrieved chunks, produce a JSON object.\n\n"
        f"Answer: {answer_text[:1000]}\n\n"
        f"Retrieved chunks:\n{chunks_summary}\n\n"
        f"Output ONLY valid JSON with this schema:\n"
        f'{{"confidence": "high|medium|low", "evidence": [{{"document": "...", "page": N, "chunk_id": "...", "snippet": "...", "support": "direct|inferred|absent"}}]}}\n\n'
        f"Rules: confidence=high if answer is fully supported, medium if partially, low if mostly inferred/absent."
    )
    try:
        response = get_fast_llm().invoke([HumanMessage(content=prompt)])
        result = _parse_json_response(response.content, AnswerResult)
        return result.confidence, [e.model_dump() for e in result.evidence]
    except Exception:
        fallback = [{"document": c["doc_id"], "page": c["page_num"], "chunk_id": c["chunk_id"], "snippet": c["snippet"], "support": "inferred"} for c in state["chunks"][:3]]
        return "low", fallback


def run(query: str) -> dict[str, Any]:
    """Blocking version — used by evaluate.py."""
    state = run_pipeline_only(query)

    prompt = ANSWER_SYSTEM.format(query=state["original_query"], documents=_build_docs_text(state))
    response = get_llm().invoke([HumanMessage(content=prompt)])
    try:
        result = _parse_json_response(response.content, AnswerResult)
        answer, confidence = result.answer, result.confidence
        evidence = [e.model_dump() for e in result.evidence]
    except Exception:
        answer = response.content.strip()
        confidence = "low"
        evidence = [{"document": c["doc_id"], "page": c["page_num"], "chunk_id": c["chunk_id"], "snippet": c["snippet"], "support": "inferred"} for c in state["chunks"][:3]]

    return {
        "answer": answer,
        "confidence": confidence,
        "evidence": evidence,
        "self_correction": state["self_correction"],
        "_debug": {
            "original_query": state["original_query"],
            "final_query": state["current_query"],
            "chunks": state["chunks"],
        },
    }


if __name__ == "__main__":
    result = run("Which parties are involved in the NDA and what are their roles?")
    print(json.dumps(result, indent=2, ensure_ascii=False))
