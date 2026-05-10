"""
3-layer diagnostic evaluation for NDA RAG.

Usage:
    python evaluate.py --dataset /path/to/kleister-nda --split dev-0 --limit 5

Layers:
  1. INGESTION  — is the expected answer anywhere in the indexed chunks?
  2. RETRIEVAL  — did the top-K results include it?
  3. GENERATION — did the LLM produce the right answer?

Metrics:
  Custom : index_hit, retrieval_hit, F1, pairwise_correctness (LLM judge)
  Ragas  : faithfulness, answer_relevancy, context_precision, context_recall
"""

import argparse
import csv
import json
import lzma
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

from langchain_core.messages import HumanMessage

from agent import run, get_fast_llm
from config import DOCS_DIR, PARENTS_DIR, OLLAMA_HOST, LLM_MODEL, FAST_MODEL, EMBED_MODEL, BASE_DIR, OPENROUTER_API_KEY, OPENROUTER_MODEL
from ingest import index_documents, reset_index

_judge = None

def _judge_llm():
    """OpenRouter when key is set, otherwise local 3B."""
    global _judge
    if _judge is None:
        if OPENROUTER_API_KEY:
            from langchain_openai import ChatOpenAI
            _judge = ChatOpenAI(
                model=OPENROUTER_MODEL,
                api_key=OPENROUTER_API_KEY,
                base_url="https://openrouter.ai/api/v1",
                temperature=0,
                timeout=120,
                max_retries=1,
            )
            print(f"Judge: OpenRouter ({OPENROUTER_MODEL})")
        else:
            _judge = get_fast_llm()
            print(f"Judge: local ({FAST_MODEL})")
    return _judge


FIELD_QUERIES = {
    "effective_date": (
        "What is the effective date of this NDA? "
        "Return ONLY the date in YYYY-MM-DD format."
    ),
    "jurisdiction": (
        "What jurisdiction governs this NDA? "
        "Return ONLY the jurisdiction name."
    ),
    "party": (
        "List ALL parties to this NDA. "
        "Return ONLY the party names, one per line."
    ),
    "term": (
        "What is the term/duration of this NDA? "
        "Return in the format: number_units (e.g. 2_years, 18_months)."
    ),
}

JUDGE_PROMPT = """You are a fair judge comparing two answers to a question about an NDA document.

Question: {question}

Answer A:
{answer_a}

Answer B:
{answer_b}

Instructions:
1. Analyze both answers step by step — consider factual accuracy, completeness, and relevance.
2. One answer is from a RAG system, the other is the reference. You do not know which is which.
3. State which answer is better, or if they are equivalent.

Think through your reasoning carefully, then end with exactly one line:
VERDICT: A_better | B_better | tie"""


# ── Helpers ────────────────────────────────────────────────────────────────


def _load_tsv(path: Path) -> list[list[str]]:
    opener = lzma.open if path.suffix == ".xz" else open
    with opener(path, "rt", encoding="utf-8") as f:
        return list(csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE))


def _parse_expected(path: Path) -> list[dict[str, list[str]]]:
    records: list[dict[str, list[str]]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            fields: dict[str, list[str]] = defaultdict(list)
            for pair in line.strip().split(" "):
                if "=" in pair:
                    k, _, v = pair.partition("=")
                    fields[k.strip()].append(v.strip())
            records.append(dict(fields))
    return records


def _tokenize(value: str) -> set[str]:
    return {t for t in re.split(r"[\s_.,;:!?()\[\]{}/&|]+", value.strip().upper()) if t}


def _fuzzy_match(a: str, b: str, threshold: float = 0.5) -> bool:
    """Token-level Jaccard similarity -- handles 'Commonwealth of Massachusetts' vs 'Massachusetts'."""
    ta, tb = _tokenize(a), _tokenize(b)
    if not ta or not tb:
        return False
    if ta & tb:
        smaller = min(len(ta), len(tb))
        return len(ta & tb) / smaller >= threshold
    return False


def _values_match(a: str, b: str) -> bool:
    """Fuzzy match with duration equivalence (e.g. '24_months' == '2_years')."""
    if _fuzzy_match(a, b):
        return True
    ma, mb = _term_to_months(a), _term_to_months(b)
    if ma is not None and mb is not None:
        return ma == mb
    return False


def _f1(predicted: list[str], expected: list[str]) -> float:
    if not predicted and not expected:
        return 1.0
    if not predicted or not expected:
        return 0.0
    tp = sum(1 for p in predicted if any(_values_match(p, e) for e in expected))
    matched_exp = sum(1 for e in expected if any(_values_match(e, p) for p in predicted))
    p = tp / len(predicted)
    r = matched_exp / len(expected)
    return 2 * p * r / (p + r) if p + r else 0.0


def _term_to_months(value: str) -> int | None:
    """Normalize a term like '2_years' or '24_months' to total months for comparison."""
    m = re.search(r"(\d+)\s*[_\s]*(year|month|week|day)s?", value, re.IGNORECASE)
    if not m:
        return None
    n, unit = int(m.group(1)), m.group(2).lower()
    if unit == "year":
        return n * 12
    if unit == "month":
        return n
    if unit == "week":
        return round(n / 4.33)
    if unit == "day":
        return round(n / 30)
    return None


def _extract_values(answer: str, field: str) -> list[str]:
    answer = answer.strip()
    if field == "effective_date":
        return re.findall(r"\d{4}-\d{2}-\d{2}", answer)[:1]
    if field == "jurisdiction":
        lines = [l.strip(" -•*") for l in answer.splitlines() if l.strip()]
        return [lines[0]] if lines else []
    if field == "party":
        lines = [l.strip(" -•*1234567890.)") for l in answer.splitlines() if l.strip()]
        return [l for l in lines if len(l) > 3][:6]
    if field == "term":
        m = re.search(r"(\d+)\s*(year|month|week|day)s?", answer, re.IGNORECASE)
        return [f"{m.group(1)}_{m.group(2).lower()}s"] if m else []
    return [answer.split(".")[0].strip()] if answer else []


# ── Layer 1: Index Scan ───────────────────────────────────────────────────


def _load_parents_by_doc() -> dict[str, list[str]]:
    by_doc: dict[str, list[str]] = defaultdict(list)
    for f in PARENTS_DIR.glob("*.json"):
        try:
            p = json.loads(f.read_text())
            by_doc[p.get("doc_id", "")].append(p.get("text", ""))
        except Exception:
            pass
    return dict(by_doc)


def _text_contains_value(text: str, value: str) -> bool:
    """Check if any token window in text fuzzy-matches the value."""
    val_tokens = _tokenize(value)
    if not val_tokens:
        return False
    text_upper = text.upper()
    return any(vt in text_upper for vt in val_tokens if len(vt) > 2)


def _index_has_value(parent_texts: list[str], values: list[str]) -> bool:
    return any(_text_contains_value(t, v) for v in values for t in parent_texts)


# ── Layer 2: Retrieval Hit ────────────────────────────────────────────────


def _retrieval_hit(contexts: list[str], values: list[str]) -> bool:
    return any(_text_contains_value(c, v) for v in values for c in contexts)


# ── Pairwise LLM Judge ───────────────────────────────────────────────────


def _pairwise_judge(question: str, rag_answer: str, reference: str) -> tuple[str, str]:
    swapped = random.random() < 0.5
    a, b = (reference, rag_answer) if swapped else (rag_answer, reference)
    prompt = JUDGE_PROMPT.format(question=question, answer_a=a, answer_b=b)
    try:
        reasoning = _judge_llm().invoke([HumanMessage(content=prompt)]).content.strip()
    except Exception as e:
        print(f"    judge error: {e}")
        return "error", str(e)

    match = re.search(r"VERDICT:\s*(A_better|B_better|tie)", reasoning, re.IGNORECASE)
    raw = match.group(1).lower() if match else "tie"

    de_swap = {"a_better": "reference_better", "b_better": "rag_better"} if swapped \
         else {"a_better": "rag_better", "b_better": "reference_better"}
    return de_swap.get(raw, "tie"), reasoning


# ── Diagnosis ─────────────────────────────────────────────────────────────


def _diagnose(index_hit: bool, ret_hit: bool, faithful: float | None, correct: bool) -> str:
    if not index_hit:
        return "INGESTION"
    if not ret_hit:
        return "RETRIEVAL"
    if faithful is not None and faithful < 0.5:
        return "HALLUCINATION"
    if not correct:
        return "GENERATION"
    return "OK"


# ── Ragas Batch ───────────────────────────────────────────────────────────


FAITHFULNESS_PROMPT = """Given a question, an answer, and the retrieved context passages, rate how faithful the answer is to the context.
A faithful answer only makes claims supported by the context. An unfaithful answer invents facts not in the context.

Question: {question}
Answer: {answer}
Context: {context}

Rate faithfulness from 0.0 (completely unfaithful/hallucinated) to 1.0 (fully grounded in context).
Respond with ONLY a JSON object: {{"score": <float>, "reason": "<brief explanation>"}}"""

RELEVANCY_PROMPT = """Given a question and an answer, rate how relevant the answer is to the question.

Question: {question}
Answer: {answer}

Rate relevancy from 0.0 (completely off-topic) to 1.0 (directly addresses the question).
Respond with ONLY a JSON object: {{"score": <float>, "reason": "<brief explanation>"}}"""


def _llm_metrics(samples: list[dict]) -> list[dict]:
    """Score faithfulness and relevancy using the local LLM."""
    results = []
    for i, s in enumerate(samples):
        if not s["answer"]:
            results.append({})
            continue
        ctx = "\n---\n".join(s["contexts"][:3]) or "No context."
        row: dict = {}
        for name, prompt_tpl in [("faithfulness", FAITHFULNESS_PROMPT), ("answer_relevancy", RELEVANCY_PROMPT)]:
            try:
                kw = {"question": s["query"], "answer": s["answer"][:500]}
                if "context" in prompt_tpl:
                    kw["context"] = ctx[:2000]
                resp = _judge_llm().invoke([HumanMessage(content=prompt_tpl.format(**kw))]).content
                m = re.search(r'"score"\s*:\s*([\d.]+)', resp)
                row[name] = float(m.group(1)) if m else None
            except Exception:
                row[name] = None
        results.append(row)
        f, r = row.get("faithfulness", "?"), row.get("answer_relevancy", "?")
        print(f"  metrics [{i+1}/{len(samples)}]: faith={f} relev={r}")
    return results


# ── Main ──────────────────────────────────────────────────────────────────


def evaluate(dataset_dir: Path, split: str = "dev-0", limit: int | None = None) -> None:
    split_dir = dataset_dir / split
    in_file = split_dir / "in.tsv.xz"
    if not in_file.exists():
        in_file = split_dir / "in.tsv"
    if not in_file.exists():
        print(f"Input file not found in {split_dir}")
        sys.exit(1)

    in_rows = _load_tsv(in_file)
    expected_records = _parse_expected(split_dir / "expected.tsv")
    if limit:
        in_rows, expected_records = in_rows[:limit], expected_records[:limit]

    parents_by_doc = _load_parents_by_doc()
    collected: list[dict] = []

    # ── Phase 1: Pipeline + index/retrieval checks + pairwise judge ───────

    for i, (row, expected) in enumerate(zip(in_rows, expected_records)):
        if not row:
            continue
        doc_filename = row[0].strip()
        doc_path = dataset_dir / "documents" / doc_filename

        if not doc_path.exists():
            print(f"[{i+1}] SKIP — {doc_filename} not found")
            continue

        dest = DOCS_DIR / doc_filename
        if not dest.exists():
            import shutil
            shutil.copy(doc_path, dest)
            index_documents([dest])
            parents_by_doc = _load_parents_by_doc()

        print(f"\n[{i+1}/{len(in_rows)}] {doc_filename}")

        for field, query in FIELD_QUERIES.items():
            if field not in expected:
                continue

            exp_values = expected[field]
            full_query = f"Document: {doc_filename}\n{query}"

            doc_parents = parents_by_doc.get(doc_filename, [])
            index_hit = _index_has_value(doc_parents, exp_values)

            try:
                result = run(full_query, doc_id=doc_filename)
                answer = result["answer"]
                confidence = result.get("confidence", "low")
                contexts = [p["text"] for p in result["_debug"].get("parents", [])]
                if not contexts:
                    contexts = [c["text"] for c in result["_debug"].get("chunks", [])]
                if answer == "NOT_FOUND":
                    answer = ""
                predicted = _extract_values(answer, field)
            except Exception as e:
                print(f"  {field}: ERROR — {e}")
                answer, confidence, contexts, predicted = "", "low", [], []

            ret_hit = _retrieval_hit(contexts, exp_values)
            f1 = _f1(predicted, exp_values)

            ref_str = " | ".join(exp_values)
            if answer and ref_str:
                verdict, reasoning = _pairwise_judge(full_query, answer, ref_str)
            else:
                verdict, reasoning = "no_answer", ""
            correct = verdict in ("rag_better", "tie")

            diagnosis = _diagnose(index_hit, ret_hit, None, correct)

            record = {
                "doc": doc_filename, "field": field, "query": full_query,
                "expected": exp_values, "predicted": predicted,
                "answer": answer, "confidence": confidence, "contexts": contexts,
                "index_hit": index_hit, "retrieval_hit": ret_hit, "f1": f1,
                "pairwise_verdict": verdict, "pairwise_reasoning": reasoning,
                "diagnosis": diagnosis,
            }
            collected.append(record)

            idx_s = "IDX:✓" if index_hit else "IDX:✗"
            ret_s = "RET:✓" if ret_hit else "RET:✗"
            print(f"  {field}: {idx_s} {ret_s} F1={f1:.3f} judge={verdict} → {diagnosis}")
            print(f"    pred={predicted}  exp={exp_values}")

    if not collected:
        print("No samples collected.")
        return

    # ── Phase 2: Ragas metrics (batch) ────────────────────────────────────

    print(f"\nScoring faithfulness + relevancy on {len(collected)} samples…")
    ragas_scores = _llm_metrics(collected)
    for rec, scores in zip(collected, ragas_scores):
        rec["faithfulness"] = scores.get("faithfulness")
        rec["answer_relevancy"] = scores.get("answer_relevancy")
        rec["context_precision"] = scores.get("context_precision")
        rec["context_recall"] = scores.get("context_recall")
        if rec["faithfulness"] is not None and rec["faithfulness"] < 0.5 and rec["diagnosis"] == "GENERATION":
            rec["diagnosis"] = "HALLUCINATION"

    # ── Phase 3: Write JSONL + summary ────────────────────────────────────

    out_path = BASE_DIR / "eval_results.jsonl"
    with open(out_path, "w") as f:
        for rec in collected:
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
    print(f"\nDetailed results → {out_path}")

    _print_summary(collected)


def _safe_avg(records: list[dict], key: str) -> str:
    vals = [r[key] for r in records if r.get(key) is not None]
    return f"{sum(vals) / len(vals):.3f}" if vals else "  n/a"


def _print_summary(collected: list[dict]) -> None:
    print("\n" + "=" * 76)
    print(f"{'Field':<18} {'F1':>6} {'HitR':>6} {'Faith':>6} {'Relev':>6} {'Judge':>12} {'N':>4}")
    print("-" * 76)

    all_diag: dict[str, int] = defaultdict(int)
    all_f1: list[float] = []

    for field in FIELD_QUERIES:
        recs = [r for r in collected if r["field"] == field]
        if not recs:
            continue

        hit_r = sum(r["retrieval_hit"] for r in recs) / len(recs)
        n_correct = sum(1 for r in recs if r.get("pairwise_verdict") in ("rag_better", "tie"))

        print(
            f"{field:<18} {_safe_avg(recs, 'f1'):>6} {hit_r:>6.3f} "
            f"{_safe_avg(recs, 'faithfulness'):>6} {_safe_avg(recs, 'answer_relevancy'):>6} "
            f"{n_correct}/{len(recs):>10} {len(recs):>4}"
        )
        all_f1.extend(r["f1"] for r in recs)
        for r in recs:
            all_diag[r["diagnosis"]] += 1

    print("-" * 76)
    overall = sum(all_f1) / len(all_f1) if all_f1 else 0.0
    print(f"{'Overall':<18} {overall:>6.3f}")
    print(f"\nDiagnosis: {dict(all_diag)}")
    print("=" * 76)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="3-layer diagnostic evaluation for NDA RAG")
    parser.add_argument("--dataset", required=True, help="Path to kleister-nda repo root")
    parser.add_argument("--split", default="dev-0", choices=["dev-0", "train", "test-A"])
    parser.add_argument("--limit", type=int, default=None, help="Limit number of documents")
    parser.add_argument("--reset", action="store_true", help="Reset index before evaluating")
    args = parser.parse_args()

    if args.reset:
        reset_index()

    evaluate(Path(args.dataset), split=args.split, limit=args.limit)
