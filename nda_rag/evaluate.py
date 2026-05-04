"""
Evaluation script for the NDA RAG system against the Kleister NDA dataset.

Usage:
    python evaluate.py --split dev-0 --dataset /path/to/kleister-nda --limit 20

The script queries the RAG system for each of the 4 NDA fields, compares against
expected.tsv, and computes F1 scores (uppercased, matching the dataset metric).
"""

import argparse
import csv
import io
import json
import lzma
import re
import sys
from collections import defaultdict
from pathlib import Path

from agent import run
from ingest import index_documents, reset_index
from config import DOCS_DIR


FIELD_QUERIES = {
    "effective_date": (
        "What is the effective date of this NDA agreement? "
        "Return the date in YYYY-MM-DD format."
    ),
    "jurisdiction": (
        "Under which state or country jurisdiction is this contract signed? "
        "Return only the jurisdiction name."
    ),
    "party": (
        "Who are all the parties to this NDA agreement? "
        "List each party name separately."
    ),
    "term": (
        "What is the duration or term of this NDA agreement? "
        "Return in the format: number_units (e.g. 2_years, 18_months)."
    ),
}


def _load_tsv(path: Path) -> list[dict]:
    opener = lzma.open if path.suffix == ".xz" else open
    rows = []
    with opener(path, "rt", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
        for row in reader:
            rows.append(row)
    return rows


def _parse_expected(path: Path) -> list[dict[str, list[str]]]:
    records: list[dict[str, list[str]]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            fields: dict[str, list[str]] = defaultdict(list)
            for pair in line.split(" "):
                if "=" in pair:
                    key, _, value = pair.partition("=")
                    fields[key.strip()].append(value.strip())
            records.append(dict(fields))
    return records


def _normalise(value: str) -> str:
    return re.sub(r"[\s:]+", "_", value.strip()).upper()


def _f1(predicted: list[str], expected: list[str]) -> float:
    pred_set = {_normalise(v) for v in predicted}
    exp_set = {_normalise(v) for v in expected}
    if not pred_set and not exp_set:
        return 1.0
    if not pred_set or not exp_set:
        return 0.0
    tp = len(pred_set & exp_set)
    precision = tp / len(pred_set)
    recall = tp / len(exp_set)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _extract_values_from_answer(answer: str, field: str) -> list[str]:
    answer = answer.strip()
    if field == "effective_date":
        dates = re.findall(r"\d{4}-\d{2}-\d{2}", answer)
        return dates[:1]
    if field == "jurisdiction":
        lines = [l.strip(" -•*") for l in answer.splitlines() if l.strip()]
        return [lines[0]] if lines else []
    if field == "party":
        lines = [l.strip(" -•*1234567890.)") for l in answer.splitlines() if l.strip()]
        return [l for l in lines if len(l) > 3][:6]
    if field == "term":
        match = re.search(r"(\d+)\s*(year|month|week|day)s?", answer, re.IGNORECASE)
        if match:
            return [f"{match.group(1)}_{match.group(2).lower()}s"]
    return [answer.split(".")[0].strip()] if answer else []


def evaluate(dataset_dir: Path, split: str = "dev-0", limit: int | None = None) -> None:
    split_dir = dataset_dir / split
    in_file = split_dir / "in.tsv.xz"
    expected_file = split_dir / "expected.tsv"

    if not in_file.exists():
        in_file = split_dir / "in.tsv"
    if not in_file.exists():
        print(f"Input file not found in {split_dir}")
        sys.exit(1)

    in_rows = _load_tsv(in_file)
    expected_records = _parse_expected(expected_file)

    if limit:
        in_rows = in_rows[:limit]
        expected_records = expected_records[:limit]

    field_f1s: dict[str, list[float]] = defaultdict(list)

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

        print(f"[{i+1}/{len(in_rows)}] {doc_filename}")

        for field, query in FIELD_QUERIES.items():
            if field not in expected:
                field_f1s[field].append(1.0)
                continue

            full_query = f"Document: {doc_filename}\n{query}"
            try:
                result = run(full_query)
                predicted = _extract_values_from_answer(result["answer"], field)
            except Exception as e:
                print(f"  {field}: ERROR — {e}")
                predicted = []

            exp_values = expected[field]
            score = _f1(predicted, exp_values)
            field_f1s[field].append(score)
            print(f"  {field}: predicted={predicted} expected={exp_values} F1={score:.3f}")

    print("\n" + "=" * 60)
    print(f"{'Field':<20} {'F1':>8} {'N':>6}")
    print("-" * 36)
    all_scores: list[float] = []
    for field in FIELD_QUERIES:
        scores = field_f1s[field]
        avg = sum(scores) / len(scores) if scores else 0.0
        print(f"{field:<20} {avg:>8.3f} {len(scores):>6}")
        all_scores.extend(scores)
    overall = sum(all_scores) / len(all_scores) if all_scores else 0.0
    print("-" * 36)
    print(f"{'Overall':<20} {overall:>8.3f} {len(all_scores):>6}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="Path to kleister-nda repo root")
    parser.add_argument("--split", default="dev-0", choices=["dev-0", "train", "test-A"])
    parser.add_argument("--limit", type=int, default=None, help="Limit number of documents")
    parser.add_argument("--reset", action="store_true", help="Reset index before evaluating")
    args = parser.parse_args()

    if args.reset:
        reset_index()

    evaluate(Path(args.dataset), split=args.split, limit=args.limit)
