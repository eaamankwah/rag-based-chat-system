#!/usr/bin/env python3
"""
End-to-end batch evaluation driver for the NASA RAG system.

Loads a set of test questions (test_questions.json by default), runs each
one through the full RAG pipeline (retrieve -> generate -> evaluate), and
prints/saves a per-question breakdown plus an aggregate (mean) for every
RAGAS metric.

Usage:
    python batch_evaluate.py --openai-key YOUR_KEY \
        --chroma-dir ./chroma_db_openai \
        --collection-name nasa_space_missions_text \
        --test-file test_questions.json \
        --output evaluation_results.json
"""

import argparse
import json
import os
import sys
from typing import Dict, List

import rag_client
import llm_client
import ragas_evaluator


def load_test_questions(path: str) -> List[Dict]:
    """Load test questions from a JSON file (test_questions.json format)."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list) or not data:
        raise ValueError(f"{path} must contain a non-empty JSON list of test questions")
    return data


def run_batch_evaluation(openai_key: str, chroma_dir: str, collection_name: str,
                          test_file: str, model: str = "gpt-3.5-turbo",
                          n_results: int = 3, base_url: str = None) -> Dict:
    """Run the full RAG + RAGAS pipeline over every question in test_file."""

    if not openai_key or not openai_key.strip():
        raise ValueError(
            "No OpenAI API key was provided (--openai-key was empty). "
            "This usually means an environment variable like $OPENAI_API_KEY "
            "was referenced but never actually exported in this terminal "
            "session. Run `echo \"$OPENAI_API_KEY\"` to check - if it prints "
            "nothing, run `export OPENAI_API_KEY=\"sk-...\"` first, or pass "
            "the key directly: --openai-key \"sk-...\"."
        )

    os.environ["OPENAI_API_KEY"] = openai_key
    os.environ["CHROMA_OPENAI_API_KEY"] = openai_key

    # Resolve a custom endpoint (e.g. classroom Vocareum proxy). The openai
    # SDK only auto-reads OPENAI_BASE_URL, not the older OPENAI_API_BASE
    # name some setup guides use, so normalize to the former here.
    resolved_base_url = base_url or os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_API_BASE")
    if resolved_base_url:
        os.environ["OPENAI_BASE_URL"] = resolved_base_url

    collection, success, error = rag_client.initialize_rag_system(chroma_dir, collection_name)
    if not success:
        raise RuntimeError(f"Could not open ChromaDB collection: {error}")

    test_cases = load_test_questions(test_file)

    per_question_results = []
    metric_totals: Dict[str, List[float]] = {}

    for case in test_cases:
        question = case["question"]
        reference = case.get("reference")

        docs_result = rag_client.retrieve_documents(
            collection, question, n_results,
            mission_filter=case.get("mission")
        )

        documents = docs_result["documents"][0] if docs_result and docs_result.get("documents") else []
        metadatas = docs_result["metadatas"][0] if docs_result and docs_result.get("metadatas") else []
        context = rag_client.format_context(documents, metadatas)

        answer = llm_client.generate_response(openai_key, question, context, [], model)

        scores = ragas_evaluator.evaluate_response_quality(question, answer, documents, reference)

        result_row = {
            "id": case.get("id"),
            "mission": case.get("mission"),
            "category": case.get("category"),
            "question": question,
            "answer": answer,
            "reference": reference,
            "num_contexts_retrieved": len(documents),
            "scores": scores,
        }
        per_question_results.append(result_row)

        if "error" not in scores:
            for metric_name, value in scores.items():
                metric_totals.setdefault(metric_name, []).append(value)

    aggregate = {
        metric_name: (sum(values) / len(values) if values else 0.0)
        for metric_name, values in metric_totals.items()
    }

    return {"per_question": per_question_results, "aggregate": aggregate}


def main():
    parser = argparse.ArgumentParser(description="Batch-evaluate the NASA RAG system with RAGAS")
    parser.add_argument("--openai-key", required=True, help="OpenAI API key")
    parser.add_argument("--base-url", default=None,
                        help="Custom OpenAI-compatible endpoint, e.g. https://openai.vocareum.com/v1 "
                             "for a classroom Vocareum key. Defaults to OPENAI_BASE_URL / "
                             "OPENAI_API_BASE environment variables if set.")
    parser.add_argument("--chroma-dir", default="./chroma_db_openai", help="ChromaDB persist directory")
    parser.add_argument("--collection-name", default="nasa_space_missions_text", help="Collection name")
    parser.add_argument("--test-file", default="test_questions.json", help="Path to test questions JSON")
    parser.add_argument("--model", default="gpt-3.5-turbo", help="OpenAI chat model for answer generation")
    parser.add_argument("--n-results", type=int, default=3, help="Number of documents to retrieve per question")
    parser.add_argument("--output", default="evaluation_results.json", help="Where to save the JSON report")
    args = parser.parse_args()

    try:
        results = run_batch_evaluation(
            openai_key=args.openai_key,
            chroma_dir=args.chroma_dir,
            collection_name=args.collection_name,
            test_file=args.test_file,
            model=args.model,
            n_results=args.n_results,
            base_url=args.base_url,
        )
    except Exception as e:
        print(f"Batch evaluation failed: {e}", file=sys.stderr)
        sys.exit(1)

    print("=" * 70)
    print("PER-QUESTION RESULTS")
    print("=" * 70)
    for row in results["per_question"]:
        print(f"\n[{row['id']}] ({row['category']} / {row['mission']}) {row['question']}")
        print(f"  Answer: {row['answer'][:200]}{'...' if len(row['answer']) > 200 else ''}")
        print(f"  Scores: {row['scores']}")

    print("\n" + "=" * 70)
    print("AGGREGATE METRICS (mean across all questions)")
    print("=" * 70)
    for metric_name, value in results["aggregate"].items():
        print(f"  {metric_name}: {value:.3f}")

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nFull report saved to {args.output}")


if __name__ == "__main__":
    main()