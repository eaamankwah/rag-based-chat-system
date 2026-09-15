import os
import logging
import warnings

# ragas.llms.base unconditionally imports langchain_google_vertexai (needed
# only for our earlier compatibility patch - this project never actually
# uses Vertex AI). That import chain pulls in google.cloud.aiplatform /
# google.cloud.vectorsearch_v1beta, which check the interpreter's Python
# version at import time and emit a FutureWarning on Python versions
# nearing end-of-life (e.g. 3.10 as of late 2026). It's not actionable
# here - we don't call any Vertex/Google Cloud API - so it's suppressed
# before triggering the import chain below.
warnings.filterwarnings("ignore", category=FutureWarning, module=r"google\.api_core.*")

from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# RAGAS imports
try:
    from ragas import SingleTurnSample
    from ragas.metrics import BleuScore, LLMContextPrecisionWithReference, ResponseRelevancy, Faithfulness, RougeScore
    from ragas import evaluate
    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False


def _resolve_openai_key() -> Optional[str]:
    """Find an OpenAI API key from the environment.

    chat.py stores the user-entered key under CHROMA_OPENAI_API_KEY (for
    ChromaDB's embedding function); fall back to that if the standard
    OPENAI_API_KEY isn't set so the evaluator LLM/embeddings can still
    authenticate.
    """
    return os.getenv("OPENAI_API_KEY") or os.getenv("CHROMA_OPENAI_API_KEY")


def _resolve_base_url() -> Optional[str]:
    """Resolve a custom OpenAI-compatible base URL (e.g. Vocareum proxy).

    Checks both the current OPENAI_BASE_URL and the older, still commonly
    documented OPENAI_API_BASE, since the openai SDK (and langchain_openai,
    which wraps it) only auto-reads the former.
    """
    return os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_API_BASE")


def evaluate_response_quality(question: str, answer: str, contexts: List[str],
                               reference: Optional[str] = None) -> Dict[str, float]:
    """Evaluate response quality using RAGAS metrics.

    Always computes (no ground-truth answer required):
        - response_relevancy: how relevant/on-topic the answer is to the question
        - faithfulness: how well the answer is grounded in the retrieved context

    Additionally, when a `reference` (ground-truth) answer is supplied,
    also computes reference-based metrics:
        - bleu_score
        - rouge_score
        - context_precision (via LLMContextPrecisionWithReference, which
          asks the evaluator LLM whether each retrieved chunk was useful
          for producing the reference answer - semantic judgment, not
          literal string overlap)

    Args:
        question: The user's question
        answer: The model's generated answer
        contexts: The retrieved context chunks used to generate the answer
        reference: Optional ground-truth answer for reference-based metrics

    Returns:
        Dictionary mapping metric name -> score (0.0-1.0), or an
        {"error": ...} dictionary if evaluation could not be performed.
    """
    if not RAGAS_AVAILABLE:
        return {"error": "RAGAS not available"}

    # Handle empty or malformed inputs with a clear error message (no crashes)
    if not question or not isinstance(question, str):
        return {"error": "A non-empty 'question' string is required"}
    if not answer or not isinstance(answer, str):
        return {"error": "A non-empty 'answer' string is required"}
    if not contexts or not isinstance(contexts, list) or not any(contexts):
        return {"error": "A non-empty list of 'contexts' strings is required"}

    api_key = _resolve_openai_key()
    if not api_key:
        return {"error": "No OpenAI API key found (set OPENAI_API_KEY)"}

    try:
        # Create evaluator LLM with model gpt-3.5-turbo
        evaluator_llm = LangchainLLMWrapper(
            ChatOpenAI(model="gpt-3.5-turbo", api_key=api_key, base_url=_resolve_base_url(), temperature=0)
        )

        # Create evaluator_embeddings with model text-embedding-3-small
        evaluator_embeddings = LangchainEmbeddingsWrapper(
            OpenAIEmbeddings(model="text-embedding-3-small", api_key=api_key, base_url=_resolve_base_url())
        )

        # Define an instance for each metric to evaluate
        response_relevancy_metric = ResponseRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings)
        faithfulness_metric = Faithfulness(llm=evaluator_llm)

        sample = SingleTurnSample(
            user_input=question,
            response=answer,
            retrieved_contexts=[c for c in contexts if c],
            reference=reference,
        )

        results: Dict[str, float] = {}

        # Evaluate the response using the metrics
        try:
            results["response_relevancy"] = float(response_relevancy_metric.single_turn_score(sample))
        except Exception as e:
            logger.error(f"response_relevancy failed: {e}")
            results["response_relevancy"] = 0.0

        try:
            results["faithfulness"] = float(faithfulness_metric.single_turn_score(sample))
        except Exception as e:
            logger.error(f"faithfulness failed: {e}")
            results["faithfulness"] = 0.0

        # Additional reference-based metrics - only meaningful when a
        # ground-truth reference answer is available (e.g. batch evaluation
        # against evaluation_dataset.txt / test_questions.json). These all
        # score directly off `sample` (user_input, retrieved_contexts,
        # response, reference) - no extra fields needed.
        if reference:
            try:
                results["bleu_score"] = float(BleuScore().single_turn_score(sample))
            except Exception as e:
                logger.error(f"bleu_score failed: {e}")

            try:
                results["rouge_score"] = float(RougeScore().single_turn_score(sample))
            except Exception as e:
                logger.error(f"rouge_score failed: {e}")

            try:
                # LLM-judged context precision: asks the evaluator LLM
                # whether each retrieved chunk was actually useful for
                # arriving at the reference answer, and rewards chunks
                # that rank higher. This is deliberately NOT
                # NonLLMContextPrecisionWithReference, which compares
                # retrieved chunks against a reference_contexts field via
                # raw string similarity - since we only have a short
                # reference *answer* (not hand-curated reference *context
                # passages*), that string-similarity comparison against
                # long, differently-worded retrieved chunks always scores
                # ~0 regardless of retrieval quality. Judging relevance
                # semantically via the LLM instead of literal text overlap
                # fixes this without requiring hand-curated reference
                # contexts for every test question.
                precision_metric = LLMContextPrecisionWithReference(llm=evaluator_llm)
                results["context_precision"] = float(precision_metric.single_turn_score(sample))
            except Exception as e:
                logger.error(f"context_precision failed: {e}")

        # Return the evaluation results
        return results

    except Exception as e:
        logger.error(f"RAGAS evaluation failed: {e}")
        return {"error": f"RAGAS evaluation failed: {str(e)}"}


def evaluate_batch(test_cases: List[Dict]) -> Dict:
    """Run evaluate_response_quality over a batch of test cases.

    Args:
        test_cases: list of dicts, each with keys "question", "answer",
            "contexts" (list[str]), and optionally "reference".

    Returns:
        A dict with a per-question breakdown and the aggregate (mean) for
        each metric across the batch.
    """
    per_question = []
    metric_totals: Dict[str, List[float]] = {}

    for case in test_cases:
        question = case.get("question", "")
        answer = case.get("answer", "")
        contexts = case.get("contexts", [])
        reference = case.get("reference")

        scores = evaluate_response_quality(question, answer, contexts, reference)
        per_question.append({"question": question, "scores": scores})

        if "error" not in scores:
            for metric_name, value in scores.items():
                metric_totals.setdefault(metric_name, []).append(value)

    aggregate = {
        metric_name: (sum(values) / len(values) if values else 0.0)
        for metric_name, values in metric_totals.items()
    }

    return {"per_question": per_question, "aggregate": aggregate}
