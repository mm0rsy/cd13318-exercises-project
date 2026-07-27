import json
import os
import numpy as np
from pathlib import Path
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from typing import Dict, List, Optional

# RAGAS imports
try:
    from ragas import SingleTurnSample
    from ragas.metrics import BleuScore, NonLLMContextPrecisionWithReference, ResponseRelevancy, Faithfulness, RougeScore
    from ragas import evaluate
    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False

def evaluate_response_quality(question: str, answer: str, contexts: List[str]) -> Dict[str, float]:
    """Evaluate response quality using RAGAS metrics"""
    if not RAGAS_AVAILABLE:
        return {"error": "RAGAS not available"}
    
    if not question or not answer:
        return {"error": "Question and answer are required"}

    if not contexts:
        contexts = [""]

    try:
        evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-3.5-turbo"))
        evaluator_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings(model="text-embedding-3-small"))

        faithfulness_metric = Faithfulness(llm=evaluator_llm)
        relevancy_metric = ResponseRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings)
        bleu_metric = BleuScore()
        rouge_metric = RougeScore()

        sample = SingleTurnSample(
            user_input=question,
            response=answer,
            retrieved_contexts=contexts,
        )

        import asyncio

        loop = asyncio.new_event_loop()
        try:
            results = {
                "faithfulness": loop.run_until_complete(faithfulness_metric.single_turn_ascore(sample)),
                "answer_relevancy": loop.run_until_complete(relevancy_metric.single_turn_ascore(sample)),
                "bleu_score": loop.run_until_complete(bleu_metric.single_turn_ascore(sample)),
                "rouge_score": loop.run_until_complete(rouge_metric.single_turn_ascore(sample)),
            }
        finally:
            loop.close()

        return results

    except Exception as e:
        return {"error": str(e)}


def batch_evaluate(
    rag_retrieve_fn,
    rag_generate_fn,
    dataset_path: str = "evaluation_dataset.txt",
) -> Dict:
    """
    Load questions from evaluation_dataset.txt or test_questions.json,
    run end-to-end RAG for each, evaluate with RAGAS metrics, and return
    per-question results plus aggregate statistics.

    Args:
        rag_retrieve_fn: Callable(question) -> (contexts: List[str], context_str: str)
        rag_generate_fn: Callable(question, context_str) -> answer: str
        dataset_path:    Path to evaluation_dataset.txt or test_questions.json

    Returns:
        {
          "per_question": [ {question, answer, contexts, scores}, ... ],
          "aggregate":    { metric_name: {mean, min, max}, ... }
        }
    """
    path = Path(dataset_path)
    if not path.exists():
        return {"error": f"Dataset file not found: {dataset_path}"}

    # Load questions — support .txt (one per line) or .json (list of strings or objects)
    questions: List[str] = []
    try:
        if path.suffix == ".json":
            data = json.loads(path.read_text(encoding="utf-8"))
            for item in data:
                if isinstance(item, str):
                    questions.append(item)
                elif isinstance(item, dict):
                    questions.append(item.get("question", ""))
        else:
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line and not line.startswith("#"):
                    questions.append(line)
    except Exception as e:
        return {"error": f"Failed to load dataset: {e}"}

    if not questions:
        return {"error": "No questions found in dataset file"}

    per_question_results = []
    all_scores: Dict[str, List[float]] = {}

    for question in questions:
        try:
            contexts, context_str = rag_retrieve_fn(question)
            answer = rag_generate_fn(question, context_str)
            scores = evaluate_response_quality(question, answer, contexts)
        except Exception as e:
            scores = {"error": str(e)}
            contexts, answer = [], ""

        record = {
            "question": question,
            "answer": answer,
            "contexts": contexts,
            "scores": scores,
        }
        per_question_results.append(record)

        for metric, value in scores.items():
            if isinstance(value, (int, float)) and not np.isnan(value):
                all_scores.setdefault(metric, []).append(float(value))

    # Build aggregate stats
    aggregate = {}
    for metric, values in all_scores.items():
        aggregate[metric] = {
            "mean": float(np.mean(values)),
            "min":  float(np.min(values)),
            "max":  float(np.max(values)),
            "count": len(values),
        }

    return {"per_question": per_question_results, "aggregate": aggregate}
