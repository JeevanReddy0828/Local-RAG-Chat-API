"""
RAG System Evaluation Module

Defines and implements both intrinsic and extrinsic metrics for evaluating
RAG system performance.

Intrinsic Metrics (Retrieval Quality):
- Precision@K: Fraction of retrieved docs that are relevant
- Recall@K: Fraction of relevant docs that are retrieved
- MRR (Mean Reciprocal Rank): Position of first relevant result
- NDCG (Normalized Discounted Cumulative Gain): Ranking quality
- Similarity Score Distribution: Embedding similarity statistics

Extrinsic Metrics (Answer Quality):
- Answer Relevance: How well answer addresses the question
- Faithfulness: Is answer grounded in retrieved context
- Answer Similarity: Fuzzy match with expected answer
- Task Success Rate: Binary success/fail for specific tasks
- Hallucination Detection: Facts not in source documents

Usage:
    python -m app.evaluation --eval-file eval_data.jsonl --output results.json
"""

import json
import logging
import statistics
import sys
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime

import numpy as np
from rapidfuzz import fuzz
from sentence_transformers import SentenceTransformer

from .rag import RagStore
from .config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class EvalSample:
    """Single evaluation sample."""
    question: str
    expected_answer: Optional[str] = None
    expected_sources: Optional[List[str]] = None
    task_type: Optional[str] = None  # "summary", "factual", "comparison", etc.
    difficulty: Optional[str] = None  # "easy", "medium", "hard"


@dataclass
class IntrinsicMetrics:
    """Retrieval quality metrics."""
    precision_at_k: float = 0.0
    recall_at_k: float = 0.0
    mrr: float = 0.0  # Mean Reciprocal Rank
    ndcg: float = 0.0  # Normalized Discounted Cumulative Gain
    avg_similarity: float = 0.0
    min_similarity: float = 0.0
    max_similarity: float = 0.0
    similarity_std: float = 0.0
    num_retrieved: int = 0
    num_relevant_retrieved: int = 0


@dataclass
class ExtrinsicMetrics:
    """Answer quality metrics."""
    answer_relevance: float = 0.0  # 0-100
    faithfulness: float = 0.0  # 0-100 (grounded in context)
    answer_similarity: float = 0.0  # 0-100 (vs expected)
    task_success: bool = False
    hallucination_score: float = 0.0  # 0-100 (lower is better)
    response_length: int = 0
    latency_ms: float = 0.0


@dataclass
class SampleResult:
    """Complete evaluation result for one sample."""
    question: str
    generated_answer: str
    expected_answer: Optional[str]
    retrieved_sources: List[str]
    expected_sources: Optional[List[str]]
    intrinsic: IntrinsicMetrics
    extrinsic: ExtrinsicMetrics
    task_type: Optional[str] = None


@dataclass
class EvaluationReport:
    """Aggregated evaluation report."""
    timestamp: str
    num_samples: int
    
    # Aggregated Intrinsic Metrics
    avg_precision_at_k: float = 0.0
    avg_recall_at_k: float = 0.0
    avg_mrr: float = 0.0
    avg_ndcg: float = 0.0
    avg_similarity: float = 0.0
    
    # Aggregated Extrinsic Metrics
    avg_answer_relevance: float = 0.0
    avg_faithfulness: float = 0.0
    avg_answer_similarity: float = 0.0
    task_success_rate: float = 0.0
    avg_hallucination_score: float = 0.0
    avg_latency_ms: float = 0.0
    
    # Per-task breakdown
    metrics_by_task: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # Individual results
    samples: List[SampleResult] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════════════════
# INTRINSIC METRICS (RETRIEVAL QUALITY)
# ═══════════════════════════════════════════════════════════════════════════════


class IntrinsicEvaluator:
    """Evaluates retrieval quality."""
    
    def __init__(self):
        self.embedder = SentenceTransformer(settings.embed_model)
    
    def precision_at_k(
        self,
        retrieved_sources: List[str],
        relevant_sources: List[str],
    ) -> float:
        """
        Precision@K: What fraction of retrieved docs are relevant?
        
        P@K = |relevant ∩ retrieved| / |retrieved|
        """
        if not retrieved_sources:
            return 0.0
        
        relevant_set = set(relevant_sources) if relevant_sources else set()
        retrieved_set = set(retrieved_sources)
        
        relevant_retrieved = len(relevant_set & retrieved_set)
        return relevant_retrieved / len(retrieved_set)
    
    def recall_at_k(
        self,
        retrieved_sources: List[str],
        relevant_sources: List[str],
    ) -> float:
        """
        Recall@K: What fraction of relevant docs were retrieved?
        
        R@K = |relevant ∩ retrieved| / |relevant|
        """
        if not relevant_sources:
            return 1.0  # If no expected sources, consider it success
        
        relevant_set = set(relevant_sources)
        retrieved_set = set(retrieved_sources)
        
        relevant_retrieved = len(relevant_set & retrieved_set)
        return relevant_retrieved / len(relevant_set)
    
    def mrr(
        self,
        retrieved_sources: List[str],
        relevant_sources: List[str],
    ) -> float:
        """
        Mean Reciprocal Rank: Position of first relevant result.
        
        MRR = 1 / rank_of_first_relevant
        """
        if not relevant_sources:
            return 1.0
        
        relevant_set = set(relevant_sources)
        
        for i, source in enumerate(retrieved_sources, 1):
            if source in relevant_set:
                return 1.0 / i
        
        return 0.0
    
    def ndcg(
        self,
        retrieved_sources: List[str],
        relevant_sources: List[str],
        scores: Optional[List[float]] = None,
    ) -> float:
        """
        Normalized Discounted Cumulative Gain.
        
        Measures ranking quality - relevant docs should be ranked higher.
        """
        if not retrieved_sources:
            return 0.0
        
        relevant_set = set(relevant_sources) if relevant_sources else set()
        
        # Binary relevance: 1 if relevant, 0 otherwise
        relevances = [1 if src in relevant_set else 0 for src in retrieved_sources]
        
        # DCG
        dcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(relevances))
        
        # Ideal DCG (all relevant docs at top)
        ideal_relevances = sorted(relevances, reverse=True)
        idcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(ideal_relevances))
        
        if idcg == 0:
            return 0.0
        
        return dcg / idcg
    
    def similarity_stats(
        self,
        scores: List[float],
    ) -> Dict[str, float]:
        """Calculate similarity score statistics."""
        if not scores:
            return {
                "avg": 0.0,
                "min": 0.0,
                "max": 0.0,
                "std": 0.0,
            }
        
        return {
            "avg": statistics.mean(scores),
            "min": min(scores),
            "max": max(scores),
            "std": statistics.stdev(scores) if len(scores) > 1 else 0.0,
        }
    
    def evaluate(
        self,
        retrieved_items: List[Dict[str, Any]],
        relevant_sources: Optional[List[str]] = None,
    ) -> IntrinsicMetrics:
        """Compute all intrinsic metrics."""
        
        retrieved_sources = [item["source"] for item in retrieved_items]
        scores = [float(item.get("score", 0.0)) for item in retrieved_items]
        
        relevant_sources = relevant_sources or []
        
        sim_stats = self.similarity_stats(scores)
        
        relevant_set = set(relevant_sources)
        num_relevant = sum(1 for src in retrieved_sources if src in relevant_set)
        
        return IntrinsicMetrics(
            precision_at_k=self.precision_at_k(retrieved_sources, relevant_sources),
            recall_at_k=self.recall_at_k(retrieved_sources, relevant_sources),
            mrr=self.mrr(retrieved_sources, relevant_sources),
            ndcg=self.ndcg(retrieved_sources, relevant_sources, scores),
            avg_similarity=sim_stats["avg"],
            min_similarity=sim_stats["min"],
            max_similarity=sim_stats["max"],
            similarity_std=sim_stats["std"],
            num_retrieved=len(retrieved_items),
            num_relevant_retrieved=num_relevant,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# EXTRINSIC METRICS (ANSWER QUALITY)
# ═══════════════════════════════════════════════════════════════════════════════


class ExtrinsicEvaluator:
    """Evaluates answer quality."""
    
    def __init__(self):
        self.embedder = SentenceTransformer(settings.embed_model)
    
    def answer_similarity(
        self,
        generated: str,
        expected: str,
    ) -> float:
        """
        Fuzzy string similarity between generated and expected answer.
        
        Returns 0-100 score.
        """
        if not expected:
            return 0.0
        
        return fuzz.ratio(generated.lower(), expected.lower())
    
    def semantic_similarity(
        self,
        text1: str,
        text2: str,
    ) -> float:
        """
        Semantic similarity using embeddings.
        
        Returns 0-100 score.
        """
        if not text1 or not text2:
            return 0.0
        
        embeddings = self.embedder.encode(
            [f"query: {text1}", f"query: {text2}"],
            normalize_embeddings=True,
        )
        
        # Cosine similarity (already normalized)
        similarity = np.dot(embeddings[0], embeddings[1])
        
        # Convert to 0-100 scale
        return float((similarity + 1) * 50)
    
    def answer_relevance(
        self,
        question: str,
        answer: str,
    ) -> float:
        """
        How relevant is the answer to the question?
        
        Uses semantic similarity between question and answer.
        Returns 0-100 score.
        """
        return self.semantic_similarity(question, answer)
    
    def faithfulness(
        self,
        answer: str,
        context_chunks: List[str],
    ) -> float:
        """
        Is the answer grounded in the provided context?
        
        Measures semantic similarity between answer and context.
        Returns 0-100 score.
        """
        if not context_chunks:
            return 0.0
        
        # Combine context
        combined_context = " ".join(context_chunks)
        
        # Semantic similarity
        return self.semantic_similarity(answer, combined_context)
    
    def hallucination_score(
        self,
        answer: str,
        context_chunks: List[str],
    ) -> float:
        """
        Estimate hallucination risk (inverse of faithfulness).
        
        Higher score = more likely hallucination.
        Returns 0-100 score (lower is better).
        """
        faith = self.faithfulness(answer, context_chunks)
        return 100 - faith
    
    def task_success(
        self,
        answer: str,
        expected: Optional[str],
        task_type: Optional[str],
        threshold: float = 50.0,
    ) -> bool:
        """
        Binary task success based on task type.
        
        Different thresholds for different task types.
        """
        if not expected:
            # If no expected answer, check if answer is non-empty and not an error
            return bool(answer) and "error" not in answer.lower()
        
        # Task-specific thresholds
        thresholds = {
            "summary": 40.0,  # Summaries can vary
            "factual": 60.0,  # Facts should be more precise
            "comparison": 50.0,
            "extraction": 70.0,  # Extraction should be exact
        }
        
        task_threshold = thresholds.get(task_type, threshold)
        
        # Use semantic similarity for flexible matching
        similarity = self.semantic_similarity(answer, expected)
        
        return similarity >= task_threshold
    
    def evaluate(
        self,
        question: str,
        generated_answer: str,
        expected_answer: Optional[str],
        context_chunks: List[str],
        task_type: Optional[str] = None,
        latency_ms: float = 0.0,
    ) -> ExtrinsicMetrics:
        """Compute all extrinsic metrics."""
        
        return ExtrinsicMetrics(
            answer_relevance=self.answer_relevance(question, generated_answer),
            faithfulness=self.faithfulness(generated_answer, context_chunks),
            answer_similarity=self.answer_similarity(generated_answer, expected_answer or ""),
            task_success=self.task_success(generated_answer, expected_answer, task_type),
            hallucination_score=self.hallucination_score(generated_answer, context_chunks),
            response_length=len(generated_answer),
            latency_ms=latency_ms,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# COMBINED EVALUATOR
# ═══════════════════════════════════════════════════════════════════════════════


class RAGEvaluator:
    """Complete RAG system evaluator."""
    
    def __init__(self):
        self.rag = RagStore()
        self.intrinsic = IntrinsicEvaluator()
        self.extrinsic = ExtrinsicEvaluator()
    
    def evaluate_sample(
        self,
        sample: EvalSample,
        session_id: str = "__eval__",
    ) -> SampleResult:
        """Evaluate a single sample."""
        import time
        
        # Retrieve
        start = time.perf_counter()
        retrieved = self.rag.retrieve(session_id, sample.question)
        
        # Generate
        result = self.rag.answer(session_id, sample.question)
        latency_ms = (time.perf_counter() - start) * 1000
        
        generated_answer = result["answer"]
        retrieved_sources = list(set(item["source"] for item in retrieved))
        context_chunks = [item["text"] for item in retrieved]
        
        # Intrinsic metrics
        intrinsic_metrics = self.intrinsic.evaluate(
            retrieved_items=retrieved,
            relevant_sources=sample.expected_sources,
        )
        
        # Extrinsic metrics
        extrinsic_metrics = self.extrinsic.evaluate(
            question=sample.question,
            generated_answer=generated_answer,
            expected_answer=sample.expected_answer,
            context_chunks=context_chunks,
            task_type=sample.task_type,
            latency_ms=latency_ms,
        )
        
        return SampleResult(
            question=sample.question,
            generated_answer=generated_answer,
            expected_answer=sample.expected_answer,
            retrieved_sources=retrieved_sources,
            expected_sources=sample.expected_sources,
            intrinsic=intrinsic_metrics,
            extrinsic=extrinsic_metrics,
            task_type=sample.task_type,
        )
    
    def evaluate_batch(
        self,
        samples: List[EvalSample],
        session_id: str = "__eval__",
    ) -> EvaluationReport:
        """Evaluate multiple samples and aggregate results."""
        
        results: List[SampleResult] = []
        
        for i, sample in enumerate(samples, 1):
            logger.info(f"Evaluating sample {i}/{len(samples)}: {sample.question[:50]}...")
            try:
                result = self.evaluate_sample(sample, session_id)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to evaluate sample: {e}")
        
        if not results:
            return EvaluationReport(
                timestamp=datetime.now().isoformat(),
                num_samples=0,
            )
        
        # Aggregate metrics
        report = EvaluationReport(
            timestamp=datetime.now().isoformat(),
            num_samples=len(results),
            samples=results,
        )
        
        # Intrinsic aggregation
        report.avg_precision_at_k = statistics.mean(r.intrinsic.precision_at_k for r in results)
        report.avg_recall_at_k = statistics.mean(r.intrinsic.recall_at_k for r in results)
        report.avg_mrr = statistics.mean(r.intrinsic.mrr for r in results)
        report.avg_ndcg = statistics.mean(r.intrinsic.ndcg for r in results)
        report.avg_similarity = statistics.mean(r.intrinsic.avg_similarity for r in results)
        
        # Extrinsic aggregation
        report.avg_answer_relevance = statistics.mean(r.extrinsic.answer_relevance for r in results)
        report.avg_faithfulness = statistics.mean(r.extrinsic.faithfulness for r in results)
        report.avg_answer_similarity = statistics.mean(r.extrinsic.answer_similarity for r in results)
        report.task_success_rate = sum(1 for r in results if r.extrinsic.task_success) / len(results)
        report.avg_hallucination_score = statistics.mean(r.extrinsic.hallucination_score for r in results)
        report.avg_latency_ms = statistics.mean(r.extrinsic.latency_ms for r in results)
        
        # Per-task breakdown
        task_results: Dict[str, List[SampleResult]] = {}
        for r in results:
            task = r.task_type or "unknown"
            if task not in task_results:
                task_results[task] = []
            task_results[task].append(r)
        
        for task, task_samples in task_results.items():
            report.metrics_by_task[task] = {
                "count": len(task_samples),
                "avg_precision": statistics.mean(r.intrinsic.precision_at_k for r in task_samples),
                "avg_recall": statistics.mean(r.intrinsic.recall_at_k for r in task_samples),
                "avg_relevance": statistics.mean(r.extrinsic.answer_relevance for r in task_samples),
                "avg_faithfulness": statistics.mean(r.extrinsic.faithfulness for r in task_samples),
                "success_rate": sum(1 for r in task_samples if r.extrinsic.task_success) / len(task_samples),
            }
        
        return report
    
    def print_report(self, report: EvaluationReport):
        """Print formatted evaluation report."""
        
        print("\n" + "=" * 80)
        print("RAG SYSTEM EVALUATION REPORT")
        print("=" * 80)
        print(f"Timestamp: {report.timestamp}")
        print(f"Samples Evaluated: {report.num_samples}")
        
        print("\n" + "-" * 40)
        print("INTRINSIC METRICS (Retrieval Quality)")
        print("-" * 40)
        print(f"  Precision@K:      {report.avg_precision_at_k:.4f}")
        print(f"  Recall@K:         {report.avg_recall_at_k:.4f}")
        print(f"  MRR:              {report.avg_mrr:.4f}")
        print(f"  NDCG:             {report.avg_ndcg:.4f}")
        print(f"  Avg Similarity:   {report.avg_similarity:.4f}")
        
        print("\n" + "-" * 40)
        print("EXTRINSIC METRICS (Answer Quality)")
        print("-" * 40)
        print(f"  Answer Relevance: {report.avg_answer_relevance:.2f}/100")
        print(f"  Faithfulness:     {report.avg_faithfulness:.2f}/100")
        print(f"  Answer Similarity:{report.avg_answer_similarity:.2f}/100")
        print(f"  Task Success Rate:{report.task_success_rate:.2%}")
        print(f"  Hallucination:    {report.avg_hallucination_score:.2f}/100 (lower is better)")
        print(f"  Avg Latency:      {report.avg_latency_ms:.0f}ms")
        
        if report.metrics_by_task:
            print("\n" + "-" * 40)
            print("METRICS BY TASK TYPE")
            print("-" * 40)
            for task, metrics in report.metrics_by_task.items():
                print(f"\n  [{task.upper()}] (n={metrics['count']})")
                print(f"    Precision:    {metrics['avg_precision']:.4f}")
                print(f"    Recall:       {metrics['avg_recall']:.4f}")
                print(f"    Relevance:    {metrics['avg_relevance']:.2f}/100")
                print(f"    Faithfulness: {metrics['avg_faithfulness']:.2f}/100")
                print(f"    Success Rate: {metrics['success_rate']:.2%}")
        
        print("\n" + "-" * 40)
        print("SAMPLE DETAILS")
        print("-" * 40)
        for i, sample in enumerate(report.samples[:10], 1):  # Show first 10
            status = "✓" if sample.extrinsic.task_success else "✗"
            print(f"\n  {i}. [{status}] {sample.question[:60]}...")
            print(f"     Sources: {sample.retrieved_sources}")
            print(f"     Precision: {sample.intrinsic.precision_at_k:.2f} | "
                  f"Relevance: {sample.extrinsic.answer_relevance:.0f} | "
                  f"Faithful: {sample.extrinsic.faithfulness:.0f}")
        
        if len(report.samples) > 10:
            print(f"\n  ... and {len(report.samples) - 10} more samples")
        
        print("\n" + "=" * 80)


# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════


def load_eval_data(path: str) -> List[EvalSample]:
    """
    Load evaluation data from JSONL file.
    
    Expected format per line:
    {
        "question": "...",
        "expected_answer": "...",  // optional
        "expected_sources": ["file1.txt", "file2.txt"],  // optional
        "task_type": "summary" | "factual" | "extraction",  // optional
        "difficulty": "easy" | "medium" | "hard"  // optional
    }
    """
    samples = []
    filepath = Path(path)
    
    if not filepath.exists():
        logger.error(f"Evaluation file not found: {path}")
        return []
    
    with open(filepath, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                data = json.loads(line)
                samples.append(EvalSample(
                    question=data.get("question", ""),
                    expected_answer=data.get("expected_answer"),
                    expected_sources=data.get("expected_sources"),
                    task_type=data.get("task_type"),
                    difficulty=data.get("difficulty"),
                ))
            except json.JSONDecodeError as e:
                logger.warning(f"Invalid JSON on line {line_num}: {e}")
    
    return samples


def save_report(report: EvaluationReport, path: str):
    """Save evaluation report to JSON file."""
    
    # Convert dataclasses to dicts
    def to_dict(obj):
        if hasattr(obj, "__dataclass_fields__"):
            return {k: to_dict(v) for k, v in asdict(obj).items()}
        elif isinstance(obj, list):
            return [to_dict(item) for item in obj]
        elif isinstance(obj, dict):
            return {k: to_dict(v) for k, v in obj.items()}
        return obj
    
    with open(path, "w", encoding="utf-8") as f:
        json.dump(to_dict(report), f, indent=2, ensure_ascii=False)
    
    logger.info(f"Report saved to {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════


def main():
    """Run evaluation from command line."""
    import argparse
    
    parser = argparse.ArgumentParser(description="RAG System Evaluation")
    parser.add_argument(
        "--eval-file",
        default="eval_data.jsonl",
        help="Path to evaluation data (JSONL)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output path for JSON report",
    )
    parser.add_argument(
        "--session",
        default="__eval__",
        help="Session ID to use for evaluation",
    )
    
    args = parser.parse_args()
    
    # Load data
    samples = load_eval_data(args.eval_file)
    if not samples:
        logger.error("No evaluation samples found")
        sys.exit(1)
    
    logger.info(f"Loaded {len(samples)} evaluation samples")
    
    # Run evaluation
    evaluator = RAGEvaluator()
    report = evaluator.evaluate_batch(samples, args.session)
    
    # Print report
    evaluator.print_report(report)
    
    # Save if output specified
    if args.output:
        save_report(report, args.output)


if __name__ == "__main__":
    main()