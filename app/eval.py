"""
Evaluation framework for RAG system.

Metrics:
- Recall@K: Does the expected source appear in retrieved results?
- Answer Similarity: Fuzzy match between generated and expected answers

Usage:
    python -m app.eval
"""

import json
import sys
import logging
from typing import Dict, Any, List
from pathlib import Path

from rapidfuzz import fuzz

from .rag import RagStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    """
    Load evaluation data from JSONL file.
    
    Expected format per line:
    {
        "question": "...",
        "expected_answer": "...",
        "expected_source": "filename.txt"
    }
    """
    rows = []
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
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                logger.warning(f"Invalid JSON on line {line_num}: {e}")
    
    return rows


# ═══════════════════════════════════════════════════════════════════════════════
# METRICS
# ═══════════════════════════════════════════════════════════════════════════════


def compute_recall_at_k(expected_source: str, retrieved_sources: List[str]) -> bool:
    """Check if expected source is in retrieved sources."""
    if not expected_source:
        return False
    return expected_source in retrieved_sources


def compute_answer_similarity(generated: str, expected: str) -> float:
    """
    Compute fuzzy similarity between generated and expected answers.
    
    Returns score 0-100.
    """
    if not expected:
        return 0.0
    return fuzz.ratio(generated.lower(), expected.lower())


# ═══════════════════════════════════════════════════════════════════════════════
# EVALUATION
# ═══════════════════════════════════════════════════════════════════════════════


class EvalResults:
    """Container for evaluation results."""
    
    def __init__(self):
        self.total = 0
        self.recall_hits = 0
        self.similarity_sum = 0.0
        self.details: List[Dict[str, Any]] = []
    
    def add_result(
        self,
        question: str,
        expected_source: str,
        expected_answer: str,
        retrieved_sources: List[str],
        generated_answer: str,
    ):
        """Add a single evaluation result."""
        self.total += 1
        
        # Recall
        recall_hit = compute_recall_at_k(expected_source, retrieved_sources)
        if recall_hit:
            self.recall_hits += 1
        
        # Similarity
        similarity = compute_answer_similarity(generated_answer, expected_answer)
        self.similarity_sum += similarity
        
        # Store details
        self.details.append({
            "question": question,
            "expected_source": expected_source,
            "expected_answer": expected_answer,
            "retrieved_sources": retrieved_sources,
            "generated_answer": generated_answer,
            "recall_hit": recall_hit,
            "similarity": similarity,
        })
    
    @property
    def recall_at_k(self) -> float:
        """Recall@K metric."""
        return self.recall_hits / self.total if self.total > 0 else 0.0
    
    @property
    def avg_similarity(self) -> float:
        """Average answer similarity."""
        return self.similarity_sum / self.total if self.total > 0 else 0.0
    
    def print_report(self):
        """Print evaluation report."""
        print("\n" + "=" * 70)
        print("EVALUATION RESULTS")
        print("=" * 70)
        
        for i, detail in enumerate(self.details, 1):
            print(f"\n--- Sample {i} ---")
            print(f"Q: {detail['question']}")
            print(f"Expected source: {detail['expected_source']}")
            print(f"Retrieved sources: {detail['retrieved_sources']}")
            print(f"Recall hit: {'✓' if detail['recall_hit'] else '✗'}")
            print(f"Similarity: {detail['similarity']:.1f}")
            
            # Truncate long answers
            answer = detail['generated_answer']
            if len(answer) > 300:
                answer = answer[:300] + "..."
            print(f"Answer: {answer}")
        
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        print(f"Total samples: {self.total}")
        print(f"Recall@K: {self.recall_at_k:.4f} ({self.recall_hits}/{self.total})")
        print(f"Avg similarity: {self.avg_similarity:.2f}")
        print("=" * 70 + "\n")


def run_evaluation(eval_file: str = "eval_data.jsonl") -> EvalResults:
    """
    Run evaluation on test data.
    
    Args:
        eval_file: Path to JSONL evaluation file
        
    Returns:
        EvalResults object with metrics
    """
    # Load test data
    data = load_jsonl(eval_file)
    
    if not data:
        logger.error("No evaluation data found")
        return EvalResults()
    
    logger.info(f"Loaded {len(data)} evaluation samples")
    
    # Initialize RAG
    rag = RagStore()
    results = EvalResults()
    
    # Evaluation session ID
    eval_session = "__eval__"
    
    for item in data:
        question = item.get("question", "")
        expected_answer = item.get("expected_answer", "")
        expected_source = item.get("expected_source", "")
        
        if not question:
            logger.warning("Skipping item without question")
            continue
        
        # Retrieve
        retrieved = rag.retrieve(eval_session, question)
        retrieved_sources = list(set(r["source"] for r in retrieved))
        
        # Generate
        output = rag.answer(eval_session, question)
        generated_answer = output["answer"]
        
        # Record result
        results.add_result(
            question=question,
            expected_source=expected_source,
            expected_answer=expected_answer,
            retrieved_sources=retrieved_sources,
            generated_answer=generated_answer,
        )
    
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════


def main():
    """Run evaluation from command line."""
    eval_file = sys.argv[1] if len(sys.argv) > 1 else "eval_data.jsonl"
    
    logger.info(f"Running evaluation with: {eval_file}")
    results = run_evaluation(eval_file)
    
    if results.total == 0:
        logger.error("No samples evaluated")
        sys.exit(1)
    
    results.print_report()


if __name__ == "__main__":
    main()
