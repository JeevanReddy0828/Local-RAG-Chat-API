import json
from typing import Dict, Any, List
from rapidfuzz import fuzz

from .rag import RagStore


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main():
    rag = RagStore()
    data = load_jsonl("eval_data.jsonl")

    total = 0
    recall_hits = 0
    sim_sum = 0.0

    for item in data:
        q = item["question"]
        exp_ans = item.get("expected_answer", "")
        exp_src = item.get("expected_source", "")

        retrieved = rag.retrieve(q)
        got_sources = [r["source"] for r in retrieved]

        hit = (exp_src in got_sources) if exp_src else False
        if hit:
            recall_hits += 1

        # generate answer (no memory needed for eval)
        out = rag.answer(session_id="__eval__", query=q)
        ans = out["answer"]

        sim = fuzz.ratio(ans.lower(), exp_ans.lower()) if exp_ans else 0
        sim_sum += sim
        total += 1

        print("\n---")
        print("Q:", q)
        print("Expected source:", exp_src)
        print("Retrieved sources:", got_sources)
        print("Recall hit:", hit)
        print("Similarity:", sim)
        print("Answer:", ans[:300], "..." if len(ans) > 300 else "")

    if total == 0:
        print("No eval samples found.")
        return

    recall_at_k = recall_hits / total
    avg_sim = sim_sum / total

    print("\n==========================")
    print("Total samples:", total)
    print("Recall@K:", round(recall_at_k, 4))
    print("Avg answer similarity:", round(avg_sim, 2))
    print("==========================\n")


if __name__ == "__main__":
    main()
