import os
import json
from typing import Iterator, List, Dict, Any, Optional, Tuple

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from .config import settings
from .memory import format_history, append_turn, get_active_file
from .ollama_client import ollama_generate, ollama_generate_stream


DOC_LEVEL_TRIGGERS = [
    "summary of the document",
    "summarize the document",
    "summarize the given document",
    "summarize this document",
    "what is this document",
    "describe this document",
    "what kind of document",
    "overview of the document",
    "summary",
]


class RagStore:
    """
    Per-session RAG store:
    data/index/<session_id>/faiss.index
    data/index/<session_id>/meta.json
    """

    def __init__(self):
        self.embedder = SentenceTransformer(settings.embed_model)
        self._stores: Dict[str, Dict[str, Any]] = {}  # session_id -> {index, meta, paths}

    def _paths(self, session_id: str) -> Tuple[str, str, str]:
        sess_dir = os.path.join(settings.index_dir, session_id)
        os.makedirs(sess_dir, exist_ok=True)
        index_path = os.path.join(sess_dir, "faiss.index")
        meta_path = os.path.join(sess_dir, "meta.json")
        return sess_dir, index_path, meta_path

    def _load_store(self, session_id: str) -> Dict[str, Any]:
        if session_id in self._stores:
            return self._stores[session_id]

        _, index_path, meta_path = self._paths(session_id)

        store = {"index": None, "meta": [], "index_path": index_path, "meta_path": meta_path}

        if os.path.exists(index_path) and os.path.exists(meta_path):
            store["index"] = faiss.read_index(index_path)
            with open(meta_path, "r", encoding="utf-8") as f:
                store["meta"] = json.load(f)

        self._stores[session_id] = store
        return store

    def _save_store(self, session_id: str) -> None:
        store = self._load_store(session_id)
        idx = store["index"]
        if idx is None:
            return
        faiss.write_index(idx, store["index_path"])
        with open(store["meta_path"], "w", encoding="utf-8") as f:
            json.dump(store["meta"], f, ensure_ascii=False, indent=2)

    def clear_session_index(self, session_id: str) -> None:
        # in-memory
        self._stores.pop(session_id, None)
        # on-disk
        sess_dir, index_path, meta_path = self._paths(session_id)
        if os.path.exists(index_path):
            os.remove(index_path)
        if os.path.exists(meta_path):
            os.remove(meta_path)

    def add_texts(self, session_id: str, chunks: List[str], source: str) -> int:
        if not chunks:
            return 0

        store = self._load_store(session_id)

        docs = [f"passage: {c}" for c in chunks]
        emb = self.embedder.encode(docs, normalize_embeddings=True)
        emb = np.array(emb, dtype="float32")

        if store["index"] is None:
            dim = emb.shape[1]
            store["index"] = faiss.IndexFlatIP(dim)

        start_id = len(store["meta"])
        store["index"].add(emb)

        for i, c in enumerate(chunks):
            store["meta"].append({"id": start_id + i, "text": c, "source": source})

        self._save_store(session_id)
        return len(chunks)

    def retrieve(self, session_id: str, query: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        store = self._load_store(session_id)
        if store["index"] is None or not store["meta"]:
            return []

        active_file = get_active_file(session_id)
        q_lower = query.lower()

        # ✅ Document-level queries: return ALL chunks of active file
        if active_file and any(t in q_lower for t in DOC_LEVEL_TRIGGERS):
            return [{**m, "score": 1.0} for m in store["meta"] if m["source"] == active_file]

        # ✅ Normal semantic retrieval (still scoped to active_file)
        k = top_k or settings.top_k

        q = f"query: {query}"
        q_emb = self.embedder.encode([q], normalize_embeddings=True)
        q_emb = np.array(q_emb, dtype="float32")

        scores, ids = store["index"].search(q_emb, k * 3)

        results = []
        for idx, score in zip(ids[0], scores[0]):
            if idx == -1:
                continue
            item = {**store["meta"][int(idx)], "score": float(score)}
            if active_file and item["source"] != active_file:
                continue
            results.append(item)
            if len(results) >= k:
                break

        return results

    def _build_prompt(self, session_id: str, query: str, ctx_items: List[Dict[str, Any]]) -> str:
        history = format_history(session_id)

        ctx_lines = []
        for x in ctx_items:
            src = x.get("source", "unknown")
            cid = x.get("id", "na")
            score = float(x.get("score", 0.0))
            ctx_lines.append(f"[source={src} | chunk_id={cid} | score={score:.3f}]\n{x['text']}")

        context = "\n\n---\n\n".join(ctx_lines) if ctx_lines else ""

        parts = [
            "You are a helpful assistant.",
            "Use ONLY the provided context to answer.",
            "If the context is insufficient, say: \"I don't know based on the provided documents.\"",
            "If asked to summarize a document, produce a concise high-level summary of the entire document.",
            "",
        ]

        if history:
            parts.append("Conversation so far:\n" + history + "\n")

        if context:
            parts.append("Context:\n" + context + "\n")

        parts.append(f"Question: {query}\nAnswer:")
        return "\n".join(parts).strip()

    def answer(self, session_id: str, query: str) -> Dict[str, Any]:
        ctx_items = self.retrieve(session_id, query)
        prompt = self._build_prompt(session_id, query, ctx_items)

        append_turn(session_id, "user", query)
        text = ollama_generate(prompt)
        append_turn(session_id, "assistant", text)

        # sources summary (unique)
        best_by_source: Dict[str, float] = {}
        for x in ctx_items:
            best_by_source[x["source"]] = max(best_by_source.get(x["source"], 0.0), float(x["score"]))

        sources = [{"source": s, "best_score": sc} for s, sc in sorted(best_by_source.items(), key=lambda t: t[1], reverse=True)]
        return {"answer": text, "sources": sources, "active_file": get_active_file(session_id)}

    def answer_stream(self, session_id: str, query: str) -> Iterator[str]:
        ctx_items = self.retrieve(session_id, query)
        prompt = self._build_prompt(session_id, query, ctx_items)

        append_turn(session_id, "user", query)

        collected: List[str] = []
        for chunk in ollama_generate_stream(prompt):
            collected.append(chunk)
            yield chunk

        final_text = "".join(collected).strip()
        if final_text:
            append_turn(session_id, "assistant", final_text)
