"""
RAG (Retrieval-Augmented Generation) engine.
"""

import os
import json
import logging
from typing import Iterator, List, Dict, Any, Optional, Tuple

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from .config import settings
from .memory import format_history, append_turn, get_active_file
from .ollama_client import ollama_generate, ollama_generate_stream

logger = logging.getLogger(__name__)


# Document-level query triggers
DOC_LEVEL_TRIGGERS = ["summary", "summarize", "summarise", "overview"]


def _is_document_level_query(query: str) -> bool:
    """Determine if query requires full-document context."""
    q_lower = query.lower().strip()
    return any(trigger in q_lower for trigger in DOC_LEVEL_TRIGGERS)


class RagStore:
    """Per-session RAG store with FAISS indexing."""

    def __init__(self):
        """Initialize the RAG store with embedding model."""
        logger.info(f"Loading embedding model: {settings.embed_model}")
        self.embedder = SentenceTransformer(settings.embed_model)
        self._embedding_dim = self.embedder.get_sentence_embedding_dimension()
        logger.info(f"Embedding dimension: {self._embedding_dim}")
        self._stores: Dict[str, Dict[str, Any]] = {}

    def _paths(self, session_id: str) -> Tuple[str, str, str]:
        """Get file paths for a session's index."""
        sess_dir = os.path.join(settings.index_dir, session_id)
        os.makedirs(sess_dir, exist_ok=True)
        index_path = os.path.join(sess_dir, "faiss.index")
        meta_path = os.path.join(sess_dir, "meta.json")
        return sess_dir, index_path, meta_path

    def _load_store(self, session_id: str) -> Dict[str, Any]:
        """Load or create store for a session."""
        if session_id in self._stores:
            return self._stores[session_id]

        _, index_path, meta_path = self._paths(session_id)

        store = {
            "index": None,
            "meta": [],
            "index_path": index_path,
            "meta_path": meta_path,
        }

        if os.path.exists(index_path) and os.path.exists(meta_path):
            try:
                store["index"] = faiss.read_index(index_path)
                with open(meta_path, "r", encoding="utf-8") as f:
                    store["meta"] = json.load(f)
                logger.debug(f"Loaded index for session {session_id}: {len(store['meta'])} chunks")
            except Exception as e:
                logger.error(f"Failed to load index for {session_id}: {e}")
                store["index"] = None
                store["meta"] = []

        self._stores[session_id] = store
        return store

    def _save_store(self, session_id: str) -> None:
        """Persist store to disk."""
        store = self._load_store(session_id)
        idx = store["index"]
        
        if idx is None:
            return
            
        try:
            faiss.write_index(idx, store["index_path"])
            with open(store["meta_path"], "w", encoding="utf-8") as f:
                json.dump(store["meta"], f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Failed to save index for {session_id}: {e}")

    def clear_session_index(self, session_id: str) -> None:
        """Clear FAISS index and metadata for a session."""
        self._stores.pop(session_id, None)
        _, index_path, meta_path = self._paths(session_id)
        for path in [index_path, meta_path]:
            if os.path.exists(path):
                try:
                    os.remove(path)
                except Exception as e:
                    logger.error(f"Failed to remove {path}: {e}")
        logger.info(f"Cleared index for session {session_id}")

    def add_texts(self, session_id: str, chunks: List[str], source: str) -> int:
        """Add text chunks to session's FAISS index."""
        if not chunks:
            return 0

        store = self._load_store(session_id)

        docs = [f"passage: {c}" for c in chunks]
        embeddings = self.embedder.encode(docs, normalize_embeddings=True)
        embeddings = np.array(embeddings, dtype="float32")

        if store["index"] is None:
            dim = embeddings.shape[1]
            store["index"] = faiss.IndexFlatIP(dim)

        start_id = len(store["meta"])
        store["index"].add(embeddings)

        for i, chunk in enumerate(chunks):
            store["meta"].append({
                "id": start_id + i,
                "text": chunk,
                "source": source,
            })

        self._save_store(session_id)
        logger.info(f"Session {session_id}: Added {len(chunks)} chunks from '{source}'")
        return len(chunks)

    def retrieve(self, session_id: str, query: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """Retrieve relevant chunks for a query."""
        store = self._load_store(session_id)
        
        if store["index"] is None or store["index"].ntotal == 0:
            return []

        active_file = get_active_file(session_id)

        # Document-level: return first 2 chunks only
        if active_file and _is_document_level_query(query):
            file_chunks = [m for m in store["meta"] if m["source"] == active_file]
            # Only take first 2 chunks to keep context small
            results = [{**m, "score": 1.0} for m in file_chunks[:2]]
            logger.debug(f"Document-level retrieval: {len(results)} chunks")
            return results

        # Semantic retrieval
        k = top_k or settings.top_k
        q_emb = self.embedder.encode([f"query: {query}"], normalize_embeddings=True)
        q_emb = np.array(q_emb, dtype="float32")

        search_k = min(k * 3, store["index"].ntotal)
        scores, indices = store["index"].search(q_emb, search_k)

        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx == -1:
                continue
            item = {**store["meta"][int(idx)], "score": float(score)}
            if active_file and item["source"] != active_file:
                continue
            results.append(item)
            if len(results) >= k:
                break

        return results

    def _build_prompt(self, session_id: str, query: str, context_items: List[Dict[str, Any]]) -> str:
        """Build a simple prompt - matching what worked in test."""
        # Combine context chunks
        context_text = "\n\n".join([item["text"] for item in context_items])
        
        # Simple prompt format - this worked in test_ollama_direct.py
        prompt = f"Summarize this document in 2-3 sentences:\n\n{context_text}\n\nSummary:"
        
        return prompt

    def answer(self, session_id: str, query: str) -> Dict[str, Any]:
        """Generate answer for a query (synchronous)."""
        context_items = self.retrieve(session_id, query)
        prompt = self._build_prompt(session_id, query, context_items)

        append_turn(session_id, "user", query)
        
        try:
            answer_text = ollama_generate(prompt)
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            answer_text = f"Error generating response: {str(e)}"

        append_turn(session_id, "assistant", answer_text)

        sources = list(set(item["source"] for item in context_items))

        return {
            "answer": answer_text,
            "sources": [{"source": s, "best_score": 1.0} for s in sources],
            "active_file": get_active_file(session_id),
        }

    def answer_stream(self, session_id: str, query: str) -> Iterator[str]:
        """Generate answer for a query (streaming)."""
        context_items = self.retrieve(session_id, query)
        prompt = self._build_prompt(session_id, query, context_items)

        append_turn(session_id, "user", query)

        collected: List[str] = []
        try:
            for chunk in ollama_generate_stream(prompt):
                collected.append(chunk)
                yield chunk
        except Exception as e:
            logger.error(f"Streaming generation failed: {e}")
            error_msg = f"Error: {str(e)}"
            collected.append(error_msg)
            yield error_msg

        final_text = "".join(collected).strip()
        if final_text:
            append_turn(session_id, "assistant", final_text)

    def get_session_stats(self, session_id: str) -> Dict[str, Any]:
        """Get statistics for a session's index."""
        store = self._load_store(session_id)
        
        if not store["meta"]:
            return {"session_id": session_id, "total_chunks": 0, "sources": []}

        source_counts: Dict[str, int] = {}
        for item in store["meta"]:
            src = item["source"]
            source_counts[src] = source_counts.get(src, 0) + 1

        return {
            "session_id": session_id,
            "total_chunks": len(store["meta"]),
            "sources": [{"source": s, "chunks": c} for s, c in source_counts.items()],
            "active_file": get_active_file(session_id),
        }