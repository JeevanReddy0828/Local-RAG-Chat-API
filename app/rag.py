"""
RAG (Retrieval-Augmented Generation) engine.

Core responsibilities:
- Per-session FAISS index management
- Intent-aware retrieval (full-doc vs semantic top-k)
- Prompt construction with context and history
- LLM generation (sync and streaming)
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


# ═══════════════════════════════════════════════════════════════════════════════
# INTENT DETECTION
# ═══════════════════════════════════════════════════════════════════════════════


# Queries matching these patterns trigger full-document retrieval
DOC_LEVEL_TRIGGERS = [
    "summary of the document",
    "summarize the document",
    "summarize the given document",
    "summarize this document",
    "what is this document",
    "describe this document",
    "what kind of document",
    "overview of the document",
    "tell me about this document",
    "what does this document contain",
    "document overview",
    "full summary",
    "summary",
]


def _is_document_level_query(query: str) -> bool:
    """
    Determine if query requires full-document context.
    
    Args:
        query: User query string
        
    Returns:
        True if query matches document-level patterns
    """
    q_lower = query.lower().strip()
    return any(trigger in q_lower for trigger in DOC_LEVEL_TRIGGERS)


# ═══════════════════════════════════════════════════════════════════════════════
# RAG STORE
# ═══════════════════════════════════════════════════════════════════════════════


class RagStore:
    """
    Per-session RAG store with FAISS indexing.
    
    Each session gets its own:
    - FAISS index (data/index/{session_id}/faiss.index)
    - Metadata store (data/index/{session_id}/meta.json)
    
    This ensures complete session isolation.
    """

    def __init__(self):
        """Initialize the RAG store with embedding model."""
        logger.info(f"Loading embedding model: {settings.embed_model}")
        self.embedder = SentenceTransformer(settings.embed_model)
        self._embedding_dim = self.embedder.get_sentence_embedding_dimension()
        logger.info(f"Embedding dimension: {self._embedding_dim}")
        
        # In-memory cache: session_id -> {index, meta, paths}
        self._stores: Dict[str, Dict[str, Any]] = {}

    # ═══════════════════════════════════════════════════════════════════════════
    # INDEX MANAGEMENT
    # ═══════════════════════════════════════════════════════════════════════════

    def _paths(self, session_id: str) -> Tuple[str, str, str]:
        """
        Get file paths for a session's index.
        
        Returns:
            Tuple of (session_dir, index_path, meta_path)
        """
        sess_dir = os.path.join(settings.index_dir, session_id)
        os.makedirs(sess_dir, exist_ok=True)
        index_path = os.path.join(sess_dir, "faiss.index")
        meta_path = os.path.join(sess_dir, "meta.json")
        return sess_dir, index_path, meta_path

    def _load_store(self, session_id: str) -> Dict[str, Any]:
        """
        Load or create store for a session.
        
        Loads from disk if exists, otherwise creates empty store.
        """
        if session_id in self._stores:
            return self._stores[session_id]

        _, index_path, meta_path = self._paths(session_id)

        store = {
            "index": None,
            "meta": [],
            "index_path": index_path,
            "meta_path": meta_path,
        }

        # Load existing index from disk
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
            logger.debug(f"Saved index for session {session_id}")
        except Exception as e:
            logger.error(f"Failed to save index for {session_id}: {e}")

    def clear_session_index(self, session_id: str) -> None:
        """
        Clear FAISS index and metadata for a session.
        
        Removes both in-memory and on-disk data.
        """
        # Clear in-memory
        self._stores.pop(session_id, None)
        
        # Clear on-disk
        _, index_path, meta_path = self._paths(session_id)
        for path in [index_path, meta_path]:
            if os.path.exists(path):
                try:
                    os.remove(path)
                    logger.debug(f"Removed {path}")
                except Exception as e:
                    logger.error(f"Failed to remove {path}: {e}")

        logger.info(f"Cleared index for session {session_id}")

    # ═══════════════════════════════════════════════════════════════════════════
    # INDEXING
    # ═══════════════════════════════════════════════════════════════════════════

    def add_texts(
        self,
        session_id: str,
        chunks: List[str],
        source: str,
    ) -> int:
        """
        Add text chunks to session's FAISS index.
        
        Args:
            session_id: Session identifier
            chunks: List of text chunks to index
            source: Source filename for metadata
            
        Returns:
            Number of chunks added
        """
        if not chunks:
            return 0

        store = self._load_store(session_id)

        # Embed chunks (E5 models expect "passage: " prefix for documents)
        docs = [f"passage: {c}" for c in chunks]
        embeddings = self.embedder.encode(docs, normalize_embeddings=True)
        embeddings = np.array(embeddings, dtype="float32")

        # Create index if needed
        if store["index"] is None:
            dim = embeddings.shape[1]
            store["index"] = faiss.IndexFlatIP(dim)  # Inner product (cosine with normalized vectors)
            logger.debug(f"Created new FAISS index for session {session_id}, dim={dim}")

        # Add to index
        start_id = len(store["meta"])
        store["index"].add(embeddings)

        # Store metadata
        for i, chunk in enumerate(chunks):
            store["meta"].append({
                "id": start_id + i,
                "text": chunk,
                "source": source,
            })

        self._save_store(session_id)
        logger.info(f"Session {session_id}: Added {len(chunks)} chunks from '{source}'")
        
        return len(chunks)

    # ═══════════════════════════════════════════════════════════════════════════
    # RETRIEVAL
    # ═══════════════════════════════════════════════════════════════════════════

    def retrieve(
        self,
        session_id: str,
        query: str,
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant chunks for a query.
        
        Uses intent-aware retrieval:
        - Document-level queries → all chunks from active file
        - Fact-based queries → semantic top-k search
        
        Args:
            session_id: Session identifier
            query: User query
            top_k: Number of results (defaults to settings.top_k)
            
        Returns:
            List of retrieved chunks with scores
        """
        store = self._load_store(session_id)
        
        if store["index"] is None or not store["meta"]:
            logger.warning(f"Session {session_id}: No index available for retrieval")
            return []

        active_file = get_active_file(session_id)

        # ─────────────────────────────────────────────────────────────────────
        # Document-level queries: return ALL chunks from active file
        # ─────────────────────────────────────────────────────────────────────
        if active_file and _is_document_level_query(query):
            results = [
                {**m, "score": 1.0}
                for m in store["meta"]
                if m["source"] == active_file
            ]
            logger.debug(f"Document-level retrieval: {len(results)} chunks from '{active_file}'")
            return results

        # ─────────────────────────────────────────────────────────────────────
        # Semantic retrieval: top-k with active file filter
        # ─────────────────────────────────────────────────────────────────────
        k = top_k or settings.top_k

        # Embed query (E5 models expect "query: " prefix for queries)
        q_emb = self.embedder.encode([f"query: {query}"], normalize_embeddings=True)
        q_emb = np.array(q_emb, dtype="float32")

        # Search (retrieve extra to account for filtering)
        search_k = min(k * 3, store["index"].ntotal)
        scores, indices = store["index"].search(q_emb, search_k)

        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx == -1:
                continue
                
            item = {**store["meta"][int(idx)], "score": float(score)}
            
            # Filter by active file if set
            if active_file and item["source"] != active_file:
                continue
                
            results.append(item)
            
            if len(results) >= k:
                break

        logger.debug(f"Semantic retrieval: {len(results)} chunks (query: '{query[:50]}...')")
        return results

    # ═══════════════════════════════════════════════════════════════════════════
    # PROMPT CONSTRUCTION
    # ═══════════════════════════════════════════════════════════════════════════

    def _build_prompt(
        self,
        session_id: str,
        query: str,
        context_items: List[Dict[str, Any]],
    ) -> str:
        """
        Build prompt with system instructions, history, context, and query.
        
        Args:
            session_id: Session identifier
            query: User query
            context_items: Retrieved context chunks
            
        Returns:
            Complete prompt string
        """
        # Get conversation history
        history = format_history(session_id, max_turns=settings.max_history_turns)

        # Format context chunks
        context_lines = []
        for item in context_items:
            src = item.get("source", "unknown")
            chunk_id = item.get("id", "na")
            score = float(item.get("score", 0.0))
            context_lines.append(
                f"[source={src} | chunk_id={chunk_id} | score={score:.3f}]\n{item['text']}"
            )
        context = "\n\n---\n\n".join(context_lines) if context_lines else ""

        # Build prompt
        parts = [
            "You are a helpful assistant answering questions about documents.",
            "",
            "Instructions:",
            "- Use ONLY the provided context to answer the question.",
            "- If the context doesn't contain enough information, say: \"I don't have enough information in the provided documents to answer that.\"",
            "- For summary requests, provide a comprehensive overview of the document's main points.",
            "- Be concise but thorough.",
            "",
        ]

        if history:
            parts.append("=== Conversation History ===")
            parts.append(history)
            parts.append("")

        if context:
            parts.append("=== Document Context ===")
            parts.append(context)
            parts.append("")

        parts.append(f"Question: {query}")
        parts.append("")
        parts.append("Answer:")

        return "\n".join(parts)

    # ═══════════════════════════════════════════════════════════════════════════
    # GENERATION
    # ═══════════════════════════════════════════════════════════════════════════

    def answer(
        self,
        session_id: str,
        query: str,
    ) -> Dict[str, Any]:
        """
        Generate answer for a query (synchronous).
        
        Args:
            session_id: Session identifier
            query: User query
            
        Returns:
            Dict with answer, sources, and active_file
        """
        # Retrieve context
        context_items = self.retrieve(session_id, query)
        
        # Build prompt
        prompt = self._build_prompt(session_id, query, context_items)

        # Record user turn
        append_turn(session_id, "user", query)
        
        # Generate response
        try:
            answer_text = ollama_generate(prompt)
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            answer_text = f"Error generating response: {str(e)}"

        # Record assistant turn
        append_turn(session_id, "assistant", answer_text)

        # Aggregate sources (best score per source)
        best_by_source: Dict[str, float] = {}
        for item in context_items:
            source = item["source"]
            score = float(item["score"])
            best_by_source[source] = max(best_by_source.get(source, 0.0), score)

        sources = [
            {"source": s, "best_score": round(sc, 4)}
            for s, sc in sorted(best_by_source.items(), key=lambda t: t[1], reverse=True)
        ]

        return {
            "answer": answer_text,
            "sources": sources,
            "active_file": get_active_file(session_id),
        }

    def answer_stream(
        self,
        session_id: str,
        query: str,
    ) -> Iterator[str]:
        """
        Generate answer for a query (streaming).
        
        Yields tokens as they are generated.
        
        Args:
            session_id: Session identifier
            query: User query
            
        Yields:
            Token strings
        """
        # Retrieve context
        context_items = self.retrieve(session_id, query)
        
        # Build prompt
        prompt = self._build_prompt(session_id, query, context_items)

        # Record user turn
        append_turn(session_id, "user", query)

        # Stream response
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

        # Record complete assistant response
        final_text = "".join(collected).strip()
        if final_text:
            append_turn(session_id, "assistant", final_text)

    # ═══════════════════════════════════════════════════════════════════════════
    # STATS
    # ═══════════════════════════════════════════════════════════════════════════

    def get_session_stats(self, session_id: str) -> Dict[str, Any]:
        """
        Get statistics for a session's index.
        
        Returns:
            Dict with chunk count, sources, etc.
        """
        store = self._load_store(session_id)
        
        if not store["meta"]:
            return {
                "session_id": session_id,
                "total_chunks": 0,
                "sources": [],
            }

        # Count chunks per source
        source_counts: Dict[str, int] = {}
        for item in store["meta"]:
            src = item["source"]
            source_counts[src] = source_counts.get(src, 0) + 1

        return {
            "session_id": session_id,
            "total_chunks": len(store["meta"]),
            "sources": [
                {"source": s, "chunks": c}
                for s, c in source_counts.items()
            ],
            "active_file": get_active_file(session_id),
        }
