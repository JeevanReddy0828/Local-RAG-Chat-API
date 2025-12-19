"""
Configuration module for the RAG API.

All settings can be overridden via environment variables or a .env file.
"""

from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings with sensible defaults."""

    # ═══════════════════════════════════════════════════════════════════════════
    # STORAGE PATHS
    # ═══════════════════════════════════════════════════════════════════════════
    data_dir: str = Field(default="data", description="Base data directory")
    raw_dir: str = Field(default="data/raw", description="Uploaded documents storage")
    index_dir: str = Field(default="data/index", description="FAISS index storage (per-session)")

    # ═══════════════════════════════════════════════════════════════════════════
    # EMBEDDING MODEL
    # ═══════════════════════════════════════════════════════════════════════════
    embed_model: str = Field(
        default="intfloat/e5-small-v2",
        description="SentenceTransformer model for embeddings (384-dim)",
    )

    # ═══════════════════════════════════════════════════════════════════════════
    # RETRIEVAL
    # ═══════════════════════════════════════════════════════════════════════════
    top_k: int = Field(default=4, ge=1, le=20, description="Number of chunks to retrieve")

    # ═══════════════════════════════════════════════════════════════════════════
    # CHUNKING
    # ═══════════════════════════════════════════════════════════════════════════
    chunk_max_chars: int = Field(
        default=1400,
        ge=200,
        le=8000,
        description="Maximum characters per chunk",
    )
    chunk_overlap_chars: int = Field(
        default=250,
        ge=0,
        le=1000,
        description="Overlap between consecutive chunks",
    )

    # ═══════════════════════════════════════════════════════════════════════════
    # OLLAMA LLM
    # ═══════════════════════════════════════════════════════════════════════════
    ollama_enabled: bool = Field(default=True, description="Enable Ollama for LLM inference")
    ollama_base_url: str = Field(
        default="http://localhost:11434",
        description="Ollama API base URL",
    )
    ollama_model: str = Field(default="mistral", description="Ollama model name")

    # ═══════════════════════════════════════════════════════════════════════════
    # GENERATION PARAMETERS
    # ═══════════════════════════════════════════════════════════════════════════
    max_new_tokens: int = Field(
        default=512,
        ge=64,
        le=4096,
        description="Maximum tokens to generate",
    )
    temperature: float = Field(
        default=0.2,
        ge=0.0,
        le=2.0,
        description="Sampling temperature (lower = more deterministic)",
    )

    # ═══════════════════════════════════════════════════════════════════════════
    # SESSION MANAGEMENT
    # ═══════════════════════════════════════════════════════════════════════════
    max_history_turns: int = Field(
        default=8,
        ge=1,
        le=50,
        description="Maximum conversation turns to include in context",
    )

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Global settings instance
settings = Settings()
