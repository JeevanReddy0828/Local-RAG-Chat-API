"""
Configuration module for the RAG API.
"""

from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings with sensible defaults."""

    # Storage paths
    data_dir: str = Field(default="data", description="Base data directory")
    raw_dir: str = Field(default="data/raw", description="Uploaded documents storage")
    index_dir: str = Field(default="data/index", description="FAISS index storage")

    # Embedding model
    embed_model: str = Field(
        default="intfloat/e5-small-v2",
        description="SentenceTransformer model for embeddings",
    )

    # Retrieval
    top_k: int = Field(default=4, ge=1, le=20, description="Number of chunks to retrieve")

    # Chunking
    chunk_max_chars: int = Field(default=1000, ge=200, le=8000)
    chunk_overlap_chars: int = Field(default=0, ge=0, le=1000)

    # Ollama LLM
    ollama_enabled: bool = Field(default=True)
    ollama_base_url: str = Field(default="http://localhost:11435")
    ollama_model: str = Field(default="llama3.2")

    # Generation
    max_new_tokens: int = Field(default=300, ge=64, le=4096)
    temperature: float = Field(default=0.3, ge=0.0, le=2.0)

    # Session
    max_history_turns: int = Field(default=8, ge=1, le=50)

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "ignore"


settings = Settings()