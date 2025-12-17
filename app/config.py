from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Storage
    data_dir: str = "data"
    raw_dir: str = "data/raw"
    index_dir: str = "data/index"

    # RAG
    embed_model: str = "intfloat/e5-small-v2"
    top_k: int = 4

    # Chunking
    chunk_max_chars: int = 1400
    chunk_overlap_chars: int = 250

    # Ollama (your machine is listening on 11435)
    ollama_enabled: bool = True
    ollama_base_url: str = "http://localhost:11435"
    ollama_model: str = "mistral"

    # Generation
    max_new_tokens: int = 256
    temperature: float = 0.2

    class Config:
        env_file = ".env"


settings = Settings()
