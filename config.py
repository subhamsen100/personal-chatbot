from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Ollama
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen3-vl:8b"
    ollama_embed_model: str = "nomic-embed-text"

    # Vector store
    vector_store_path: str = "data/indexes"

    # SQLite memory
    db_path: str = "db/chat_memory.db"

    # Chunking
    chunk_size: int = 512
    chunk_overlap: int = 64

    # Retrieval
    top_k: int = 5

    # Memory management
    max_history_turns: int = 20
    summarize_after_turns: int = 16

    # Paths
    docs_path: str = "data/docs"

    # App
    app_name: str = "personal_knowledge_base"
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


settings = Settings()

# Ensure runtime directories exist
for _dir in [settings.vector_store_path, settings.docs_path, Path(settings.db_path).parent]:
    Path(_dir).mkdir(parents=True, exist_ok=True)
