from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    groq_api_key: str = ""
    gemini_api_key: str = ""
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str = ""
    logfire_token: str = ""
    qdrant_collection: str = "documents"
    gemini_model: str = "models/gemini-embedding-001"
    groq_model: str = "llama-3.3-70b-versatile"
    chunk_size: int = 512
    chunk_overlap: int = 64
    cache_ttl: int = 300
    cache_maxsize: int = 100
    embedding_dim: int = 768

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }


settings = Settings()
