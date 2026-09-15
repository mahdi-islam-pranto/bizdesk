from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]

# main environment setting for the app
class Settings(BaseSettings):
    app_name: str = "BizDesk (Enterprise Company Support Agent)"
    app_env: str = "development"
    openai_api_key: str = ""
    tavily_api_key: str = ""
    pinecone_api_key: str = ""
    pinecone_index_name: str = "daraz-faq-rag"
    pinecone_namespace: str = "daraz-faq-kb"
    embedding_model: str = "text-embedding-3-small"
    openai_model: str = "gpt-4o-mini"
    top_k: int = 4
    max_retries: int = 1
    admin_api_key: str = "change-in-production"
    audit_db_path: str = str(BASE_DIR / "data" / "conversation.db")
    upload_dir: str = str(BASE_DIR / "uploads")
    sample_kb_dir: str = str(BASE_DIR / "data" / "sample_kb")


    model_config = SettingsConfigDict(env_file=str(BASE_DIR / ".env"), extra="ignore")



@lru_cache
def get_settings() -> Settings:
    return Settings()