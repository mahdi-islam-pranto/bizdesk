# test settings
from app.core.config import get_settings

settings = get_settings()

print(f"APP NAME: {settings.app_name}")
print(f"OPENAI KEY: {settings.openai_api_key}")
print(f"OPENAI MODEL: {settings.openai_model}")
print(f"All conversation storage: {settings.audit_db_path}")

# test logging
from app.core.logging import configure_logging
log = configure_logging()
print("logging configured successfully")

# test data ingestion
from app.services.ingestion import load_file, chunk_documents
from pathlib import Path

test_file_path = Path("data/sample_kb/Daraz.pdf")
loaded_docs = load_file(test_file_path)
chunked_docs = chunk_documents(loaded_docs)
print(f"Loaded {len(loaded_docs)} documents from {test_file_path}")
print(f"Chunked into {len(chunked_docs)} smaller documents")