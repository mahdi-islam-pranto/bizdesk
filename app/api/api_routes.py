from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException, Header
from pydantic import BaseModel, Field
from app.core.config import get_settings
from app.rag.workflow import ask
from app.rag.vector_store import add_documents
from app.services.ingestion import load_file, chunk_documents, SUPPORTED
from app.services.store_conversations import add_conversation_audit, get_token_usage_totals

router = APIRouter(prefix="/api")

settings = get_settings()

# api request input
class ChatRequest(BaseModel):
    question: str = Field(min_length=2, max_length=3000)


@router.get("/health")
def health():
    return {"status": "ok", "service": settings.app_name}


# cumulative token usage across all stored conversations
@router.get("/token_usage")
def token_usage_totals():
    return get_token_usage_totals()

# chat router
@router.post("/chat")
def chat(payload: ChatRequest):
    try:
        # execute full workflow
        result = ask(payload.question)

        input_tokens = result.get("input_tokens", 0)
        output_tokens = result.get("output_tokens", 0)
        token_usage = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "llm_calls": result.get("llm_calls", 0),
            "by_node": result.get("token_usage_by_node", []),
        }

        # store conversation with its token usage to db
        add_conversation_audit(
            payload.question, result["source_used"], result.get("trace", []), token_usage
        )

        return {
            "answer": result["answer"],
            "source_used": result["source_used"],
            "trace": result.get("trace", []),
            "citations": result.get("citations", []),
            "rewritten_query": result.get("current_query", payload.question),
            "token_usage": token_usage,
        }

    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc



# router for upload documents
@router.post("/upload_document")
async def ingest(file: UploadFile = File(...), x_admin_key: str = Header(default="")):
    if x_admin_key != settings.admin_api_key:
        raise HTTPException(status_code=401, detail="Invalid admin key")
    
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in SUPPORTED:
        raise HTTPException(status_code=400, detail=f"Supported: {', '.join(sorted(SUPPORTED))}")
    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)
    dest = upload_dir / Path(file.filename).name
    dest.write_bytes(await file.read())
    docs = load_file(dest)
    chunks = chunk_documents(docs)
    ids = add_documents(chunks)
    return {"message": "Document indexed", "file": dest.name, "chunks": len(chunks), "ids_created": len(ids)}