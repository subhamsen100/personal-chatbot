"""
FastAPI backend for the Personal Knowledge Base AI.

Run:
    source chatenv/bin/activate
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload
"""

import logging
import shutil
import tempfile
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from agent.rag_agent import rag_agent
from config import settings
from storage.vector_store import vector_store

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


# lifespan 

@asynccontextmanager
async def lifespan(_: FastAPI):
    log.info("Personal Knowledge Base AI started.")
    log.info("Vector store chunks: %d", vector_store.stats()["total_chunks"])
    yield
    log.info("Shutting down.")


# app 

app = FastAPI(
    title="Personal Knowledge Base AI",
    description="Offline, grounded RAG assistant powered by Google ADK + Ollama",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# schemas ─

class ChatRequest(BaseModel):
    session_id: str
    message: str


class ChatResponse(BaseModel):
    session_id: str
    answer: str


class NewSessionResponse(BaseModel):
    session_id: str


# routes

@app.get("/api/health")
async def health():
    kb = vector_store.stats()
    return {
        "status": "ok",
        "model": settings.ollama_model,
        "embed_model": settings.ollama_embed_model,
        "kb_chunks": kb["total_chunks"],
        "kb_sources": kb["sources"],
    }


@app.post("/api/sessions", response_model=NewSessionResponse)
async def new_session():
    """Create a new conversation session."""
    session_id = rag_agent.new_session()
    return {"session_id": session_id}


@app.get("/api/sessions")
async def list_sessions():
    """List all sessions ordered by most recently active."""
    return rag_agent.list_sessions()


@app.get("/api/sessions/{session_id}/history")
async def get_history(session_id: str, last_n: int = 50):
    """Retrieve message history for a session."""
    return rag_agent.get_history(session_id, last_n=last_n)


@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete a session and all its messages."""
    rag_agent.delete_session(session_id)
    return {"deleted": session_id}


@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """Send a message and receive a grounded response."""
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty.")
    result = await rag_agent.chat(req.session_id, req.message)
    return result


@app.post("/api/ingest/file")
async def ingest_file(file: UploadFile = File(...)):
    """
    Upload and ingest a document (PDF, DOCX, MD, TXT).
    The file is saved to data/docs/ then indexed.
    """
    allowed = {".pdf", ".md", ".markdown", ".docx", ".txt", ".text"}
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Allowed: {allowed}",
        )

    # Save to docs directory
    dest = Path(settings.docs_path) / file.filename
    with dest.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        result = rag_agent.ingest_file(dest)
    except Exception as exc:
        dest.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=str(exc))

    return result


@app.post("/api/ingest/directory")
async def ingest_directory():
    """Ingest all supported documents already present in data/docs/."""
    docs_path = Path(settings.docs_path)
    if not docs_path.exists():
        raise HTTPException(status_code=404, detail="docs directory not found")
    results = rag_agent.ingest_directory(docs_path)
    return {"ingested": results, "total_files": len(results)}


@app.get("/api/kb/stats")
async def kb_stats():
    """Knowledge base statistics."""
    return vector_store.stats()


@app.delete("/api/kb")
async def clear_kb():
    """Clear the entire knowledge base (vector store only; docs are kept)."""
    vector_store.clear()
    return {"status": "knowledge base cleared"}
