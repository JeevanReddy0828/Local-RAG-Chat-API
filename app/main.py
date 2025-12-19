"""
FastAPI application for Local RAG Chat API.

Endpoints:
- POST /upload - Upload and index documents
- POST /chat - Synchronous chat
- GET /chat/stream - Streaming chat (SSE)
- POST /memory/clear - Clear session memory
- POST /index/clear - Clear session index
- GET /health - Health check
- GET /stats/{session_id} - Session statistics
"""

import os
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, Form, Query, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .config import settings
from .rag import RagStore
from .memory import clear_session, set_active_file, get_session_count
from .chunking import load_text_for_file, chunk_text
from .ollama_client import check_ollama_health, OllamaError

# ═══════════════════════════════════════════════════════════════════════════════
# LOGGING SETUP
# ═══════════════════════════════════════════════════════════════════════════════

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# APP LIFECYCLE
# ═══════════════════════════════════════════════════════════════════════════════


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown logic."""
    # Startup
    logger.info("=" * 60)
    logger.info("Starting Local RAG Chat API")
    logger.info(f"Ollama URL: {settings.ollama_base_url}")
    logger.info(f"Ollama Model: {settings.ollama_model}")
    logger.info(f"Embedding Model: {settings.embed_model}")
    logger.info("=" * 60)
    
    # Check Ollama health
    if settings.ollama_enabled:
        if check_ollama_health():
            logger.info("✓ Ollama is healthy")
        else:
            logger.warning("✗ Ollama is not reachable - generation will fail")
    
    # Ensure directories exist
    os.makedirs(settings.raw_dir, exist_ok=True)
    os.makedirs(settings.index_dir, exist_ok=True)
    logger.info(f"Data directory: {settings.data_dir}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Local RAG Chat API")


# ═══════════════════════════════════════════════════════════════════════════════
# FASTAPI APP
# ═══════════════════════════════════════════════════════════════════════════════

app = FastAPI(
    title="Local RAG Chat API",
    description="Session-aware RAG system with intent-aware retrieval and streaming",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware (adjust origins for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static UI
ui_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "ui")
if os.path.exists(ui_dir):
    app.mount("/ui", StaticFiles(directory=ui_dir, html=True), name="ui")
    logger.info(f"Mounted UI from {ui_dir}")

# Initialize RAG store
rag = RagStore()


# ═══════════════════════════════════════════════════════════════════════════════
# MODELS
# ═══════════════════════════════════════════════════════════════════════════════


class ChatRequest(BaseModel):
    """Request body for chat endpoint."""
    session_id: str = Field(..., description="Unique session identifier")
    query: str = Field(..., description="User question or query")


class UploadResponse(BaseModel):
    """Response for upload endpoint."""
    session_id: str
    file: str
    type: str
    chunks_added: int
    active_file: str


class ChatResponse(BaseModel):
    """Response for chat endpoint."""
    answer: str
    sources: list
    active_file: str | None


class HealthResponse(BaseModel):
    """Response for health endpoint."""
    status: str
    ollama_healthy: bool
    active_sessions: int


# ═══════════════════════════════════════════════════════════════════════════════
# ROUTES
# ═══════════════════════════════════════════════════════════════════════════════


@app.get("/", include_in_schema=False)
def root():
    """Redirect root to UI."""
    return RedirectResponse(url="/ui")


@app.get("/health", response_model=HealthResponse, tags=["System"])
def health_check():
    """
    Health check endpoint.
    
    Returns system status including Ollama connectivity.
    """
    ollama_ok = check_ollama_health() if settings.ollama_enabled else False
    
    return HealthResponse(
        status="healthy",
        ollama_healthy=ollama_ok,
        active_sessions=get_session_count(),
    )


@app.post("/upload", response_model=UploadResponse, tags=["Documents"])
async def upload_document(
    session_id: str = Form(..., description="Unique session identifier"),
    file: UploadFile = File(..., description="Document to upload"),
):
    """
    Upload and index a document.
    
    Supported formats: .docx, .ipynb, .txt, .md
    
    The uploaded document becomes the active document for the session.
    """
    filename = file.filename or "uploaded.txt"
    save_path = os.path.join(settings.raw_dir, filename)

    logger.info(f"Upload: session={session_id}, file={filename}")

    # Save file
    try:
        contents = await file.read()
        with open(save_path, "wb") as f:
            f.write(contents)
    except Exception as e:
        logger.error(f"Failed to save file: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")

    # Parse and chunk
    try:
        text, kind = load_text_for_file(save_path)
        chunks = chunk_text(
            text,
            max_chars=settings.chunk_max_chars,
            overlap_chars=settings.chunk_overlap_chars,
        )
    except Exception as e:
        logger.error(f"Failed to parse file: {e}")
        raise HTTPException(status_code=400, detail=f"Failed to parse file: {str(e)}")

    if not chunks:
        raise HTTPException(status_code=400, detail="No content could be extracted from file")

    # Index chunks
    added = rag.add_texts(session_id=session_id, chunks=chunks, source=filename)
    set_active_file(session_id, filename)

    return UploadResponse(
        session_id=session_id,
        file=filename,
        type=kind,
        chunks_added=added,
        active_file=filename,
    )


@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
def chat(req: ChatRequest):
    """
    Chat with documents (synchronous).
    
    Returns complete response after generation finishes.
    """
    logger.info(f"Chat: session={req.session_id}, query='{req.query[:50]}...'")
    
    try:
        result = rag.answer(req.session_id, req.query)
        return ChatResponse(
            answer=result["answer"],
            sources=result["sources"],
            active_file=result["active_file"],
        )
    except OllamaError as e:
        logger.error(f"Ollama error: {e}")
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/chat/stream", tags=["Chat"])
def chat_stream(
    session_id: str = Query(..., description="Unique session identifier"),
    query: str = Query(..., description="User question or query"),
):
    """
    Chat with documents (streaming via SSE).
    
    Returns tokens as they are generated using Server-Sent Events.
    
    Event format:
    - `data: [START]` - Generation started
    - `data: <token>` - Generated tokens
    - `data: [END]` - Generation complete
    """
    logger.info(f"Stream: session={session_id}, query='{query[:50]}...'")

    def event_generator():
        yield "data: [START]\n\n"

        try:
            for chunk in rag.answer_stream(session_id, query):
                # Escape newlines for SSE format
                safe_chunk = chunk.replace("\n", "\\n")
                yield f"data: {safe_chunk}\n\n"
        except OllamaError as e:
            logger.error(f"Stream error: {e}")
            yield f"data: [ERROR] {str(e)}\n\n"
        except Exception as e:
            logger.error(f"Stream error: {e}")
            yield f"data: [ERROR] Generation failed\n\n"

        yield "data: [END]\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


@app.post("/memory/clear", tags=["Session"])
def clear_memory(session_id: str = Query(..., description="Session to clear")):
    """
    Clear conversation memory for a session.
    
    Preserves the document index.
    """
    clear_session(session_id)
    logger.info(f"Memory cleared: session={session_id}")
    
    return {"ok": True, "session_id": session_id}


@app.post("/index/clear", tags=["Session"])
def clear_index(session_id: str = Query(..., description="Session to clear")):
    """
    Clear document index and memory for a session.
    
    Removes all indexed documents and conversation history.
    """
    rag.clear_session_index(session_id)
    clear_session(session_id)
    logger.info(f"Index cleared: session={session_id}")
    
    return {"ok": True, "session_id": session_id, "cleared": True}


@app.get("/stats/{session_id}", tags=["Session"])
def get_stats(session_id: str):
    """
    Get statistics for a session.
    
    Returns chunk counts and source information.
    """
    stats = rag.get_session_stats(session_id)
    return stats
