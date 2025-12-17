import os

from fastapi import FastAPI, UploadFile, File, Form, Query
from fastapi.responses import JSONResponse, StreamingResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .config import settings
from .rag import RagStore
from .memory import clear_session, set_active_file
from .chunking import load_text_for_file, chunk_text

app = FastAPI(title="Local RAG Chat API", version="1.0.0")

app.mount("/ui", StaticFiles(directory="ui", html=True), name="ui")


@app.get("/")
def root():
    return RedirectResponse(url="/ui")


rag = RagStore()
os.makedirs(settings.raw_dir, exist_ok=True)
os.makedirs(settings.index_dir, exist_ok=True)


class ChatRequest(BaseModel):
    session_id: str
    query: str


@app.post("/upload")
async def upload(
    session_id: str = Form(...),
    file: UploadFile = File(...),
):
    filename = file.filename or "uploaded.txt"
    save_path = os.path.join(settings.raw_dir, filename)

    contents = await file.read()
    with open(save_path, "wb") as f:
        f.write(contents)

    text, kind = load_text_for_file(save_path)
    chunks = chunk_text(
        text,
        max_chars=settings.chunk_max_chars,
        overlap_chars=settings.chunk_overlap_chars,
    )

    added = rag.add_texts(session_id=session_id, chunks=chunks, source=filename)
    set_active_file(session_id, filename)

    return {
        "session_id": session_id,
        "file": filename,
        "type": kind,
        "chunks_added": added,
        "active_file": filename,
    }


@app.post("/chat")
def chat(req: ChatRequest):
    out = rag.answer(req.session_id, req.query)
    return JSONResponse(out)


@app.get("/chat/stream")
def chat_stream(session_id: str = Query(...), query: str = Query(...)):
    def event_gen():
        yield "data: [START]\n\n"

        for chunk in rag.answer_stream(session_id, query):
            safe_chunk = chunk.replace("\n", "\\n")
            yield f"data: {safe_chunk}\n\n"

        yield "data: [END]\n\n"

    return StreamingResponse(event_gen(), media_type="text/event-stream")



@app.post("/memory/clear")
def memory_clear(session_id: str = Query(...)):
    clear_session(session_id)
    return {"ok": True, "session_id": session_id}


@app.post("/index/clear")
def index_clear(session_id: str = Query(...)):
    """
    Clears FAISS + meta for ONLY this session.
    """
    rag.clear_session_index(session_id)
    clear_session(session_id)
    return {"ok": True, "session_id": session_id, "cleared": True}
