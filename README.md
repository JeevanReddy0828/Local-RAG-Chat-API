# Local RAG Chat API (Session-Aware, Streaming)

A **local Retrieval-Augmented Generation (RAG) system** built with **Mistral-7B, FAISS, and FastAPI**, supporting **session-aware document uploads, intent-aware retrieval, and real-time streaming responses** — without relying on external LLM APIs.

This project demonstrates **production-grade RAG design patterns** such as document isolation, active-file tracking, adaptive retrieval strategies, and streaming inference.

---

## 🚀 Features

- Local LLM inference using **Mistral-7B (Hugging Face Transformers)**
- **Session-aware document uploads** with active-file tracking
- **Intent-aware retrieval**
  - Full-document retrieval for summaries and document-level questions
  - Semantic top-k retrieval for fact-based queries
- **FAISS vector search** with SentenceTransformers (E5)
- Robust **document chunking**
  - Supports `.docx` and `.ipynb`
- **Real-time token streaming** using Server-Sent Events (SSE)
- Conversation memory per session
- CPU / GPU compatible with quantized models
- Optional **Ollama fallback** for local inference
- No external APIs required

---

## 🧠 Architecture

```
Client (UI)
    │
    ▼
FastAPI
├── Upload Endpoint
│   └── Chunking + Embeddings
│       └── FAISS Index
│
├── Chat Endpoint
│   ├── Session Memory
│   ├── Active Document Resolver
│   ├── Intent-Aware Retrieval
│   └── Prompt Construction
│
└── Streaming Endpoint (SSE)
    └── Token-by-token output
```

---

## 🧱 Tech Stack

- **LLM**: Mistral-7B (Transformers)
- **Embeddings**: `intfloat/e5-small-v2`
- **Vector Store**: FAISS
- **Backend**: FastAPI, Uvicorn
- **Streaming**: Server-Sent Events (SSE)
- **Chunking**: Custom logic for DOCX / IPYNB
- **Optional**: Ollama for fallback inference

---

## 📂 Project Structure

```
hf-rag-api/
├── app/
│   ├── main.py              # API routes + streaming
│   ├── rag.py               # RAG logic (retrieval + generation)
│   ├── chunking.py          # File loading and chunking
│   ├── memory.py            # Session memory & active-file tracking
│   ├── config.py            # Configuration
│   └── ollama_client.py     # Optional Ollama fallback
│
├── index/                   # FAISS index (generated at runtime)
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 🛠 Setup

### 1️⃣ Create virtual environment

```bash
python -m venv .venv
source .venv/bin/activate    # macOS / Linux
.venv\Scripts\activate       # Windows
```

### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ (Optional) Set Hugging Face cache location

```bash
# Windows
setx HF_HOME E:\hf_cache

# macOS / Linux
export HF_HOME=/path/to/hf_cache
```

### 4️⃣ Run the server

```bash
uvicorn app.main:app --reload
```

Server runs at:

```
http://127.0.0.1:8000
```

---

## 📤 Upload a Document (Session-Aware)

```http
POST /upload
```

**Parameters**

| Name         | Description                 |
| ------------ | --------------------------- |
| `session_id` | Unique session identifier   |
| `file`       | `.docx` or `.ipynb` file    |

**Example response**

```json
{
  "file": "resume.docx",
  "type": "docx",
  "chunks_added": 6,
  "active_file": "resume.docx"
}
```

Uploading a new document automatically updates the **active document** for that session.

---

## 💬 Chat with the Document

```http
GET /chat
```

**Query Parameters**

| Name         | Description               |
| ------------ | ------------------------- |
| `session_id` | Unique session identifier |
| `query`      | Your question             |

### Example queries

- "Summarize the document"
- "What skills are mentioned?"
- "Explain the projects section"

---

## ⚡ Streaming Chat (SSE)

```http
GET /chat/stream
```

Streams tokens in real time:

```
data: [START]
data: This document is a professional resume...
data: It highlights experience in...
data: [END]
```

---

## 🧠 Retrieval Strategy

| Query Type                       | Retrieval Behavior          |
| -------------------------------- | --------------------------- |
| "Summary of the document"        | All chunks from active file |
| "What kind of document is this?" | All chunks from active file |
| "What tools are mentioned?"      | Semantic top-k              |
| "Where is AWS used?"             | Semantic + file filter      |

This design prevents:

- Cross-document leakage
- Empty summaries
- Hallucinated context

---

## 🔒 Session Isolation

- Each session maintains:
  - Independent conversation memory
  - An active document
- Queries are **strictly scoped** to the active document
- Uploading a new file updates the session context automatically

---

## 🧪 Known Limitations

- Large models may require CPU offloading or GPU memory
- FAISS index is local (single-node)
- Authentication not included (demo-focused)

---

## 🌱 Future Improvements

- Multi-document selection per session
- Hybrid retrieval (BM25 + FAISS)
- Chunk-level citations in UI
- RAG evaluation metrics
- Docker + GPU deployment
- Full web UI with document explorer


## 📜 License

MIT License