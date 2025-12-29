# 🔍 Local RAG Chat API

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-18-61DAFB.svg?logo=react&logoColor=white)](https://reactjs.org/)
[![FAISS](https://img.shields.io/badge/FAISS-Vector_Search-blue.svg)](https://github.com/facebookresearch/faiss)
[![Ollama](https://img.shields.io/badge/Ollama-Local_LLM-black.svg)](https://ollama.ai/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg?logo=docker&logoColor=white)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A **production-grade, local Retrieval-Augmented Generation (RAG) system** built with **LLaMA/Mistral, FAISS, and FastAPI**. Features session-aware document management, intent-aware retrieval strategies, real-time streaming responses, and a built-in evaluation framework — all running locally without external API dependencies.

---

## 🎯 What is This?

This is a document question-answering system that lets you upload documents and ask questions about them. The AI answers based only on your document content, not from its general knowledge, which prevents hallucinations and ensures accurate responses.

When you upload a document like a resume or report, the system first extracts the text and splits it into smaller pieces called chunks. This is necessary because AI models have limited context windows and can't process very large documents at once. Each chunk is then converted into a numerical representation called an embedding using a Sentence Transformer model. These embeddings capture the semantic meaning of the text, so similar content will have similar numbers.

The embeddings are stored in a FAISS index, which is Facebook's library for fast similarity search. When you ask a question, your question is also converted into an embedding, and FAISS quickly finds the chunks most relevant to your query by comparing vector similarities. The system then sends these relevant chunks along with your question to a local LLM running through Ollama (LLaMA or Mistral). The model reads the context and generates an answer based specifically on your document.

The backend is built with FastAPI, providing REST endpoints for uploading, chatting, and streaming responses. The React frontend offers a clean interface with real-time streaming, showing tokens as they're generated. Everything runs locally on your machine, ensuring privacy since no data is sent to external servers.

---

## 🔄 How It Works

```
┌──────────────┐     ┌──────────────┐      ┌──────────────┐
│   Upload     │────▶│   Chunking   │────▶│  Embedding   │
│   Document   │     │  Split text  │      │  E5-small-v2 │
└──────────────┘     └──────────────┘      └──────┬───────┘
                                                  │
                                                  ▼
┌──────────────┐      ┌──────────────┐      ┌──────────────┐
│   Answer     │◀────│   LLaMA/     │◀──── │    FAISS     │
│   Display    │      │   Mistral    │      │   Search     │
└──────────────┘      └──────────────┘      └──────────────┘
                            ▲                     ▲
                            │                     │
                     ┌──────┴─────────────────────┴──────┐
                     │         User Question             │
                     └───────────────────────────────────┘
```

**Step-by-Step Flow:**

1. **Upload** → You upload a .docx, .txt, or .md file
2. **Chunking** → Text is split into smaller pieces (~1000 chars each)
3. **Embedding** → Each chunk becomes a 384-dimensional vector
4. **Indexing** → Vectors stored in FAISS for fast search
5. **Question** → Your question is also converted to a vector
6. **Retrieval** → FAISS finds the most similar chunks
7. **Generation** → Relevant chunks + question go to the LLM
8. **Streaming** → Answer streams back in real-time

---

## 📋 Table of Contents

- [Key Features](#-key-features)
- [Architecture](#-architecture)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Quick Start](#-quick-start)
- [API Reference](#-api-reference)
- [Retrieval Strategy](#-retrieval-strategy)
- [Evaluation Framework](#-evaluation-framework)
- [Docker Deployment](#-docker-deployment)
- [Configuration](#-configuration)
- [Troubleshooting](#-troubleshooting)
- [Known Limitations](#-known-limitations)
- [Future Roadmap](#-future-roadmap)
- [License](#-license)

---

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| **🔒 100% Local** | Runs entirely on your machine — no data sent to external servers |
| **🤖 Local LLM Inference** | LLaMA/Mistral via Ollama — no external APIs, full data privacy |
| **📁 Session-Aware Uploads** | Per-session document isolation with active-file tracking |
| **🎯 Intent-Aware Retrieval** | Adaptive strategy: full-document for summaries, semantic top-k for facts |
| **⚡ Real-Time Streaming** | Server-Sent Events (SSE) for token-by-token output |
| **🧠 Conversation Memory** | Multi-turn context retention per session |
| **📊 Evaluation Framework** | Built-in Precision, Recall, MRR, NDCG, and answer similarity metrics |
| **🌐 React UI** | Modern, responsive chat interface with dark theme |
| **🐳 Docker + GPU** | Production deployment with NVIDIA CUDA support |

---

## 🏗 Architecture

```
┌────────────────────────────────────────────────────────────────────────────┐
│                              CLIENT LAYER                                  │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│    ┌──────────────┐         ┌──────────────┐         ┌──────────────┐      │
│    │  React UI    │         │  REST Client │         │   cURL/SDK   │      │
│    │  (port 3000) │         │  (Postman)   │         │              │      │
│    └──────┬───────┘         └──────┬───────┘         └──────┬───────┘      │
│           │                        │                        │              │
└───────────┼────────────────────────┼────────────────────────┼──────────────┘
            │                        │                        │
            ▼                        ▼                        ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              API LAYER (FastAPI)                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│    ┌────────────────┐    ┌────────────────┐    ┌────────────────┐           │
│    │  POST /upload  │    │  POST /chat    │    │ GET /chat/stream│          │
│    │                │    │                │    │     (SSE)      │           │
│    └───────┬────────┘    └───────┬────────┘    └───────┬────────┘           │
│            │                     │                     │                    │
└────────────┼─────────────────────┼─────────────────────┼────────────────────┘
             │                     │                     │
             ▼                     ▼                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              RAG ENGINE                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                        DOCUMENT PROCESSING                          │    │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐               │    │
│  │  │  chunking.py │  │   ingest.py  │  │  E5 Embedder │               │    │
│  │  │  .docx/.ipynb│  │  .txt/.md    │  │  (384-dim)   │               │    │
│  │  └──────────────┘  └──────────────┘  └──────────────┘               │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                    │                                        │
│                                    ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                         RETRIEVAL LAYER                             │    │
│  │  ┌──────────────────────────────────────────────────────────────┐   │    │
│  │  │                    Intent Detection                          │   │    │
│  │  │  "summarize" / "overview" → Full Document Retrieval          │   │    │
│  │  │  Fact-based queries       → Semantic Top-K (FAISS)           │   │    │
│  │  └──────────────────────────────────────────────────────────────┘   │    │
│  │                              │                                      │    │
│  │  ┌──────────────┐  ┌────────┴───────┐  ┌──────────────┐             │    │
│  │  │ FAISS Index  │  │ Active File    │  │ Session      │             │    │
│  │  │ (IndexFlatIP)│  │ Filter         │  │ Isolation    │             │    │
│  │  └──────────────┘  └────────────────┘  └──────────────┘             │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                    │                                        │
│                                    ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                        GENERATION LAYER                             │    │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐               │    │
│  │  │  memory.py   │  │ Prompt Build │  │ Ollama Client│               │    │
│  │  │  (History)   │  │ (Context+Q)  │  │(Mistral)     │               │    │
│  │  └──────────────┘  └──────────────┘  └──────────────┘               │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              STORAGE LAYER                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│    data/                                                                    │
│    ├── raw/                    # Uploaded documents                         │
│    │   └── {filename}                                                       │
│    └── index/                  # Per-session FAISS indexes                  │
│        └── {session_id}/                                                    │
│            ├── faiss.index     # Vector embeddings                          │
│            └── meta.json       # Chunk metadata + sources                   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 🧱 Tech Stack

| Layer | Technology | Purpose |
|-------|------------|---------|
| **LLM** | LLaMA 3.2 / Mistral (via Ollama) | Local text generation |
| **Embeddings** | `intfloat/e5-small-v2` | 384-dim semantic vectors |
| **Vector Store** | FAISS (IndexFlatIP) | Similarity search |
| **Backend** | FastAPI + Uvicorn | Async API server |
| **Frontend** | React + Vite | Modern chat UI |
| **Streaming** | Server-Sent Events | Real-time token output |
| **Document Parsing** | python-docx, json | DOCX/IPYNB support |
| **Evaluation** | Sentence Transformers, RapidFuzz | Metrics scoring |
| **Containerization** | Docker + NVIDIA Container Toolkit | GPU-accelerated deployment |

---

## 📂 Project Structure

```
Local-RAG-Chat-API/
│
├── app/                          # Core application
│   ├── __init__.py               # Package init
│   ├── main.py                   # FastAPI routes & endpoints
│   ├── rag.py                    # RAG engine (retrieval + generation)
│   ├── chunking.py               # Document parsing (DOCX/IPYNB)
│   ├── ingest.py                 # Text file ingestion
│   ├── memory.py                 # Session memory & active-file tracking
│   ├── config.py                 # Pydantic settings
│   ├── ollama_client.py          # Ollama API client (sync + streaming)
│   └── evaluation.py             # Evaluation framework
│
├── ui/                           # Simple HTML frontend
│   └── index.html                # Vanilla JS chat interface
│
├── ui-react/                     # React frontend (recommended)
│   ├── src/
│   │   ├── App.jsx               # Main React component
│   │   └── main.jsx              # Entry point
│   ├── index.html
│   ├── package.json
│   ├── vite.config.js
│   └── README.md
│
├── data/                         # Runtime data (gitignored)
│   ├── raw/                      # Uploaded documents
│   └── index/                    # Per-session FAISS indexes
│
├── eval_data.jsonl               # Evaluation test cases
├── requirements.txt              # Python dependencies
├── Dockerfile                    # Container build
├── docker-compose.yml            # GPU orchestration
├── .gitignore                    # Git exclusions
├── .env.example                  # Environment template
└── README.md                     # Documentation
```

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.10 - 3.12** (Python 3.13+ not yet supported by PyTorch)
- **Node.js 18+** (for React UI)
- [Ollama](https://ollama.ai/) installed and running

### 1️⃣ Install Ollama & Model

```bash
# Install Ollama from https://ollama.ai/

# Pull a model
ollama pull llama3.2
# or
ollama pull mistral

# Start Ollama server
ollama serve
```

### 2️⃣ Clone & Setup Environment

```bash
git clone https://github.com/JeevanReddy0828/Local-RAG-Chat-API.git
cd Local-RAG-Chat-API
```

**Linux/macOS:**
```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

**Windows (PowerShell):**
```powershell
py -3.11 -m venv .venv311
.venv311\Scripts\activate
```

### 3️⃣ Install Dependencies

```bash
# Install PyTorch (GPU)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install remaining dependencies
pip install -r requirements.txt
```

### 4️⃣ Configure Environment

Create a `.env` file:

```env
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2
```

> **Note:** If Ollama runs on a different port (e.g., 11435), update the URL accordingly.

### 5️⃣ Start Backend

```bash
uvicorn app.main:app --reload
```

Backend runs at: **http://127.0.0.1:8000**

### 6️⃣ Start React Frontend

```bash
cd ui-react
npm install
npm run dev
```

Frontend runs at: **http://localhost:3000**

### 7️⃣ Use It!

1. Open http://localhost:3000
2. Upload a document (.docx, .txt, .md)
3. Ask questions like "summarize" or "what skills are mentioned?"

---

## 📡 API Reference

### Upload Document

```http
POST /upload
Content-Type: multipart/form-data
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `session_id` | string | Unique session identifier |
| `file` | file | Document (.docx, .ipynb, .txt, .md) |

### Chat (Synchronous)

```http
POST /chat
Content-Type: application/json

{
  "session_id": "user123",
  "query": "What skills are mentioned?"
}
```

### Chat (Streaming)

```http
GET /chat/stream?session_id=user123&query=summarize
Accept: text/event-stream
```

### Other Endpoints

- `GET /health` — Health check
- `GET /stats/{session_id}` — Session statistics
- `POST /index/clear?session_id=xxx` — Clear index
- `POST /memory/clear?session_id=xxx` — Clear memory

---

## 🎯 Retrieval Strategy

The system uses **intent-aware retrieval**:

| Query Pattern | Strategy |
|---------------|----------|
| "Summarize", "Overview" | Top chunks from active file |
| Fact-based queries | Semantic Top-K via FAISS |

---

## 📊 Evaluation Framework

Run evaluation with built-in metrics:

```bash
python -m app.evaluation --eval-file eval_data.jsonl --output results.json
```

**Metrics:** Precision@K, Recall@K, MRR, NDCG, Answer Relevance, Faithfulness, Task Success Rate

---

## ⚙️ Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama endpoint |
| `OLLAMA_MODEL` | `llama3.2` | Model name |
| `EMBED_MODEL` | `intfloat/e5-small-v2` | Embedding model |
| `TOP_K` | `4` | Chunks to retrieve |
| `CHUNK_MAX_CHARS` | `1000` | Max chunk size |
| `MAX_NEW_TOKENS` | `300` | Generation limit |
| `TEMPERATURE` | `0.3` | Sampling temperature |

---

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| "Connection refused" | Ensure Ollama is running: `ollama serve` |
| Word repetition | Use latest `App.jsx` or disable streaming |
| Slow responses | First request loads model; use smaller model |
| Module not found | Activate venv, run `pip install -r requirements.txt` |

---

## 🗺 Future Roadmap

- [ ] PDF Support
- [ ] Hybrid Retrieval (BM25 + FAISS)
- [ ] Multi-Document Selection
- [ ] Source Highlighting in UI
- [ ] Authentication
- [ ] Redis Session Store

---

## 🧑‍💻 Author

**Jeevan Reddy**  
Software Engineer | ML/NLP Enthusiast

[![GitHub](https://img.shields.io/badge/GitHub-Follow-black)](https://github.com/JeevanReddy0828)


## 📜 License

MIT License — see [LICENSE](LICENSE) for details.
