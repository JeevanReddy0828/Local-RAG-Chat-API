# 🔍 Local RAG Chat API

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/downloads/)
[![PyTorch 2.4](https://img.shields.io/badge/PyTorch-2.4-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![CUDA 12.1](https://img.shields.io/badge/CUDA-12.1-76B900.svg?logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![FAISS](https://img.shields.io/badge/FAISS-Vector_Search-blue.svg)](https://github.com/facebookresearch/faiss)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg?logo=docker&logoColor=white)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A **production-grade, local Retrieval-Augmented Generation (RAG) system** built with **Mistral-7B, FAISS, and FastAPI**. Features session-aware document management, intent-aware retrieval strategies, real-time streaming responses, and a built-in evaluation framework — all running locally without external API dependencies.

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
- [Known Limitations](#-known-limitations)
- [Future Roadmap](#-future-roadmap)
- [License](#-license)

---

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| **🤖 Local LLM Inference** | Mistral-7B via Ollama — no external APIs, full data privacy |
| **📁 Session-Aware Uploads** | Per-session document isolation with active-file tracking |
| **🎯 Intent-Aware Retrieval** | Adaptive strategy: full-document for summaries, semantic top-k for facts |
| **⚡ Real-Time Streaming** | Server-Sent Events (SSE) for token-by-token output |
| **🧠 Conversation Memory** | Multi-turn context retention per session |
| **📊 Evaluation Framework** | Built-in Recall@K and answer similarity metrics |
| **🐳 Docker + GPU Ready** | Production deployment with NVIDIA CUDA support |
| **🌐 Web UI ** | Clean, basic functional chat interface |

---

## 🏗 Architecture

```
┌────────────────────────────────────────────────────────────────────────────┐
│                              CLIENT LAYER                                  │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│    ┌──────────────┐         ┌──────────────┐         ┌──────────────┐      │
│    │   Web UI     │         │  REST Client │         │   cURL/SDK   │      │
│    │  (index.html)│         │  (Postman)   │         │              │      │
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
│  │  │  (History)   │  │ (Context+Q)  │  │ (Mistral-7B) │               │    │
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
| **LLM** | Mistral-7B (via Ollama) | Local text generation |
| **Deep Learning** | PyTorch 2.4 + CUDA 12.1 | GPU-accelerated inference |
| **Embeddings** | `intfloat/e5-small-v2` | 384-dim semantic vectors |
| **Vector Store** | FAISS (IndexFlatIP) | Similarity search |
| **Backend** | FastAPI + Uvicorn | Async API server |
| **Streaming** | Server-Sent Events | Real-time token output |
| **Document Parsing** | python-docx, json | DOCX/IPYNB support |
| **Evaluation** | RapidFuzz | Answer similarity scoring |
| **Containerization** | Docker + NVIDIA Container Toolkit | GPU-accelerated deployment |

---

## 📂 Project Structure

```
hf-rag-api/
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
│   └── eval.py                   # Evaluation framework
│
├── ui/                           # Frontend
│   └── index.html                # Streaming chat interface
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

- Python 3.10+
- [Ollama](https://ollama.ai/) with Mistral model
- **For GPU acceleration:**
  - NVIDIA GPU with CUDA support
  - CUDA 12.1+ and cuDNN 9+
  - PyTorch 2.4+ with CUDA support
  - [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) (for Docker)

### 1️⃣ Clone & Setup Environment

```bash
git clone https://github.com/yourusername/hf-rag-api.git
cd hf-rag-api

python -m venv .venv
source .venv/bin/activate    # Linux/macOS
# .venv\Scripts\activate     # Windows
```

### 2️⃣ Install Dependencies

**For GPU (CUDA 12.1) - Recommended:**
```bash
# Install PyTorch with CUDA support first
pip install torch --index-url https://download.pytorch.org/whl/cu121

# Install remaining dependencies
pip install -r requirements.txt

# Verify CUDA is available
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"
```

**For CPU only:**
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

### 3️⃣ Start Ollama with Mistral

```bash
# Install Ollama (if not installed)
curl -fsSL https://ollama.ai/install.sh | sh

# Pull and run Mistral
ollama pull mistral
ollama serve  # Runs on port 11434 by default
```

### 3️⃣ Configure (Optional)

```bash
cp .env.example .env
# Edit .env to customize ports, models, etc.
```

### 4️⃣ Run the Server

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 5️⃣ Open the UI

Navigate to **http://localhost:8000** → auto-redirects to the chat UI.

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

**Response:**
```json
{
  "session_id": "user123",
  "file": "resume.docx",
  "type": "docx",
  "chunks_added": 8,
  "active_file": "resume.docx"
}
```

### Chat (Synchronous)

```http
POST /chat
Content-Type: application/json
```

**Request:**
```json
{
  "session_id": "user123",
  "query": "What skills are mentioned in the document?"
}
```

**Response:**
```json
{
  "answer": "The document mentions Python, FastAPI, and machine learning skills...",
  "sources": [
    {"source": "resume.docx", "best_score": 0.847}
  ],
  "active_file": "resume.docx"
}
```

### Chat (Streaming)

```http
GET /chat/stream?session_id=user123&query=Summarize%20the%20document
Accept: text/event-stream
```

**Response (SSE):**
```
data: [START]
data: The document is a professional resume...
data: It highlights experience in...
data: [END]
```

### Clear Session Index

```http
POST /index/clear?session_id=user123
```

### Clear Session Memory

```http
POST /memory/clear?session_id=user123
```

---

## 🎯 Retrieval Strategy

The system uses **intent-aware retrieval** to optimize context selection:

| Query Pattern | Detection | Retrieval Strategy |
|---------------|-----------|-------------------|
| "Summarize the document" | Keyword match | **All chunks** from active file |
| "What is this document about?" | Keyword match | **All chunks** from active file |
| "Overview of the document" | Keyword match | **All chunks** from active file |
| "What skills are mentioned?" | Semantic | **Top-K** (default: 4) via FAISS |
| "Where is Python used?" | Semantic | **Top-K** with active-file filter |

### Intent Detection Triggers

```python
DOC_LEVEL_TRIGGERS = [
    "summary of the document",
    "summarize the document",
    "summarize this document",
    "what is this document",
    "describe this document",
    "what kind of document",
    "overview of the document",
    "summary",
]
```

### Why This Matters

| Problem | Solution |
|---------|----------|
| Empty summaries from top-k only | Full-document retrieval for summary queries |
| Cross-document contamination | Active-file filtering |
| Hallucinated context | Strict source scoping |

---

## 📊 Evaluation Framework

Built-in evaluation with Recall@K and answer similarity metrics.

### Prepare Test Data

Create `eval_data.jsonl`:
```jsonl
{"question": "What programming languages are mentioned?", "expected_answer": "Python, JavaScript, and SQL", "expected_source": "resume.docx"}
{"question": "Summarize the document", "expected_answer": "A professional resume highlighting software engineering experience", "expected_source": "resume.docx"}
```

### Run Evaluation

```bash
python -m app.eval
```

### Output

```
---
Q: What programming languages are mentioned?
Expected source: resume.docx
Retrieved sources: ['resume.docx']
Recall hit: True
Similarity: 78
Answer: The document mentions Python, JavaScript, SQL...

==========================
Total samples: 2
Recall@K: 1.0
Avg answer similarity: 82.5
==========================
```

---

## 🐳 Docker Deployment

### Prerequisites for GPU

```bash
# Verify NVIDIA driver
nvidia-smi

# Install NVIDIA Container Toolkit (if not installed)
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

### GPU-Accelerated (Recommended)

```bash
# Ensure NVIDIA Container Toolkit is installed
docker compose up --build
```

### CPU-Only

```bash
# Modify docker-compose.yml to remove GPU reservation
docker compose up --build
```

### Docker Compose Configuration

```yaml
services:
  rag-api:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./data:/app/data
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
```

---

## ⚙️ Configuration

All settings via environment variables or `.env` file:

| Variable | Default | Description |
|----------|---------|-------------|
| `DATA_DIR` | `data` | Base data directory |
| `RAW_DIR` | `data/raw` | Uploaded documents |
| `INDEX_DIR` | `data/index` | FAISS indexes |
| `EMBED_MODEL` | `intfloat/e5-small-v2` | Embedding model |
| `TOP_K` | `4` | Retrieval count |
| `CHUNK_MAX_CHARS` | `1400` | Max chunk size |
| `CHUNK_OVERLAP_CHARS` | `250` | Chunk overlap |
| `OLLAMA_ENABLED` | `true` | Use Ollama |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama endpoint |
| `OLLAMA_MODEL` | `mistral` | LLM model |
| `MAX_NEW_TOKENS` | `256` | Generation limit |
| `TEMPERATURE` | `0.2` | Sampling temperature |

---

## ⚠️ Known Limitations

| Limitation | Mitigation |
|------------|------------|
| Single-node FAISS | Suitable for demo/small-scale; use Pinecone/Weaviate for production |
| No authentication | Add OAuth2/JWT for production deployment |
| In-memory session state | Add Redis for horizontal scaling |
| Ollama dependency | Ensure Ollama is running before API starts |

---

## 🗺 Future Roadmap

- [ ] **Hybrid Retrieval** — BM25 + FAISS fusion
- [ ] **Multi-Document Selection** — Query across multiple active files
- [ ] **Chunk-Level Citations** — UI displays source chunks with highlights
- [ ] **Redis Session Store** — Horizontal scaling support
- [ ] **Authentication** — OAuth2 / API key support
- [ ] **Observability** — OpenTelemetry tracing + Prometheus metrics
- [ ] **PDF Support** — PyMuPDF integration
- [ ] **Reranking** — Cross-encoder reranking for improved precision

---

## 🧑‍💻 Author

**Jeevan Reddy**  
Software Engineer | ML/NLP Enthusiast

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue)](https://linkedin.com/in/yourprofile)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black)](https://github.com/yourusername)

---
## 📌 Resume Bullet Point

> Designed and built a **production-grade local RAG system** using **PyTorch, CUDA, Mistral-7B, FAISS, and FastAPI** with **intent-aware retrieval** (full-document vs. semantic top-k), **session isolation**, **SSE streaming**, and a **built-in evaluation framework** — demonstrating GPU-accelerated ML inference and production patterns for document Q&A without external API dependencies.

---

## 📜 License

MIT License — see [LICENSE](LICENSE) for details.
