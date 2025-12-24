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
| **🌐 Web UI Included** | Clean, functional chat interface out of the box |

---

## 🏗 Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              CLIENT LAYER                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
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
│    ┌────────────────┐    ┌────────────────┐    ┌────────────────┐          │
│    │  POST /upload  │    │  POST /chat    │    │ GET /chat/stream│          │
│    │                │    │                │    │     (SSE)      │          │
│    └───────┬────────┘    └───────┬────────┘    └───────┬────────┘          │
│            │                     │                     │                    │
└────────────┼─────────────────────┼─────────────────────┼────────────────────┘
             │                     │                     │
             ▼                     ▼                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              RAG ENGINE                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                        DOCUMENT PROCESSING                          │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │   │
│  │  │  chunking.py │  │   ingest.py  │  │  E5 Embedder │              │   │
│  │  │  .docx/.ipynb│  │  .txt/.md    │  │  (384-dim)   │              │   │
│  │  └──────────────┘  └──────────────┘  └──────────────┘              │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│                                    ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                         RETRIEVAL LAYER                             │   │
│  │  ┌──────────────────────────────────────────────────────────────┐  │   │
│  │  │                    Intent Detection                          │  │   │
│  │  │  "summarize" / "overview" → Full Document Retrieval          │  │   │
│  │  │  Fact-based queries       → Semantic Top-K (FAISS)           │  │   │
│  │  └──────────────────────────────────────────────────────────────┘  │   │
│  │                              │                                      │   │
│  │  ┌──────────────┐  ┌────────┴───────┐  ┌──────────────┐           │   │
│  │  │ FAISS Index  │  │ Active File    │  │ Session      │           │   │
│  │  │ (IndexFlatIP)│  │ Filter         │  │ Isolation    │           │   │
│  │  └──────────────┘  └────────────────┘  └──────────────┘           │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│                                    ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                        GENERATION LAYER                             │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │   │
│  │  │  memory.py   │  │ Prompt Build │  │ Ollama Client│              │   │
│  │  │  (History)   │  │ (Context+Q)  │  │ (Mistral-7B) │              │   │
│  │  └──────────────┘  └──────────────┘  └──────────────┘              │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
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

- **Python 3.9 - 3.12** (Python 3.13+ not yet supported by PyTorch)
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
```

**Linux/macOS:**
```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

**Windows (PowerShell):**
```powershell
# Check available Python versions
py --list

# Create venv with Python 3.11 (or 3.12)
py -3.11 -m venv .venv311
.venv311\Scripts\activate
```

> ⚠️ **Important:** PyTorch requires Python 3.9-3.12. If you have Python 3.13+, you must specify an older version when creating the virtual environment.

### 2️⃣ Install Dependencies

**For GPU (CUDA 12.1) - Recommended:**
```bash
# Install PyTorch with CUDA support first
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install remaining dependencies
pip install -r requirements.txt

# Verify CUDA is available
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"
```

**For CPU only:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

### 3️⃣ Start Ollama with Mistral

**Linux/macOS:**
```bash
# Install Ollama (if not installed)
curl -fsSL https://ollama.ai/install.sh | sh

# Pull and run Mistral
ollama pull mistral
ollama serve  # Runs on port 11434 by default
```

**Windows:**
```powershell
# Download and install from https://ollama.ai/download/windows
# Then:
ollama pull mistral
ollama serve
```

> 📝 **Note:** Ollama may run on port `11434` or `11435`. Check your Ollama output and update `.env` if needed:
> ```
> OLLAMA_BASE_URL=http://localhost:11435
> ```

### 4️⃣ Configure (Optional)

```bash
cp .env.example .env
# Edit .env to customize ports, models, etc.
```

### 5️⃣ Run the Server

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 6️⃣ Open the UI

**Option A: Simple HTML UI**

Navigate to **http://localhost:8000** → auto-redirects to the basic chat UI.

**Option B: React UI (Recommended)**

```bash
cd ui-react
npm install
npm run dev
```

Opens at **http://localhost:3000** with:
- 🌙 Dark theme minimal design
- 📱 Fully responsive
- ⚡ Real-time streaming
- 📊 Index statistics
- 💾 Session management

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

Comprehensive evaluation with **intrinsic** (retrieval quality) and **extrinsic** (answer quality) metrics.

### Metrics Overview

| Category | Metric | Description |
|----------|--------|-------------|
| **Intrinsic** | Precision@K | Fraction of retrieved docs that are relevant |
| | Recall@K | Fraction of relevant docs that are retrieved |
| | MRR | Mean Reciprocal Rank - position of first relevant result |
| | NDCG | Normalized Discounted Cumulative Gain - ranking quality |
| | Similarity Stats | Embedding similarity distribution |
| **Extrinsic** | Answer Relevance | How well answer addresses the question (0-100) |
| | Faithfulness | Is answer grounded in retrieved context (0-100) |
| | Answer Similarity | Fuzzy match with expected answer (0-100) |
| | Task Success Rate | Binary success/fail for specific task types |
| | Hallucination Score | Facts not in source documents (lower is better) |

### Prepare Test Data

Create `eval_data.jsonl`:
```jsonl
{"question": "What programming languages are mentioned?", "expected_answer": "Python, JavaScript, SQL", "task_type": "factual", "difficulty": "easy"}
{"question": "Summarize the document", "expected_answer": "A professional resume for a software engineer", "task_type": "summary", "difficulty": "medium"}
{"question": "What certifications does the person have?", "expected_answer": "AWS Machine Learning Associate", "task_type": "extraction", "difficulty": "easy"}
```

### Run Evaluation

```bash
# Basic evaluation
python -m app.evaluation --eval-file eval_data.jsonl

# Save results to JSON
python -m app.evaluation --eval-file eval_data.jsonl --output results.json
```

### Sample Output

```
================================================================================
RAG SYSTEM EVALUATION REPORT
================================================================================
Timestamp: 2024-12-20T15:30:00
Samples Evaluated: 10

----------------------------------------
INTRINSIC METRICS (Retrieval Quality)
----------------------------------------
  Precision@K:      0.8500
  Recall@K:         0.9000
  MRR:              0.9200
  NDCG:             0.8800
  Avg Similarity:   0.7650

----------------------------------------
EXTRINSIC METRICS (Answer Quality)
----------------------------------------
  Answer Relevance: 82.50/100
  Faithfulness:     78.30/100
  Answer Similarity:71.20/100
  Task Success Rate:80.00%
  Hallucination:    21.70/100 (lower is better)
  Avg Latency:      1250ms

----------------------------------------
METRICS BY TASK TYPE
----------------------------------------
  [SUMMARY] (n=3)
    Precision:    0.9000
    Success Rate: 100.00%

  [FACTUAL] (n=4)
    Precision:    0.8500
    Success Rate: 75.00%

  [EXTRACTION] (n=3)
    Precision:    0.8000
    Success Rate: 66.67%
================================================================================
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