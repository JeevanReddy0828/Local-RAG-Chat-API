# Hugging Face RAG Chat API 🚀

A simple **LLM-powered Retrieval-Augmented Generation (RAG) API** built **from scratch** using **Hugging Face open-source models**.  
You can upload documents, index them locally, and chat with an LLM that answers questions using your data.

This project runs **fully locally** (no OpenAI, no paid APIs).

---

## ✨ Features

- 📄 Upload `.txt` or `.md` documents
- 🔍 Semantic search using embeddings + FAISS
- 🤖 Question answering with Hugging Face LLMs
- 📚 Source-aware answers (shows which document was used)
- ⚡ FastAPI backend with Swagger UI
- 🧠 Fully local inference (CPU friendly by default)

---

## 🧠 Architecture (High Level)

hf-rag-api/
│
├── app/
│ ├── main.py # FastAPI app
│ ├── rag.py # RAG logic (FAISS + LLM)
│ ├── ingest.py # Document chunking
│ └── config.py # Central config
│
├── data/
│ ├── raw/ # Uploaded documents
│ └── index/ # FAISS index + metadata
│
├── requirements.txt
└── README.md

---

## ⚙️ Setup Instructions

### 1️⃣ Create virtual environment

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

## Install dependencies
```bash
pip install -r requirements.txt
```

## Run the Application
```bash
uvicorn app.main:app --reload
```

##API will be available at:

http://127.0.0.1:8000


###Swagger UI:
http://127.0.0.1:8000/docs

## Upload a Document

Use /upload endpoint in Swagger UI or via curl:
```bash
curl -X POST "http://127.0.0.1:8000/upload" \
  -F "file=@example.txt"

###Supported formats:
-.txt
-.md


## Upload a Document

Endpoint: /chat

Example request:


