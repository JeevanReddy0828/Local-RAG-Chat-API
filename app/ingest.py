import os
from .config import settings


def chunk_text(text: str, chunk_size: int, overlap: int):
    text = text.replace("\r\n", "\n").strip()
    if not text:
        return []

    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - overlap
        if start < 0:
            start = 0
        if end == len(text):
            break
    return chunks


def load_raw_files(raw_dir="data/raw"):
    files = []
    if not os.path.exists(raw_dir):
        return files

    for name in os.listdir(raw_dir):
        path = os.path.join(raw_dir, name)
        if os.path.isfile(path) and (name.endswith(".txt") or name.endswith(".md")):
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                files.append((name, f.read()))
    return files
