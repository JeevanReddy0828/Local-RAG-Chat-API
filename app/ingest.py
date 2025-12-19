"""
Legacy text file ingestion module.

Note: For new code, prefer using chunking.py which supports
more file formats (DOCX, IPYNB) and better chunking strategies.

This module is kept for backward compatibility.
"""

import os
from typing import List, Tuple

from .config import settings


def chunk_text(
    text: str,
    chunk_size: int = 1400,
    overlap: int = 250,
) -> List[str]:
    """
    Simple character-based text chunking.
    
    For better semantic chunking, use chunking.py instead.
    
    Args:
        text: Input text
        chunk_size: Maximum characters per chunk
        overlap: Character overlap between chunks
        
    Returns:
        List of text chunks
    """
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
        
        # Move start with overlap
        start = end - overlap
        if start < 0:
            start = 0
        if end == len(text):
            break
    
    return chunks


def load_raw_files(raw_dir: str = None) -> List[Tuple[str, str]]:
    """
    Load all text files from raw directory.
    
    Args:
        raw_dir: Directory path (defaults to settings.raw_dir)
        
    Returns:
        List of (filename, content) tuples
    """
    raw_dir = raw_dir or settings.raw_dir
    files = []
    
    if not os.path.exists(raw_dir):
        return files

    supported_extensions = {".txt", ".md"}
    
    for name in os.listdir(raw_dir):
        path = os.path.join(raw_dir, name)
        
        if not os.path.isfile(path):
            continue
            
        ext = os.path.splitext(name)[1].lower()
        if ext not in supported_extensions:
            continue
        
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
            files.append((name, content))
        except Exception as e:
            print(f"Warning: Failed to read {name}: {e}")
    
    return files
