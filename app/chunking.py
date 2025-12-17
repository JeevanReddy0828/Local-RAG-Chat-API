import json
import re
from pathlib import Path
from typing import List, Tuple

from docx import Document


def _clean(text: str) -> str:
    text = text.replace("\u00a0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _split_blocks(text: str) -> List[str]:
    # split by blank lines (better semantic blocks)
    return [b.strip() for b in re.split(r"\n\s*\n", text) if b.strip()]


def _chunk_by_char_budget(blocks: List[str], max_chars: int, overlap_chars: int) -> List[str]:
    chunks: List[str] = []
    buf: List[str] = []
    cur = 0

    def flush():
        nonlocal buf, cur
        if not buf:
            return
        chunk = "\n\n".join(buf).strip()
        if chunk:
            chunks.append(chunk)
        buf = []
        cur = 0

    for b in blocks:
        if len(b) > max_chars:
            # hard split giant blocks
            for i in range(0, len(b), max_chars):
                sub = b[i : i + max_chars].strip()
                if sub:
                    chunks.append(sub)
            continue

        if cur + len(b) + 2 > max_chars:
            flush()

        buf.append(b)
        cur += len(b) + 2

    flush()

    # simple overlap
    if overlap_chars > 0 and len(chunks) > 1:
        out = [chunks[0]]
        for i in range(1, len(chunks)):
            prev = chunks[i - 1]
            overlap = prev[-overlap_chars:] if len(prev) > overlap_chars else prev
            out.append((overlap + "\n\n" + chunks[i]).strip())
        chunks = out

    return chunks


def parse_docx(path: str) -> str:
    doc = Document(path)
    lines = []
    for p in doc.paragraphs:
        t = p.text.strip()
        if t:
            lines.append(t)
    return _clean("\n".join(lines))


def parse_ipynb(path: str) -> str:
    data = json.loads(Path(path).read_text(encoding="utf-8", errors="ignore"))
    out = []
    for cell in data.get("cells", []):
        ctype = cell.get("cell_type")
        src = "".join(cell.get("source", [])).strip()
        if not src:
            continue

        if ctype == "markdown":
            out.append(f"## Notebook Markdown\n{src}")
        elif ctype == "code":
            out.append(f"## Notebook Code\n```python\n{src}\n```")
        else:
            out.append(src)
    return _clean("\n\n".join(out))


def parse_text_like(path: str) -> str:
    return _clean(Path(path).read_text(encoding="utf-8", errors="ignore"))


def load_text_for_file(filepath: str) -> Tuple[str, str]:
    """
    Returns (text, kind)
    """
    p = Path(filepath)
    ext = p.suffix.lower()

    if ext == ".docx":
        return parse_docx(filepath), "docx"
    if ext == ".ipynb":
        return parse_ipynb(filepath), "ipynb"
    return parse_text_like(filepath), "text"


def chunk_text(text: str, max_chars: int = 1400, overlap_chars: int = 250) -> List[str]:
    blocks = _split_blocks(text)
    return _chunk_by_char_budget(blocks, max_chars=max_chars, overlap_chars=overlap_chars)
