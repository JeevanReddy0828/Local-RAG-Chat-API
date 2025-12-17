from typing import Iterator
import requests

from .config import settings


def ollama_generate(prompt: str) -> str:
    url = f"{settings.ollama_base_url}/api/generate"
    payload = {
        "model": settings.ollama_model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": settings.temperature,
            "num_predict": settings.max_new_tokens,
        },
    }
    r = requests.post(url, json=payload, timeout=600)
    r.raise_for_status()
    data = r.json()
    return data.get("response", "").strip()


def ollama_generate_stream(prompt: str) -> Iterator[str]:
    """
    Yields token chunks as strings.
    Ollama streams JSON lines like: {"response":"...", "done":false}
    """
    url = f"{settings.ollama_base_url}/api/generate"
    payload = {
        "model": settings.ollama_model,
        "prompt": prompt,
        "stream": True,
        "options": {
            "temperature": settings.temperature,
            "num_predict": settings.max_new_tokens,
        },
    }

    with requests.post(url, json=payload, stream=True, timeout=600) as r:
        r.raise_for_status()
        for line in r.iter_lines(decode_unicode=True):
            if not line:
                continue
            # Each line is JSON
            try:
                import json

                obj = json.loads(line)
            except Exception:
                continue

            if obj.get("done"):
                break

            chunk = obj.get("response", "")
            if chunk:
                yield chunk
