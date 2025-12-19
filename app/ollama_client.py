"""
Ollama API client for LLM inference.

Supports both synchronous and streaming generation modes.
"""

import json
import logging
from typing import Iterator, Optional

import requests
from requests.exceptions import RequestException, Timeout, ConnectionError

from .config import settings

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# EXCEPTIONS
# ═══════════════════════════════════════════════════════════════════════════════


class OllamaError(Exception):
    """Base exception for Ollama client errors."""
    pass


class OllamaConnectionError(OllamaError):
    """Raised when connection to Ollama fails."""
    pass


class OllamaTimeoutError(OllamaError):
    """Raised when Ollama request times out."""
    pass


class OllamaGenerationError(OllamaError):
    """Raised when generation fails."""
    pass


# ═══════════════════════════════════════════════════════════════════════════════
# HEALTH CHECK
# ═══════════════════════════════════════════════════════════════════════════════


def check_ollama_health() -> bool:
    """
    Check if Ollama is running and accessible.
    
    Returns:
        True if Ollama is healthy, False otherwise
    """
    try:
        url = f"{settings.ollama_base_url}/api/tags"
        response = requests.get(url, timeout=5)
        return response.status_code == 200
    except RequestException:
        return False


def get_available_models() -> list:
    """
    Get list of available models from Ollama.
    
    Returns:
        List of model names, or empty list on error
    """
    try:
        url = f"{settings.ollama_base_url}/api/tags"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        return [m.get("name") for m in data.get("models", [])]
    except RequestException as e:
        logger.warning(f"Failed to get Ollama models: {e}")
        return []


# ═══════════════════════════════════════════════════════════════════════════════
# SYNCHRONOUS GENERATION
# ═══════════════════════════════════════════════════════════════════════════════


def ollama_generate(
    prompt: str,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    timeout: int = 600,
) -> str:
    """
    Generate text using Ollama (synchronous).
    
    Args:
        prompt: Input prompt
        model: Model name (defaults to settings.ollama_model)
        temperature: Sampling temperature (defaults to settings.temperature)
        max_tokens: Max tokens to generate (defaults to settings.max_new_tokens)
        timeout: Request timeout in seconds
        
    Returns:
        Generated text
        
    Raises:
        OllamaConnectionError: If connection fails
        OllamaTimeoutError: If request times out
        OllamaGenerationError: If generation fails
    """
    url = f"{settings.ollama_base_url}/api/generate"
    
    payload = {
        "model": model or settings.ollama_model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature if temperature is not None else settings.temperature,
            "num_predict": max_tokens or settings.max_new_tokens,
        },
    }

    try:
        logger.debug(f"Ollama generate: model={payload['model']}, prompt_len={len(prompt)}")
        
        response = requests.post(url, json=payload, timeout=timeout)
        response.raise_for_status()
        
        data = response.json()
        result = data.get("response", "").strip()
        
        logger.debug(f"Ollama response: {len(result)} chars")
        return result

    except ConnectionError as e:
        logger.error(f"Ollama connection failed: {e}")
        raise OllamaConnectionError(
            f"Cannot connect to Ollama at {settings.ollama_base_url}. "
            "Ensure Ollama is running with 'ollama serve'."
        ) from e

    except Timeout as e:
        logger.error(f"Ollama request timed out after {timeout}s")
        raise OllamaTimeoutError(
            f"Ollama request timed out after {timeout} seconds."
        ) from e

    except RequestException as e:
        logger.error(f"Ollama generation failed: {e}")
        raise OllamaGenerationError(f"Ollama generation failed: {e}") from e


# ═══════════════════════════════════════════════════════════════════════════════
# STREAMING GENERATION
# ═══════════════════════════════════════════════════════════════════════════════


def ollama_generate_stream(
    prompt: str,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    timeout: int = 600,
) -> Iterator[str]:
    """
    Generate text using Ollama with streaming.
    
    Yields tokens as they are generated.
    
    Args:
        prompt: Input prompt
        model: Model name (defaults to settings.ollama_model)
        temperature: Sampling temperature (defaults to settings.temperature)
        max_tokens: Max tokens to generate (defaults to settings.max_new_tokens)
        timeout: Request timeout in seconds
        
    Yields:
        Token strings as they are generated
        
    Raises:
        OllamaConnectionError: If connection fails
        OllamaTimeoutError: If request times out
        OllamaGenerationError: If generation fails
    """
    url = f"{settings.ollama_base_url}/api/generate"
    
    payload = {
        "model": model or settings.ollama_model,
        "prompt": prompt,
        "stream": True,
        "options": {
            "temperature": temperature if temperature is not None else settings.temperature,
            "num_predict": max_tokens or settings.max_new_tokens,
        },
    }

    try:
        logger.debug(f"Ollama stream: model={payload['model']}, prompt_len={len(prompt)}")
        
        with requests.post(url, json=payload, stream=True, timeout=timeout) as response:
            response.raise_for_status()
            
            token_count = 0
            for line in response.iter_lines(decode_unicode=True):
                if not line:
                    continue

                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse Ollama stream line: {line[:100]}")
                    continue

                # Check for completion
                if obj.get("done"):
                    logger.debug(f"Ollama stream complete: {token_count} tokens")
                    break

                # Yield token
                chunk = obj.get("response", "")
                if chunk:
                    token_count += 1
                    yield chunk

    except ConnectionError as e:
        logger.error(f"Ollama connection failed: {e}")
        raise OllamaConnectionError(
            f"Cannot connect to Ollama at {settings.ollama_base_url}. "
            "Ensure Ollama is running with 'ollama serve'."
        ) from e

    except Timeout as e:
        logger.error(f"Ollama stream timed out after {timeout}s")
        raise OllamaTimeoutError(
            f"Ollama stream timed out after {timeout} seconds."
        ) from e

    except RequestException as e:
        logger.error(f"Ollama stream failed: {e}")
        raise OllamaGenerationError(f"Ollama stream failed: {e}") from e
