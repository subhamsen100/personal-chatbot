"""Embedding generation using Ollama's nomic-embed-text model."""

import numpy as np
import ollama

from config import settings


def embed_text(text: str) -> list[float]:
    """Embed a single string. Returns a list of floats."""
    response = ollama.embeddings(
        model=settings.ollama_embed_model,
        prompt=text,
    )
    return response["embedding"]


def embed_batch(texts: list[str]) -> np.ndarray:
    """Embed a list of strings. Returns shape (N, dim) float32 array."""
    vectors = [embed_text(t) for t in texts]
    return np.array(vectors, dtype=np.float32)
