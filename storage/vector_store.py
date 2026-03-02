"""FAISS-backed vector store for document chunks."""

import json
from pathlib import Path

import faiss
import numpy as np

from config import settings
from ingestion.embedder import embed_batch, embed_text


class FAISSVectorStore:
    """Persistent FAISS index with JSON metadata sidecar."""

    def __init__(self, store_path: str | None = None):
        self._path = Path(store_path or settings.vector_store_path)
        self._path.mkdir(parents=True, exist_ok=True)
        self._index_file = self._path / "index.faiss"
        self._meta_file = self._path / "metadata.json"
        self._index: faiss.Index | None = None
        self._metadata: list[dict] = []
        self._load()

    # persistence─

    def _load(self) -> None:
        if self._index_file.exists() and self._meta_file.exists():
            self._index = faiss.read_index(str(self._index_file))
            self._metadata = json.loads(self._meta_file.read_text())

    def _save(self) -> None:
        if self._index is not None:
            faiss.write_index(self._index, str(self._index_file))
        self._meta_file.write_text(json.dumps(self._metadata, ensure_ascii=False))

    # public API──

    def add_chunks(self, chunks: list[dict]) -> None:
        """Embed and index a list of chunk dicts (must have 'text' and 'source')."""
        if not chunks:
            return

        texts = [c["text"] for c in chunks]
        vectors = embed_batch(texts)

        if self._index is None:
            dim = vectors.shape[1]
            self._index = faiss.IndexFlatL2(dim)

        start = len(self._metadata)
        self._index.add(vectors)
        for i, chunk in enumerate(chunks):
            self._metadata.append({**chunk, "id": start + i})

        self._save()
        print(f"[vector_store] Indexed {len(chunks)} chunks. Total: {self._index.ntotal}")

    def search(self, query: str, top_k: int | None = None) -> list[dict]:
        """Return top-k most similar chunks for a query string."""
        if self._index is None or self._index.ntotal == 0:
            return []

        k = min(top_k or settings.top_k, self._index.ntotal)
        q_vec = np.array([embed_text(query)], dtype=np.float32)
        distances, indices = self._index.search(q_vec, k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if 0 <= idx < len(self._metadata):
                results.append({**self._metadata[idx], "score": float(dist)})
        return results

    def is_empty(self) -> bool:
        return self._index is None or self._index.ntotal == 0

    def stats(self) -> dict:
        return {
            "total_chunks": self._index.ntotal if self._index else 0,
            "sources": list({m["source"] for m in self._metadata}),
        }

    def clear(self) -> None:
        self._index = None
        self._metadata = []
        if self._index_file.exists():
            self._index_file.unlink()
        if self._meta_file.exists():
            self._meta_file.unlink()


# Module-level singleton
vector_store = FAISSVectorStore()
