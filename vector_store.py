"""
vector_store.py
----------------
A lightweight FAISS-based vector store used by the AI Study Assistant.

Embeddings are generated locally through Ollama's `nomic-embed-text`
model (no paid API, no internet connection required after the model has
been pulled once). The store keeps the raw text chunk and its source
filename alongside each vector so retrieved results can cite where they
came from.
"""

import os
import numpy as np
import faiss
import requests

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_EMBED_MODEL = os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text")


def get_embedding(text: str) -> list:
    """Request a single embedding vector from Ollama for the given text."""
    url = f"{OLLAMA_HOST}/api/embeddings"
    payload = {"model": OLLAMA_EMBED_MODEL, "prompt": text}
    try:
        response = requests.post(url, json=payload, timeout=120)
        response.raise_for_status()
    except requests.exceptions.ConnectionError as exc:
        raise RuntimeError(
            "Could not connect to Ollama for embeddings. Make sure Ollama is "
            f"running and the model is pulled (`ollama pull {OLLAMA_EMBED_MODEL}`)."
        ) from exc
    except requests.exceptions.HTTPError as exc:
        raise RuntimeError(f"Ollama returned an error while embedding text: {exc}") from exc

    data = response.json()
    embedding = data.get("embedding")
    if not embedding:
        raise RuntimeError("Ollama did not return an embedding vector.")
    return embedding


class VectorStore:
    """Simple in-memory FAISS flat L2 index with parallel text/metadata lists."""

    def __init__(self):
        self.index = None
        self.dimension = None
        self.chunks = []       # list[str]
        self.metadatas = []    # list[dict] e.g. {"source": filename}

    def is_empty(self) -> bool:
        return self.index is None or self.index.ntotal == 0

    def add_texts(self, texts: list, metadatas: list):
        """Embed and add a batch of text chunks to the FAISS index."""
        vectors = [get_embedding(t) for t in texts]
        vectors_np = np.array(vectors, dtype="float32")

        if self.index is None:
            self.dimension = vectors_np.shape[1]
            self.index = faiss.IndexFlatL2(self.dimension)

        self.index.add(vectors_np)
        self.chunks.extend(texts)
        self.metadatas.extend(metadatas)

    def similarity_search(self, query: str, k: int = 4):
        """Return the top-k most similar chunks (with metadata) to the query."""
        if self.is_empty():
            return []

        query_vec = np.array([get_embedding(query)], dtype="float32")
        k = min(k, self.index.ntotal)
        distances, indices = self.index.search(query_vec, k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
            results.append({
                "text": self.chunks[idx],
                "metadata": self.metadatas[idx],
                "distance": float(dist),
            })
        return results

    def get_all_text(self, max_chars: int = 12000) -> str:
        """Return concatenated stored text, truncated, for summary/quiz generation."""
        joined = "\n\n".join(self.chunks)
        return joined[:max_chars]

    def clear(self):
        self.index = None
        self.dimension = None
        self.chunks = []
        self.metadatas = []


# A single shared in-memory store used by app.py (simple single-user local app).
store = VectorStore()
