import os
import time
import pickle
from typing import List
import numpy as np
from aoai_client import get_aoai_client


CACHE_DIR = os.path.join(".cache")
CACHE_FILE = os.path.join(CACHE_DIR, "embeddings.pkl")


def _ensure_cache():
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR, exist_ok=True)


def _load_cache():
    if not os.path.exists(CACHE_FILE):
        return {}
    try:
        with open(CACHE_FILE, "rb") as f:
            return pickle.load(f)
    except Exception:
        return {}


def _save_cache(cache):
    _ensure_cache()
    with open(CACHE_FILE, "wb") as f:
        pickle.dump(cache, f)


def embed_texts(texts: List[str], model: str = "text-embedding-3-small", batch_size: int = 16) -> np.ndarray:
    """Batch embed a list of texts, using a simple on-disk cache to avoid repeat calls.

    Returns an (n, d) numpy array.
    """
    cache = _load_cache()
    pending_indices = []
    results = []

    # preserve order
    for i, t in enumerate(texts):
        key = t
        if key in cache:
            results.append(np.array(cache[key]))
        else:
            results.append(None)
            pending_indices.append(i)

    if pending_indices:
        client = get_aoai_client()
        # send in batches
        for batch_start in range(0, len(pending_indices), batch_size):
            batch_indices = pending_indices[batch_start : batch_start + batch_size]
            batch_texts = [texts[i] for i in batch_indices]
            resp = client.embeddings.create(input=batch_texts, model=model)
            for idx, item in zip(batch_indices, resp.data):
                emb = np.array(item.embedding)
                cache[texts[idx]] = emb.tolist()
                results[idx] = emb
            time.sleep(0.01)

        _save_cache(cache)

    return np.vstack(results)


def embed_text(text: str, model: str = "text-embedding-3-small") -> np.ndarray:
    return embed_texts([text], model=model)[0]


def cosine_similarity_matrix(embs: np.ndarray) -> np.ndarray:
    # normalized dot product matrix
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    normalized = embs / np.clip(norms, 1e-12, None)
    return normalized @ normalized.T
