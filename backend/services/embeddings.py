from __future__ import annotations

import json
import os
import pickle
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np

# ── lazy imports (keep startup fast) ─────────────────────────────────────────
_model = None
_index = None
_chunks: List[str] = []
_metadata: List[Dict] = []

INDEX_PATH = Path(os.getenv("FAISS_INDEX_PATH", "faiss_index.bin"))
META_PATH  = Path(os.getenv("FAISS_META_PATH",  "faiss_meta.pkl"))

MODEL_NAME = "all-MiniLM-L6-v2"


# ── model loading ─────────────────────────────────────────────────────────────

def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(MODEL_NAME)
    return _model


# ── FAISS index ───────────────────────────────────────────────────────────────

def _get_index(dim: int = 384):
    global _index
    if _index is None:
        import faiss
        _index = faiss.IndexFlatL2(dim)
    return _index


def build_index(chunks: List[str], metadata: Optional[List[Dict]] = None) -> None:
    """Encode chunks and build a new FAISS index (overwrites existing)."""
    global _index, _chunks, _metadata

    import faiss

    model = _get_model()
    embeddings = model.encode(chunks, show_progress_bar=False, batch_size=32)
    embeddings = np.array(embeddings, dtype="float32")

    dim = embeddings.shape[1]
    _index = faiss.IndexFlatL2(dim)
    _index.add(embeddings)

    _chunks = list(chunks)
    _metadata = metadata if metadata else [{"id": f"doc_{i}"} for i in range(len(chunks))]

    # Persist
    faiss.write_index(_index, str(INDEX_PATH))
    with open(META_PATH, "wb") as f:
        pickle.dump({"chunks": _chunks, "metadata": _metadata}, f)


def load_index() -> bool:
    """Load persisted index from disk. Returns True if successful."""
    global _index, _chunks, _metadata

    if not INDEX_PATH.exists() or not META_PATH.exists():
        return False

    try:
        import faiss
        _index = faiss.read_index(str(INDEX_PATH))
        with open(META_PATH, "rb") as f:
            data = pickle.load(f)
        _chunks   = data["chunks"]
        _metadata = data["metadata"]
        return True
    except Exception:
        return False


def index_ready() -> bool:
    return _index is not None and len(_chunks) > 0


# ── search ────────────────────────────────────────────────────────────────────

def search(query: str, top_k: int = 5) -> List[Tuple[str, float, Dict]]:
    """
    Return top-k (chunk, score, metadata) tuples.
    Score is L2 distance (lower = more similar).
    """
    if not index_ready():
        load_index()
    if not index_ready():
        return []

    model = _get_model()
    q_emb = model.encode([query], show_progress_bar=False)
    q_emb = np.array(q_emb, dtype="float32")

    distances, indices = _index.search(q_emb, min(top_k, len(_chunks)))

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx < 0:
            continue
        results.append((_chunks[idx], float(dist), _metadata[idx]))
    return results


# ── PCA projection for visualisation ─────────────────────────────────────────

def get_pca_points(max_points: int = 200) -> List[Dict]:
    """
    Return 2-D PCA projection of all indexed embeddings.
    Used by the frontend Embeddings Visualization panel.
    """
    if not index_ready():
        load_index()
    if not index_ready():
        return []

    from sklearn.decomposition import PCA

    # Re-encode (FAISS index stores only the vectors, not raw embeddings easily)
    model = _get_model()
    sample = _chunks[:max_points]
    embeddings = model.encode(sample, show_progress_bar=False, batch_size=32)
    embeddings = np.array(embeddings, dtype="float32")

    n_components = min(2, embeddings.shape[0], embeddings.shape[1])
    pca = PCA(n_components=n_components)
    coords = pca.fit_transform(embeddings)

    points = []
    for i, (x, y) in enumerate(coords):
        points.append({
            "x": float(x),
            "y": float(y),
            "label": f"doc_{i}",
            "preview": sample[i][:60],
        })
    return points


# ── index stats ───────────────────────────────────────────────────────────────

def get_stats() -> Dict:
    if not index_ready():
        load_index()
    return {
        "total_vectors": len(_chunks),
        "dimensions":    _index.d if _index else 0,
        "model":         MODEL_NAME,
        "index_ready":   index_ready(),
    }