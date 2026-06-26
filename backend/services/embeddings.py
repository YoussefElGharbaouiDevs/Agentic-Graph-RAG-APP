from __future__ import annotations

import json
import os
import pickle
import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
from sklearn.decomposition import PCA

# ── lazy imports (keep startup fast) ─────────────────────────────────────────
_model = None
_index = None
_chunks: List[str] = []
_metadata: List[Dict] = []

INDEX_PATH = Path(os.getenv("FAISS_INDEX_PATH", "faiss_index.bin"))
META_PATH  = Path(os.getenv("FAISS_META_PATH",  "faiss_meta.pkl"))

# ── Supported embedding models ──────────────────────────────────────────────
EMBEDDING_MODELS = {
    "sentence-transformers": "all-MiniLM-L6-v2",
    "tfidf": "TF-IDF",  # Placeholder for sklearn TfidfVectorizer
    "word2vec": "word2vec-google-news-300",  # Placeholder
    "glove": "glove-wiki-gigaword-300",  # Placeholder
    "bert": "bert-base-uncased",
    "openai": "text-embedding-ada-002",
    "instructor": "hkunlp/instructor-xl",
}

CURRENT_MODEL = "sentence-transformers"


# ── model loading ─────────────────────────────────────────────────────────────

_LOADED_MODEL_NAME: Optional[str] = None

def _get_model(model_name: str = CURRENT_MODEL):
    global _model, _LOADED_MODEL_NAME
    if _model is None or _LOADED_MODEL_NAME != model_name:
        if model_name == "sentence-transformers":
            from sentence_transformers import SentenceTransformer
            _model = SentenceTransformer(EMBEDDING_MODELS[model_name])
        elif model_name == "tfidf":
            from sklearn.feature_extraction.text import TfidfVectorizer
            _model = TfidfVectorizer(max_features=384)  # Match dim
        elif model_name in {"word2vec", "glove"}:
            try:
                import gensim.downloader as api
                _model = api.load(EMBEDDING_MODELS[model_name])
            except Exception:
                _model = None
        elif model_name == "bert":
            from transformers import BertModel, BertTokenizer
            tokenizer = BertTokenizer.from_pretrained(EMBEDDING_MODELS[model_name])
            model = BertModel.from_pretrained(EMBEDDING_MODELS[model_name])
            _model = {"tokenizer": tokenizer, "model": model}
        elif model_name == "openai":
            _model = EMBEDDING_MODELS[model_name]
        elif model_name == "instructor":
            from sentence_transformers import SentenceTransformer
            _model = SentenceTransformer(EMBEDDING_MODELS[model_name])

        if _model is not None:
            _LOADED_MODEL_NAME = model_name
        else:
            _LOADED_MODEL_NAME = None
    return _model


def _text_to_tokens(text: str) -> List[str]:
    return re.findall(r"\w+", text.lower())


def _average_word_vectors(chunks: List[str], model) -> List[List[float]]:
    vectors = []
    dim = getattr(model, "vector_size", 300)
    for text in chunks:
        tokens = [t for t in _text_to_tokens(text) if t in model]
        if not tokens:
            vectors.append([0.0] * dim)
            continue
        embeddings = [model[t] for t in tokens]
        avg = np.mean(np.array(embeddings, dtype="float32"), axis=0)
        vectors.append(avg.tolist())
    return vectors


def _bert_encode_texts(chunks: List[str], model_info) -> List[List[float]]:
    tokenizer = model_info["tokenizer"]
    model = model_info["model"]
    try:
        import torch
    except Exception:
        return [[0.0] * 768 for _ in chunks]

    model.eval()
    embeddings = []
    with torch.no_grad():
        for text in chunks:
            inputs = tokenizer(
                text,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            outputs = model(**inputs)
            if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                emb = outputs.pooler_output[0].cpu().numpy()
            else:
                emb = outputs.last_hidden_state[0, 0].cpu().numpy()
            embeddings.append(emb.tolist())
    return embeddings


def _encode_openai_texts(chunks: List[str], model_name: str) -> List[List[float]]:
    try:
        import openai
    except Exception:
        return [[0.0] * 1536 for _ in chunks]

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return [[0.0] * 1536 for _ in chunks]

    openai.api_key = api_key
    output = []
    for i in range(0, len(chunks), 10):
        batch = chunks[i:i + 10]
        try:
            response = openai.Embedding.create(
                input=batch,
                model=EMBEDDING_MODELS[model_name],
            )
            output.extend(item["embedding"] for item in response["data"])
        except Exception:
            output.extend([[0.0] * 1536 for _ in batch])
    return output


# ── FAISS index ───────────────────────────────────────────────────────────────

def _get_index(dim: int = 384):
    global _index
    if _index is None:
        import faiss
        _index = faiss.IndexFlatL2(dim)
    return _index


def build_index(chunks: List[str], metadata: Optional[List[Dict]] = None, model_name: str = CURRENT_MODEL) -> None:
    """Encode chunks and build a new FAISS index (overwrites existing)."""
    global _index, _chunks, _metadata, CURRENT_MODEL

    import faiss

    CURRENT_MODEL = model_name
    model = _get_model(model_name)
    embeddings = encode_chunks(chunks, model_name)
    embeddings = np.array(embeddings, dtype="float32")

    dim = embeddings.shape[1]
    _index = faiss.IndexFlatL2(dim)
    _index.add(embeddings)

    _chunks = list(chunks)
    _metadata = metadata if metadata else [{"id": f"doc_{i}"} for i in range(len(chunks))]

    # Persist
    faiss.write_index(_index, str(INDEX_PATH))
    with open(META_PATH, "wb") as f:
        pickle.dump({"chunks": _chunks, "metadata": _metadata, "model": CURRENT_MODEL}, f)


def load_index() -> bool:
    """Load persisted index from disk."""
    if not INDEX_PATH.exists() or not META_PATH.exists():
        return False

    try:
        import faiss
        global _index, _chunks, _metadata, CURRENT_MODEL
        _index = faiss.read_index(str(INDEX_PATH))
        with open(META_PATH, "rb") as f:
            data = pickle.load(f)
        _chunks   = data["chunks"]
        _metadata = data["metadata"]
        CURRENT_MODEL = data.get("model", "sentence-transformers")
        return True
    except Exception:
        return False


def index_ready() -> bool:
    return _index is not None and len(_chunks) > 0

def encode_chunks(chunks: List[str], model_name: str) -> List[List[float]]:
    """Encode a list of chunks using the specified model."""
    model = _get_model(model_name)
    if model_name in {"sentence-transformers", "instructor"}:
        return model.encode(chunks, show_progress_bar=False, batch_size=32).tolist()
    elif model_name == "tfidf":
        return model.fit_transform(chunks).toarray().tolist()
    elif model_name in {"word2vec", "glove"}:
        if model is not None:
            return _average_word_vectors(chunks, model)
        return [[0.0] * 300 for _ in chunks]
    elif model_name == "bert":
        if model is not None:
            return _bert_encode_texts(chunks, model)
        return [[0.0] * 768 for _ in chunks]
    elif model_name == "openai":
        return _encode_openai_texts(chunks, model_name)
    return [[0.0] * 384 for _ in chunks]

def encode_query(query: str, model_name: str) -> List[float]:
    """Encode a single query using the specified model."""
    model = _get_model(model_name)
    if model_name in {"sentence-transformers", "instructor"}:
        return model.encode([query], show_progress_bar=False)[0].tolist()
    elif model_name == "tfidf":
        return model.transform([query]).toarray()[0].tolist()
    elif model_name in {"word2vec", "glove"}:
        if model is not None:
            return _average_word_vectors([query], model)[0]
        return [0.0] * 300
    elif model_name == "bert":
        if model is not None:
            return _bert_encode_texts([query], model)[0]
        return [0.0] * 768
    elif model_name == "openai":
        return _encode_openai_texts([query], model_name)[0]
    return [0.0] * 384


def search(query: str, top_k: int = 5, model_name: Optional[str] = None):
    if not index_ready():
        load_index()
    if not index_ready():
        return []

    if model_name and model_name != CURRENT_MODEL:
        # Rebuild index with new model if needed
        if _chunks:
            build_index(_chunks, _metadata, model_name)

    q_emb = encode_query(query, model_name or CURRENT_MODEL)
    q_emb = np.array(q_emb, dtype="float32")

    distances, indices = _index.search(q_emb, min(top_k, len(_chunks)))

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx < 0:
            continue
        results.append((_chunks[idx], float(dist), _metadata[idx]))
    return results


# ── PCA projection for visualisation ─────────────────────────────────────────

def get_pca_points(max_points: int = 200, query: str = "") -> List[Dict]:
    """
    Return 2-D PCA projection of stored indexed embeddings.
    Used by the frontend Embeddings Visualization panel.
    """
    if not index_ready():
        load_index()
    if not index_ready():
        return []

    max_points = min(max_points, len(_chunks))
    if max_points <= 0:
        return []

    embeddings = []
    try:
        import faiss
        for i in range(max_points):
            emb = _index.reconstruct(i)
            embeddings.append(np.array(emb, dtype="float32"))
        embeddings = np.vstack(embeddings)
    except Exception:
        # Fallback to text re-encoding if reconstruct is unavailable.
        model = _get_model("sentence-transformers")
        if model is None:
            return []
        sample = _chunks[:max_points]
        embeddings = np.array(model.encode(sample, show_progress_bar=False, batch_size=32), dtype="float32")

    n_components = min(2, embeddings.shape[0], embeddings.shape[1])
    if n_components < 2:
        return []

    pca = PCA(n_components=n_components)
    reduced = pca.fit_transform(embeddings)

    nearest_indices = set()
    if query and index_ready():
        try:
            q_emb = encode_query(query, CURRENT_MODEL)
            q_emb = np.array([q_emb], dtype="float32")
            _, indices = _index.search(q_emb, 5) # Top 5 nearest
            nearest_indices = set(indices[0].tolist())
        except Exception:
            pass

    points = []
    for i, (x, y) in enumerate(reduced):
        chunk_snippet = _chunks[i][:60] + "..." if len(_chunks[i]) > 60 else _chunks[i]
        points.append({
            "x": round(float(x), 4),
            "y": round(float(y), 4),
            "chunk": chunk_snippet,
            "preview": chunk_snippet,
            "is_nearest": i in nearest_indices,
        })
    return points


# ── index stats ───────────────────────────────────────────────────────────────

def get_stats():
    if not index_ready():
        load_index()
    return {
        "total_vectors": len(_chunks),
        "dimensions":    _index.d if _index else 0,
        "model":         CURRENT_MODEL,
        "index_ready":   index_ready(),
    }