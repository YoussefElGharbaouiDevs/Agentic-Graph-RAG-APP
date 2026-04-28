from __future__ import annotations

import re
from typing import List, Dict, Tuple

import numpy as np

from services import embeddings as emb_service


def _tokenize(text: str) -> List[str]:
    return re.findall(r'\w+', text.lower())


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def top_k_semantic(query: str, top_k: int = 5) -> List[Dict]:
    results = emb_service.search(query, top_k)
    return [
        {"chunk": chunk, "score": round(1.0 / (1.0 + dist), 4), "method": "top_k_semantic", "meta": meta}
        for chunk, dist, meta in results
    ]


def cosine_similarity(query: str, top_k: int = 5) -> List[Dict]:
    if not emb_service.index_ready():
        emb_service.load_index()
    chunks = emb_service._chunks
    if not chunks:
        return []
    model = emb_service._get_model()
    q_emb = model.encode([query], show_progress_bar=False)[0]
    c_embs = model.encode(chunks[:500], show_progress_bar=False, batch_size=32)
    scores = [_cosine(q_emb, c) for c in c_embs]
    top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    return [
        {"chunk": chunks[i], "score": round(scores[i], 4), "method": "cosine_similarity",
         "meta": emb_service._metadata[i] if i < len(emb_service._metadata) else {}}
        for i in top_idx
    ]


def bm25_search(query: str, top_k: int = 5) -> List[Dict]:
    from rank_bm25 import BM25Okapi
    if not emb_service.index_ready():
        emb_service.load_index()
    chunks = emb_service._chunks
    if not chunks:
        return []
    tokenized_corpus = [_tokenize(c) for c in chunks]
    bm25 = BM25Okapi(tokenized_corpus)
    scores = bm25.get_scores(_tokenize(query))
    top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    max_score = max(scores) if max(scores) > 0 else 1.0
    return [
        {"chunk": chunks[i], "score": round(float(scores[i]) / max_score, 4), "method": "bm25",
         "meta": emb_service._metadata[i] if i < len(emb_service._metadata) else {}}
        for i in top_idx if scores[i] > 0
    ]


def hybrid_bm25(query: str, top_k: int = 5, alpha: float = 0.6) -> List[Dict]:
    from rank_bm25 import BM25Okapi
    if not emb_service.index_ready():
        emb_service.load_index()
    chunks = emb_service._chunks
    if not chunks:
        return []
    model = emb_service._get_model()
    q_emb = model.encode([query], show_progress_bar=False)[0]
    c_embs = model.encode(chunks[:500], show_progress_bar=False, batch_size=32)
    sem_scores = np.array([_cosine(q_emb, c) for c in c_embs])
    tokenized_corpus = [_tokenize(c) for c in chunks[:500]]
    bm25 = BM25Okapi(tokenized_corpus)
    bm25_raw = bm25.get_scores(_tokenize(query))
    bm25_scores = np.array(bm25_raw[:500])
    sem_norm = (sem_scores - sem_scores.min()) / (sem_scores.ptp() + 1e-9)
    bm25_norm = (bm25_scores - bm25_scores.min()) / (bm25_scores.ptp() + 1e-9)
    fused = alpha * sem_norm + (1 - alpha) * bm25_norm
    top_idx = fused.argsort()[::-1][:top_k]
    return [
        {"chunk": chunks[i], "score": round(float(fused[i]), 4), "method": "hybrid_bm25",
         "meta": emb_service._metadata[i] if i < len(emb_service._metadata) else {}}
        for i in top_idx
    ]


def mmr_search(query: str, top_k: int = 5, lambda_param: float = 0.5) -> List[Dict]:
    if not emb_service.index_ready():
        emb_service.load_index()
    chunks = emb_service._chunks
    if not chunks:
        return []
    model = emb_service._get_model()
    q_emb = model.encode([query], show_progress_bar=False)[0]
    c_embs = model.encode(chunks[:200], show_progress_bar=False, batch_size=32)
    relevance = np.array([_cosine(q_emb, c) for c in c_embs])
    selected, remaining = [], list(range(len(chunks[:200])))
    while len(selected) < top_k and remaining:
        if not selected:
            best = max(remaining, key=lambda i: relevance[i])
        else:
            best = max(
                remaining,
                key=lambda i: lambda_param * relevance[i]
                - (1 - lambda_param) * max(_cosine(c_embs[i], c_embs[s]) for s in selected),
            )
        selected.append(best)
        remaining.remove(best)
    return [
        {"chunk": chunks[i], "score": round(float(relevance[i]), 4), "method": "mmr",
         "meta": emb_service._metadata[i] if i < len(emb_service._metadata) else {}}
        for i in selected
    ]


RETRIEVAL_METHODS = {
    "top_k_semantic":    top_k_semantic,
    "cosine_similarity": cosine_similarity,
    "bm25":              bm25_search,
    "hybrid_bm25":       hybrid_bm25,
    "mmr":               mmr_search,
}


def select_best_retrieval(query: str) -> str:
    words = query.split()
    if len(words) <= 3:
        return "bm25"
    if len(words) >= 15:
        return "mmr"
    return "hybrid_bm25"


def retrieve(query: str, method: str = "auto", top_k: int = 5) -> Tuple[str, List[Dict]]:
    if method == "auto":
        method = select_best_retrieval(query)
    fn = RETRIEVAL_METHODS.get(method, hybrid_bm25)
    results = fn(query, top_k)
    return method, results


def build_summary(results: List[Dict], query: str) -> str:
    if not results:
        return "No relevant documents found."
    top = results[:3]
    excerpts = [r["chunk"][:300] for r in top]
    return " [...] ".join(excerpts)