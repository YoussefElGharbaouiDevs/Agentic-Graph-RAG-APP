from __future__ import annotations

import os
import re
from typing import List, Dict, Tuple, Optional

import numpy as np

from services import embeddings as emb_service
from openai import OpenAI


def _tokenize(text: str) -> List[str]:
    return re.findall(r'\w+', text.lower())


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def top_k_semantic(query: str, top_k: int = 5, model_name: Optional[str] = None) -> List[Dict]:
    results = emb_service.search(query, top_k, model_name)
    return [
        {"chunk": chunk, "score": round(1.0 / (1.0 + dist), 4), "method": "top_k_semantic", "meta": meta}
        for chunk, dist, meta in results
    ]


def cosine_similarity(query: str, top_k: int = 5, model_name: Optional[str] = None) -> List[Dict]:
    if not emb_service.index_ready():
        emb_service.load_index()
    chunks = emb_service._chunks
    if not chunks:
        return []
    model = emb_service._get_model(model_name or emb_service.CURRENT_MODEL)
    q_emb = emb_service.encode_query(query, model_name or emb_service.CURRENT_MODEL)
    q_emb = np.array(q_emb, dtype="float32")
    c_embs = emb_service.encode_chunks(chunks[:500], model_name or emb_service.CURRENT_MODEL)
    c_embs = np.array(c_embs, dtype="float32")
    scores = [_cosine(q_emb, c) for c in c_embs]
    top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    return [
        {"chunk": chunks[i], "score": round(scores[i], 4), "method": "cosine_similarity",
         "meta": emb_service._metadata[i] if i < len(emb_service._metadata) else {}}
        for i in top_idx
    ]

def bm25_search(query: str, top_k: int = 5, model_name: Optional[str] = None) -> List[Dict]:
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


def hybrid_bm25(query: str, top_k: int = 5, model_name: Optional[str] = None, alpha: float = 0.6) -> List[Dict]:
    from rank_bm25 import BM25Okapi
    if not emb_service.index_ready():
        emb_service.load_index()
    chunks = emb_service._chunks
    if not chunks:
        return []
    model = emb_service._get_model(model_name or emb_service.CURRENT_MODEL)
    q_emb = emb_service.encode_query(query, model_name or emb_service.CURRENT_MODEL)
    q_emb = np.array(q_emb, dtype="float32")
    c_embs = emb_service.encode_chunks(chunks[:500], model_name or emb_service.CURRENT_MODEL)
    c_embs = np.array(c_embs, dtype="float32")
    sem_scores = np.array([_cosine(q_emb, c) for c in c_embs])
    tokenized_corpus = [_tokenize(c) for c in chunks[:500]]
    bm25 = BM25Okapi(tokenized_corpus)
    bm25_raw = bm25.get_scores(_tokenize(query))
    bm25_scores = np.array(bm25_raw[:500])
    sem_norm = (sem_scores - sem_scores.min()) / (np.ptp(sem_scores) + 1e-9)
    bm25_norm = (bm25_scores - bm25_scores.min()) / (np.ptp(bm25_scores) + 1e-9)
    fused = alpha * sem_norm + (1 - alpha) * bm25_norm
    top_idx = fused.argsort()[::-1][:top_k]
    return [
        {"chunk": chunks[i], "score": round(float(fused[i]), 4), "method": "hybrid_bm25",
         "meta": emb_service._metadata[i] if i < len(emb_service._metadata) else {}}
        for i in top_idx
    ]


def mmr_search(query: str, top_k: int = 5, model_name: Optional[str] = None, lambda_param: float = 0.5) -> List[Dict]:
    if not emb_service.index_ready():
        emb_service.load_index()
    chunks = emb_service._chunks
    if not chunks:
        return []
    q_emb = emb_service.encode_query(query, model_name or emb_service.CURRENT_MODEL)
    q_emb = np.array(q_emb, dtype="float32")
    c_embs = emb_service.encode_chunks(chunks[:200], model_name or emb_service.CURRENT_MODEL)
    c_embs = np.array(c_embs, dtype="float32")
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


def retrieve(query: str, method: str = "auto", top_k: int = 5, model_name: Optional[str] = None) -> Tuple[str, List[Dict]]:
    if method == "auto":
        method = select_best_retrieval(query)
    fn = RETRIEVAL_METHODS.get(method, hybrid_bm25)
    results = fn(query, top_k, model_name)
    return method, results


def build_summary(results: List[Dict], query: str) -> str:
    if not results:
        return "No relevant documents found."
    
    # Detect query language to enforce correct response language
    english_indicators = {"what", "who", "where", "when", "why", "how", "which",
                          "is", "are", "was", "were", "the", "a", "an", "does",
                          "do", "did", "can", "could", "would", "explain", "describe"}
    query_words = set(query.lower().split())
    is_english = len(query_words & english_indicators) >= 1
    lang_instruction = (
        "IMPORTANT: The user asked in English. You MUST answer in English, "
        "even if the context is in French."
        if is_english else
        "IMPORTANT: L'utilisateur a posé la question en français. Vous DEVEZ répondre en français."
    )
    
    try:
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            client = OpenAI(api_key=api_key)
            context = " [...] ".join([r.get("chunk", r.get("path", ""))[:500] for r in results[:5]])
            
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": (
                        f"You are a scientific assistant. {lang_instruction} "
                        "Answer clearly and concisely based strictly on the provided context. "
                        "Do NOT include raw text excerpts in your answer."
                    )},
                    {"role": "user", "content": f"Question: {query}\n\nContext: {context}\n\nAnswer:"}
                ],
                max_tokens=600,
                temperature=0.3
            )
            return response.choices[0].message.content
    except Exception as e:
        print(f"[build_summary] OpenAI error: {e}")
    
    # Fallback: return raw excerpts
    top = results[:3]
    excerpts = [r["chunk"][:300] for r in top]
    return " [...] ".join(excerpts)