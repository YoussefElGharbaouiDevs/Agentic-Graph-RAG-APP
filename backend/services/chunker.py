from __future__ import annotations

import re
from pathlib import Path
from typing import List, Tuple

from pypdf import PdfReader


# ── helpers ───────────────────────────────────────────────────────────────────

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract full text from a PDF file."""
    reader = PdfReader(pdf_path)
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages)


def _clean(text: str) -> str:
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {2,}', ' ', text)
    return text.strip()


# ── strategy implementations ──────────────────────────────────────────────────

def chunk_fixed_size(text: str, size: int = 500) -> List[str]:
    """Strategy 1 – fixed character windows, no overlap."""
    text = _clean(text)
    return [text[i:i+size] for i in range(0, len(text), size) if text[i:i+size].strip()]


def chunk_sentences(text: str, max_sentences: int = 5) -> List[str]:
    """Strategy 2 – group N sentences per chunk."""
    text = _clean(text)
    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    chunks = []
    for i in range(0, len(sentences), max_sentences):
        chunk = " ".join(sentences[i:i+max_sentences])
        if chunk:
            chunks.append(chunk)
    return chunks


def chunk_paragraphs(text: str) -> List[str]:
    """Strategy 3 – split on blank lines (paragraphs)."""
    text = _clean(text)
    chunks = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]
    return chunks


def chunk_semantic(text: str, threshold: float = 0.75) -> List[str]:
    """
    Strategy 4 – merge sentences while cosine similarity stays above threshold.
    Falls back to sentence chunking if sentence-transformers is unavailable.
    """
    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np

        model = SentenceTransformer("all-MiniLM-L6-v2")
        sentences = re.split(r'(?<=[.!?])\s+', _clean(text))
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]

        if not sentences:
            return chunk_sentences(text)

        embeddings = model.encode(sentences, show_progress_bar=False)

        def cosine(a, b):
            return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

        chunks, current = [], [sentences[0]]
        for i in range(1, len(sentences)):
            sim = cosine(embeddings[i-1], embeddings[i])
            if sim >= threshold:
                current.append(sentences[i])
            else:
                chunks.append(" ".join(current))
                current = [sentences[i]]
        if current:
            chunks.append(" ".join(current))
        return [c for c in chunks if c.strip()]

    except Exception:
        return chunk_sentences(text)


def chunk_sliding_window(text: str, size: int = 400, overlap: int = 100) -> List[str]:
    """Strategy 5 – overlapping fixed-size windows."""
    text = _clean(text)
    step = max(1, size - overlap)
    chunks = []
    for i in range(0, len(text), step):
        chunk = text[i:i+size]
        if chunk.strip():
            chunks.append(chunk)
    return chunks


def chunk_recursive(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """Strategy 6 – RecursiveCharacterTextSplitter (uses LangChain if available,
    otherwise a pure-Python recursive splitter)."""
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        docs = splitter.create_documents([text])
        return [d.page_content if hasattr(d, "page_content") else d for d in docs]
    except Exception:
        # Pure-Python fallback: split on separators in order of preference
        separators = ["\n\n", "\n", ". ", " "]
        chunks: List[str] = []

        def _split(txt: str, seps: List[str]) -> List[str]:
            if not seps or len(txt) <= chunk_size:
                return [txt] if txt.strip() else []
            sep = seps[0]
            parts = txt.split(sep)
            result, current = [], ""
            for part in parts:
                candidate = current + sep + part if current else part
                if len(candidate) <= chunk_size:
                    current = candidate
                else:
                    if current:
                        result.append(current)
                    # part itself may be too big → recurse
                    if len(part) > chunk_size:
                        result.extend(_split(part, seps[1:]))
                        current = ""
                    else:
                        current = part
            if current:
                result.append(current)
            return result

        raw = _split(_clean(text), separators)
        # Apply overlap
        if overlap == 0:
            return [c.strip() for c in raw if c.strip()]
        merged: List[str] = []
        for i, chunk in enumerate(raw):
            if i == 0:
                merged.append(chunk)
            else:
                tail = merged[-1][-overlap:] if len(merged[-1]) >= overlap else merged[-1]
                merged.append(tail + " " + chunk)
        return [c.strip() for c in merged if c.strip()]


def chunk_token_based(text: str, max_tokens: int = 200) -> List[str]:
    """Strategy 7 – approximate split by word count (1 word ≈ 1.3 tokens)."""
    text = _clean(text)
    words = text.split()
    approx_words = int(max_tokens / 1.3)
    chunks = []
    for i in range(0, len(words), approx_words):
        chunk = " ".join(words[i:i+approx_words])
        if chunk.strip():
            chunks.append(chunk)
    return chunks


# ── strategy registry ─────────────────────────────────────────────────────────

STRATEGIES = {
    "fixed_size":     chunk_fixed_size,
    "sentences":      chunk_sentences,
    "paragraphs":     chunk_paragraphs,
    "semantic":       chunk_semantic,
    "sliding_window": chunk_sliding_window,
    "recursive":      chunk_recursive,      # ← best method
    "token_based":    chunk_token_based,
}


# ── auto-select best method ───────────────────────────────────────────────────

def select_best_method(text: str) -> str:
    """
    Heuristic: choose chunking method based on corpus characteristics.
    - Very long paragraphs  → recursive (handles hierarchy)
    - Short, dense text     → sliding_window
    - Well-structured text  → paragraphs
    - Default               → recursive
    """
    avg_para_len = sum(len(p) for p in text.split("\n\n")) / max(1, text.count("\n\n"))
    paragraph_count = text.count("\n\n")

    if paragraph_count > 50 and avg_para_len < 300:
        return "paragraphs"
    if avg_para_len > 800:
        return "sliding_window"
    return "recursive"


# ── public API ────────────────────────────────────────────────────────────────

def run_chunking(
    text: str,
    method: str = "auto",
) -> Tuple[str, List[str]]:
    """
    Run chunking on `text`.

    Parameters
    ----------
    text   : plain text to chunk
    method : strategy name or "auto" for automatic selection

    Returns
    -------
    (method_used, list_of_chunks)
    """
    if method == "auto":
        method = select_best_method(text)

    fn = STRATEGIES.get(method, chunk_recursive)
    chunks = fn(text)

    # safety: remove empty chunks
    chunks = [c.strip() for c in chunks if c.strip()]
    return method, chunks


def get_all_strategies_info(text: str) -> dict:
    """Return stats for all 7 strategies (used for the frontend display)."""
    results = {}
    for name, fn in STRATEGIES.items():
        try:
            chunks = fn(text[:5000])  # sample for speed
            results[name] = {
                "count": len(chunks),
                "avg_len": int(sum(len(c) for c in chunks) / max(1, len(chunks))),
                "sample": chunks[0][:200] if chunks else "",
            }
        except Exception as e:
            results[name] = {"count": 0, "avg_len": 0, "sample": str(e)}
    return results