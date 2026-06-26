from fastapi import APIRouter, Query
from pathlib import Path
from typing import Dict, Any
from models.schemas import VectorialResponse, ChunkInfo, EmbeddingInfo, RetrievalResult
from services import chunker as chk_svc
from services import embeddings as emb_svc
from services import retriever as ret_svc

router = APIRouter()

def _get_data_text(file_name: str = "echauffements_stratospherique_cleaned.pdf") -> str:
    data_path = Path(__file__).resolve().parent.parent / "data" / file_name
    if data_path.exists():
        try:
            return chk_svc.extract_text_from_pdf(str(data_path))
        except Exception:
            return ""
    return ""


@router.get("/chunking", response_model=Dict[str, Any])
def chunking_analysis():
    """Return analysis of all 7 chunking methods for the UI."""
    emb_stats = emb_svc.get_stats()
    chunks = emb_svc._chunks

    if not chunks:
        emb_svc.load_index()
        chunks = emb_svc._chunks

    text_sample = _get_data_text("echauffements_stratospherique_cleaned.pdf")
    if not text_sample:
        text_sample = " ".join(chunks[:10]) if chunks else "Sample text for chunking analysis."

    analysis = chk_svc.get_all_strategies_info(text_sample)
    best_method = chk_svc.select_best_method(text_sample)

    return {
        "best_method": best_method,
        "strategies": analysis,
        "total_chunks": len(chunks),
    }


@router.get("/vectorial", response_model=VectorialResponse)
def vectorial_rag(
    query: str = Query(..., description="User question"),
    method: str = Query("auto", description="Retrieval method or auto"),
    model: str = Query("sentence-transformers", description="Embedding model to use"),
    file_name: str = Query("echauffements_stratospherique_cleaned.pdf", description="Data file to base chunking analysis on"),
):
    if not emb_svc.index_ready():
        emb_svc.load_index()

    chunks = emb_svc._chunks
    text_sample = _get_data_text(file_name)
    if not text_sample:
        text_sample = " ".join(chunks[:10]) if chunks else query

    best_method = chk_svc.select_best_method(text_sample)
    avg_chunk_size = int(sum(len(c) for c in chunks) / max(1, len(chunks))) if chunks else 0
    sample_chunks = [c[:200] for c in chunks[:3]] if chunks else []

    hierarchy = []
    if chunks:
        doc_node = {"doc_id": "Document Complet", "chunks": []}
        for i, c in enumerate(chunks[:3]):
            sub_chunks = [s.strip() + "." for s in c.split('.') if s.strip()]
            doc_node["chunks"].append({
                "id": f"Chunk {i+1}",
                "text": c[:150] + "...",
                "sub_chunks": sub_chunks[:3]
            })
        hierarchy.append(doc_node)

    chunk_info = ChunkInfo(
        method=best_method,
        total_chunks=len(chunks),
        avg_chunk_size=avg_chunk_size,
        sample_chunks=sample_chunks,
        hierarchy=hierarchy
    )

    emb_stats = emb_svc.get_stats()
    emb_info = EmbeddingInfo(
        total_vectors=emb_stats["total_vectors"],
        dimensions=emb_stats["dimensions"],
        pca_points=emb_svc.get_pca_points(max_points=150, query=query),
    )

    used_method, results = ret_svc.retrieve(query, method=method, top_k=5, model_name=model)
    summary = ret_svc.build_summary(results, query)
    relevant_docs = [
        RetrievalResult(
            doc_id=r["meta"].get("id", f"doc_{i}"),
            excerpt=r["chunk"][:300],
            score=r["score"],
        )
        for i, r in enumerate(results)
    ]

    return VectorialResponse(
        best_method=best_method,
        chunk_info=chunk_info,
        embedding_info=emb_info,
        retrieval_method=used_method,
        relevant_docs=relevant_docs,
        summary=summary,
    )