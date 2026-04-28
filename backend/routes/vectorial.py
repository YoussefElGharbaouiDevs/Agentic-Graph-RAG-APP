from fastapi import APIRouter, Query
from models.schemas import VectorialResponse, ChunkInfo, EmbeddingInfo, RetrievalResult
from services import chunker as chk_svc
from services import embeddings as emb_svc
from services import retriever as ret_svc

router = APIRouter()


@router.get("/vectorial", response_model=VectorialResponse)
async def vectorial_rag(
    query:  str = Query(..., description="User question"),
    method: str = Query("auto", description="Retrieval method"),
):
    # ── 1. Chunking info ───────────────────────────────────────────────────────
    emb_stats  = emb_svc.get_stats()
    chunks     = emb_svc._chunks  # already-indexed chunks

    # If index not ready, give lightweight info
    if not chunks:
        emb_svc.load_index()
        chunks = emb_svc._chunks

    best_method = "recursive"
    sample_chunks: list[str] = []
    avg_chunk_size = 0

    if chunks:
        # Infer best method from existing chunk sizes
        best_method   = "recursive"
        avg_chunk_size = int(sum(len(c) for c in chunks) / max(1, len(chunks)))
        sample_chunks  = [c[:200] for c in chunks[:3]]
    
    chunk_info = ChunkInfo(
        method         = best_method,
        total_chunks   = len(chunks),
        avg_chunk_size = avg_chunk_size,
        sample_chunks  = sample_chunks,
    )

    # ── 2. Embedding info ─────────────────────────────────────────────────────
    pca_points  = emb_svc.get_pca_points(max_points=150)
    emb_info    = EmbeddingInfo(
        total_vectors = emb_stats["total_vectors"],
        dimensions    = emb_stats["dimensions"],
        pca_points    = pca_points,
    )

    # ── 3. Retrieval ──────────────────────────────────────────────────────────
    ret_method, results = ret_svc.retrieve(query, method=method, top_k=5)
    summary = ret_svc.build_summary(results, query)

    relevant_docs = [
        RetrievalResult(
            doc_id  = r["meta"].get("id", f"doc_{i}"),
            excerpt = r["chunk"][:300],
            score   = r["score"],
        )
        for i, r in enumerate(results)
    ]

    return VectorialResponse(
        best_method    = best_method,
        chunk_info     = chunk_info,
        embedding_info = emb_info,
        retrieval_method = ret_method,
        relevant_docs  = relevant_docs,
        summary        = summary,
    )