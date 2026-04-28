from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any


# ── Query ──────────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="User question")

class QueryResponse(BaseModel):
    query: str
    query_type: str                  # "semantic" | "systematic" | "hybrid"
    decision: str                    # "Graph RAG" | "Vectorial RAG"
    confidence: float                # 0.0 – 1.0
    routed_to: str
    answer: str
    sources: List[str] = []
    policy_path: List[str] = []      # steps the agent took


# ── Vectorial RAG ──────────────────────────────────────────────────────────────

class ChunkInfo(BaseModel):
    method: str
    total_chunks: int
    avg_chunk_size: int
    sample_chunks: List[str] = []

class EmbeddingInfo(BaseModel):
    total_vectors: int
    dimensions: int
    pca_points: List[Dict[str, float]] = []  # [{x, y, label}]

class RetrievalResult(BaseModel):
    doc_id: str
    excerpt: str
    score: float

class VectorialResponse(BaseModel):
    best_method: str
    chunk_info: ChunkInfo
    embedding_info: EmbeddingInfo
    retrieval_method: str
    relevant_docs: List[RetrievalResult] = []
    summary: str


# ── Graph RAG ──────────────────────────────────────────────────────────────────

class GraphNode(BaseModel):
    id: str
    label: str
    group: int = 0

class GraphEdge(BaseModel):
    source: str
    target: str
    relation: str

class GraphAura(BaseModel):
    nodes: List[GraphNode] = []
    edges: List[GraphEdge] = []

class Community(BaseModel):
    id: int
    members: List[str]
    size: int

class GraphResponse(BaseModel):
    graph_aura: GraphAura
    modularity: float
    num_clusters: int
    communities: List[Community] = []
    centrality_top: List[Dict[str, Any]] = []
    semantic_paths: List[str] = []


# ── Agentic RAG ────────────────────────────────────────────────────────────────

class QTableEntry(BaseModel):
    state: str
    action_graph: float
    action_vectorial: float

class AgenticResponse(BaseModel):
    query: str
    state_features: Dict[str, float]
    chosen_action: str
    confidence: float
    reward: float
    q_table_snapshot: List[QTableEntry] = []
    decision_path: List[str] = []
    final_answer: str
    provenance: str