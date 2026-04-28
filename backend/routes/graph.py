"""
routes/graph.py  –  GET /graph?query=...

Runs the full Graph RAG pipeline and returns:
- Neo4j Aura graph  (nodes + edges for visualisation)
- Louvain communities (modularity + cluster list)
- Advanced graph analysis (centrality + semantic paths)
"""

from fastapi import APIRouter, Query
from models.schemas import GraphResponse, GraphAura, GraphNode, GraphEdge, Community
from services import graph_rag as graph_svc
from services import embeddings as emb_svc

router = APIRouter()


@router.get("/graph", response_model=GraphResponse)
async def graph_rag(
    query: str = Query(..., description="User question"),
):
    # Load existing chunks to build / refresh the graph
    if not emb_svc.index_ready():
        emb_svc.load_index()

    chunks = emb_svc._chunks

    # Extract entities & relations from ALL chunks
    entities, relations = graph_svc.build_graph_from_chunks(chunks)

    # Query the graph for relevant paths
    graph_results = graph_svc.query_graph(query, top_k=10)

    # Aggregate all graph info
    info = graph_svc.get_graph_info(entities, relations)

    # Build response
    nodes = [GraphNode(id=n["id"], label=n["label"], group=n.get("group", 0))
             for n in info["graph_aura"]["nodes"]]
    edges = [GraphEdge(source=e["source"], target=e["target"], relation=e["relation"])
             for e in info["graph_aura"]["edges"]]

    communities = [
        Community(id=c["id"], members=c["members"][:10], size=c["size"])
        for c in info["communities"][:8]
    ]

    return GraphResponse(
        graph_aura      = GraphAura(nodes=nodes, edges=edges),
        modularity      = info["modularity"],
        num_clusters    = info["num_clusters"],
        communities     = communities,
        centrality_top  = info["centrality_top"],
        semantic_paths  = info["semantic_paths"],
    )