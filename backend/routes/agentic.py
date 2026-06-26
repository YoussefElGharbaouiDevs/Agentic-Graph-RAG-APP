from fastapi import APIRouter, Query
from models.schemas import AgenticResponse, QTableEntry
from services import agent as agent_svc
from services import retriever as ret_svc
from services import graph_rag as graph_svc
from services import embeddings as emb_svc

router = APIRouter()


@router.get("/agentic", response_model=AgenticResponse)
def agentic_rag(
    query: str = Query(..., description="User question"),
):
    # ── 0. Input validation ───────────────────────────────────────────────────
    query = query.strip()
    if len(query) < 5 or len(query.split()) < 2:
        return AgenticResponse(
            query           = query,
            state_features  = {"semantic_score": 0, "entity_count": 0, "complexity": 0, "question_type": 0},
            chosen_action   = "Rejected",
            confidence      = 0.0,
            reward          = 0.0,
            q_table_snapshot= [],
            decision_path   = ["Query rejected: too short (< 2 words)"],
            final_answer    = (
                "Votre question est trop courte. Veuillez formuler une question complète.\n"
                "Your question is too short. Please ask a complete question."
            ),
            provenance      = "Input Validation",
            cypher_queries  = [],
        )

    # ── 1. Initial agent decision ─────────────────────────────────────────────
    decision = agent_svc.run_agent(query)
    action   = decision["action_index"]

    # ── 2. Execute chosen pipeline ────────────────────────────────────────────
    results: list = []
    final_answer  = ""
    provenance    = ""
    cypher_queries = []

    # Dynamic top_k: systematic queries cast a wider retrieval net
    top_k = 8 if decision.get("query_type") == "systematic" else 5

    if action == 1:  # Graph RAG
        if not emb_svc.index_ready():
            emb_svc.load_index()
        chunks = emb_svc._chunks[:100]
        entities, relations = graph_svc.build_graph_from_chunks(chunks)
        results = graph_svc.query_graph(query, top_k=top_k)

        if results:
            paths = [r.get("path", "") for r in results if r.get("path")]
            cypher_queries = list(set([r.get("cypher_query", "") for r in results if r.get("cypher_query")]))
            for r in results:
                if r.get("path") and not r.get("chunk"):
                    r["chunk"] = f"Graph relation: {r['path']}"
            final_answer = ret_svc.build_summary(results, query)
            provenance   = "Neo4j Graph RAG"
        else:
            # Graph returned nothing → fallback to Vectorial RAG
            print("[agentic] Graph RAG returned 0 results → falling back to Vectorial RAG")
            method, results = ret_svc.retrieve(query, method="auto", top_k=top_k)
            final_answer = ret_svc.build_summary(results, query)
            provenance   = f"Fallback: FAISS Vectorial RAG ({method})"

    else:  # Vectorial RAG
        method, results = ret_svc.retrieve(query, method="auto", top_k=top_k)
        final_answer = ret_svc.build_summary(results, query)
        provenance   = f"FAISS Vectorial RAG ({method})"

    # Check for out-of-context
    if not results or (action == 0 and not any(r.get("score", 0) > 0.1 for r in results)) or (action == 1 and not results):
        final_answer = "I don't know."
        provenance = "Out of context"

    # ── 3. Reward + Q-table update ────────────────────────────────────────────
    reward = agent_svc.compute_reward(action, results, query)
    agent_svc.update_q_table(
        state    = decision["state"],
        action   = action,
        reward   = reward,
    )

    # ── 4. Extended decision path ─────────────────────────────────────────────
    decision_path = decision["decision_path"] + [
        f"Pipeline executed: {provenance}",
        f"Results obtained: {len(results)} items",
        f"Reward: {reward:.3f}",
        "Q-table updated ✓",
    ]

    # ── 5. Q-table snapshot ───────────────────────────────────────────────────
    q_snapshot_raw = agent_svc.get_q_table_snapshot()
    q_snapshot = [
        QTableEntry(
            state             = row["state"],
            action_graph      = row["action_graph"],
            action_vectorial  = row["action_vectorial"],
        )
        for row in q_snapshot_raw[:10]
    ]

    return AgenticResponse(
        query            = query,
        state_features   = decision["state_features"],
        chosen_action    = decision["chosen_action"],
        confidence       = decision["confidence"],
        reward           = reward,
        q_table_snapshot = q_snapshot,
        decision_path    = decision_path,
        final_answer     = final_answer or "No answer generated.",
        provenance       = provenance,
        cypher_queries   = cypher_queries,
    )