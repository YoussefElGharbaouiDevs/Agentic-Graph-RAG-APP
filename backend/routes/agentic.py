from fastapi import APIRouter, Query
from models.schemas import AgenticResponse, QTableEntry
from services import agent as agent_svc
from services import retriever as ret_svc
from services import graph_rag as graph_svc
from services import embeddings as emb_svc

router = APIRouter()


@router.get("/agentic", response_model=AgenticResponse)
async def agentic_rag(
    query: str = Query(..., description="User question"),
):
    # ── 1. Initial agent decision ─────────────────────────────────────────────
    decision = agent_svc.run_agent(query)
    action   = decision["action_index"]

    # ── 2. Execute chosen pipeline ────────────────────────────────────────────
    results: list = []
    final_answer  = ""
    provenance    = ""

    if action == 1:  # Graph RAG
        if not emb_svc.index_ready():
            emb_svc.load_index()
        chunks = emb_svc._chunks[:100]
        entities, relations = graph_svc.build_graph_from_chunks(chunks)
        results = graph_svc.query_graph(query, top_k=5)

        paths = [f"{r.get('node','')} →[{r.get('rel','')}]→ {r.get('neighbor','')}"
                 for r in results if r.get("node")]
        final_answer = "\n".join(paths) if paths else "No graph paths found."
        provenance   = "Neo4j Graph RAG"

    else:  # Vectorial RAG
        method, results = ret_svc.retrieve(query, method="auto", top_k=5)
        final_answer = ret_svc.build_summary(results, query)
        provenance   = f"FAISS Vectorial RAG ({method})"

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
    )