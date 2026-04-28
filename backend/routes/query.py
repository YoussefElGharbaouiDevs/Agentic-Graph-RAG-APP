from fastapi import APIRouter, HTTPException
from models.schemas import QueryRequest, QueryResponse
from services import agent as agent_svc
from services import retriever as ret_svc
from services import graph_rag as graph_svc

router = APIRouter()


@router.post("/query", response_model=QueryResponse)
async def process_query(req: QueryRequest):
    query = req.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    # Step 1 – agent decision (no results yet → no reward update)
    decision = agent_svc.run_agent(query)
    action   = decision["action_index"]

    # Step 2 – execute chosen pipeline
    answer  = ""
    sources = []

    if action == 1:  # Graph RAG
        graph_results = graph_svc.query_graph(query, top_k=5)
        answer  = _format_graph_answer(graph_results, query)
        sources = [r.get("node", "") for r in graph_results if r.get("node")]

    else:            # Vectorial RAG
        method, results = ret_svc.retrieve(query, method="auto", top_k=5)
        answer  = ret_svc.build_summary(results, query)
        sources = [r["meta"].get("id", f"doc_{i}") for i, r in enumerate(results)]

    # Step 3 – update Q-table with reward
    all_results = graph_results if action == 1 else results   # type: ignore[possibly-undefined]
    reward = agent_svc.compute_reward(action, all_results, query)
    agent_svc.update_q_table(decision["state"], action, reward)

    return QueryResponse(
        query        = query,
        query_type   = decision["query_type"],
        decision     = decision["chosen_action"],
        confidence   = decision["confidence"],
        routed_to    = decision["chosen_action"],
        answer       = answer or "No answer found.",
        sources      = sources[:5],
        policy_path  = decision["decision_path"],
    )


def _format_graph_answer(graph_results: list, query: str) -> str:
    if not graph_results:
        return "No graph results found for this query."
    lines = []
    for r in graph_results[:5]:
        node = r.get("node", "")
        rel  = r.get("rel", "")
        nb   = r.get("neighbor", "")
        if node and rel and nb:
            lines.append(f"• {node} —[{rel}]→ {nb}")
        elif node:
            lines.append(f"• {node}")
    return "\n".join(lines) if lines else "Graph traversal returned no paths."