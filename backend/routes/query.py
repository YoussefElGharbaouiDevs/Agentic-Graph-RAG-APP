from fastapi import APIRouter, HTTPException
from models.schemas import QueryRequest, QueryResponse
from services import agent as agent_svc
from services import retriever as ret_svc
from services import graph_rag as graph_svc
import os

router = APIRouter()


def _synthesize_with_llm(context: str, query: str) -> str:
    """Use OpenAI to synthesize a real answer from raw context."""
    # Detect query language
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
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        prompt = (
            f"You are a scientific assistant. {lang_instruction} "
            f"Use ONLY the context provided. "
            f"If the context does not contain enough information, say so clearly.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\n\n"
            f"Answer concisely in 2-4 sentences:"
        )
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
            temperature=0.2,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"[query] LLM synthesis failed: {e}")
        return context  # fallback to raw context



@router.post("/query", response_model=QueryResponse)
def process_query(req: QueryRequest):
    query = req.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    # Minimum length guard — reject single words/letters
    word_count = len(query.split())
    if len(query) < 5 or word_count < 2:
        return QueryResponse(
            query      = query,
            query_type = "unknown",
            decision   = "Rejected",
            confidence = 0.0,
            routed_to  = "None",
            answer     = (
                "Votre question est trop courte pour être traitée. "
                "Veuillez formuler une question complète.\n\n"
                "Your question is too short to process. "
                "Please ask a complete question."
            ),
            sources    = [],
            policy_path = ["Query rejected: too short (< 2 words)"],
        )

    # Step 1 – agent decision
    decision = agent_svc.run_agent(query)
    action   = decision["action_index"]

    answer  = ""
    sources = []
    graph_results = []
    results = []

    # Dynamic top_k: systematic queries need wider retrieval net
    top_k = 8 if decision["query_type"] == "systematic" else 5

    if action == 1:  # Graph RAG
        graph_results = graph_svc.query_graph(query, top_k=top_k)
        sources = [r.get("path", "") for r in graph_results if r.get("path")]

        if graph_results:
            # Build context from graph paths and synthesize with LLM
            context = "\n".join(
                f"- {r.get('path', '')} (depth={r.get('depth', 1)})"
                for r in graph_results[:8]
            )
            answer = _synthesize_with_llm(context, query)
        else:
            # Graph returned nothing → fall back to vectorial
            print("[query] Graph RAG returned no results, falling back to vectorial.")
            method, results = ret_svc.retrieve(query, method="auto", top_k=top_k)
            sources = [r["meta"].get("id", f"doc_{i}") for i, r in enumerate(results)]
            if results:
                context = "\n".join(r["chunk"][:1500] for r in results[:top_k])
                answer = _synthesize_with_llm(context, query)

    else:  # Vectorial RAG
        method, results = ret_svc.retrieve(query, method="auto", top_k=top_k)
        sources = [r["meta"].get("id", f"doc_{i}") for i, r in enumerate(results)]
        if results:
            context = "\n".join(r["chunk"][:1500] for r in results[:top_k])
            answer = _synthesize_with_llm(context, query)

    # Final fallback — truly no context found
    if not answer or not sources:
        answer = "I could not find enough information to answer this question. / Je n'ai pas trouvé suffisamment d'informations pour répondre à cette question."
        sources = []

    # Step 3 – update Q-table with reward
    all_results = graph_results if action == 1 else results
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
        state        = decision["state"]
    )


@router.post("/feedback")
def process_feedback(req: dict): # using dict or import FeedbackRequest
    from models.schemas import FeedbackRequest
    try:
        feedback = FeedbackRequest(**req)
        # Action map: 1 = Graph, 0 = Vectorial
        action_idx = 1 if "Graph" in feedback.action else 0
        agent_svc.update_q_table(feedback.state, action_idx, feedback.reward_adjustment)
        return {"status": "success", "message": "Q-Table updated with manual feedback."}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))