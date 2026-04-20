# Agentic Graph RAG

A full-stack AI retrieval system that combines Vectorial RAG (FAISS) and Graph RAG (Neo4j) under a Q-Learning agent that intelligently routes each query to the most appropriate pipeline.

Built with **FastAPI**, **LangChain**, **React + TypeScript**, **FAISS**, and **Neo4j Aura**.

---

## Architecture overview

```
User Query
    │
    ▼
React Frontend (TypeScript)
    │  REST / JSON
    ▼
FastAPI Backend
    ├── POST /query      ← Q-Learning agent classifies & routes
    ├── GET  /vectorial  ← Chunking + FAISS embeddings + retrieval
    ├── GET  /graph      ← Neo4j Aura + Louvain + Cypher
    └── GET  /agentic    ← Full agent loop with reward & Q-table update
         │
    ┌────┴────┐
    ▼         ▼
 FAISS      Neo4j Aura
 Index      Knowledge Graph
```

The Q-Learning agent extracts features from each query (semantic score, entity count, query complexity) and selects an action — Vectorial RAG or Graph RAG — based on a learned Q-table. After each response the agent computes a reward and updates the table.

---

## Project structure

```
agentic-graph-rag/
├── backend/
│   ├── main.py                  # FastAPI app entry point
│   ├── routes/
│   │   ├── query.py             # POST /query
│   │   ├── vectorial.py         # GET  /vectorial
│   │   ├── graph.py             # GET  /graph
│   │   └── agentic.py           # GET  /agentic
│   ├── services/
│   │   ├── chunker.py           # 7 chunking strategies (LangChain)
│   │   ├── embeddings.py        # sentence-transformers + FAISS
│   │   ├── retriever.py         # Top-k, cosine, BM25 hybrid search
│   │   ├── graph_rag.py         # Neo4j driver + Cypher + Louvain
│   │   └── agent.py             # Q-Learning agent (Q-table, reward)
│   ├── models/
│   │   └── schemas.py           # Pydantic request/response models
│   ├── .env                     # secrets — never commit this
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── App.tsx
│   │   ├── api/
│   │   │   └── client.ts        # axios instance
│   │   ├── tabs/
│   │   │   ├── QueryTab.tsx
│   │   │   ├── VectorialTab.tsx
│   │   │   ├── GraphTab.tsx
│   │   │   └── AgenticTab.tsx
│   │   └── components/          # charts, graph canvas, reward monitor
│   ├── public/
│   └── package.json
├── .gitignore
└── README.md
```

---

## Prerequisites

- Python 3.10+
- Node.js 18+
- A [Neo4j Aura](https://neo4j.com/cloud/aura/) free instance
- Git

---

## Getting started

### 1. Clone the repository

```bash
git clone https://github.com/your-username/agentic-graph-rag.git
cd agentic-graph-rag
```

### 2. Backend setup

```bash
cd backend
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Create a `.env` file in the `backend/` directory:

```env
NEO4J_URI=bolt+s://your-instance.databases.neo4j.io
NEO4J_USER=neo4j
NEO4J_PASSWORD=your-password
```

Start the server:

```bash
uvicorn main:app --reload
```

The API will be available at `http://localhost:8000`.
Interactive docs at `http://localhost:8000/docs`.

### 3. Frontend setup

```bash
cd frontend
npm install
npm start
```

The UI will be available at `http://localhost:3000`.

---

## API reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/query` | Classifies the query (semantic / systematic / hybrid) and returns routing decision + confidence score |
| `GET` | `/vectorial` | Runs the full Vectorial RAG pipeline: chunking → embeddings → FAISS retrieval → summary |
| `GET` | `/graph` | Runs the Graph RAG pipeline: Neo4j knowledge graph → Louvain communities → Cypher query |
| `GET` | `/agentic` | Full agent loop: feature extraction → Q-table action → pipeline execution → reward → Q-table update |

### Example request

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the relations between transformer layers?"}'
```

### Example response

```json
{
  "analysis": "Systematic",
  "routing": "Graph RAG",
  "confidence": 0.92
}
```

---

## Key concepts

### Vectorial RAG pipeline

1. Documents are split using one of 7 LangChain chunking strategies (auto-selected per corpus).
2. Chunks are embedded with `sentence-transformers/all-MiniLM-L6-v2`.
3. Embeddings are indexed in FAISS using inner product (cosine similarity).
4. At query time, Top-k semantic search retrieves the most relevant chunks.
5. Retrieved context is passed to the LLM for final answer generation.

### Graph RAG pipeline

1. Entities and relations from documents are stored as nodes and edges in Neo4j Aura.
2. Louvain community detection groups related concepts (modularity score reported in the UI).
3. At query time, Cypher queries traverse the knowledge graph for structured retrieval.
4. Graph results are fused with vectorial context for the final answer.

### Q-Learning agent

The agent maintains a Q-table of shape `(n_states, 2)` where the two actions are Vectorial RAG and Graph RAG. At each query:

1. Features are extracted (relational keywords, entity count, query length).
2. Features are mapped to a discrete state index.
3. The agent picks an action using epsilon-greedy policy.
4. The chosen pipeline executes.
5. A reward is computed from response quality.
6. The Q-table is updated via the Bellman equation.

---

## Requirements

### Backend (`requirements.txt`)

```
fastapi
uvicorn
langchain
langchain-community
sentence-transformers
faiss-cpu
neo4j
python-dotenv
pydantic
numpy
```

### Frontend

```
react
typescript
axios
react-router-dom
recharts
```

---

## Environment variables

| Variable | Description |
|----------|-------------|
| `NEO4J_URI` | Bolt URI for your Neo4j Aura instance |
| `NEO4J_USER` | Neo4j username (default: `neo4j`) |
| `NEO4J_PASSWORD` | Neo4j password |

Never commit `.env` to version control. Use `.env.example` as a template.

---

## Development tips

- Run `http://localhost:8000/docs` to test all endpoints interactively via Swagger UI before touching the frontend.
- The Q-table starts at zero. Run 50–100 synthetic queries at startup to pre-train the agent before a demo.
- FAISS index files (`*.index`) are excluded from git — rebuild them from your documents on each fresh clone.
- Use `recharts` for the reward monitor curve and a force-directed graph library (e.g. `react-force-graph`) for the Neo4j visualization tab.

---

## Roadmap

- [ ] Persist Q-table to disk between sessions
- [ ] Add hybrid action (run both pipelines and merge results)
- [ ] Extend action space to include retrieval strategy selection (Top-k vs BM25 vs hybrid)
- [ ] Add user feedback button to provide explicit reward signal
- [ ] Docker Compose setup for one-command deployment

---

## References

- [FastAPI documentation](https://fastapi.tiangolo.com)
- [LangChain documentation](https://python.langchain.com)
- [Neo4j Graph Data Science — Louvain](https://neo4j.com/docs/graph-data-science/current/algorithms/louvain/)
- [FAISS — Facebook AI Similarity Search](https://github.com/facebookresearch/faiss)
- [sentence-transformers](https://www.sbert.net)
- [DataCamp FastAPI tutorial](https://www.datacamp.com/tutorial/introduction-fastapi-tutorial)