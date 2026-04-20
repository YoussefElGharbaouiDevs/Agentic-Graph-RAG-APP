## Project Structure

project/
├── backend/
│   ├── main.py              # FastAPI app, CORS, route registration
│   ├── routes/
│   │   ├── query.py         # POST /query  → Q-Learning classification
│   │   ├── vectorial.py     # GET  /vectorial → chunking + FAISS
│   │   ├── graph.py         # GET  /graph  → Neo4j + Louvain
│   │   └── agentic.py       # GET  /agentic → agent decision + reward
│   ├── services/
│   │   ├── chunker.py       # 7 chunking strategies (LangChain splitters)
│   │   ├── embeddings.py    # sentence-transformers + FAISS index
│   │   ├── retriever.py     # Top-k, cosine, BM25 hybrid search
│   │   ├── graph_rag.py     # Neo4j driver, Cypher queries, Louvain
│   │   └── agent.py         # Q-table, feature extraction, reward fn
│   └── models/
│       └── schemas.py       # Pydantic request/response models
└── frontend/
    ├── src/
    │   ├── App.tsx
    │   ├── tabs/
    │   │   ├── QueryTab.tsx
    │   │   ├── VectorialTab.tsx
    │   │   ├── GraphTab.tsx
    │   │   └── AgenticTab.tsx
    │   ├── components/       # Charts, graph canvas, reward monitor
    │   └── api/
    │       └── client.ts     # axios instance pointing at FastAPI
    └── package.json