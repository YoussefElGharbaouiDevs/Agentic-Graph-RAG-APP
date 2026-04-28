import os
import traceback
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

from services import embeddings as emb_svc
from services import chunker as chk_svc
from routes import query, vectorial, graph, agentic

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[startup] Checking FAISS index …")
    if emb_svc.load_index():
        stats = emb_svc.get_stats()
        print(f"[startup] Loaded existing index: {stats['total_vectors']} vectors.")
    else:
        pdf_path = Path(os.getenv("PDF_PATH", "data/13.pdf"))
        if pdf_path.exists():
            print(f"[startup] Building index from {pdf_path} …")
            text = chk_svc.extract_text_from_pdf(str(pdf_path))
            method, chunks = chk_svc.run_chunking(text, method="auto")
            print(f"[startup] Chunked with '{method}': {len(chunks)} chunks.")
            emb_svc.build_index(chunks)
            print(f"[startup] FAISS index built: {len(chunks)} vectors.")
        else:
            print(f"[startup] WARNING: PDF not found at {pdf_path}.")
    yield
    print("[shutdown] Bye.")

app = FastAPI(
    title="Agentic Graph RAG API",
    version="1.0.0",
    lifespan=lifespan,
)

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    print("ERROR:", traceback.format_exc())
    return JSONResponse(status_code=500, content={"detail": str(exc)})

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(query.router,     tags=["Query"])
app.include_router(vectorial.router, tags=["Vectorial RAG"])
app.include_router(graph.router,     tags=["Graph RAG"])
app.include_router(agentic.router,   tags=["Agentic RAG"])

@app.get("/")
async def root():
    stats = emb_svc.get_stats()
    return {
        "status": "ok",
        "index_ready": stats["index_ready"],
        "total_vectors": stats["total_vectors"],
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)