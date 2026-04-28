import os
import time
from dotenv import load_dotenv
from pathlib import Path
from langchain_openai import ChatOpenAI
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_neo4j import Neo4jGraph  # <-- Updated Import
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. Robust path handling
env_path = Path(__file__).resolve().parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

class GraphExtractor:
    def __init__(self):
        # 2. Connect to Neo4j Aura Cloud (Updated for bolt+ssc bypass)
        self.graph = Neo4jGraph(
            url=os.getenv("NEO4J_URI"),
            username=os.getenv("NEO4J_USERNAME"),
            password=os.getenv("NEO4J_PASSWORD"),
            refresh_schema=False
        )
        
        # 3. Initialize the OpenAI Brain
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("CRITICAL: OPENAI_API_KEY is missing from .env")
            
        self.llm = ChatOpenAI(temperature=0, model="gpt-4o-mini", api_key=api_key)
        self.transformer = LLMGraphTransformer(llm=self.llm)

    def process_large_document(self, full_text):
        """Safely processes massive texts without hitting API limits."""
        print("-> Chunking document to respect Free API limits...")
        
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = splitter.split_text(full_text)
        
        print(f"-> Total chunks to process: {len(chunks)}")
        
        for i, chunk_text in enumerate(chunks):
            print(f"\n--- Extracting Chunk {i+1}/{len(chunks)} ---")
            try:
                doc = Document(page_content=chunk_text)
                graph_documents = self.transformer.convert_to_graph_documents([doc])
                
                if graph_documents:
                    self.graph.add_graph_documents(graph_documents)
                    print(f"✓ Chunk {i+1} injected into Neo4j Cloud.")
                else:
                    print(f"⚠ No relationships found in Chunk {i+1}.")
                
                if i < len(chunks) - 1:
                    print("Pausing for 20 seconds to avoid OpenAI ban...")
                    time.sleep(20)
                    
            except Exception as e:
                print(f"[ERROR] Failed on chunk {i+1}: {e}")
                print("Pausing for 60 seconds before trying the next one...")
                time.sleep(60)

# ==========================================
# TESTING BLOCK
# ==========================================
if __name__ == "__main__":
    extractor = GraphExtractor()
    
    sample_thesis_text = (
        "Les ondes de Rossby sont des ondes planétaires géantes qui se forment dans l'atmosphère. "
        "Lorsqu'elles se propagent vers la stratosphère, les ondes de Rossby perturbent "
        "fortement le vortex polaire, ce qui provoque un réchauffement stratosphérique majeur. " * 10
    )
    
    print("--- STARTING AI GRAPH EXTRACTION PIPELINE ---")
    extractor.process_large_document(sample_thesis_text)
    print("\n--- PIPELINE COMPLETE ---")