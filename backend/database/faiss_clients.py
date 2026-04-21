import os
import uuid
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

class FaissClient:
    def __init__(self, index_path="faiss_index"):
        self.index_path = index_path
        # Initialize the local HuggingFace embedding model
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.vector_store = None
        
        # In-memory dictionary to store our Parent Chunks.
        # In a massive production system, this would be a NoSQL DB like MongoDB.
        self.parent_store = {} 

    def hierarchical_chunking_and_store(self, clean_text: str):
        """Implements Document -> Chunks -> Sub-Chunks hierarchy."""
        print("1. Starting Hierarchical Chunking (Recursive Splitting)...")
        
        # Level 1: Parent Chunks (Large blocks for context)
        parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000, 
            chunk_overlap=200
        )
        
        # Level 2: Sub-Chunks (Small blocks for accurate semantic search)
        child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=400, 
            chunk_overlap=50
        )

        parent_chunks = parent_splitter.split_text(clean_text)
        print(f"   -> Created {len(parent_chunks)} Parent Chunks.")

        sub_chunk_documents = []

        # Iterate through the large chunks to break them down further
        for parent_text in parent_chunks:
            # Generate a unique string ID for this parent
            parent_id = str(uuid.uuid4())
            
            # Save the full parent text in our key-value store
            self.parent_store[parent_id] = parent_text
            
            # Split the parent into smaller sub-chunks
            sub_chunks = child_splitter.split_text(parent_text)
            
            # Create LangChain Document objects and link them to their parent
            for sub_text in sub_chunks:
                doc = Document(
                    page_content=sub_text,
                    metadata={"parent_id": parent_id}
                )
                sub_chunk_documents.append(doc)
                
        print(f"   -> Created {len(sub_chunk_documents)} Sub-Chunks.")

        print("2. Generating embeddings for Sub-Chunks and building FAISS index...")
        self.vector_store = FAISS.from_documents(sub_chunk_documents, self.embeddings)

        print(f"3. Saving index locally to '{self.index_path}'...")
        self.vector_store.save_local(self.index_path)
        print("   -> Success! Vector database created.")

    def search_with_parent_retrieval(self, query: str, k: int = 3):
        """Searches Sub-Chunks but returns the full context of the Parent Chunk."""
        if not self.vector_store:
            self.vector_store = FAISS.load_local(
                self.index_path, 
                self.embeddings, 
                allow_dangerous_deserialization=True
            )
        
        print(f"\nSearching vector store for: '{query}'")
        
        # 1. Find the most relevant small Sub-Chunks
        sub_chunk_results = self.vector_store.similarity_search(query, k=k)
        
        # 2. Retrieve their original Parent Chunks using the metadata ID
        retrieved_parents = []
        seen_parent_ids = set() # O(1) lookup to prevent duplicates
        
        for doc in sub_chunk_results:
            parent_id = doc.metadata.get("parent_id")
            if parent_id and parent_id not in seen_parent_ids:
                retrieved_parents.append(self.parent_store[parent_id])
                seen_parent_ids.add(parent_id)
                
        return retrieved_parents

# ==========================================
# TESTING BLOCK (Runs only when executed directly in the terminal)
# ==========================================
if __name__ == "__main__":
    client = FaissClient()
    
    # Mocking the thesis text
    mock_thesis_text = (
        "Les ondes de Rossby sont des ondes planétaires qui se forment dans l'atmosphère. "
        "Elles jouent un rôle crucial dans la dynamique de la stratosphère. "
        "Lorsqu'elles se propagent vers le haut, elles peuvent déferler et perturber le vortex polaire. "
        "Cette perturbation est souvent la cause principale des échauffements stratosphériques majeurs (SSW). "
        "Ces événements provoquent une inversion de la circulation des vents et une forte hausse de température. "
    ) * 30 # Multiplying the text to force multiple chunks
    
    print("--- STARTING DATA INGESTION TEST ---")
    client.hierarchical_chunking_and_store(mock_thesis_text)
    
    print("\n--- STARTING RETRIEVAL TEST ---")
    test_question = "Qu'est-ce qui cause les échauffements stratosphériques majeurs ?"
    
    retrieved_docs = client.search_with_parent_retrieval(test_question, k=2)
    
    for i, text in enumerate(retrieved_docs):
        print(f"\n[Result {i+1} (Full Parent Context)]:\n{text[:200]}...") # Printing first 200 chars for brevity