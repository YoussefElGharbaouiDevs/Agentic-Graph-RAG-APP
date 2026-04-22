import os
from neo4j import GraphDatabase
from dotenv import load_dotenv
from pathlib import Path

# 1. Robust path handling to find your .env file
env_path = Path(__file__).resolve().parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

class Neo4jClient:
    def __init__(self):
        # 2. Get and clean credentials
        self.uri = os.getenv("NEO4J_URI", "").strip()
        self.user = os.getenv("NEO4J_USERNAME", "").strip()
        self.password = os.getenv("NEO4J_PASSWORD", "").strip()
        
        # Security check to prevent the "URI scheme b''" error
        if not self.uri:
            raise ValueError("CRITICAL: NEO4J_URI is empty. Check your .env file.")
            
        print(f"Connecting to: {self.uri}")
        
        # 3. Initialize the driver
        self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))

    def close(self):
        if self.driver:
            self.driver.close()

    def create_relationship(self, entity1, relation, entity2):
        """Creates or merges nodes and their directed relationship."""
        with self.driver.session() as session:
            query = (
                "MERGE (a:Entity {name: $entity1}) "
                "MERGE (b:Entity {name: $entity2}) "
                "MERGE (a)-[r:RELATION {type: $relation}]->(b) "
                "RETURN a, r, b"
            )
            session.run(query, entity1=entity1, relation=relation, entity2=entity2)

    def get_graph_metrics(self):
        """Retrieves node count for the 'Analyse Graphique Avancée' requirement."""
        with self.driver.session() as session:
            query = "MATCH (n) RETURN count(n) as node_count"
            result = session.run(query)
            return result.single()["node_count"]

# ==========================================
# TESTING BLOCK (Line 53 starts here)
# ==========================================
if __name__ == "__main__":
    client = None
    try:
        client = Neo4jClient()
        print("--- STARTING NEO4J CONNECTION TEST ---")
        
        print("1. Injecting test relationship...")
        client.create_relationship("Onde de Rossby", "perturbe", "Vortex Polaire")
        
        count = client.get_graph_metrics()
        print(f"2. Success! Total nodes in Cloud: {count}")
        
    except Exception as e:
        print(f"\n[ERROR]: {e}")
    finally:
        if client:
            client.close()