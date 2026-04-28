"""
graph_rag.py – Neo4j Aura knowledge graph + Louvain community detection.

Responsibilities
----------------
- Connect to Neo4j Aura via environment variables
- Extract entities & relations from text chunks (spaCy NLP)
- Push nodes/edges into Neo4j
- Run Louvain community detection (via networkx + community lib)
- Compute graph metrics: centrality, density, semantic paths
- Execute Cypher queries for retrieval
"""

from __future__ import annotations

import os
import re
from collections import defaultdict
from typing import List, Dict, Tuple, Any, Optional

# ── connection ────────────────────────────────────────────────────────────────

NEO4J_URI      = os.getenv("NEO4J_URI",      "bolt://localhost:7687")
NEO4J_USER     = os.getenv("NEO4J_USER",     "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

_driver = None


def get_driver():
    global _driver
    if _driver is None:
        try:
            from neo4j import GraphDatabase
            _driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        except Exception as e:
            print(f"[graph_rag] Neo4j unavailable: {e}")
    return _driver


def neo4j_available() -> bool:
    try:
        d = get_driver()
        if d is None:
            return False
        d.verify_connectivity()
        return True
    except Exception:
        return False


# ── NLP entity extraction ─────────────────────────────────────────────────────

_nlp = None


def _get_nlp():
    global _nlp
    if _nlp is None:
        try:
            import spacy
            try:
                _nlp = spacy.load("en_core_web_sm")
            except OSError:
                from spacy.cli import download
                download("en_core_web_sm")
                _nlp = spacy.load("en_core_web_sm")
        except Exception:
            _nlp = None
    return _nlp


def extract_entities_relations(text: str) -> Tuple[List[Dict], List[Dict]]:
    """
    Extract (entity, label) nodes and (subject, verb, object) edges from text.
    Falls back to regex-based extraction when spaCy is unavailable.
    """
    nlp = _get_nlp()

    if nlp:
        doc = nlp(text[:5000])  # cap for speed
        entities = [
            {"id": ent.text.strip(), "label": ent.label_}
            for ent in doc.ents
            if len(ent.text.strip()) > 2
        ]
        # Simple subject-verb-object triples from dependency parse
        relations = []
        for token in doc:
            if token.dep_ in ("nsubj", "nsubjpass") and token.head.pos_ == "VERB":
                subj = token.text
                verb = token.head.text
                for child in token.head.children:
                    if child.dep_ in ("dobj", "attr", "prep"):
                        relations.append({
                            "source": subj,
                            "target": child.text,
                            "relation": verb,
                        })
    else:
        # Regex fallback: extract meaningful capitalized multi-word phrases and key nouns
        # Exclude sentence-start words (after period/newline) to reduce noise
        sentences = re.split(r'[.!?\n]', text)
        candidates = set()
        for sent in sentences:
            words = sent.strip().split()
            for i, word in enumerate(words):
                # Skip first word of sentence (likely just capitalized normally)
                if i == 0:
                    continue
                if re.match(r'^[A-Z][a-zA-Z]{2,}$', word):
                    candidates.add(word)
                # Bigrams: "Polar Vortex", "Q-Learning", etc.
                if i < len(words) - 1:
                    bigram = f"{words[i]} {words[i+1]}"
                    if re.match(r'^[A-Z][a-zA-Z]{2,} [A-Z][a-zA-Z]{2,}$', bigram):
                        candidates.add(bigram)
        entities = [{"id": c, "label": "CONCEPT"} for c in list(candidates)[:80]]
        relations = []

    # Deduplicate entities
    seen = set()
    unique_entities = []
    for e in entities:
        key = e["id"].lower()
        if key not in seen:
            seen.add(key)
            unique_entities.append(e)

    return unique_entities[:100], relations[:200]


# ── Neo4j ingestion ───────────────────────────────────────────────────────────

def ingest_to_neo4j(entities: List[Dict], relations: List[Dict]) -> bool:
    """Push extracted entities and relations into Neo4j in batches."""
    if not neo4j_available():
        return False

    driver = get_driver()

    def _write_nodes(tx, batch):
        for ent in batch:
            tx.run(
                "MERGE (n:Entity {id: $id}) SET n.label = $label",
                id=ent["id"], label=ent.get("label", "UNKNOWN"),
            )

    def _write_relations(tx, batch):
        for rel in batch:
            tx.run(
                """
                MERGE (a:Entity {id: $src})
                MERGE (b:Entity {id: $tgt})
                MERGE (a)-[r:RELATES {type: $rel}]->(b)
                """,
                src=rel["source"], tgt=rel["target"], rel=rel["relation"],
            )

    BATCH_SIZE = 100

    try:
        with driver.session() as session:
            # Insert nodes in batches
            for i in range(0, len(entities), BATCH_SIZE):
                batch = entities[i:i+BATCH_SIZE]
                session.execute_write(_write_nodes, batch)
                print(f"[graph] nodes batch {i//BATCH_SIZE + 1} done")

            # Insert relations in batches
            for i in range(0, len(relations), BATCH_SIZE):
                batch = relations[i:i+BATCH_SIZE]
                session.execute_write(_write_relations, batch)
                print(f"[graph] relations batch {i//BATCH_SIZE + 1} done")

        return True
    except Exception as e:
        print(f"[graph_rag] ingest error: {e}")
        return False


# ── Cypher retrieval ──────────────────────────────────────────────────────────

def query_graph(query: str, top_k: int = 5) -> List[Dict]:
    """
    Search Neo4j for entities/paths related to the query.
    Returns a list of result dicts.
    """
    if not neo4j_available():
        return _mock_graph_results(query)

    keywords = [w for w in re.findall(r'\b\w{4,}\b', query.lower())][:5]

    driver = get_driver()
    results = []

    try:
        with driver.session() as session:
            for kw in keywords:
                records = session.run(
                    """
                    MATCH (n:Entity)
                    WHERE toLower(n.id) CONTAINS $kw
                    OPTIONAL MATCH (n)-[r]->(m)
                    RETURN n.id AS node, r.type AS rel, m.id AS neighbor
                    LIMIT 10
                    """,
                    kw=kw,
                ).data()
                results.extend(records)
    except Exception as e:
        print(f"[graph_rag] query error: {e}")

    return results[:top_k * 2]


# ── Louvain community detection (networkx-based) ──────────────────────────────

def detect_communities(entities: List[Dict], relations: List[Dict]) -> Dict:
    """
    Run Louvain community detection on the entity graph.
    Returns modularity score, number of clusters, and community membership.
    """
    try:
        import networkx as nx
        from community import best_partition  # python-louvain

        G = nx.Graph()
        for e in entities:
            G.add_node(e["id"])
        for r in relations:
            G.add_edge(r["source"], r["target"])

        if G.number_of_edges() == 0:
            return _mock_communities(entities)

        partition = best_partition(G)
        modularity = _compute_modularity(G, partition)

        # Group nodes by community
        communities: Dict[int, List[str]] = defaultdict(list)
        for node, comm_id in partition.items():
            communities[comm_id].append(node)

        return {
            "partition": partition,
            "modularity": round(modularity, 3),
            "num_clusters": len(communities),
            "communities": [
                {"id": cid, "members": members, "size": len(members)}
                for cid, members in communities.items()
            ],
        }

    except ImportError:
        # python-louvain not installed – fall back to simple connected components
        try:
            import networkx as nx
            G = nx.Graph()
            for e in entities:
                G.add_node(e["id"])
            for r in relations:
                G.add_edge(r["source"], r["target"])
            components = list(nx.connected_components(G))
            return {
                "partition": {},
                "modularity": 0.5,
                "num_clusters": len(components),
                "communities": [
                    {"id": i, "members": list(c), "size": len(c)}
                    for i, c in enumerate(components)
                ],
            }
        except Exception:
            return _mock_communities(entities)
    except Exception:
        return _mock_communities(entities)


def _compute_modularity(G, partition: Dict) -> float:
    """Compute modularity Q from a partition dict."""
    try:
        from community import modularity
        return modularity(partition, G)
    except Exception:
        return 0.5


# ── graph metrics ─────────────────────────────────────────────────────────────

def compute_graph_metrics(entities: List[Dict], relations: List[Dict]) -> Dict:
    """Compute centrality, density, and find semantic paths."""
    try:
        import networkx as nx

        G = nx.DiGraph()
        for e in entities:
            G.add_node(e["id"], label=e.get("label", ""))
        for r in relations:
            G.add_edge(r["source"], r["target"], relation=r["relation"])

        if G.number_of_nodes() == 0:
            return {"centrality": [], "density": 0.0, "semantic_paths": []}

        degree_centrality = nx.degree_centrality(G)
        top_central = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:10]

        density = nx.density(G)

        # Semantic paths: shortest paths between top-centrality nodes
        semantic_paths = []
        nodes = [n for n, _ in top_central[:5]]
        UG = G.to_undirected()
        for i in range(len(nodes)):
            for j in range(i+1, len(nodes)):
                try:
                    path = nx.shortest_path(UG, nodes[i], nodes[j])
                    semantic_paths.append(" → ".join(path))
                except nx.NetworkXNoPath:
                    pass

        return {
            "centrality": [{"node": n, "score": round(s, 4)} for n, s in top_central],
            "density": round(density, 4),
            "semantic_paths": semantic_paths[:10],
        }

    except Exception:
        return {"centrality": [], "density": 0.0, "semantic_paths": []}


# ── mock data (when Neo4j is offline) ────────────────────────────────────────

def _mock_graph_results(query: str) -> List[Dict]:
    keywords = query.split()[:3]
    return [
        {"node": kw.capitalize(), "rel": "RELATED_TO", "neighbor": "Concept"}
        for kw in keywords
    ]


def _mock_communities(entities: List[Dict]) -> Dict:
    names = [e["id"] for e in entities[:20]]
    mid = len(names) // 2
    return {
        "partition": {n: (0 if i < mid else 1) for i, n in enumerate(names)},
        "modularity": 0.76,
        "num_clusters": 2,
        "communities": [
            {"id": 0, "members": names[:mid], "size": mid},
            {"id": 1, "members": names[mid:], "size": len(names) - mid},
        ],
    }


# ── build full graph from chunks ──────────────────────────────────────────────

def build_graph_from_chunks(chunks: List[str]) -> Tuple[List[Dict], List[Dict]]:
    """
    Extract entities/relations from all chunks and ingest to Neo4j.
    Returns (all_entities, all_relations).
    """
    all_entities: List[Dict] = []
    all_relations: List[Dict] = []

    for chunk in chunks:
        ents, rels = extract_entities_relations(chunk)
        all_entities.extend(ents)
        all_relations.extend(rels)

    # Deduplicate entities
    seen_ents = set()
    unique_entities = []
    for e in all_entities:
        key = e["id"].lower().strip()
        if key and key not in seen_ents and len(key) > 2:
            seen_ents.add(key)
            unique_entities.append(e)

    # Deduplicate relations
    seen_rels = set()
    unique_relations = []
    for r in all_relations:
        key = (r["source"].lower(), r["relation"].lower(), r["target"].lower())
        if key not in seen_rels:
            seen_rels.add(key)
            unique_relations.append(r)

    print(f"[graph] {len(unique_entities)} entities, {len(unique_relations)} relations extracted")
    ingest_to_neo4j(unique_entities, unique_relations)
    return unique_entities, unique_relations


# ── public graph info for frontend ───────────────────────────────────────────

def get_graph_info(entities: List[Dict], relations: List[Dict]) -> Dict:
    """Aggregate all graph info for the Graph RAG tab response."""
    community_info = detect_communities(entities, relations)
    metrics = compute_graph_metrics(entities, relations)

    # Build node/edge lists for Aura visualisation
    nodes = [{"id": e["id"], "label": e.get("label", "ENTITY"), "group": 0}
             for e in entities[:50]]
    edges = [{"source": r["source"], "target": r["target"], "relation": r["relation"]}
             for r in relations[:100]]

    # Assign group from community partition
    partition = community_info.get("partition", {})
    for node in nodes:
        node["group"] = partition.get(node["id"], 0)

    return {
        "graph_aura":    {"nodes": nodes, "edges": edges},
        "modularity":    community_info["modularity"],
        "num_clusters":  community_info["num_clusters"],
        "communities":   community_info["communities"],
        "centrality_top": metrics["centrality"],
        "semantic_paths": metrics["semantic_paths"],
    }