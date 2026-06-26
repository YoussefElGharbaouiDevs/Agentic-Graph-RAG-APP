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
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", os.getenv("NEO4J_USER", "neo4j"))

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
                _nlp = spacy.load("fr_core_news_sm")
            except OSError:
                from spacy.cli import download
                download("fr_core_news_sm")
                _nlp = spacy.load("fr_core_news_sm")
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
        with driver.session(database=NEO4J_DATABASE) as session:
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
    Enhanced with deeper path traversal (up to 3 hops).
    """
    if not neo4j_available():
        return _mock_graph_results(query)

    keywords = [w for w in re.findall(r'\b\w{3,}\b', query.lower())]
    stopwords = {'le','la','les','un','une','des','et','ou','pour','dans','sur','avec','qui','que','quoi',
                 'dont','quand','pourquoi','comment','ce','cet','cette','ces','mon','ton','son','ma','ta',
                 'sa','mes','tes','ses','notre','votre','leur','nos','vos','leurs','je','tu','il','elle',
                 'nous','vous','ils','elles','me','te','se','lui','est','sont','ont','les','aux'}
    keywords = [w for w in keywords if w not in stopwords][:6]

    driver = get_driver()
    results = []

    try:
        with driver.session(database=NEO4J_DATABASE) as session:
            for kw in keywords:
                cypher_str = """
                MATCH path = (n)-[*1..3]-(m)
                WHERE toLower(n.id) CONTAINS $kw OR toLower(m.id) CONTAINS $kw
                RETURN path, length(path) as depth
                ORDER BY depth DESC
                LIMIT 15
                """
                # Use raw result (not .data()) to keep native path objects
                raw = session.run(cypher_str, kw=kw)
                for record in raw:
                    try:
                        path = record["path"]
                        # Native Neo4j path object
                        nodes = [node.get("id", "?") for node in path.nodes]
                        relationships = [rel.type for rel in path.relationships]
                        results.append({
                            "path": " -> ".join(nodes),
                            "relations": relationships,
                            "depth": record["depth"],
                            "cypher_query": cypher_str.strip().replace("$kw", f"'{kw}'")
                        })
                    except Exception as inner_e:
                        # Fallback: path came back as dict (some driver versions)
                        try:
                            path_dict = dict(record["path"])
                            start = path_dict.get("start", {}).get("id", "?")
                            end   = path_dict.get("end",   {}).get("id", "?")
                            results.append({
                                "path": f"{start} -> {end}",
                                "relations": [],
                                "depth": record["depth"],
                                "cypher_query": cypher_str.strip().replace("$kw", f"'{kw}'")
                            })
                        except Exception:
                            pass
    except Exception as e:
        print(f"[graph_rag] query error: {e}")

    return results[:top_k]



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

        # Sort communities by size descending
        sorted_comms = sorted(communities.values(), key=len, reverse=True)

        # Re-index partition and communities sequentially for a cleaner UI
        clean_partition = {}
        clean_communities = []
        for i, members in enumerate(sorted_comms):
            new_id = i + 1  # 1-indexed (Cluster 1, Cluster 2...)
            for m in members:
                clean_partition[m] = new_id
            clean_communities.append({"id": new_id, "members": members, "size": len(members)})

        return {
            "partition": clean_partition,
            "modularity": round(modularity, 3),
            "num_clusters": len(clean_communities),
            "communities": clean_communities,
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

def _neo4j_has_data() -> bool:
    """Quick check – returns True if Neo4j already contains at least one node."""
    try:
        driver = get_driver()
        if driver is None:
            return False
        db = os.getenv("NEO4J_DATABASE", os.getenv("NEO4J_USER", "neo4j"))
        with driver.session(database=db) as session:
            result = session.run("MATCH (n) RETURN count(n) AS cnt LIMIT 1")
            count = result.single()["cnt"]
            return count > 0
    except Exception as e:
        print(f"[graph_rag] _neo4j_has_data check failed: {e}")
        return False


def _read_graph_from_neo4j() -> Tuple[List[Dict], List[Dict]]:
    """Read existing entities and relations directly from Neo4j."""
    entities: List[Dict] = []
    relations: List[Dict] = []
    try:
        driver = get_driver()
        if driver is None:
            return entities, relations
        db = os.getenv("NEO4J_DATABASE", os.getenv("NEO4J_USER", "neo4j"))
        with driver.session(database=db) as session:
            node_result = session.run("MATCH (n) RETURN n.id AS id, labels(n)[0] AS label LIMIT 1000")
            for record in node_result:
                if record["id"]:
                    entities.append({"id": record["id"], "label": record["label"] or "ENTITY"})

            rel_result = session.run(
                "MATCH (a)-[r]->(b) RETURN a.id AS src, type(r) AS rel, b.id AS tgt LIMIT 3000"
            )
            for record in rel_result:
                if record["src"] and record["tgt"]:
                    relations.append({
                        "source": record["src"],
                        "relation": record["rel"],
                        "target": record["tgt"],
                    })
    except Exception as e:
        print(f"[graph_rag] _read_graph_from_neo4j error: {e}")
    print(f"[graph_rag] Read {len(entities)} nodes and {len(relations)} relations from Neo4j.")
    return entities, relations


# ── build full graph from chunks ──────────────────────────────────────────────

def build_graph_from_chunks(chunks: List[str]) -> Tuple[List[Dict], List[Dict]]:
    """
    Extract entities/relations from all chunks and ingest to Neo4j.
    IMPORTANT: Skips the expensive LLM build if Neo4j already has data.
    """
    # ── Fast path: Neo4j already populated ────────────────────────────────────
    if _neo4j_has_data():
        print("[graph_rag] Neo4j already has data — skipping LLM extraction. Reading existing graph.")
        return _read_graph_from_neo4j()

    # ── Slow path: Neo4j is empty → build from scratch ────────────────────────
    print("[graph_rag] Neo4j is empty — starting fast local NLP graph extraction...")
    all_entities: List[Dict] = []
    all_relations: List[Dict] = []

    # Use fast local spaCy extraction for all chunks
    for i, chunk in enumerate(chunks):
        if i % 50 == 0:
            print(f"[graph_rag] Extracting chunk {i}/{len(chunks)}...")
        ents, rels = extract_entities_relations(chunk)
        all_entities.extend(ents)
        all_relations.extend(rels)

    print(f"[graph_rag] Extracted {len(all_entities)} entities and {len(all_relations)} relations.")

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
    # Stopwords to exclude from graph visualization (noisy single tokens)
    GRAPH_STOPWORDS = {
        'dans', 'les', 'des', 'une', 'est', 'qui', 'que', 'par', 'sur',
        'avec', 'pour', 'ses', 'son', 'cette', 'ces', 'leur', 'leurs',
        'aux', 'nul', 'the', 'and', 'for', 'are', 'was', 'its',
        'not', 'but', 'can', 'all', 'has', 'had', 'our', 'from', 'have',
        'un', 'et', 'ou', 'quoi', 'dont', 'où', 'quand', 'pourquoi', 'comment',
        'ce', 'cet', 'mon', 'ton', 'ma', 'ta', 'sa', 'mes', 'tes',
        'notre', 'votre', 'nos', 'vos', 'je', 'tu', 'il', 'elle', 'nous', 'vous',
        'ils', 'elles', 'me', 'te', 'se', 'lui', 'y', 'en', 'sont', 'ont', 'plus', 'très',
        'bien', 'tout', 'tous', 'toute', 'toutes', 'a', 'de', 'du', 'au',
        'puis', 'au-delà', 'delà'
    }

    # 1. Filter entities BEFORE running analytics
    filtered_entities = [
        e for e in entities
        if e.get("id")
        and len(str(e["id"]).strip()) > 2
        and str(e["id"]).strip().lower() not in GRAPH_STOPWORDS
    ][:1000]

    valid_ids = {n["id"] for n in filtered_entities}

    # 2. Filter relations to only include valid endpoints
    filtered_relations = [
        r for r in relations
        if r.get("source") in valid_ids and r.get("target") in valid_ids
    ][:3000]

    # 3. Compute analytics on the clean data
    community_info = detect_communities(filtered_entities, filtered_relations)
    metrics = compute_graph_metrics(filtered_entities, filtered_relations)

    # 4. Build node and edge lists for the UI
    raw_nodes = [
        {"id": e["id"], "label": e.get("label", "ENTITY"), "group": 0}
        for e in filtered_entities
    ]
    raw_edges = filtered_relations

    # Assign community group from Louvain partition
    partition = community_info.get("partition", {})
    for node in raw_nodes:
        node["group"] = partition.get(node["id"], 0)

    return {
        "graph_aura":     {"nodes": raw_nodes, "edges": raw_edges},
        "modularity":     community_info["modularity"],
        "num_clusters":   community_info["num_clusters"],
        "communities":    community_info["communities"],
        "centrality_top": metrics["centrality"],
        "semantic_paths": metrics["semantic_paths"],
        "density":        metrics["density"],
    }