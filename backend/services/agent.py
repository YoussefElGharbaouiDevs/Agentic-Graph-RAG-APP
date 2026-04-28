from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# ── hyper-parameters ──────────────────────────────────────────────────────────

ALPHA       = 0.1    # learning rate
GAMMA       = 0.9    # discount factor
EPSILON     = 0.1    # exploration rate (ε-greedy)
Q_TABLE_PATH = Path(os.getenv("Q_TABLE_PATH", "q_table.json"))

ACTIONS = {0: "Vectorial RAG", 1: "Graph RAG"}

# ── relational keywords that push toward Graph RAG ────────────────────────────
GRAPH_KEYWORDS = {
    "relation", "relations", "related", "link", "linked", "connect", "connected",
    "network", "path", "paths", "between", "structure", "hierarchy", "graph",
    "community", "communities", "entity", "entities", "who", "which", "where",
    "neighbour", "neighbor", "cluster", "clusters",
}

# ── semantic keywords that push toward Vectorial RAG ─────────────────────────
SEMANTIC_KEYWORDS = {
    "what", "how", "explain", "describe", "summarise", "summary", "meaning",
    "context", "about", "topic", "theme", "why", "impact", "effect", "cause",
    "trend", "evolution", "analysis", "analyse", "analyse",
}


# ── Q-table management ────────────────────────────────────────────────────────

def _load_q_table() -> Dict[str, List[float]]:
    if Q_TABLE_PATH.exists():
        try:
            with open(Q_TABLE_PATH) as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _save_q_table(table: Dict[str, List[float]]) -> None:
    try:
        with open(Q_TABLE_PATH, "w") as f:
            json.dump(table, f, indent=2)
    except Exception:
        pass


_q_table: Dict[str, List[float]] = _load_q_table()


# ── feature extraction ────────────────────────────────────────────────────────

def extract_features(query: str) -> Dict[str, float]:
    """
    Return a feature dict for the query.
    All values are in [0, 1].
    """
    words = re.findall(r'\b\w+\b', query.lower())
    if not words:
        return {"semantic_score": 0.5, "entity_count": 0.0, "complexity": 0.5, "question_type": 0.0}

    graph_hits    = sum(1 for w in words if w in GRAPH_KEYWORDS)
    semantic_hits = sum(1 for w in words if w in SEMANTIC_KEYWORDS)

    semantic_score = semantic_hits / max(1, semantic_hits + graph_hits)
    entity_count   = min(1.0, len([w for w in words if w[0].isupper()]) / max(1, len(words)))
    complexity     = min(1.0, len(words) / 30.0)

    # Question type encoding
    first_word = words[0]
    if first_word in {"who", "which", "where"}:
        question_type = 1.0
    elif first_word in {"why", "how"}:
        question_type = 0.5
    else:
        question_type = 0.0

    return {
        "semantic_score":  round(semantic_score,  3),
        "entity_count":    round(entity_count,    3),
        "complexity":      round(complexity,       3),
        "question_type":   round(question_type,   3),
    }


# ── state discretisation ──────────────────────────────────────────────────────

def _discretise(features: Dict[str, float]) -> str:
    """Map continuous features to a discrete state key."""
    def level(v: float) -> str:
        if v < 0.33:
            return "L"
        if v < 0.67:
            return "M"
        return "H"

    return (
        level(features["semantic_score"])
        + level(features["entity_count"])
        + level(features["complexity"])
        + level(features["question_type"])
    )


def _get_q_values(state: str) -> List[float]:
    if state not in _q_table:
        _q_table[state] = [0.5, 0.5]  # initialise with equal probability
    return _q_table[state]


# ── policy ────────────────────────────────────────────────────────────────────

def choose_action(features: Dict[str, float], explore: bool = False) -> Tuple[int, float]:
    """
    ε-greedy policy.

    Returns
    -------
    (action_index, confidence)
      action_index : 0=Vectorial, 1=Graph
      confidence   : softmax-derived probability of chosen action
    """
    state = _discretise(features)
    q_vals = _get_q_values(state)

    if explore and np.random.random() < EPSILON:
        action = np.random.randint(0, 2)
    else:
        action = int(np.argmax(q_vals))

    # Confidence via softmax
    exp_q = np.exp(np.array(q_vals) - np.max(q_vals))
    probs = exp_q / exp_q.sum()
    confidence = float(probs[action])

    return action, confidence


# ── reward function ───────────────────────────────────────────────────────────

def compute_reward(action: int, results: List[Dict], query: str) -> float:
    """
    Compute reward based on:
    - Number of relevant results returned
    - Whether chosen action matches query type

    Returns a float in [-1, 1].
    """
    features = extract_features(query)

    # Correctness signal
    if action == 1:  # Graph RAG chosen
        expected = features["entity_count"] + features["question_type"]
    else:            # Vectorial RAG chosen
        expected = features["semantic_score"] + (1 - features["question_type"])

    result_quality = min(1.0, len(results) / 5.0)
    reward = 0.5 * expected + 0.5 * result_quality
    return round(float(reward), 4)


# ── Q-table update ────────────────────────────────────────────────────────────

def update_q_table(
    state: str,
    action: int,
    reward: float,
    next_features: Optional[Dict[str, float]] = None,
) -> None:
    """Standard Q-learning update: Q(s,a) ← Q(s,a) + α[r + γ·maxQ(s') - Q(s,a)]"""
    global _q_table

    q_vals = _get_q_values(state)
    current_q = q_vals[action]

    if next_features:
        next_state = _discretise(next_features)
        max_next_q = max(_get_q_values(next_state))
    else:
        max_next_q = 0.0

    new_q = current_q + ALPHA * (reward + GAMMA * max_next_q - current_q)
    _q_table[state][action] = round(new_q, 6)
    _save_q_table(_q_table)


# ── query type classification ─────────────────────────────────────────────────

def classify_query(features: Dict[str, float]) -> str:
    """Return 'semantic', 'systematic', or 'hybrid' label for the frontend."""
    sem  = features["semantic_score"]
    ent  = features["entity_count"]
    qt   = features["question_type"]

    graph_signal = ent + qt
    sem_signal   = sem

    if abs(graph_signal - sem_signal) < 0.2:
        return "hybrid"
    if graph_signal > sem_signal:
        return "systematic"
    return "semantic"


# ── public API ────────────────────────────────────────────────────────────────

def run_agent(query: str, results_after: Optional[List[Dict]] = None) -> Dict:
    """
    Full agent loop:
    1. Extract features
    2. Choose action
    3. (Optionally) compute reward and update Q-table
    4. Return decision dict for the API

    Parameters
    ----------
    query         : user query
    results_after : retrieval results (if already obtained) for reward calc
    """
    features   = extract_features(query)
    state      = _discretise(features)
    action, confidence = choose_action(features, explore=True)
    query_type = classify_query(features)

    reward = 0.0
    if results_after is not None:
        reward = compute_reward(action, results_after, query)
        update_q_table(state, action, reward)

    q_vals = _get_q_values(state)
    q_snapshot = [
        {"state": state, "action_graph": round(q_vals[1], 4), "action_vectorial": round(q_vals[0], 4)}
    ]

    decision_path = [
        f"Query received: '{query[:60]}...' " if len(query) > 60 else f"Query: '{query}'",
        f"Feature extraction → semantic={features['semantic_score']}, entities={features['entity_count']}",
        f"State discretised → '{state}'",
        f"Q-values → Vectorial={q_vals[0]:.3f}, Graph={q_vals[1]:.3f}",
        f"Decision → {ACTIONS[action]} (confidence {confidence:.0%})",
    ]
    if results_after is not None:
        decision_path.append(f"Reward computed → {reward:.3f}")
        decision_path.append("Q-table updated")

    return {
        "query":           query,
        "query_type":      query_type,
        "state_features":  features,
        "state":           state,
        "chosen_action":   ACTIONS[action],
        "action_index":    action,
        "confidence":      round(confidence, 4),
        "reward":          reward,
        "q_table_snapshot": q_snapshot,
        "decision_path":   decision_path,
    }


def get_q_table_snapshot() -> List[Dict]:
    """Return full Q-table for frontend display."""
    return [
        {
            "state": state,
            "action_vectorial": round(vals[0], 4),
            "action_graph":     round(vals[1], 4),
        }
        for state, vals in _q_table.items()
    ]


# Optional import guard
from typing import Optional