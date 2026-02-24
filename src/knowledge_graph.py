import pandas as pd
import numpy as np
import networkx as nx
import json
import os
import ast
from sklearn.metrics.pairwise import cosine_similarity

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED    = os.path.join(BASE_DIR, "data", "processed")
TRAIN_DIR    = os.path.join(BASE_DIR, "data", "train")

# ── Helper: safely parse list columns saved as strings in CSV ──────────────
def safe_parse_list(val):
    """Convert string representation of list back to actual list."""
    if isinstance(val, list):
        return val
    if pd.isna(val) or val == "" or val == "[]":
        return []
    try:
        return ast.literal_eval(val)
    except (ValueError, SyntaxError):
        return []

# ── Step 1: Load Data ──────────────────────────────────────────────────────
def load_data():
    news_df = pd.read_csv(os.path.join(PROCESSED, "news_enriched.csv"))
    news_df["entity_ids"]    = news_df["entity_ids"].apply(safe_parse_list)
    news_df["entity_labels"] = news_df["entity_labels"].apply(safe_parse_list)
    news_df["spacy_entities"]= news_df["spacy_entities"].apply(safe_parse_list)

    with open(os.path.join(PROCESSED, "user_profiles.json"), "r") as f:
        user_profiles = json.load(f)

    with open(os.path.join(PROCESSED, "user_interest_profiles.json"), "r") as f:
        user_interest_profiles = json.load(f)

    return news_df, user_profiles, user_interest_profiles

# ── Step 2: Build Knowledge Graph ─────────────────────────────────────────
def build_knowledge_graph(news_df, user_profiles, sample_users=5000):
    """
    Build a heterogeneous directed graph with nodes:
      - User nodes    : prefix 'U'
      - News nodes    : prefix 'N'
      - Entity nodes  : Wikidata Q-codes e.g. 'Q317521'
      - Category nodes: plain string e.g. 'sports'

    Edges:
      - User  → News     : USER_CLICKED
      - News  → Entity   : MENTIONS
      - News  → Category : BELONGS_TO
      - Entity→ Entity   : CO_OCCURS (two entities in same article)
    """
    G = nx.DiGraph()

    print("Building Knowledge Graph...")
    print(f"  Adding news nodes and edges...")

    # ── Add News, Entity, Category nodes and their edges ──────────────────
    for _, row in news_df.iterrows():
        nid      = row["news_id"]
        category = str(row["category"]).lower().strip()
        entities = row["entity_ids"]

        # Add news node
        G.add_node(nid, node_type="news", 
                   category=category, 
                   title=str(row["title"])[:100])

        # Add category node + edge
        G.add_node(category, node_type="category")
        G.add_edge(nid, category, edge_type="BELONGS_TO")

        # Add entity nodes + edges
        for eid in entities:
            G.add_node(eid, node_type="entity")
            G.add_edge(nid, eid, edge_type="MENTIONS")

        # Co-occurrence edges between entities in same article
        for i in range(len(entities)):
            for j in range(i + 1, len(entities)):
                G.add_edge(entities[i], entities[j], edge_type="CO_OCCURS")
                G.add_edge(entities[j], entities[i], edge_type="CO_OCCURS")

    print(f"  Adding user nodes and edges (sample: {sample_users} users)...")

    # ── Add User nodes (sample to keep graph manageable) ──────────────────
    # Using a sample avoids a 50k-user graph that's slow to query
    sampled_users = list(user_profiles.keys())[:sample_users]

    for uid in sampled_users:
        clicked = user_profiles[uid]
        G.add_node(uid, node_type="user")
        for nid in clicked:
            if G.has_node(nid):
                G.add_edge(uid, nid, edge_type="USER_CLICKED")

    print(f"\n✅ Knowledge Graph built")
    print(f"   Total nodes : {G.number_of_nodes():,}")
    print(f"   Total edges : {G.number_of_edges():,}")

    # Node type breakdown
    type_counts = {}
    for _, data in G.nodes(data=True):
        t = data.get("node_type", "unknown")
        type_counts[t] = type_counts.get(t, 0) + 1
    print(f"   Node types  : {type_counts}")

    return G, sampled_users

# ── Step 3: KG Similarity Score ───────────────────────────────────────────
def compute_kg_similarity(user_id, news_id, user_interest_profiles, 
                           news_df_lookup, G):
    """
    KG similarity = weighted combination of:
    1. Entity Jaccard overlap (user entities ∩ article entities)
    2. Category match bonus
    3. Entity embedding cosine similarity (if TransE vectors available)
    """
    # Get user's entity set
    if user_id not in user_interest_profiles:
        return 0.0
    user_entities = set(user_interest_profiles[user_id].get("entity_ids", []))
    user_cats     = user_interest_profiles[user_id].get("category_weights", {})

    # Get article's entity set
    if news_id not in news_df_lookup:
        return 0.0
    article       = news_df_lookup[news_id]
    article_ents  = set(article.get("entity_ids", []))
    article_cat   = str(article.get("category", "")).lower().strip()

    # 1. Jaccard similarity on entities
    if len(user_entities) == 0 and len(article_ents) == 0:
        jaccard = 0.0
    elif len(user_entities | article_ents) == 0:
        jaccard = 0.0
    else:
        intersection = len(user_entities & article_ents)
        union        = len(user_entities | article_ents)
        jaccard      = intersection / union

    # 2. Category match bonus
    cat_bonus = user_cats.get(article_cat, 0.0)

    # 3. Graph connectivity bonus
    # Check if user has 2-hop connection to article via shared entities
    graph_bonus = 0.0
    if G.has_node(user_id) and G.has_node(news_id):
        # Direct click = strongest signal
        if G.has_edge(user_id, news_id):
            graph_bonus = 1.0
        else:
            # Check shared entity neighbors
            user_neighbors    = set(G.successors(user_id))
            article_neighbors = set(G.predecessors(news_id))
            shared            = user_neighbors & article_neighbors
            graph_bonus       = min(len(shared) * 0.1, 0.5)

    # Weighted combination
    kg_score = (0.4 * jaccard) + (0.3 * cat_bonus) + (0.3 * graph_bonus)
    return round(kg_score, 4)


# ── Step 4: Batch KG Scoring ──────────────────────────────────────────────
def get_kg_scores_for_user(user_id, candidate_news_ids, 
                            user_interest_profiles, news_df_lookup, G):
    """Compute KG scores for a list of candidate articles for one user."""
    scores = {}
    for nid in candidate_news_ids:
        scores[nid] = compute_kg_similarity(
            user_id, nid, user_interest_profiles, news_df_lookup, G
        )
    return scores


# ── Step 5: Incremental KG Update (Dynamic KG feature) ───────────────────
def add_new_article_to_kg(G, news_id, category, entity_ids, title=""):
    """
    Incrementally add a new article node to the existing graph.
    This is the 'Dynamic KG' feature — no full rebuild needed.
    """
    if G.has_node(news_id):
        print(f"Article {news_id} already in graph")
        return G

    # Add news node
    G.add_node(news_id, node_type="news", 
               category=category.lower(), title=title[:100])

    # Add category edge
    G.add_node(category.lower(), node_type="category")
    G.add_edge(news_id, category.lower(), edge_type="BELONGS_TO")

    # Add entity edges
    for eid in entity_ids:
        G.add_node(eid, node_type="entity")
        G.add_edge(news_id, eid, edge_type="MENTIONS")

    # Co-occurrence edges
    for i in range(len(entity_ids)):
        for j in range(i + 1, len(entity_ids)):
            G.add_edge(entity_ids[i], entity_ids[j], edge_type="CO_OCCURS")
            G.add_edge(entity_ids[j], entity_ids[i], edge_type="CO_OCCURS")

    print(f"✅ New article {news_id} added to KG")
    print(f"   Category : {category}")
    print(f"   Entities : {entity_ids}")
    print(f"   New graph size: {G.number_of_nodes()} nodes, "
          f"{G.number_of_edges()} edges")
    return G


# ── Main: Build and Test ───────────────────────────────────────────────────
if __name__ == "__main__":
    import time

    # Load data
    print("Loading data...")
    news_df, user_profiles, user_interest_profiles = load_data()

    # Build news lookup dict for fast access
    news_df_lookup = {}
    for _, row in news_df.iterrows():
        news_df_lookup[row["news_id"]] = {
            "category"  : row["category"],
            "entity_ids": row["entity_ids"],
            "title"     : row["title"]
        }
    print(f"✅ News lookup built: {len(news_df_lookup)} articles")

    # Build KG
    start = time.time()
    G, sampled_users = build_knowledge_graph(news_df, user_profiles, 
                                              sample_users=5000)
    print(f"   Build time  : {time.time() - start:.1f}s")

    # Test KG similarity
    print("\n── KG Similarity Test ──")
    test_user = "U13740"
    test_articles = list(news_df["news_id"].head(20))

    scores = get_kg_scores_for_user(
        test_user, test_articles,
        user_interest_profiles, news_df_lookup, G
    )

    # Show top scoring articles
    top = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:5]
    print(f"User: {test_user}")
    print(f"Top category: {user_interest_profiles[test_user]['top_category']}")
    print(f"\nTop 5 KG-scored articles:")
    for nid, score in top:
        title = news_df_lookup[nid]["title"][:65]
        cat   = news_df_lookup[nid]["category"]
        print(f"  [{cat}] {title} → KG score: {score}")

    # Test incremental update
    print("\n── Dynamic KG Update Test ──")
    G = add_new_article_to_kg(
        G,
        news_id    = "N99999",
        category   = "technology",
        entity_ids = ["Q317521", "Q11696"],   # Elon Musk, Tesla
        title      = "Test: Elon Musk announces new Tesla model"
    )

    # Save graph
    print("\nSaving graph...")
    import pickle
    graph_path = os.path.join(PROCESSED, "knowledge_graph.pkl")
    with open(graph_path, "wb") as f:
        pickle.dump({
            "graph"         : G,
            "sampled_users" : sampled_users,
            "news_df_lookup": news_df_lookup
        }, f)
    print(f"✅ Graph saved to data/processed/knowledge_graph.pkl")

def load_transe_embeddings(train_dir=None):
    """
    Load Microsoft's pre-trained TransE entity embeddings from MIND.
    Format: Q-code TAB val1 TAB val2 ... TAB val100
    Returns: dict {entity_id: np.array(100,)}
    """
    if train_dir is None:
        train_dir = os.path.join(BASE_DIR, "data", "train")

    path = os.path.join(train_dir, "entity_embedding.vec")
    if not os.path.exists(path):
        print(f"⚠️  entity_embedding.vec not found at {path}")
        return {}

    entity_embeddings = {}
    skipped = 0

    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                skipped += 1
                continue
            entity_id = parts[0].strip()
            try:
                vector = np.array(parts[1:], dtype=np.float32)
                if len(vector) == 100:
                    # L2 normalize for cosine similarity via dot product
                    norm = np.linalg.norm(vector)
                    if norm > 0:
                        vector = vector / norm
                    entity_embeddings[entity_id] = vector
                else:
                    skipped += 1
            except ValueError:
                skipped += 1
                continue

    print(f"✅ TransE embeddings loaded")
    print(f"   Entities loaded : {len(entity_embeddings)}")
    print(f"   Skipped         : {skipped}")
    print(f"   Vector dim      : 100")
    return entity_embeddings


if __name__ == "__main__":
    # Quick test
    transe = load_transe_embeddings()
    if transe:
        sample = list(transe.items())[:3]
        for eid, vec in sample:
            print(f"  {eid} → shape: {vec.shape}, norm: {np.linalg.norm(vec):.4f}")
