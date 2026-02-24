import pandas as pd
import numpy as np
import json
import os
import pickle
import ast
import sys

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED = os.path.join(BASE_DIR, "data", "processed")

# Add src to path for knowledge_graph import
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from knowledge_graph import load_transe_embeddings

# ── Weights (w1 + w2 + w3 = 1.0) ──────────────────────────────────────────
W_KG       = 0.4
W_SEMANTIC = 0.6
W_TREND    = 0.0

# ── Helper ─────────────────────────────────────────────────────────────────
def safe_parse(val):
    if isinstance(val, list): return val
    if pd.isna(val) or str(val) in ("", "[]"): return []
    try: return ast.literal_eval(str(val))
    except: return []


# ══════════════════════════════════════════════════════════════════════════
# CLASS: HybridNewsRecommender
# ══════════════════════════════════════════════════════════════════════════
class HybridNewsRecommender:

    def __init__(self):
        self.news_df                = None
        self.news_df_lookup         = {}
        self.user_profiles          = {}
        self.user_interest_profiles = {}
        self.article_embeddings     = None
        self.news_id_to_idx         = {}
        self.news_id_list           = []
        self.user_vectors           = {}
        self.user_vector_ids        = []
        self.kg_graph               = None
        self.sampled_users          = []
        self.time_decay_scores      = {}
        self.entity_trend_scores    = {}
        self.entity_label_map       = {}
        self.transe_embeddings      = {}
        self.is_loaded              = False

    # ── Load All Modules ───────────────────────────────────────────────────
    def load(self):
        print("Loading HybridNewsRecommender...")

        # 1. News data
        self.news_df = pd.read_csv(os.path.join(PROCESSED, "news_enriched.csv"))
        self.news_df["entity_ids"]    = self.news_df["entity_ids"].apply(safe_parse)
        self.news_df["entity_labels"] = self.news_df["entity_labels"].apply(safe_parse)
        self.news_df_lookup = {
            row["news_id"]: {
                "category"     : str(row["category"]).lower(),
                "entity_ids"   : row["entity_ids"],
                "entity_labels": row["entity_labels"],
                "title"        : str(row["title"]),
                "abstract"     : str(row.get("abstract", ""))
            }
            for _, row in self.news_df.iterrows()
        }
        print(f"  ✅ News data      : {len(self.news_df_lookup)} articles")

        # 2. User profiles
        with open(os.path.join(PROCESSED, "user_profiles.json")) as f:
            self.user_profiles = json.load(f)
        with open(os.path.join(PROCESSED, "user_interest_profiles.json")) as f:
            self.user_interest_profiles = json.load(f)
        print(f"  ✅ User profiles  : {len(self.user_profiles)} users")

        # 3. Semantic embeddings
        self.article_embeddings = np.load(
            os.path.join(PROCESSED, "article_embeddings.npy")
        )
        with open(os.path.join(PROCESSED, "news_id_list.json")) as f:
            self.news_id_list = json.load(f)
        self.news_id_to_idx = {
            nid: idx for idx, nid in enumerate(self.news_id_list)
        }
        user_vector_array = np.load(
            os.path.join(PROCESSED, "user_vectors.npy")
        )
        with open(os.path.join(PROCESSED, "user_vector_ids.json")) as f:
            self.user_vector_ids = json.load(f)
        self.user_vectors = {
            uid: user_vector_array[i]
            for i, uid in enumerate(self.user_vector_ids)
        }
        print(f"  ✅ Embeddings     : {self.article_embeddings.shape}")
        print(f"  ✅ User vectors   : {len(self.user_vectors)}")

        # 4. Knowledge Graph
        with open(os.path.join(PROCESSED, "knowledge_graph.pkl"), "rb") as f:
            kg_data = pickle.load(f)
        self.kg_graph      = kg_data["graph"]
        self.sampled_users = kg_data["sampled_users"]
        print(f"  ✅ KG graph       : {self.kg_graph.number_of_nodes()} nodes")

        # 5. Trend data
        with open(os.path.join(PROCESSED, "trend_data.json")) as f:
            trend_data = json.load(f)
        self.time_decay_scores   = trend_data["time_decay_scores"]
        self.entity_trend_scores = trend_data["entity_trend_scores"]
        print(f"  ✅ Trend data     : {len(self.time_decay_scores)} scores")

        # 6. Pre-build entity_id → label mapping ONCE
        self.entity_label_map = {}
        for _, row in self.news_df.iterrows():
            eids    = row["entity_ids"]
            elabels = row["entity_labels"]
            if isinstance(eids, list) and isinstance(elabels, list):
                for eid, elabel in zip(eids, elabels):
                    if eid not in self.entity_label_map and elabel:
                        self.entity_label_map[eid] = elabel
        print(f"  ✅ Entity labels  : {len(self.entity_label_map)} mapped")

        # 7. TransE embeddings
        self.transe_embeddings = load_transe_embeddings()
        print(f"  ✅ TransE         : {len(self.transe_embeddings)} entities")

        self.is_loaded = True
        print("\n✅ All modules loaded — recommender ready\n")

    # ── Semantic Score (vectorized) ────────────────────────────────────────
    def _semantic_score(self, user_id, candidate_ids):
        if user_id not in self.user_vectors:
            return {nid: 0.0 for nid in candidate_ids}

        user_vec  = self.user_vectors[user_id].reshape(1, -1)
        indices   = [self.news_id_to_idx[nid]
                     for nid in candidate_ids
                     if nid in self.news_id_to_idx]
        valid_ids = [nid for nid in candidate_ids
                     if nid in self.news_id_to_idx]

        if not indices:
            return {nid: 0.0 for nid in candidate_ids}

        candidate_matrix = self.article_embeddings[indices]
        scores_array     = np.dot(candidate_matrix, user_vec.T).flatten()
        scores_array     = np.maximum(scores_array, 0.0)

        scores = {nid: 0.0 for nid in candidate_ids}
        for nid, score in zip(valid_ids, scores_array):
            scores[nid] = round(float(score), 4)
        return scores

    # ── KG Score (with TransE) ─────────────────────────────────────────────
    def _kg_score(self, user_id, candidate_ids):
        if user_id not in self.user_interest_profiles:
            return {nid: 0.0 for nid in candidate_ids}

        user_profile  = self.user_interest_profiles[user_id]
        user_entities = set(user_profile.get("entity_ids", []))
        cat_weights   = user_profile.get("category_weights", {})

        # Pre-compute user TransE mean vector ONCE per user call
        user_transe_vecs = [
            self.transe_embeddings[e]
            for e in user_entities
            if e in self.transe_embeddings
        ]
        if user_transe_vecs:
            user_transe_mean = np.mean(user_transe_vecs, axis=0)
            norm = np.linalg.norm(user_transe_mean)
            user_transe_mean = user_transe_mean / norm if norm > 0 \
                               else user_transe_mean
        else:
            user_transe_mean = None

        scores = {}
        for nid in candidate_ids:
            if nid not in self.news_df_lookup:
                scores[nid] = 0.0
                continue

            article      = self.news_df_lookup[nid]
            art_entities = set(article.get("entity_ids", []))
            art_cat      = article.get("category", "")

            # Signal 1: Jaccard entity overlap
            union   = user_entities | art_entities
            jaccard = len(user_entities & art_entities) / len(union) \
                      if len(union) > 0 else 0.0

            # Signal 2: Category weight
            cat_bonus = cat_weights.get(art_cat, 0.0)

            # Signal 3: Graph connectivity
            graph_bonus = 0.0
            if self.kg_graph.has_node(user_id) and \
               self.kg_graph.has_node(nid):
                if self.kg_graph.has_edge(user_id, nid):
                    graph_bonus = 1.0
                else:
                    u_nb    = set(self.kg_graph.successors(user_id))
                    a_nb    = set(self.kg_graph.predecessors(nid))
                    graph_bonus = min(len(u_nb & a_nb) * 0.1, 0.5)

            # Signal 4: TransE semantic similarity
            transe_sim = 0.0
            if user_transe_mean is not None:
                art_transe_vecs = [
                    self.transe_embeddings[e]
                    for e in art_entities
                    if e in self.transe_embeddings
                ]
                if art_transe_vecs:
                    art_transe_mean = np.mean(art_transe_vecs, axis=0)
                    norm = np.linalg.norm(art_transe_mean)
                    if norm > 0:
                        art_transe_mean = art_transe_mean / norm
                    transe_sim = float(
                        np.dot(user_transe_mean, art_transe_mean)
                    )
                    transe_sim = max(transe_sim, 0.0)

            scores[nid] = round(
                (0.30 * jaccard) +
                (0.25 * cat_bonus) +
                (0.20 * graph_bonus) +
                (0.25 * transe_sim),
                4
            )
        return scores

    # ── Trend Score ────────────────────────────────────────────────────────
    def _trend_score(self, user_id, candidate_ids):
        cat_weights = {}
        if user_id in self.user_interest_profiles:
            cat_weights = self.user_interest_profiles[user_id].get(
                "category_weights", {}
            )
        scores = {}
        for nid in candidate_ids:
            if nid not in self.news_df_lookup:
                scores[nid] = 0.0
                continue

            article      = self.news_df_lookup[nid]
            art_cat      = article.get("category", "")
            art_entities = article.get("entity_ids", [])

            cat_weight   = cat_weights.get(art_cat, 0.05)
            time_decay   = self.time_decay_scores.get(nid, 0.1)
            entity_boost = float(np.mean([
                self.entity_trend_scores.get(eid, 0.0)
                for eid in art_entities
            ])) if art_entities else 0.0

            scores[nid] = round(
                (0.6 * cat_weight * time_decay) + (0.4 * entity_boost), 4
            )
        return scores

    # ── Explanation Generator ──────────────────────────────────────────────
    def _generate_explanation(self, user_id, news_id):
        if user_id not in self.user_interest_profiles:
            return "Recommended based on trending topics."

        user_profile  = self.user_interest_profiles[user_id]
        user_entities = set(user_profile.get("entity_ids", []))

        if news_id not in self.news_df_lookup:
            return "Recommended based on your interests."

        article      = self.news_df_lookup[news_id]
        art_entities = article.get("entity_ids", [])
        art_cat      = article.get("category", "")

        shared_labels = []
        for eid in art_entities:
            if eid in user_entities and eid in self.entity_label_map:
                shared_labels.append(self.entity_label_map[eid])

        if shared_labels:
            if len(shared_labels) == 1:
                return (f"Recommended because you read about "
                        f"'{shared_labels[0]}', who is mentioned in this article.")
            else:
                entities_str = ", ".join(f"'{e}'" for e in shared_labels[:3])
                return (f"Recommended because this article mentions "
                        f"{entities_str}, topics from your reading history.")

        top_cat = user_profile.get("top_category", "")
        if top_cat and top_cat == art_cat:
            return (f"Recommended because you frequently read "
                    f"'{top_cat}' articles.")

        return "Recommended based on your reading interests and current trends."

    # ── Core Recommendation Function ───────────────────────────────────────
    def recommend(self, user_id, top_k=10, exclude_read=True):
        if not self.is_loaded:
            raise RuntimeError("Call load() before recommend()")

        all_news_ids = self.news_id_list.copy()
        if exclude_read and user_id in self.user_profiles:
            read_set   = set(self.user_profiles[user_id])
            candidates = [nid for nid in all_news_ids if nid not in read_set]
        else:
            candidates = all_news_ids

        sem_scores   = self._semantic_score(user_id, candidates)
        top_500      = sorted(sem_scores, key=sem_scores.get, reverse=True)[:500]
        kg_scores    = self._kg_score(user_id, top_500)
        trend_scores = self._trend_score(user_id, top_500)

        results = []
        for nid in top_500:
            s_score = sem_scores.get(nid, 0.0)
            k_score = kg_scores.get(nid, 0.0)
            t_score = trend_scores.get(nid, 0.0)

            # Hybrid: only KG + Semantic — Trend kept for display only
            final = (W_KG * k_score) + (W_SEMANTIC * s_score)

            if nid in self.news_df_lookup:
                article = self.news_df_lookup[nid]
                results.append({
                    "news_id"        : nid,
                    "title"          : article["title"],
                    "category"       : article["category"],
                    "final_score"    : round(final, 4),
                    "semantic_score" : s_score,
                    "kg_score"       : k_score,
                    "trend_score"    : t_score,
                    "explanation"    : self._generate_explanation(user_id, nid)
                })

        results.sort(key=lambda x: x["final_score"], reverse=True)
        return results[:top_k]



    # ── Cold Start ─────────────────────────────────────────────────────────
    def recommend_cold_start(self, top_k=10):
        results = []
        for nid, decay in sorted(
            self.time_decay_scores.items(),
            key=lambda x: x[1], reverse=True
        )[:top_k]:
            if nid in self.news_df_lookup:
                article = self.news_df_lookup[nid]
                results.append({
                    "news_id"    : nid,
                    "title"      : article["title"],
                    "category"   : article["category"],
                    "final_score": round(decay, 4),
                    "explanation": "Trending right now — popular with all readers."
                })
        return results


# ── Main: Test the Full Pipeline ───────────────────────────────────────────
if __name__ == "__main__":
    import time

    recommender = HybridNewsRecommender()
    recommender.load()

    test_user = "U13740"
    print(f"{'='*60}")
    print(f"RECOMMENDATIONS FOR : {test_user}")
    print(f"Top category        : "
          f"{recommender.user_interest_profiles[test_user]['top_category']}")
    print(f"{'='*60}")

    t0   = time.time()
    recs = recommender.recommend(test_user, top_k=5)
    print(f"⏱  Completed in {time.time()-t0:.2f}s\n")

    for i, r in enumerate(recs, 1):
        print(f"{i}. [{r['category'].upper()}] {r['title'][:65]}")
        print(f"   Final: {r['final_score']} | "
              f"Sem: {r['semantic_score']} | "
              f"KG: {r['kg_score']} | "
              f"Trend: {r['trend_score']}")
        print(f"   💡 {r['explanation']}\n")

    test_user2 = list(recommender.user_profiles.keys())[100]
    top_cat2   = recommender.user_interest_profiles.get(
        test_user2, {}
    ).get("top_category", "unknown")
    print(f"{'='*60}")
    print(f"RECOMMENDATIONS FOR : {test_user2}")
    print(f"Top category        : {top_cat2}")
    print(f"{'='*60}")

    recs2 = recommender.recommend(test_user2, top_k=3)
    for i, r in enumerate(recs2, 1):
        print(f"{i}. [{r['category'].upper()}] {r['title'][:65]}")
        print(f"   Final: {r['final_score']} | "
              f"Sem: {r['semantic_score']} | "
              f"KG: {r['kg_score']} | "
              f"Trend: {r['trend_score']}")
        print(f"   💡 {r['explanation']}\n")

    print(f"{'='*60}")
    print("COLD START RECOMMENDATIONS (new user, no history)")
    print(f"{'='*60}")
    cold_recs = recommender.recommend_cold_start(top_k=3)
    for i, r in enumerate(cold_recs, 1):
        print(f"{i}. [{r['category'].upper()}] {r['title'][:65]}")
        print(f"   💡 {r['explanation']}\n")

    print(f"{'='*60}")
    print("SCORE DISTRIBUTION (top 10 recs for U13740)")
    print(f"{'='*60}")
    all_recs   = recommender.recommend("U13740", top_k=10)
    sem_vals   = [r["semantic_score"] for r in all_recs]
    kg_vals    = [r["kg_score"]       for r in all_recs]
    trend_vals = [r["trend_score"]    for r in all_recs]
    final_vals = [r["final_score"]    for r in all_recs]
    print(f"  Semantic — mean: {np.mean(sem_vals):.4f}  "
          f"max: {np.max(sem_vals):.4f}")
    print(f"  KG       — mean: {np.mean(kg_vals):.4f}  "
          f"max: {np.max(kg_vals):.4f}")
    print(f"  Trend    — mean: {np.mean(trend_vals):.4f}  "
          f"max: {np.max(trend_vals):.4f}")
    print(f"  Final    — mean: {np.mean(final_vals):.4f}  "
          f"max: {np.max(final_vals):.4f}")
