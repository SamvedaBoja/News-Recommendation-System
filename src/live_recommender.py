# src/live_recommender.py

import numpy as np
from collections import defaultdict
from live_store import (
    load_live_articles, encode_and_store, init_db,
    BGE_MODEL, extract_entities,
)
from live_fetcher import fetch_live_articles


class LiveRecommender:

    def __init__(self):
        init_db()
        self.articles   = []
        self.embeddings = np.zeros((0, 768))

    def refresh(self, categories=None):
        """Fetch fresh articles from NewsAPI and re-encode."""
        if categories is None:
            categories = ["general", "business", "sports",
                          "technology", "entertainment", "health"]
        all_articles = []
        for cat in categories:
            try:
                fetched = fetch_live_articles(category=cat, page_size=10)
                all_articles.extend(fetched)
            except Exception as e:
                print(f"NewsAPI error for {cat}: {e}")

        if all_articles:
            encode_and_store(all_articles)

        self.articles, self.embeddings = load_live_articles()
        print(f"Live: {len(self.articles)} articles loaded")

    def recommend_for_interests(self,
                                interests: list,
                                top_k    : int = 10,
                                category : str = "") -> list:
        """
        interests = list of topic strings entered by user
        e.g. ["cricket", "technology", "stock market"]
        Always returns a diversity-balanced result set.
        """
        if not self.articles or self.embeddings.shape[0] == 0:
            return []

        # Encode user interests as a query vector
        interest_text = " ".join(interests)
        query_vec     = BGE_MODEL.encode(interest_text, normalize_embeddings=True)

        # Extract named entities from interest text for KG scoring
        user_entities = {
            e["label"].lower()
            for e in extract_entities(interest_text)
        }

        # Semantic scores
        sem_scores = self.embeddings @ query_vec

        # Trend scores
        trend_scores = np.array([a["trend_score"] for a in self.articles])

        # KG scores (entity Jaccard + category match)
        kg_scores = np.array([
            _compute_kg_score(user_entities, interests, a)
            for a in self.articles
        ])

        # Fusion: 60% semantic + 30% KG + 10% trend
        final_scores = (0.60 * sem_scores
                      + 0.30 * kg_scores
                      + 0.10 * trend_scores)

        # Category filter (optional)
        indices = list(range(len(self.articles)))
        if category:
            indices = [i for i in indices
                       if self.articles[i]["category"] == category]

        # Rank and take a larger pool for diversity reranking
        pool_k  = min(len(indices), top_k * 5)
        top_idx = sorted(indices, key=lambda i: final_scores[i], reverse=True)[:pool_k]

        results = []
        for i in top_idx:
            art = self.articles[i]
            results.append({
                **art,
                "final_score"   : round(float(final_scores[i]), 4),
                "semantic_score": round(float(sem_scores[i]),    4),
                "kg_score"      : round(float(kg_scores[i]),     4),
                "trend_score"   : round(float(trend_scores[i]),  4),
                "explanation"   : self._explain(interests, user_entities, art),
            })

        # Always apply diversity — no toggle needed
        if not category:
            results = _diversify(results, top_k)
        else:
            results = results[:top_k]

        return results

    def trending(self, top_k: int = 10) -> list:
        if not self.articles:
            return []
        return sorted(self.articles,
                      key=lambda a: a["trend_score"],
                      reverse=True)[:top_k]

    def _explain(self, interests: list, user_entities: set, article: dict) -> str:
        title_lower = article["title"].lower()

        # Check entity match first
        art_entities = {e["label"].lower() for e in article.get("entities", [])}
        overlap = user_entities & art_entities
        if overlap:
            entity = next(iter(overlap)).title()
            return f"Matches your interest in '{entity}'"

        # Keyword match in title
        for interest in interests:
            if interest.lower() in title_lower:
                return f"Matches your interest in '{interest}'"

        # Category match
        for interest in interests:
            if (interest.lower() in article["category"].lower()
                    or article["category"].lower() in interest.lower()):
                return f"Popular in {article['category']}"

        return "Trending right now"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _compute_kg_score(user_entities: set, interests: list, article: dict) -> float:
    """Entity Jaccard overlap + category match, each weighted 50%."""
    art_entities = {e["label"].lower() for e in article.get("entities", [])}
    union        = user_entities | art_entities
    jaccard      = len(user_entities & art_entities) / len(union) if union else 0.0

    cat_match = any(
        interest.lower() in article["category"].lower()
        or article["category"].lower() in interest.lower()
        for interest in interests
    )
    cat_w = 1.0 if cat_match else 0.0

    return 0.50 * jaccard + 0.50 * cat_w


def _diversify(recs: list, top_k: int) -> list:
    """Round-robin category reranking; no single category exceeds 35% of top_k."""
    by_cat = defaultdict(list)
    for r in recs:
        by_cat[r["category"]].append(r)

    max_per_cat  = max(1, int(top_k * 0.35))
    categories   = sorted(by_cat, key=lambda c: by_cat[c][0]["final_score"], reverse=True)
    cat_ptr      = {cat: 0 for cat in categories}
    diverse_recs = []

    while len(diverse_recs) < top_k:
        added = False
        for cat in categories:
            if len(diverse_recs) >= top_k:
                break
            if sum(1 for r in diverse_recs if r["category"] == cat) >= max_per_cat:
                continue
            ptr = cat_ptr[cat]
            if ptr < len(by_cat[cat]):
                diverse_recs.append(by_cat[cat][ptr])
                cat_ptr[cat] += 1
                added = True
        if not added:
            break

    return diverse_recs[:top_k]
