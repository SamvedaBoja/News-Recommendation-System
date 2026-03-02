from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from collections import defaultdict
import sys
import os
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(BASE_DIR, "src"))

from recommender import HybridNewsRecommender

app = FastAPI(title="NewsIQ API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

recommender = HybridNewsRecommender()

@app.on_event("startup")
def startup():
    recommender.load()
    print("✅ Recommender loaded")


# ── Helper ────────────────────────────────────────────────────────────────────

def diversify(recs: list, top_k: int) -> list:
    """
    Round-robin category reranking.
    Guarantees no single category exceeds 35% of top-K results.
    Works by grouping candidates by category then interleaving them.
    """
    by_category = defaultdict(list)
    for r in recs:
        by_category[r["category"]].append(r)

    # Sort each category bucket by score descending
    for cat in by_category:
        by_category[cat].sort(key=lambda x: x["final_score"], reverse=True)

    max_per_cat  = max(1, int(top_k * 0.35))
    categories   = sorted(by_category.keys(),
                          key=lambda c: by_category[c][0]["final_score"],
                          reverse=True)
    cat_pointers = {cat: 0 for cat in categories}
    diverse_recs = []

    # Round-robin: one article per category per round
    while len(diverse_recs) < top_k:
        added = False
        for cat in categories:
            if len(diverse_recs) >= top_k:
                break
            already = sum(1 for r in diverse_recs if r["category"] == cat)
            if already >= max_per_cat:
                continue
            ptr = cat_pointers[cat]
            if ptr < len(by_category[cat]):
                diverse_recs.append(by_category[cat][ptr])
                cat_pointers[cat] += 1
                added = True
        if not added:
            break

    return diverse_recs[:top_k]


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"status": "ok", "message": "NewsIQ API running"}


@app.get("/stats")
def get_stats():
    return {
        "total_articles" : len(recommender.news_df_lookup),
        "users_indexed"  : len(recommender.user_vectors),
        "kg_nodes"       : recommender.kg_graph.number_of_nodes(),
        "kg_edges"       : recommender.kg_graph.number_of_edges(),
        "transe_entities": len(recommender.transe_embeddings),
        "evaluation": {
            "hybrid_auc" : 0.6461,
            "kg_auc"     : 0.6552,
            "hybrid_mrr" : 0.3677,
            "ndcg5"      : 0.3518
        }
    }


@app.get("/recommend/{user_id}")
def recommend(
    user_id : str,
    top_k   : int  = Query(default=10, ge=1, le=100),  # raised to 100
    category: str  = Query(default=""),
    diverse : bool = Query(default=False)
):
    if user_id not in recommender.user_profiles and \
       user_id not in recommender.user_interest_profiles:
        raise HTTPException(
            status_code=404, detail=f"User {user_id} not found"
        )

    # Fetch a larger pool when diverse mode is on so
    # diversify() has candidates from multiple categories to work with
    fetch_k = min(top_k * 10, 100) if diverse else top_k
    recs    = recommender.recommend(user_id, top_k=fetch_k)

    if category:
        recs = [r for r in recs if r["category"] == category.lower()]

    if diverse and not category:
        recs = diversify(recs, top_k)

    # Attach abstracts
    for r in recs:
        art      = recommender.news_df_lookup.get(r["news_id"], {})
        abstract = art.get("abstract", "")
        r["abstract"] = (
            abstract if abstract and abstract.lower() != "nan" else ""
        )

    return {"user_id": user_id, "recommendations": recs[:top_k]}


@app.get("/user/{user_id}/history")
def get_history(
    user_id: str,
    limit  : int = Query(default=5, ge=1, le=20)
):
    if user_id not in recommender.user_profiles:
        raise HTTPException(status_code=404, detail="User not found")

    history = recommender.user_profiles[user_id][-limit:]
    result  = []
    for nid in history:
        if nid in recommender.news_df_lookup:
            art = recommender.news_df_lookup[nid]
            result.append({
                "news_id" : nid,
                "title"   : art["title"],
                "category": art["category"],
                "abstract": art.get("abstract", "")
            })
    return {"user_id": user_id, "history": result}


@app.get("/user/{user_id}/profile")
def get_profile(user_id: str):
    if user_id not in recommender.user_interest_profiles:
        raise HTTPException(status_code=404, detail="User not found")
    profile = recommender.user_interest_profiles[user_id]
    return {
        "user_id"         : user_id,
        "top_category"    : profile.get("top_category", "unknown"),
        "category_weights": profile.get("category_weights", {}),
        "entity_count"    : len(profile.get("entity_ids", []))
    }


@app.get("/trending")
def trending(top_k: int = Query(default=10, ge=1, le=20)):
    recs = recommender.recommend_cold_start(top_k=top_k)
    for r in recs:
        art      = recommender.news_df_lookup.get(r["news_id"], {})
        abstract = art.get("abstract", "")
        r["abstract"] = (
            abstract if abstract and abstract.lower() != "nan" else ""
        )
    return {"trending": recs}


@app.get("/similar/{news_id}")
def similar(
    news_id: str,
    top_k  : int = Query(default=10, ge=1, le=20)
):
    if news_id not in recommender.news_df_lookup:
        raise HTTPException(
            status_code=404, detail=f"News {news_id} not found"
        )

    idx = recommender.news_id_to_idx.get(news_id)
    if idx is None:
        raise HTTPException(
            status_code=404, detail="No embedding for this article"
        )

    query_vec = recommender.article_embeddings[idx]
    sims      = recommender.article_embeddings @ query_vec
    sims      = np.maximum(sims, 0.0)
    top_idx   = np.argsort(-sims)

    results = []
    shown   = 0
    for i in top_idx:
        if shown >= top_k:
            break
        nid = recommender.news_id_list[i]
        if nid == news_id:
            continue
        art = recommender.news_df_lookup.get(nid, {})
        if not art:
            continue
        abstract = art.get("abstract", "")
        results.append({
            "news_id"   : nid,
            "title"     : art["title"],
            "category"  : art["category"],
            "abstract"  : abstract if abstract and abstract.lower() != "nan" else "",
            "similarity": round(float(sims[i]), 4)
        })
        shown += 1

    query_art = recommender.news_df_lookup[news_id]
    return {
        "query": {
            "news_id" : news_id,
            "title"   : query_art["title"],
            "category": query_art["category"],
            "abstract": query_art.get("abstract", "")
        },
        "similar": results
    }


@app.get("/categories")
def get_categories():
    cats = sorted(set(
        v["category"] for v in recommender.news_df_lookup.values()
    ))
    return {"categories": cats}
